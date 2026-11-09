"""PointCloud Downsampler Engine.

파일 로딩, 균일 보셀 대표점 선택, SOR, 전역 균일 소거, 내보내기 기능을 제공한다.
기존 밀도 적응형 랜덤 소거는 재현성과 공간 균일성이 떨어져 제거했다.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import open3d as o3d
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PointCloudTable:
    """XYZ와 추가 수치 컬럼을 함께 보관하는 포인트클라우드 테이블."""

    values: np.ndarray
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("포인트 데이터는 2차원 배열이어야 합니다.")
        if values.shape[1] < 3:
            raise ValueError("포인트 데이터는 최소 3개 컬럼(X, Y, Z)이 필요합니다.")
        if len(self.columns) != values.shape[1]:
            raise ValueError("컬럼 수와 데이터 폭이 일치하지 않습니다.")
        object.__setattr__(self, "values", values)

    @classmethod
    def empty(cls) -> "PointCloudTable":
        return cls(np.empty((0, 3), dtype=np.float32), ("X", "Y", "Z"))

    @property
    def count(self) -> int:
        return int(self.values.shape[0])

    @property
    def xyz(self) -> np.ndarray:
        return self.values[:, :3]

    def select_rows(self, row_indices: np.ndarray | list[int]) -> "PointCloudTable":
        indices = np.asarray(row_indices, dtype=np.int64)
        return PointCloudTable(self.values[indices], self.columns)


# ----------- 설정 -----------


@dataclass
class DownsampleConfig:
    """다운샘플링 파라미터."""

    voxel_size: float = 3.0
    sor_neighbors: int = 20
    sor_std_ratio: float = 2.0
    reduction_percent: int = 0  # 0 = 비활성, 1~99
    chunk_size: int = 500_000

    def validate(self) -> list[str]:
        """파라미터 유효성 검사."""
        errors: list[str] = []
        if self.voxel_size <= 0:
            errors.append("Voxel Size는 0보다 커야 합니다.")
        if self.sor_neighbors < 1:
            errors.append("SOR 이웃 수는 1 이상이어야 합니다.")
        if self.sor_std_ratio <= 0:
            errors.append("SOR 표준편차 비율은 0보다 커야 합니다.")
        if not (0 <= self.reduction_percent < 100):
            errors.append("감소 비율은 0~99% 범위여야 합니다.")
        return errors


# ----------- 파일 로딩 -----------

SUPPORTED_EXTENSIONS = {".txt", ".csv", ".xlsx", ".xls"}


def load_pointcloud(
    file_path: str,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PointCloudTable:
    """포인트클라우드 파일을 로드해 테이블로 반환한다.

    지원 형식: TXT, CSV, XLSX, XLS.
    첫 세 컬럼은 XYZ로 사용하고, 뒤의 수치 컬럼은 함께 보존한다.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    _emit(progress_callback, "파일 로딩 중...", 0.0)

    if ext in (".xlsx", ".xls"):
        table = _load_excel_pointcloud(file_path)
    else:
        table = _load_delimited_pointcloud(file_path, chunk_size=500_000, progress_callback=progress_callback)

    if table.count == 0:
        raise ValueError("유효한 3D 좌표 데이터가 없습니다.")

    _emit(progress_callback, f"로딩 완료: {table.count:,} 포인트", 1.0)
    logger.info(
        "파일 로드 완료: %s (%s 포인트, %d 컬럼)",
        file_path,
        f"{table.count:,}",
        len(table.columns),
    )
    return table


def _load_excel_pointcloud(file_path: str) -> PointCloudTable:
    preview = pd.read_excel(file_path, header=None, nrows=1)
    has_header = _looks_like_header(preview.iloc[0, :3].tolist()) if not preview.empty else False
    df = pd.read_excel(file_path, header=0 if has_header else None)
    return _dataframe_to_pointcloud(df, has_header=has_header)


def _load_delimited_pointcloud(
    file_path: str,
    chunk_size: int,
    progress_callback: Callable[[str, float], None] | None = None,
) -> PointCloudTable:
    has_header = _detect_delimited_header(file_path)
    delimiter = _detect_delimiter(file_path)
    chunks: list[np.ndarray] = []
    column_names: tuple[str, ...] | None = None
    total_size = max(os.path.getsize(file_path), 1)
    read_bytes = 0

    with pd.read_csv(
        file_path,
        chunksize=chunk_size,
        header=0 if has_header else None,
        sep=delimiter,
        engine="c" if delimiter in {",", "\t", ";"} else "python",
        on_bad_lines="skip",
    ) as reader:
        for chunk_idx, chunk in enumerate(reader, start=1):
            numeric_chunk, chunk_columns = _numeric_chunk_to_array(chunk, has_header=has_header)
            if column_names is None:
                column_names = chunk_columns
            if len(numeric_chunk) > 0:
                chunks.append(numeric_chunk)

            read_bytes += max(len(chunk), 1) * max(chunk.shape[1], 3) * 16
            _emit(
                progress_callback,
                f"로딩 중... ({chunk_idx} 청크)",
                min(read_bytes / total_size, 0.9),
            )

    if not chunks or column_names is None:
        raise ValueError("유효한 3D 좌표 데이터가 없습니다.")

    values = np.vstack(chunks)
    return _build_pointcloud_table(values, column_names)


def _dataframe_to_pointcloud(df: pd.DataFrame, has_header: bool) -> PointCloudTable:
    values, columns = _numeric_chunk_to_array(df, has_header=has_header)
    if len(values) == 0:
        raise ValueError("유효한 3D 좌표 데이터가 없습니다.")
    return _build_pointcloud_table(values, columns)


def _numeric_chunk_to_array(
    chunk: pd.DataFrame,
    has_header: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    numeric = chunk.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 3:
        raise ValueError("입력 파일에는 최소 3개 컬럼(X, Y, Z)이 필요합니다.")

    valid_rows = numeric.iloc[:, :3].notna().all(axis=1)
    numeric = numeric.loc[valid_rows]

    if has_header:
        columns = tuple(_clean_column_name(col, idx) for idx, col in enumerate(chunk.columns))
    else:
        columns = _default_columns(chunk.shape[1])

    return numeric.to_numpy(dtype=np.float32, copy=True), columns


def _build_pointcloud_table(values: np.ndarray, columns: tuple[str, ...]) -> PointCloudTable:
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("입력 파일에는 최소 3개 컬럼(X, Y, Z)이 필요합니다.")

    valid_columns = ~np.all(np.isnan(values), axis=0)
    valid_columns[:3] = True

    filtered_values = values[:, valid_columns]
    filtered_columns = tuple(
        column
        for column, keep in zip(columns, valid_columns, strict=False)
        if keep
    )
    return PointCloudTable(filtered_values.astype(np.float32, copy=False), filtered_columns)


def _detect_delimited_header(file_path: str) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as file:
            first_line = file.readline().strip()
    except OSError:
        return False

    if not first_line:
        return False
    values = first_line.replace(",", " ").replace("\t", " ").split()
    return _looks_like_header(values[:3])


def _detect_delimiter(file_path: str) -> str | None:
    try:
        with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as file:
            first_line = file.readline().strip()
    except OSError:
        return None

    if "\t" in first_line and "," not in first_line:
        return "\t"
    if "," in first_line and "\t" not in first_line:
        return ","
    if ";" in first_line and "," not in first_line and "\t" not in first_line:
        return ";"
    return None


def _looks_like_header(values: list[object]) -> bool:
    for value in values:
        try:
            float(str(value).strip())
        except (TypeError, ValueError):
            return True
    return False


def _clean_column_name(column: object, index: int) -> str:
    text = str(column).strip()
    if text:
        return text
    return _default_columns(index + 1)[index]


def _default_columns(width: int) -> tuple[str, ...]:
    columns = ["X", "Y", "Z"]
    if width <= 3:
        return tuple(columns[:width])
    columns.extend(f"ATTR_{index}" for index in range(4, width + 1))
    return tuple(columns)


# ----------- 처리 파이프라인 -----------


@dataclass
class ProcessResult:
    """처리 결과."""

    original_count: int = 0
    after_voxel: int = 0
    after_sor: int = 0
    after_uniform: int = 0
    final_table: PointCloudTable = field(default_factory=PointCloudTable.empty)

    @property
    def after_density(self) -> int:
        return self.after_uniform

    @property
    def final_points(self) -> np.ndarray:
        return self.final_table.xyz


def process_pointcloud(
    table: PointCloudTable,
    config: DownsampleConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> ProcessResult:
    """다운샘플링 파이프라인을 실행한다.

    1. 보셀별 대표 포인트 선택
    2. SOR (Statistical Outlier Removal)
    3. 전역 균일 소거
    """
    result = ProcessResult(original_count=table.count)

    _emit(progress_callback, "보셀 대표점 선택 중...", 0.1)
    voxel_indices = _select_voxel_representatives(table.xyz, config.voxel_size)
    voxel_table = table.select_rows(voxel_indices)
    result.after_voxel = voxel_table.count
    logger.info(
        "보셀 대표점 선택: %s -> %s",
        f"{result.original_count:,}",
        f"{result.after_voxel:,}",
    )

    _emit(progress_callback, "이상치 제거(SOR) 중...", 0.35)
    sor_indices = _select_sor_inliers(
        voxel_table.xyz,
        sor_neighbors=config.sor_neighbors,
        sor_std_ratio=config.sor_std_ratio,
    )
    sor_table = voxel_table.select_rows(sor_indices)
    result.after_sor = sor_table.count
    logger.info(
        "SOR: %s -> %s (%s 제거)",
        f"{result.after_voxel:,}",
        f"{result.after_sor:,}",
        f"{result.after_voxel - result.after_sor:,}",
    )

    target_final_count = max(
        1,
        int(round(result.original_count * (1.0 - config.reduction_percent / 100.0))),
    )

    if config.reduction_percent > 0:
        if voxel_table.count > target_final_count:
            _emit(
                progress_callback,
                f"전역 균일 소거 (-{config.reduction_percent}%) 중...",
                0.55,
            )
            sor_preferred_mask = np.zeros(voxel_table.count, dtype=bool)
            sor_preferred_mask[sor_indices] = True
            final_table = _uniform_global_downsample(
                voxel_table,
                target_count=target_final_count,
                config=config,
                progress_callback=progress_callback,
                preferred_mask=sor_preferred_mask,
            )
        else:
            logger.info(
                "전처리 결과가 이미 목표 개수 이하입니다: %s <= %s",
                f"{sor_table.count:,}",
                f"{target_final_count:,}",
            )
            final_table = sor_table
    else:
        final_table = sor_table

    result.after_uniform = final_table.count
    result.final_table = final_table

    _emit(
        progress_callback,
        f"처리 완료: {result.original_count:,} → {result.after_uniform:,}",
        1.0,
    )
    return result


def _select_voxel_representatives(points_xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points_xyz) <= 1:
        return np.arange(len(points_xyz), dtype=np.int64)

    mins = points_xyz.min(axis=0)
    voxel_keys = np.floor((points_xyz - mins) / voxel_size).astype(np.int64)
    unique_keys, ordered_indices, boundaries = _group_point_indices(voxel_keys)

    representatives = np.empty(len(unique_keys), dtype=np.int64)
    for group_idx in range(len(unique_keys)):
        start, end = boundaries[group_idx], boundaries[group_idx + 1]
        members = ordered_indices[start:end]
        representatives[group_idx] = _pick_centroid_representative(members, points_xyz[members])

    representatives.sort()
    return representatives


def _select_sor_inliers(
    points_xyz: np.ndarray,
    sor_neighbors: int,
    sor_std_ratio: float,
) -> np.ndarray:
    point_count = len(points_xyz)
    if point_count <= 2:
        return np.arange(point_count, dtype=np.int64)

    neighbor_count = min(sor_neighbors, max(point_count - 1, 1))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64, copy=False))
    _, indices = pcd.remove_statistical_outlier(
        nb_neighbors=neighbor_count,
        std_ratio=sor_std_ratio,
    )
    selected = np.asarray(indices, dtype=np.int64)
    selected.sort()
    return selected


def _uniform_global_downsample(
    table: PointCloudTable,
    target_count: int,
    config: DownsampleConfig,
    progress_callback: Callable[[str, float], None] | None = None,
    preferred_mask: np.ndarray | None = None,
) -> PointCloudTable:
    original_count = table.count

    if target_count >= original_count:
        return table

    points_xyz = table.xyz
    mins = points_xyz.min(axis=0)
    spans = np.maximum(points_xyz.max(axis=0) - mins, config.voxel_size)
    coarse_cell_size = max(
        config.voxel_size,
        float(np.cbrt(np.prod(spans) / max(target_count, 1))),
    )
    keep_ratio = target_count / max(original_count, 1)
    # Surface-like point clouds need a finer preservation grid than the global
    # cube-root estimate. Otherwise neighboring patches inside one coarse cell
    # can be fully emptied even with small reductions like 20%.
    if keep_ratio >= 0.5:
        # For mild reductions, preserving at least one point per ~1 mm cell
        # aligns better with the visual expectation of "uniform deletion".
        uniform_cell_size = max(config.voxel_size, 1.0)
    else:
        uniform_factor = float(np.clip((1.0 - keep_ratio) ** 2, 0.05, 0.25))
        uniform_cell_size = max(config.voxel_size, coarse_cell_size * uniform_factor)

    while True:
        uniform_keys = np.floor((points_xyz - mins) / uniform_cell_size).astype(np.int64)
        unique_keys, ordered_indices, boundaries = _group_point_indices(uniform_keys)
        cell_count = len(unique_keys)
        if cell_count <= target_count or uniform_cell_size >= coarse_cell_size:
            break
        uniform_cell_size = min(coarse_cell_size, uniform_cell_size * 1.5)

    cell_sizes = np.diff(boundaries)
    preferred_counts = np.zeros(cell_count, dtype=np.int64)

    cell_representatives = np.empty(cell_count, dtype=np.int64)
    for cell_idx in range(cell_count):
        start, end = boundaries[cell_idx], boundaries[cell_idx + 1]
        members = ordered_indices[start:end]
        if preferred_mask is not None:
            preferred_counts[cell_idx] = int(preferred_mask[members].sum())
        cell_representatives[cell_idx] = _pick_representative(
            members,
            points_xyz[members],
            preferred_mask,
        )
        if cell_idx % max(cell_count // 8, 1) == 0:
            _emit(
                progress_callback,
                f"전역 분포 계산 중... ({cell_idx + 1:,}/{cell_count:,} 셀)",
                0.55 + 0.15 * ((cell_idx + 1) / max(cell_count, 1)),
            )

    if target_count <= cell_count:
        cell_order = np.lexsort((unique_keys[:, 2], unique_keys[:, 1], unique_keys[:, 0]))
        chosen_positions = _evenly_spaced_positions(cell_count, target_count)
        chosen_cells = cell_order[chosen_positions]
        chosen_indices = np.sort(cell_representatives[chosen_cells])
        logger.info(
            "전역 균일 소거(셀 대표 선택): %s -> %s (-%d%%)",
            f"{original_count:,}",
            f"{len(chosen_indices):,}",
            config.reduction_percent,
        )
        return table.select_rows(chosen_indices)

    allocation_sizes = (
        np.maximum(preferred_counts, 1)
        if preferred_mask is not None
        else cell_sizes
    )
    keep_counts = _allocate_keep_counts(allocation_sizes, target_count)
    selected_indices: list[np.ndarray] = []
    for cell_idx in range(cell_count):
        start, end = boundaries[cell_idx], boundaries[cell_idx + 1]
        members = ordered_indices[start:end]
        keep_count = int(keep_counts[cell_idx])
        if keep_count <= 0:
            continue
        selected_indices.append(
            _select_uniform_members(
                members,
                points_xyz[members],
                keep_count,
                preferred_mask=preferred_mask,
            )
        )
        if cell_idx % max(cell_count // 8, 1) == 0:
            _emit(
                progress_callback,
                f"균일 포인트 선택 중... ({cell_idx + 1:,}/{cell_count:,} 셀)",
                0.72 + 0.23 * ((cell_idx + 1) / max(cell_count, 1)),
            )

    chosen = np.sort(np.concatenate(selected_indices)) if selected_indices else np.empty(0, dtype=np.int64)
    logger.info(
        "전역 균일 소거: %s -> %s (-%d%%, cell=%.6f)",
        f"{original_count:,}",
        f"{len(chosen):,}",
        config.reduction_percent,
        uniform_cell_size,
    )
    return table.select_rows(chosen)


def _group_point_indices(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    ordered_indices = np.argsort(inverse, kind="mergesort")
    sorted_inverse = inverse[ordered_indices]
    boundaries = np.concatenate((
        np.array([0], dtype=np.int64),
        np.flatnonzero(np.diff(sorted_inverse)) + 1,
        np.array([len(ordered_indices)], dtype=np.int64),
    ))
    return unique_keys, ordered_indices, boundaries


def _pick_centroid_representative(member_indices: np.ndarray, member_xyz: np.ndarray) -> int:
    centroid = member_xyz.mean(axis=0)
    distances = np.sum((member_xyz - centroid) ** 2, axis=1)
    return int(member_indices[int(np.argmin(distances))])


def _pick_representative(
    member_indices: np.ndarray,
    member_xyz: np.ndarray,
    preferred_mask: np.ndarray | None = None,
) -> int:
    if preferred_mask is None:
        return _pick_centroid_representative(member_indices, member_xyz)

    preferred_members = preferred_mask[member_indices]
    if np.any(preferred_members):
        return _pick_centroid_representative(
            member_indices[preferred_members],
            member_xyz[preferred_members],
        )
    return _pick_centroid_representative(member_indices, member_xyz)


def _allocate_keep_counts(cell_sizes: np.ndarray, target_count: int) -> np.ndarray:
    cell_count = len(cell_sizes)
    keep_counts = np.ones(cell_count, dtype=np.int64)

    if target_count <= cell_count:
        keep_counts.fill(0)
        chosen_positions = _evenly_spaced_positions(cell_count, target_count)
        keep_counts[chosen_positions] = 1
        return keep_counts

    remaining = target_count - cell_count
    capacity = np.maximum(cell_sizes - 1, 0)
    total_capacity = int(capacity.sum())
    if remaining <= 0 or total_capacity <= 0:
        return keep_counts

    quotas = remaining * (capacity / total_capacity)
    extras = np.minimum(np.floor(quotas).astype(np.int64), capacity)
    keep_counts += extras

    leftover = target_count - int(keep_counts.sum())
    if leftover <= 0:
        return keep_counts

    fractional = quotas - np.floor(quotas)
    order = np.argsort(-fractional, kind="mergesort")
    for idx in order:
        room = int(cell_sizes[idx] - keep_counts[idx])
        if room <= 0:
            continue
        add = min(room, leftover)
        keep_counts[idx] += add
        leftover -= add
        if leftover == 0:
            break

    if leftover > 0:
        by_capacity = np.argsort(-capacity, kind="mergesort")
        for idx in by_capacity:
            room = int(cell_sizes[idx] - keep_counts[idx])
            if room <= 0:
                continue
            add = min(room, leftover)
            keep_counts[idx] += add
            leftover -= add
            if leftover == 0:
                break

    return keep_counts


def _select_uniform_members(
    member_indices: np.ndarray,
    member_xyz: np.ndarray,
    keep_count: int,
    preferred_mask: np.ndarray | None = None,
) -> np.ndarray:
    point_count = len(member_indices)
    if keep_count >= point_count:
        return np.sort(member_indices.astype(np.int64, copy=False))
    if keep_count == 1:
        return np.array(
            [_pick_representative(member_indices, member_xyz, preferred_mask)],
            dtype=np.int64,
        )

    if preferred_mask is None:
        return _select_uniform_subset(member_indices, member_xyz, keep_count)

    local_preferred = preferred_mask[member_indices]
    preferred_indices = member_indices[local_preferred]
    preferred_xyz = member_xyz[local_preferred]

    if len(preferred_indices) >= keep_count:
        return _select_uniform_subset(preferred_indices, preferred_xyz, keep_count)

    chosen_parts: list[np.ndarray] = []
    if len(preferred_indices) > 0:
        chosen_parts.append(
            _select_uniform_subset(preferred_indices, preferred_xyz, len(preferred_indices))
        )

    remaining = keep_count - len(preferred_indices)
    fallback_indices = member_indices[~local_preferred]
    fallback_xyz = member_xyz[~local_preferred]
    if remaining > 0 and len(fallback_indices) > 0:
        chosen_parts.append(
            _select_uniform_subset(fallback_indices, fallback_xyz, remaining)
        )

    return np.sort(np.concatenate(chosen_parts).astype(np.int64, copy=False))


def _select_uniform_subset(
    member_indices: np.ndarray,
    member_xyz: np.ndarray,
    keep_count: int,
) -> np.ndarray:
    point_count = len(member_indices)
    if keep_count >= point_count:
        return np.sort(member_indices.astype(np.int64, copy=False))

    order = np.lexsort((
        member_indices,
        member_xyz[:, 2],
        member_xyz[:, 1],
        member_xyz[:, 0],
    ))
    sorted_members = member_indices[order]
    positions = _evenly_spaced_positions(point_count, keep_count)
    chosen = np.sort(sorted_members[positions].astype(np.int64, copy=False))
    return chosen


def _evenly_spaced_positions(total_count: int, sample_count: int) -> np.ndarray:
    if sample_count <= 0:
        return np.empty(0, dtype=np.int64)
    if sample_count >= total_count:
        return np.arange(total_count, dtype=np.int64)
    positions = ((np.arange(sample_count, dtype=np.float64) + 0.5) * total_count / sample_count).astype(np.int64)
    positions = np.clip(positions, 0, total_count - 1)
    return positions


# ----------- 내보내기 -----------


def export_pointcloud(
    points: PointCloudTable | np.ndarray,
    output_path: str,
    fmt: str = "csv",
) -> str:
    """포인트클라우드 테이블을 CSV 또는 TXT로 내보낸다."""
    if fmt == "csv":
        sep = ","
    else:
        sep = "\t"

    if isinstance(points, PointCloudTable):
        values = points.values
        columns = list(points.columns)
    else:
        values = np.asarray(points, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] < 3:
            raise ValueError("내보낼 포인트 데이터는 2차원 배열이어야 합니다.")
        columns = list(_default_columns(values.shape[1]))

    df = pd.DataFrame(values, columns=columns)
    df.to_csv(output_path, sep=sep, index=False, float_format="%.6f")
    logger.info("내보내기 완료: %s (%s 포인트)", output_path, f"{len(values):,}")
    return output_path


# ----------- 유틸리티 -----------


def _emit(
    callback: Callable[[str, float], None] | None,
    message: str,
    progress: float,
) -> None:
    """진행률 콜백 안전 호출."""
    if callback:
        try:
            callback(message, progress)
        except Exception:
            pass
