"""PointCloud Downsampler — 독립 데스크톱 앱.

포인트클라우드 파일을 로드하여 보셀 대표점 선택, SOR, 전역 균일 소거를
수행하고 CSV/TXT로 내보내는 GUI 앱.

Usage:
    python main.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

import matplotlib

matplotlib.use("TkAgg")  # Tkinter backend for embedded interactive canvas
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from engine import (
    DownsampleConfig,
    PointCloudTable,
    ProcessResult,
    export_pointcloud,
    load_pointcloud,
    process_pointcloud,
)
from version import APP_DESCRIPTION, APP_NAME, APP_VERSION

# ─── 로깅 설정 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 컬러 상수 (화이트 배경 + 고대비 보색 계열) ───
BG_DARK = "#ffffff"
BG_CARD = "#fff1d6"
BG_INPUT = "#ffffff"
FG_TEXT = "#102a43"
FG_DIM = "#486581"
FG_ACCENT = "#c2410c"
FG_SUCCESS = "#0f766e"
FG_WARN = "#7c2d12"
BORDER = "#1d4ed8"
BUTTON_SECONDARY_BG = "#163b65"
BUTTON_SECONDARY_ACTIVE = "#24558b"
GRID_COLOR = "#cbd5e1"
APP_DIR_NAME = "PointCloudDownsampler"
SETTINGS_FILE_NAME = "settings.json"


class App(tk.Tk):
    """PointCloud Downsampler 메인 윈도우."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("1100x780")
        self.minsize(900, 650)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # 아이콘 설정 시도
        try:
            self.iconbitmap(default="")
        except Exception:
            pass

        # ── 상태 변수 ──
        self.raw_cloud: PointCloudTable | None = None
        self.raw_points: np.ndarray | None = None
        self.result: ProcessResult | None = None
        self.loaded_file: str = ""
        self.default_view = {"elev": 22.0, "azim": -60.0}
        self.preview_original_color = "#2563eb"
        self.preview_result_color = "#c2410c"
        self.show_original_var = tk.BooleanVar(value=True)
        self.show_result_var = tk.BooleanVar(value=True)
        self.settings_path = self._get_settings_path()
        self._load_settings()

        # ── 스타일 ──
        self._setup_styles()

        # ── UI 구성 ──
        self._build_ui()

    # ═══════════════════════════════════════
    # 스타일
    # ═══════════════════════════════════════

    def _setup_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".", background=BG_DARK, foreground=FG_TEXT, fieldbackground=BG_INPUT)
        style.configure("TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT, font=("Segoe UI", 9))
        style.configure("Card.TLabel", background=BG_CARD, foreground=FG_TEXT, font=("Segoe UI", 9))
        style.configure("CardDim.TLabel", background=BG_CARD, foreground=FG_DIM, font=("Segoe UI", 8))
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground=FG_ACCENT)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10, "bold"), foreground=FG_ACCENT)
        style.configure("Dim.TLabel", foreground=FG_DIM, font=("Segoe UI", 8))
        style.configure("Stat.TLabel", font=("Consolas", 20, "bold"), foreground=FG_SUCCESS)
        style.configure("StatLabel.TLabel", font=("Segoe UI", 8), foreground=FG_DIM)

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            background=FG_ACCENT,
            foreground="#ffffff",
            borderwidth=0,
            padding=(16, 8),
        )
        style.map("Accent.TButton", background=[("active", "#a51018")]) # 조금 더 어두운 붉은색

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 9),
            background=BUTTON_SECONDARY_BG,
            foreground="#ffffff",
            borderwidth=1,
            padding=(12, 6),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", BUTTON_SECONDARY_ACTIVE)],
            foreground=[("active", "#ffffff")],
        )

        style.configure(
            "TEntry",
            fieldbackground=BG_INPUT,
            foreground=FG_TEXT,
            insertcolor=FG_TEXT,
            borderwidth=1,
        )

        style.configure(
            "Horizontal.TProgressbar",
            background=FG_ACCENT,
            troughcolor="#fee2e2",
            borderwidth=0,
            thickness=6,
        )

    # ═══════════════════════════════════════
    # UI 구성
    # ═══════════════════════════════════════

    def _build_ui(self) -> None:
        # 상단 헤더
        header = ttk.Frame(self)
        header.pack(fill="x", padx=20, pady=(15, 5))
        ttk.Label(header, text=f"◆ {APP_NAME}", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text=f"v{APP_VERSION}  |  {APP_DESCRIPTION}", style="Dim.TLabel").pack(
            side="left", padx=(12, 0)
        )

        # 본문: 좌측 패널 + 우측 미리보기
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # ── 좌측 패널 (설정 + 통계) ──
        left = ttk.Frame(body, width=360)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        self._build_file_section(left)
        self._build_params_section(left)
        self._build_accessibility_section(left)
        self._build_view_section(left)
        self._build_actions_section(left)
        self._build_stats_section(left)

        # ── 우측 미리보기 ──
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self._build_preview_section(right)

    def _build_file_section(self, parent: ttk.Frame) -> None:
        card = self._card(parent, "파일 선택")

        row = ttk.Frame(card, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 5))

        self.file_var = tk.StringVar(value="파일을 선택해 주세요...")
        ttk.Label(row, textvariable=self.file_var, style="CardDim.TLabel", wraplength=280).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(row, text="📂 열기", style="Secondary.TButton", command=self._open_file).pack(
            side="right"
        )

    def _build_params_section(self, parent: ttk.Frame) -> None:
        card = self._card(parent, "처리 파라미터")

        # Voxel Size
        self.voxel_var = tk.StringVar(value="3.0")
        self._param_row(card, "Voxel Size (mm)", self.voxel_var, "균일 격자 크기")

        # SOR Neighbors
        self.sor_nb_var = tk.StringVar(value="20")
        self._param_row(card, "SOR 이웃 수", self.sor_nb_var, "통계적 이상치 제거 기준 이웃")

        # SOR Std Ratio
        self.sor_std_var = tk.StringVar(value="2.0")
        self._param_row(card, "SOR 표준편차 비율", self.sor_std_var, "높을수록 관대")

        # Uniform Reduction
        self.reduction_var = tk.StringVar(value="0")
        self._param_row(card, "균일 소거율 (%)", self.reduction_var, "0=비활성, 최대 99")

    def _build_accessibility_section(self, parent: ttk.Frame) -> None:
        card = self._card(parent, "시각 설정")

        ttk.Label(
            card,
            text="포인트 색상을 직접 바꿔 시인성을 조정할 수 있습니다.",
            style="CardDim.TLabel",
        ).pack(anchor="w", padx=6, pady=(0, 6))

        self.original_color_swatch = self._color_picker_row(
            card,
            "원본 포인트 색상",
            self.preview_original_color,
            lambda: self._pick_preview_color("original"),
        )
        self.result_color_swatch = self._color_picker_row(
            card,
            "결과 포인트 색상",
            self.preview_result_color,
            lambda: self._pick_preview_color("result"),
        )

    def _build_view_section(self, parent: ttk.Frame) -> None:
        card = self._card(parent, "표시 설정")

        ttk.Label(
            card,
            text="미리보기에서 원본과 축소 결과를 각각 표시하거나 숨길 수 있습니다.",
            style="CardDim.TLabel",
        ).pack(anchor="w", padx=6, pady=(0, 6))

        options = tk.Frame(card, bg=BG_CARD)
        options.pack(fill="x", padx=5, pady=(0, 2))

        tk.Checkbutton(
            options,
            text="원본 표시",
            variable=self.show_original_var,
            command=self._redraw_current_preview,
            bg=BG_CARD,
            fg=FG_TEXT,
            activebackground=BG_CARD,
            activeforeground=FG_TEXT,
            selectcolor=BG_INPUT,
            highlightthickness=0,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        tk.Checkbutton(
            options,
            text="축소 결과 표시",
            variable=self.show_result_var,
            command=self._redraw_current_preview,
            bg=BG_CARD,
            fg=FG_TEXT,
            activebackground=BG_CARD,
            activeforeground=FG_TEXT,
            selectcolor=BG_INPUT,
            highlightthickness=0,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 0))

    def _build_actions_section(self, parent: ttk.Frame) -> None:
        card = self._card(parent, "실행")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_label = tk.StringVar(value="대기 중")

        ttk.Label(card, textvariable=self.progress_label, style="CardDim.TLabel").pack(anchor="w")
        ttk.Progressbar(card, variable=self.progress_var, maximum=1.0, mode="determinate").pack(
            fill="x", pady=(4, 8)
        )

        btn_row = ttk.Frame(card, style="Card.TFrame")
        btn_row.pack(fill="x")

        self.run_btn = ttk.Button(
            btn_row, text="▶  처리 시작", style="Accent.TButton", command=self._run_process
        )
        self.run_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.export_btn = ttk.Button(
            btn_row, text="💾 내보내기", style="Secondary.TButton", command=self._export, state="disabled"
        )
        self.export_btn.pack(side="right")

    def _build_stats_section(self, parent: ttk.Frame) -> None:
        card = self._card(parent, "처리 결과")

        grid = ttk.Frame(card, style="Card.TFrame")
        grid.pack(fill="x")
        grid.columnconfigure((0, 1), weight=1)

        self.stat_original = tk.StringVar(value="—")
        self.stat_voxel = tk.StringVar(value="—")
        self.stat_sor = tk.StringVar(value="—")
        self.stat_final = tk.StringVar(value="—")

        self._stat_cell(grid, "원본", self.stat_original, 0, 0)
        self._stat_cell(grid, "Voxel 후", self.stat_voxel, 0, 1)
        self._stat_cell(grid, "SOR 후", self.stat_sor, 1, 0)
        self._stat_cell(grid, "최종", self.stat_final, 1, 1)

    def _build_preview_section(self, parent: ttk.Frame) -> None:
        card = self._card(parent, "3D 미리보기", expand=True)

        self.fig = plt.Figure(figsize=(6, 5), dpi=100, facecolor=BG_DARK)
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor=BG_DARK)
        self.ax.view_init(**self.default_view)
        self._style_ax(self.ax)

        canvas_frame = tk.Frame(card, bg=BG_CARD)
        canvas_frame.pack(fill="both", expand=True)

        self.canvas_3d = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        canvas_widget = self.canvas_3d.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(card, bg=BG_CARD)
        toolbar_frame.pack(fill="x", pady=(6, 0))
        self.preview_toolbar = NavigationToolbar2Tk(
            self.canvas_3d,
            toolbar_frame,
            pack_toolbar=False,
        )
        self.preview_toolbar.update()
        self.preview_toolbar.pack(side="left")

        canvas_widget.bind("<MouseWheel>", self._on_mousewheel_zoom)
        canvas_widget.bind("<Button-4>", self._on_mousewheel_zoom)
        canvas_widget.bind("<Button-5>", self._on_mousewheel_zoom)

        # 범례
        legend = tk.Frame(card, bg=BG_CARD)
        legend.pack(fill="x", pady=(5, 0))
        self.original_legend_label = tk.Label(
            legend,
            text="● 원본",
            bg=BG_CARD,
            fg=self.preview_original_color,
            font=("Segoe UI", 8, "bold"),
        )
        self.original_legend_label.pack(side="left", padx=(0, 15))
        self.result_legend_label = tk.Label(
            legend,
            text="● 결과",
            bg=BG_CARD,
            fg=self.preview_result_color,
            font=("Segoe UI", 8, "bold"),
        )
        self.result_legend_label.pack(side="left")
        tk.Label(
            legend,
            text="좌클릭 회전 · 마우스 휠 확대/축소 · 툴바 Home으로 시야 초기화",
            bg=BG_CARD,
            fg=FG_DIM,
            font=("Segoe UI", 8),
        ).pack(side="right")

    # ═══════════════════════════════════════
    # 액션
    # ═══════════════════════════════════════

    def _open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="포인트클라우드 파일 선택",
            filetypes=[
                ("포인트클라우드", "*.txt *.csv *.xlsx *.xls"),
                ("모든 파일", "*.*"),
            ],
        )
        if not path:
            return

        self.loaded_file = path
        self.file_var.set(os.path.basename(path))
        self._set_progress("파일 로딩 중...", 0)

        def _load() -> None:
            try:
                cloud = load_pointcloud(path, self._progress_cb)
                self.raw_cloud = cloud
                self.raw_points = cloud.xyz
                self.after(0, lambda: self._on_loaded(cloud))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("오류", str(e)))
                self.after(0, lambda: self._set_progress("로딩 실패", 0))

        threading.Thread(target=_load, daemon=True).start()

    def _on_loaded(self, cloud: PointCloudTable) -> None:
        self.stat_original.set(f"{cloud.count:,}")
        self.stat_voxel.set("—")
        self.stat_sor.set("—")
        self.stat_final.set("—")
        self.result = None
        self.export_btn.config(state="disabled")
        self._set_progress(f"로드 완료: {cloud.count:,} 포인트", 1.0)
        self._draw_preview(cloud.xyz, None)

    def _run_process(self) -> None:
        if self.raw_cloud is None:
            messagebox.showwarning("알림", "먼저 파일을 선택해 주세요.")
            return

        # 파라미터 파싱
        try:
            config = DownsampleConfig(
                voxel_size=float(self.voxel_var.get()),
                sor_neighbors=int(self.sor_nb_var.get()),
                sor_std_ratio=float(self.sor_std_var.get()),
                reduction_percent=int(self.reduction_var.get()),
            )
        except ValueError:
            messagebox.showerror("오류", "파라미터 값을 올바르게 입력해 주세요.")
            return

        errors = config.validate()
        if errors:
            messagebox.showerror("오류", "\n".join(errors))
            return

        self.run_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self._set_progress("처리 시작...", 0)

        def _process() -> None:
            try:
                res = process_pointcloud(self.raw_cloud, config, self._progress_cb)
                self.result = res
                self.after(0, lambda: self._on_processed(res))
            except Exception as e:
                logger.error("처리 오류: %s", e)
                self.after(0, lambda: messagebox.showerror("오류", str(e)))
                self.after(0, lambda: self._set_progress("처리 실패", 0))
                self.after(0, lambda: self.run_btn.config(state="normal"))

        threading.Thread(target=_process, daemon=True).start()

    def _on_processed(self, res: ProcessResult) -> None:
        self.stat_original.set(f"{res.original_count:,}")
        self.stat_voxel.set(f"{res.after_voxel:,}")
        self.stat_sor.set(f"{res.after_sor:,}")
        self.stat_final.set(f"{res.after_uniform:,}")
        self.run_btn.config(state="normal")
        self.export_btn.config(state="normal")
        self._set_progress(
            f"완료! {res.original_count:,} → {res.after_uniform:,} "
            f"(-{100 - round(res.after_uniform / max(res.original_count, 1) * 100)}%)",
            1.0,
        )
        self._draw_preview(self.raw_points, res.final_points)

    def _export(self) -> None:
        if self.result is None or len(self.result.final_points) == 0:
            messagebox.showwarning("알림", "먼저 처리를 실행해 주세요.")
            return

        path = filedialog.asksaveasfilename(
            title="내보내기",
            defaultextension=".csv",
            initialfile=os.path.splitext(os.path.basename(self.loaded_file))[0] + "_downsampled",
            filetypes=[("CSV", "*.csv"), ("TXT", "*.txt")],
        )
        if not path:
            return

        fmt = "txt" if path.lower().endswith(".txt") else "csv"
        try:
            export_pointcloud(self.result.final_table, path, fmt)
            self._set_progress(f"저장 완료: {os.path.basename(path)}", 1.0)
            messagebox.showinfo(
                "완료",
                f"저장 완료!\n\n"
                f"파일: {os.path.basename(path)}\n"
                f"포인트 수: {len(self.result.final_points):,}\n"
                f"경로: {path}",
            )
        except Exception as e:
            messagebox.showerror("오류", str(e))

    # ═══════════════════════════════════════
    # 3D 미리보기
    # ═══════════════════════════════════════

    def _draw_preview(
        self,
        original: np.ndarray | None,
        processed: np.ndarray | None,
    ) -> None:
        current_view = {"elev": self.ax.elev, "azim": self.ax.azim}
        self.ax.cla()
        self.ax.view_init(**current_view)
        self._style_ax(self.ax)

        max_vis = 30_000  # 시각화 최대 포인트

        visible_original = original if self.show_original_var.get() else None
        visible_processed = processed if self.show_result_var.get() else None

        if visible_original is not None:
            pts = self._sample_preview_points(visible_original, max_vis)
            self.ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=self.preview_original_color, s=0.3, alpha=0.18, label="원본",
            )

        if visible_processed is not None:
            pts = self._sample_preview_points(visible_processed, max_vis)
            self.ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                c=self.preview_result_color, s=0.8, alpha=0.7, label="결과",
            )

        self.ax.set_xlabel("X", fontsize=8, color=FG_DIM)
        self.ax.set_ylabel("Y", fontsize=8, color=FG_DIM)
        self.ax.set_zlabel("Z", fontsize=8, color=FG_DIM)
        self._fit_axes_to_points(visible_original, visible_processed)

        self.canvas_3d.draw_idle()

    def _sample_preview_points(self, points: np.ndarray, max_vis: int) -> np.ndarray:
        if len(points) <= max_vis:
            return points
        positions = ((np.arange(max_vis, dtype=np.float64) + 0.5) * len(points) / max_vis).astype(np.int64)
        positions = np.clip(positions, 0, len(points) - 1)
        return points[positions]

    def _style_ax(self, ax: Any) -> None:
        ax.set_facecolor(BG_DARK)
        ax.set_position([0.02, 0.02, 0.96, 0.96])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BORDER)
        ax.yaxis.pane.set_edgecolor(BORDER)
        ax.zaxis.pane.set_edgecolor(BORDER)
        ax.tick_params(axis="both", colors=FG_DIM, labelsize=7)
        ax.grid(True, alpha=0.35, color=GRID_COLOR)

    def _fit_axes_to_points(
        self,
        original: np.ndarray | None,
        processed: np.ndarray | None,
    ) -> None:
        clouds = [
            pts for pts in (original, processed)
            if pts is not None and len(pts) > 0
        ]
        if not clouds:
            return

        combined = np.vstack(clouds)
        mins = combined.min(axis=0)
        maxs = combined.max(axis=0)
        center = (mins + maxs) / 2.0
        spans = np.maximum(maxs - mins, 1e-6)
        radius = max(float(spans.max()) * 0.52, 1.0)

        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)
        self.ax.set_box_aspect((1, 1, 1))

    def _on_mousewheel_zoom(self, event: Any) -> None:
        if self.raw_points is None and self.result is None:
            return

        step = getattr(event, "delta", 0)
        if step == 0:
            num = getattr(event, "num", None)
            if num == 4:
                step = 120
            elif num == 5:
                step = -120

        if step == 0:
            return

        scale = 0.9 if step > 0 else 1.1
        self._zoom_axes(scale)
        self.canvas_3d.draw_idle()

    def _zoom_axes(self, scale: float) -> None:
        xlim = self.ax.get_xlim3d()
        ylim = self.ax.get_ylim3d()
        zlim = self.ax.get_zlim3d()

        x_center = sum(xlim) / 2.0
        y_center = sum(ylim) / 2.0
        z_center = sum(zlim) / 2.0

        x_radius = max(abs(xlim[1] - xlim[0]) / 2.0 * scale, 1e-6)
        y_radius = max(abs(ylim[1] - ylim[0]) / 2.0 * scale, 1e-6)
        z_radius = max(abs(zlim[1] - zlim[0]) / 2.0 * scale, 1e-6)

        self.ax.set_xlim3d(x_center - x_radius, x_center + x_radius)
        self.ax.set_ylim3d(y_center - y_radius, y_center + y_radius)
        self.ax.set_zlim3d(z_center - z_radius, z_center + z_radius)
        self.ax.set_box_aspect((1, 1, 1))

    def _pick_preview_color(self, target: str) -> None:
        initial = (
            self.preview_original_color
            if target == "original"
            else self.preview_result_color
        )
        _, chosen = colorchooser.askcolor(
            color=initial,
            title="포인트 색상 선택",
            parent=self,
        )
        if not chosen:
            return

        if target == "original":
            self.preview_original_color = chosen
            self.original_color_swatch.config(bg=chosen)
        else:
            self.preview_result_color = chosen
            self.result_color_swatch.config(bg=chosen)

        self._save_settings()
        self._refresh_preview_legend()
        self._draw_preview(
            self.raw_points,
            self.result.final_points if self.result is not None else None,
        )

    def _refresh_preview_legend(self) -> None:
        if hasattr(self, "original_legend_label"):
            self.original_legend_label.config(fg=self.preview_original_color)
        if hasattr(self, "result_legend_label"):
            self.result_legend_label.config(fg=self.preview_result_color)

    def _redraw_current_preview(self) -> None:
        self._draw_preview(
            self.raw_points,
            self.result.final_points if self.result is not None else None,
        )

    def _get_settings_path(self) -> str:
        appdata_dir = os.getenv("APPDATA")
        if appdata_dir:
            base_dir = os.path.join(appdata_dir, APP_DIR_NAME)
        else:
            base_dir = os.path.join(os.path.expanduser("~"), f".{APP_DIR_NAME.lower()}")
        return os.path.join(base_dir, SETTINGS_FILE_NAME)

    def _load_settings(self) -> None:
        if not os.path.exists(self.settings_path):
            return

        try:
            with open(self.settings_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("설정 파일 로드 실패: %s", exc)
            return

        original_color = settings.get("preview_original_color")
        result_color = settings.get("preview_result_color")

        if self._is_valid_hex_color(original_color):
            self.preview_original_color = original_color
        if self._is_valid_hex_color(result_color):
            self.preview_result_color = result_color

    def _save_settings(self) -> None:
        settings_dir = os.path.dirname(self.settings_path)
        settings = {
            "preview_original_color": self.preview_original_color,
            "preview_result_color": self.preview_result_color,
        }

        try:
            os.makedirs(settings_dir, exist_ok=True)
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("설정 파일 저장 실패: %s", exc)

    def _is_valid_hex_color(self, value: Any) -> bool:
        if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
            return False
        try:
            int(value[1:], 16)
            return True
        except ValueError:
            return False

    # ═══════════════════════════════════════
    # 유틸리티
    # ═══════════════════════════════════════

    def _card(
        self, parent: ttk.Frame, title: str, expand: bool = False
    ) -> ttk.Frame:
        """섹션 카드 위젯을 생성합니다."""
        wrapper = ttk.Frame(parent)
        wrapper.pack(fill="both" if expand else "x", expand=expand, pady=(0, 8))

        ttk.Label(
            wrapper, text=f"  {title}", style="Subtitle.TLabel"
        ).pack(anchor="w", pady=(0, 4))

        card = tk.Frame(wrapper, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="both" if expand else "x", expand=expand, ipady=8, ipadx=10)
        return card

    def _param_row(
        self, parent: tk.Frame, label: str, var: tk.StringVar, hint: str
    ) -> None:
        """파라미터 입력 행을 생성합니다."""
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=3, padx=5)

        tk.Label(row, text=label, bg=BG_CARD, fg=FG_TEXT, font=("Segoe UI", 9)).pack(
            side="left"
        )
        entry = tk.Entry(
            row, textvariable=var, width=10, justify="right",
            bg=BG_INPUT, fg=FG_TEXT, insertbackground=FG_TEXT,
            font=("Consolas", 10), relief="flat", highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=FG_ACCENT,
        )
        entry.pack(side="right")

        hint_row = tk.Frame(parent, bg=BG_CARD)
        hint_row.pack(fill="x", padx=5)
        tk.Label(hint_row, text=hint, bg=BG_CARD, fg=FG_DIM, font=("Segoe UI", 7)).pack(
            anchor="e"
        )

    def _color_picker_row(
        self,
        parent: tk.Frame,
        label: str,
        color_value: str,
        command: Any,
    ) -> tk.Label:
        row = tk.Frame(parent, bg=BG_CARD)
        row.pack(fill="x", pady=3, padx=5)

        tk.Label(
            row,
            text=label,
            bg=BG_CARD,
            fg=FG_TEXT,
            font=("Segoe UI", 9),
        ).pack(side="left")

        swatch = tk.Label(
            row,
            width=3,
            bg=color_value,
            relief="solid",
            bd=1,
        )
        swatch.pack(side="right", padx=(8, 0))

        tk.Button(
            row,
            text="색상 변경",
            command=command,
            bg=BUTTON_SECONDARY_BG,
            fg="#ffffff",
            activebackground=BUTTON_SECONDARY_ACTIVE,
            activeforeground="#ffffff",
            relief="flat",
            padx=10,
            pady=3,
            cursor="hand2",
        ).pack(side="right")
        return swatch

    def _stat_cell(
        self, grid: ttk.Frame, label: str, var: tk.StringVar, row: int, col: int
    ) -> None:
        """통계 셀을 생성합니다."""
        cell = tk.Frame(grid, bg=BG_CARD, highlightbackground=BORDER, highlightthickness=1)
        cell.grid(row=row, column=col, padx=2, pady=2, sticky="nsew")
        tk.Label(cell, textvariable=var, bg=BG_CARD, fg=FG_SUCCESS, font=("Consolas", 14, "bold")).pack(
            pady=(6, 0)
        )
        tk.Label(cell, text=label, bg=BG_CARD, fg=FG_DIM, font=("Segoe UI", 8)).pack(
            pady=(0, 4)
        )

    def _set_progress(self, msg: str, val: float) -> None:
        self.progress_label.set(msg)
        self.progress_var.set(val)

    def _progress_cb(self, msg: str, val: float) -> None:
        """백그라운드 스레드에서 호출 가능한 진행률 콜백."""
        self.after(0, lambda: self._set_progress(msg, val))


if __name__ == "__main__":
    app = App()
    app.mainloop()
