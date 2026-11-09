"""Application version metadata."""

from __future__ import annotations

APP_NAME = "PointCloud Downsampler"
APP_VERSION = "1.1.5"
APP_VERSION_INFO = (1, 1, 5, 0)
APP_RELEASE_DATE = "2026-03-30"
APP_DESCRIPTION = "포인트클라우드 균일 소거/다운샘플링 도구"
APP_COMPANY = "AI TEAM PJT"
APP_COPYRIGHT = "Copyright (c) 2026 AI TEAM PJT"


def version_banner() -> str:
    return f"{APP_NAME} v{APP_VERSION} ({APP_RELEASE_DATE})"
