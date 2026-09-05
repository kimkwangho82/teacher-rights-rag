"""공용 로깅 설정. CLI 스크립트와 API 서버가 같은 포맷을 쓴다."""

import logging
import os

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | None = None) -> None:
    """LOG_LEVEL 환경변수(기본 INFO)로 레벨을 정한다. 이미 설정돼 있으면 덮어쓰지 않는다."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO), format=DEFAULT_FORMAT
    )
    # pypdf 는 손상된 객체 경고를 매우 많이 찍으므로 ERROR 이상만 표시
    logging.getLogger("pypdf").setLevel(logging.ERROR)
