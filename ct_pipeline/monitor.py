"""CT Pipeline — 성능 모니터링 및 재학습 트리거."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import redis
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[CT Monitor] %(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        logger.warning("메트릭 파일 없음: %s", metrics_path)
        return {}
    with metrics_path.open(encoding="utf-8") as f:
        return json.load(f)


def get_current_version(version_path: Path) -> str:
    if not version_path.exists():
        version_path.write_text("v1\n", encoding="utf-8")
    return version_path.read_text(encoding="utf-8").strip()


def bump_version(version_path: Path) -> tuple[str, str]:
    current = get_current_version(version_path)
    num = int(current.lstrip("v")) + 1
    new_version = f"v{num}"
    version_path.write_text(new_version + "\n", encoding="utf-8")
    return current, new_version


def get_event_count(r: redis.Redis) -> int:
    val = r.get("ct:event_count")
    return int(val) if val else 0


def get_last_retrain_count(r: redis.Redis) -> int:
    val = r.get("ct:last_retrain_count")
    return int(val) if val else 0


def set_last_retrain_count(r: redis.Redis, count: int) -> None:
    r.set("ct:last_retrain_count", count)


def check_performance(metrics: dict, thresholds: dict) -> list[str]:
    """임계값 이하인 지표 목록 반환."""
    alerts: list[str] = []

    rec = metrics.get("recommendation", {}).get("current_model", {})
    candidate = metrics.get("candidate", {})
    ranking = metrics.get("ranking", {})

    checks = {
        "HitRate@50": rec.get("HitRate@50"),
        "NDCG@50": rec.get("NDCG@50"),
        "Coverage@50": rec.get("Coverage@50"),
        "Recall@300": candidate.get("Recall@300"),
        "auc": ranking.get("auc"),
    }

    for metric, value in checks.items():
        threshold = thresholds.get(metric)
        if value is None or threshold is None:
            continue
        if value < threshold:
            alerts.append(
                f"{metric}: {value:.4f} (임계값: {threshold} 이하)"
            )

    return alerts


def run_once(config: dict, r: redis.Redis) -> None:
    version_path = BASE_DIR / config["version_path"]
    metrics_path = (BASE_DIR / config["metrics_path"]).resolve()
    thresholds = config["thresholds"]
    trigger_count = config["retrain_trigger"]["new_log_count"]

    logger.info("=" * 50)

    # ── 성능 지표 체크 ──────────────────────────────────────
    metrics = load_metrics(metrics_path)
    if metrics:
        alerts = check_performance(metrics, thresholds)
        rec = metrics.get("recommendation", {}).get("current_model", {})
        logger.info(
            "현재 성능 — HitRate@50: %.4f  NDCG@50: %.4f  Coverage@50: %.4f",
            rec.get("HitRate@50", 0),
            rec.get("NDCG@50", 0),
            rec.get("Coverage@50", 0),
        )
        if alerts:
            logger.warning("성능 저하 감지!")
            for alert in alerts:
                logger.warning("  ⚠  %s", alert)
            logger.warning("재학습을 권장합니다.")
        else:
            logger.info("모든 성능 지표 정상.")
    else:
        logger.warning("메트릭 파일을 읽을 수 없어 성능 체크를 건너뜁니다.")

    # ── 재학습 트리거 체크 ──────────────────────────────────
    total_events = get_event_count(r)
    last_retrain = get_last_retrain_count(r)
    new_events = total_events - last_retrain

    logger.info(
        "이벤트 로그 — 전체: %d건  마지막 재학습 이후: %d건  트리거 임계값: %d건",
        total_events,
        new_events,
        trigger_count,
    )

    if new_events >= trigger_count:
        old_ver, new_ver = bump_version(version_path)
        set_last_retrain_count(r, total_events)
        logger.info(
            "[CT Trigger] 신규 로그 %d건 축적 — 재학습 트리거됨. 모델 버전: %s → %s",
            new_events,
            old_ver,
            new_ver,
        )
    else:
        logger.info(
            "재학습 트리거 미달 (%d / %d건). 현재 버전: %s",
            new_events,
            trigger_count,
            get_current_version(version_path),
        )


def main() -> None:
    config = load_config()
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", config["redis"]["host"]),
        port=int(os.getenv("REDIS_PORT", config["redis"]["port"])),
        decode_responses=True,
    )

    interval_seconds = 60

    logger.info("CT 파이프라인 모니터링 시작 (체크 주기: %d초)", interval_seconds)

    while True:
        try:
            run_once(config, r)
        except Exception as exc:
            logger.error("모니터링 중 오류 발생: %s", exc)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
