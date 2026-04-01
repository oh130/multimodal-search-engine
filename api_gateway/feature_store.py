"""
Redis 기반 Feature Store.

저장 구조:
  user:{user_id}:recent_clicks   — List<item_id>, 최근 20개
  user:{user_id}:session_interest — Hash<category, score>
  user:{user_id}:click_count     — 총 클릭 수 (int)
"""

import json
import redis

RECENT_CLICKS_MAX = 20
CLICK_TTL = 60 * 60 * 24 * 7  # 7일


class RedisFeatureStore:
    def __init__(self, host: str = "redis", port: int = 6379, db: int = 0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    # ── 클릭 이벤트 ──────────────────────────────────────────
    def push_click(self, user_id: str, item_id: str) -> None:
        key = f"user:{user_id}:recent_clicks"
        self.r.lpush(key, item_id)
        self.r.ltrim(key, 0, RECENT_CLICKS_MAX - 1)
        self.r.expire(key, CLICK_TTL)
        self.r.incr(f"user:{user_id}:click_count")

    def get_recent_clicks(self, user_id: str, n: int = 10) -> list[str]:
        return self.r.lrange(f"user:{user_id}:recent_clicks", 0, n - 1)

    def get_click_count(self, user_id: str) -> int:
        val = self.r.get(f"user:{user_id}:click_count")
        return int(val) if val else 0

    # ── 세션 관심사 ──────────────────────────────────────────
    def set_session_interest(self, user_id: str, interest: dict) -> None:
        key = f"user:{user_id}:session_interest"
        self.r.set(key, json.dumps(interest), ex=CLICK_TTL)

    def get_session_interest(self, user_id: str) -> dict:
        val = self.r.get(f"user:{user_id}:session_interest")
        return json.loads(val) if val else {}

    # ── 통합 조회 ────────────────────────────────────────────
    def get_user_features(self, user_id: str) -> dict:
        return {
            "user_id": user_id,
            "recent_clicks": self.get_recent_clicks(user_id),
            "session_interest": self.get_session_interest(user_id),
            "click_count": self.get_click_count(user_id),
        }
