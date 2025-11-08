from collections import deque
from datetime import datetime

RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_store = {}

def rate_limited(key: str, max_requests: int = RATE_LIMIT_MAX_REQUESTS, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS) -> bool:
    now = datetime.utcnow().timestamp()
    dq = _rate_limit_store.get(key)
    if dq is None:
        dq = deque()
        _rate_limit_store[key] = dq
    cutoff = now - window_seconds
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= max_requests:
        return True
    dq.append(now)
    return False
