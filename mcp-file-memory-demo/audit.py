from datetime import datetime, timezone

EVENTS = []


def record_event(user_id, action, resource, allowed, reason=""):
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "action": action,
        "resource": resource,
        "allowed": bool(allowed),
        "reason": reason,
    }
    EVENTS.append(event)
    return event


def recent_events(limit=20):
    return EVENTS[-limit:]
