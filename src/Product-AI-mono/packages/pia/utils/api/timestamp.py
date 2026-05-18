from datetime import datetime, timezone


def str_UTC_ISO8601_ms_now_time():
    """현재 UTC 시간을 ISO 8601 형식의 문자열로 반환합니다. (마이크로초 단위)

    FORMAT:
        'YYYY-MM-DDTHH:MM:SS.ssssss'
    EXAMPLE:
        '2025-08-27T15:49:00.123456'
    """
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="microseconds")
