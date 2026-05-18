from pia.utils.api.timestamp import str_UTC_ISO8601_ms_now_time


def test_str_UTC_ISO8601_ms_now_time():
    timestamp = str_UTC_ISO8601_ms_now_time()  # YYYY-MM-DDTHH:MM:SS.ssssss

    assert isinstance(timestamp, str), "Timestamp should be a string"
    assert len(timestamp) == 26, "Timestamp length should be 26 characters"
