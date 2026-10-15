"""Malformed T-lane payloads rejected by both runtime and evidence parsers."""

INVALID_TASK_LANE_WIRE_SAMPLES = (
    '{"m":"neutral","t":0,"a":""}',
    '{"t":1,"a":"确认。","m":"neutral"}',
    '{"t":1,"m":"neutral","c":0,"a":"确认。"}',
    '{"t":1,"k":"new_task","m":"neutral","a":"确认。"}',
    '{"t":1,"m":"neutral","k":"new_task","a":"确认。"}',
    '{"t":1,"m":"neutral","a":"确认。","k":"new_task"}',
    '{"t":1,"m":"neutral","a":"确认。","c":0,"k":"new_task"}',
    '{"t":0,"t":1,"m":"neutral","a":"确认。"}',
    '{"t":1,"m":excited,"a":"确认。"}',
    '{"t":1,"m":"neutral","a":"   "}',
    '{"t":1,"m":"neutral","a":"继续。","c":[0,1]}',
    '{"t":1,"m":"neutral","a":"继续。","c":2}',
    '{"t":2,"m":"neutral","a":"澄清。"}',
    '{"t":2,"m":"neutral","a":"澄清。","c":0}',
    '{"t":2,"m":"neutral","a":"   "}',
)
