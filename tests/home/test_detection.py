import pytest

from fish_memory.home.detection import looks_like_home_request


@pytest.mark.parametrize(
    "text",
    [
        "客厅有哪些设备离线？",
        "把卧室灯关掉",
        "Which devices are offline?",
        "Turn off the bedroom light",
    ],
)
def test_bilingual_home_marker_launches_h(text: str) -> None:
    assert looks_like_home_request(text) is True


def test_unrelated_conversation_does_not_launch_h() -> None:
    assert looks_like_home_request("今天心情怎么样？") is False
