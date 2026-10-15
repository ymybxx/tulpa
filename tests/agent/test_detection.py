import pytest

from fish_memory.agent.detection import (
    looks_like_agent_task_correction,
    looks_like_agent_task_request,
    looks_like_agent_task_retry,
)


@pytest.mark.parametrize(
    "text",
    [
        "帮我调查这个回归并整理报告",
        "帮我找下杭州哪里可以躺平",
        "查一下杭州今天有什么活动",
        "看一下这个报错",
        "看看附近有什么烧烤店",
        "搜下最近的政策",
        "继续刚才那个任务",
        "再帮我看看附近有什么烧烤店吧",
        "然后把结果整理成表格",
        "计划一下下周的路线",
        "安排明早八点检查",
        "提醒我明天复查",
        "监控这个地址的状态",
        "执行测试并整理结果",
        "创建一个新任务",
        "写一个测试脚本",
        "生成一份报告",
        "验证这个修复",
        "更新刚才的报告",
        "把桌面上的 demo.txt 重命名为 done.txt",
        "请将这份文件移动到归档目录",
        "Please investigate this regression and prepare a report",
        "Retry the failed task",
        "Find a quiet place near Hangzhou",
        "Check the latest logs",
        "Look up nearby barbecue restaurants",
        "Search for the latest release notes",
        "Research this regression",
        "Analyze these logs",
        "Summarize the findings",
        "Write a test script",
        "Run the test suite",
        "Verify the fix",
        "Do this task now",
        "Could you please do it now?",
        "Create a task for this issue",
        "Schedule a check for tomorrow",
        "Remind me to review it",
        "Monitor this endpoint",
        "Continue the previous task",
        "Update the report",
        "Try again",
    ],
)
def test_bilingual_task_marker_opens_the_cheap_t_gate(text: str) -> None:
    assert looks_like_agent_task_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "今天心情怎么样？",
        "hello there",
        "再见",
        "这件事以后再说",
        "Do you like Hangzhou?",
        "How do I get there?",
        "I find this interesting",
        "The fun run is tomorrow",
        "This tool supports multitasking",
        "Let's have brunch",
    ],
)
def test_ordinary_conversation_does_not_open_the_cheap_t_gate(text: str) -> None:
    assert looks_like_agent_task_request(text) is False


@pytest.mark.parametrize(
    "text",
    ["把正在跑的版本改成 Python 3.13", "Change the running task to use 3.13"],
)
def test_running_task_correction_requires_an_explicit_marker(text: str) -> None:
    assert looks_like_agent_task_correction(text) is True
    assert looks_like_agent_task_correction("继续处理") is False


@pytest.mark.parametrize("text", ["重试刚才失败的任务", "retry that run"])
def test_unknown_task_retry_requires_an_explicit_marker(text: str) -> None:
    assert looks_like_agent_task_retry(text) is True
    assert looks_like_agent_task_retry("继续处理") is False
