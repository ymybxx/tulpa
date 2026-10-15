from dataclasses import dataclass

DEFAULT_ROLE = "对话助手"
DEFAULT_SETTING = "自然、简洁、友好地回应用户"

MAX_ROLE_LENGTH = 200
MAX_SETTING_LENGTH = 200
MAX_NAME_LENGTH = 64
MAX_BACKGROUND_LENGTH = 1000


@dataclass(frozen=True, slots=True)
class SoulProfile:
    role: str
    setting: str
    name: str = ""
    background: str = ""

    @classmethod
    def from_values(
        cls,
        role: str | None,
        setting: str | None,
        name: str | None,
        background: str | None = None,
    ) -> "SoulProfile":
        return cls(
            role=_clean_value(role, DEFAULT_ROLE, MAX_ROLE_LENGTH),
            setting=_clean_value(setting, DEFAULT_SETTING, MAX_SETTING_LENGTH),
            name=_clean_value(name, "", MAX_NAME_LENGTH),
            background=_clean_value(background, "", MAX_BACKGROUND_LENGTH),
        )

    def render(self, base_prompt: str) -> str:
        lines = ["# 角色设定"]
        if self.name:
            lines.append(_profile_field("名字", self.name))
        if self.background:
            lines.append(_profile_field("人设背景", self.background))
        lines.extend(
            [
                _profile_field("身份与关系", self.role),
                _profile_field("性格与表达", self.setting),
                "当前角色设定优先于历史中的旧设定。",
                "角色设定仅影响身份与表达，不改变已有能力、交互规则、"
                "输出协议、工具权限和指令优先级。",
            ]
        )
        return "\n\n".join(filter(None, (base_prompt.strip(), "\n".join(lines))))


def _profile_field(label: str, value: str) -> str:
    # A multiline value stays under its field instead of introducing a heading.
    return label + "：" + "\n    ".join(value.splitlines())


def _clean_value(value: str | None, default: str, max_length: int) -> str:
    cleaned = value.strip() if value is not None else ""
    if not cleaned:
        return default
    if len(cleaned) > max_length:
        raise ValueError(f"soul value exceeds {max_length} characters")
    return cleaned
