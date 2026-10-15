from pydantic import BaseModel, ConfigDict, Field, SecretStr

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class ExperimentalConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True, frozen=True)

    # 主动陪伴: TA decides on TA's own when to speak up.
    enabled: bool = False
    # 聊天时挑怎么回: prepare several reply lanes and let the judge pick one.
    chat_pick_enabled: bool = False
    # 说话时的表情和动作: the decision model scores each spoken segment.
    speech_facial_enabled: bool = True
    # Optional decision model (百炼 or Jev, same request shape): it judges better;
    # without one the reasoning model judges.
    api_key: SecretStr | None = None
    model: str = Field(default="jev-1.13.0", min_length=1)
    endpoint: str = Field(default=JEV_ENDPOINT, pattern=r"^https://")
    wake_seconds: float = Field(default=5, ge=1, le=3600)
    thoughts_enabled: bool = True
    thought_seconds: float = Field(default=2, ge=1, le=3600)
    audiovisual_threshold: float = Field(default=0.5, ge=0, le=1)
