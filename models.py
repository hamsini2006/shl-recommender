"""Pydantic request/response models for the chat API."""

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        assert v in ("user", "assistant"), "role must be user or assistant"
        return v


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[Message]) -> list[Message]:
        assert len(v) > 0, "messages cannot be empty"
        assert v[-1].role == "user", "last message must be from user"
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False
