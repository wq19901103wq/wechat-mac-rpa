"""SQLAlchemy 数据模型（Phase 1 MVP）——使用 SQLAlchemy 2.0 Mapped 注解以兼容 mypy。"""

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Chatroom(Base):
    """聊天会话：群聊或私聊。"""

    __tablename__ = "chatrooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chatroom_id: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    chat_type: Mapped[str] = mapped_column(Text, nullable=False, default="group")
    first_seen_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_active_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="chatroom", cascade="all, delete-orphan"
    )
    members: Mapped[list["ChatMember"]] = relationship(
        "ChatMember", back_populates="chatroom", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Chatroom(id={self.id}, chatroom_id={self.chatroom_id}, name={self.display_name})>"


class Message(Base):
    """单条聊天消息。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chatroom_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chatrooms.id"), nullable=False
    )
    local_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    server_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    wxid: Mapped[str] = mapped_column(Text, nullable=False)
    sender_display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_self: Mapped[bool] = mapped_column(Boolean, default=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_type: Mapped[str] = mapped_column(Text, nullable=False, default="text")
    image_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_at_me: Mapped[bool] = mapped_column(Boolean, default=False)
    is_revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    replied: Mapped[bool] = mapped_column(Boolean, default=False)
    reply_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    reply_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    create_time: Mapped[float] = mapped_column(Float, nullable=False)
    raw_type: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_file: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    chatroom: Mapped["Chatroom"] = relationship("Chatroom", back_populates="messages")

    __table_args__ = (
        UniqueConstraint(
            "chatroom_id",
            "wxid",
            "create_time",
            "content_hash",
            name="uix_message_unique",
        ),
        Index("idx_messages_chatroom_create_time", "chatroom_id", "create_time"),
        Index("idx_messages_wxid", "wxid"),
        Index("idx_messages_content_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return (
            f"<Message(id={self.id}, chatroom_id={self.chatroom_id}, "
            f"wxid={self.wxid}, create_time={self.create_time})>"
        )


class ChatMember(Base):
    """群成员关系。私聊也存一条 wxid=对方。"""

    __tablename__ = "chat_members"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chatroom_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("chatrooms.id"), nullable=False
    )
    wxid: Mapped[str] = mapped_column(Text, nullable=False)
    group_nickname: Mapped[str | None] = mapped_column(Text, nullable=True)
    joined_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    left_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[float | None] = mapped_column(Float, nullable=True)

    chatroom: Mapped["Chatroom"] = relationship("Chatroom", back_populates="members")

    __table_args__ = (
        UniqueConstraint("chatroom_id", "wxid", name="uix_chat_member_unique"),
        Index("idx_chat_members_chatroom", "chatroom_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<ChatMember(id={self.id}, chatroom_id={self.chatroom_id}, "
            f"wxid={self.wxid})>"
        )
