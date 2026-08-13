# 消息体架构设计（消息级回复状态）

## 核心问题

当前 `last_reply_time` / `reply_count` 是会话级别的：
- 不知道**哪条消息**回复了
- 不知道**哪条消息**还没回
- cooldown 期间的消息永久丢失

## 新设计：消息自带回复状态

```python
@dataclass
class ChatMessage:
    text: str
    sender: str
    sender_type: SenderType
    chat_name: str
    
    # === 消息级回复状态 ===
    replied: bool = False              # 是否已回复
    reply_text: str = ""              # 回复内容
    reply_time: Optional[float] = None # 回复时间戳


@dataclass
class ChatState:
    """单个聊天的完整状态"""
    chat_id: str
    chat_name: str
    messages: List[ChatMessage] = field(default_factory=list)
    _msg_ids: Set[str] = field(default_factory=set)


class GlobalStore:
    """全局存储"""
    
    def __init__(self, max_messages: int = 200):
        self.chats: Dict[str, ChatState] = {}
        self.max_messages = max_messages
    
    def merge_tick(self, chat_name: str, messages: List[ChatMessage]) -> Tuple[ChatState, List[ChatMessage]]:
        """合并 tick 消息，返回 (state, 未回复的新消息)"""
        if chat_name not in self.chats:
            self.chats[chat_name] = ChatState(
                chat_id=f"chat_{len(self.chats)}",
                chat_name=chat_name,
            )
        
        state = self.chats[chat_name]
        new_messages = []
        
        for msg in messages:
            mid = self._msg_id(msg)
            if mid not in state._msg_ids:
                state.messages.append(msg)
                state._msg_ids.add(mid)
                if not msg.replied:  # 新消息默认未回复
                    new_messages.append(msg)
        
        # 裁剪旧消息
        if len(state.messages) > self.max_messages:
            removed = state.messages[:-self.max_messages]
            state.messages = state.messages[-self.max_messages:]
            for msg in removed:
                state._msg_ids.discard(self._msg_id(msg))
        
        return state, new_messages
    
    def mark_replied(self, chat_name: str, target_msg: ChatMessage, reply_text: str):
        """标记某条消息已回复"""
        state = self.chats.get(chat_name)
        if not state:
            return
        
        # 找到对应消息，标记回复状态
        for msg in state.messages:
            if msg.text == target_msg.text and msg.sender == target_msg.sender:
                msg.replied = True
                msg.reply_text = reply_text
                msg.reply_time = time.time()
                break
    
    def get_unreplied(self, chat_name: str) -> List[ChatMessage]:
        """获取某聊天中所有未回复的消息"""
        state = self.chats.get(chat_name)
        if not state:
            return []
        return [m for m in state.messages if not m.replied and m.sender_type != SenderType.SELF]
    
    @property
    def last_reply_time(self, chat_name: str) -> Optional[float]:
        """从消息中推导最后回复时间"""
        state = self.chats.get(chat_name)
        if not state:
            return None
        replied = [m.reply_time for m in state.messages if m.replied and m.reply_time]
        return max(replied) if replied else None
    
    @property
    def reply_count(self, chat_name: str) -> int:
        """从消息中推导回复次数"""
        state = self.chats.get(chat_name)
        if not state:
            return 0
        return sum(1 for m in state.messages if m.replied)
```

## Bot 层调用

```python
def tick(self):
    result = self.perception.perceive()
    chat_name = result.chat_name
    
    # 合并 tick 消息，得到未回复的新消息
    state, unreplied_new = self.store.merge_tick(chat_name, result.messages)
    
    # 决策：回复哪条消息
    # 优先回复最早未回复的，或最新的
    if unreplied_new:
        target = unreplied_new[-1]  # 最新的未回复消息
        
        if self.policy.should_reply(target):
            reply = self.generator.generate(target, state)
            self.sender.send(reply)
            
            # 标记该消息已回复
            self.store.mark_replied(chat_name, target, reply)
```

## 优势

| 能力 | 旧架构 | 新架构 |
|------|--------|--------|
| 知道哪条消息已回复 | ❌ | ✅ |
| 知道哪条消息未回复 | ❌ | ✅ |
| cooldown 后恢复回复 | ❌ 消息已丢 | ✅ 未回复的还在 |
| 查看历史回复内容 | ❌ | ✅ `msg.reply_text` |
| 统计回复率 | ❌ | ✅ `replied / total` |
| 遗漏消息检测 | ❌ | ✅ `get_unreplied()` |

## 持久化

```json
{
  "林岚@示例交流群": {
    "chat_id": "chat_0",
    "messages": [
      {
        "text": "不太对吧",
        "sender": "wanglc",
        "sender_type": "other",
        "replied": true,
        "reply_text": "哪里不对？",
        "reply_time": 1713923405
      },
      {
        "text": "测试下",
        "sender": "wanglc",
        "sender_type": "other",
        "replied": false
      }
    ]
  }
}
```

 `_msg_ids` 启动时从 `messages` 重建，不存磁盘。
