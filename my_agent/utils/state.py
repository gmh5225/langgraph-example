from langgraph.graph import add_messages  # 导入消息添加器
from langchain_core.messages import BaseMessage  # 导入基础消息类型
from typing import TypedDict, Annotated, Sequence  # 导入类型提示工具


# 定义代理状态类型
# TypedDict 用于定义一个类型化的字典，确保类型安全
class AgentState(TypedDict):
    # messages 字段用于存储对话历史
    # Sequence[BaseMessage] 表示这是一个消息序列
    # Annotated 和 add_messages 注解表示这个序列支持自动消息追加
    # 例如：
    # state = {
    #     "messages": [
    #         HumanMessage(content="你好"),
    #         AIMessage(content="你好！有什么我可以帮你的吗？")
    #         # ... 更多消息
    #     ]
    # }
    messages: Annotated[Sequence[BaseMessage], add_messages]
