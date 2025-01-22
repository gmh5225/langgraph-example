from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from my_agent.utils.nodes import call_model, should_continue, tool_node
from my_agent.utils.state import AgentState


# 定义配置类型
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]  # 指定使用的模型名称


# 创建一个新的状态图
workflow = StateGraph(AgentState, config_schema=GraphConfig)

# 定义两个将要循环交替的节点
# agent节点负责调用语言模型进行决策
# action节点负责执行具体的工具操作
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# 设置入口点为'agent'节点
# 这意味着工作流开始时首先调用agent节点
workflow.set_entry_point("agent")

# 添加条件边（conditional edge）
workflow.add_conditional_edges(
    # 首先，我们定义起始节点为'agent'
    # 这表示这些边将在agent节点调用后被评估
    "agent",
    # 然后，传入一个函数来决定下一个要调用的节点
    should_continue,
    # 最后传入一个映射字典
    # 键是字符串，值是其他节点
    # END是一个特殊节点，表示图应该结束
    # 工作流程是：调用should_continue函数，然后将其输出
    # 与这个映射中的键进行匹配
    # 根据匹配结果决定调用哪个节点
    {
        # 如果返回'continue'，则调用action节点
        "continue": "action",
        # 如果返回'end'，则结束工作流
        "end": END,
    },
)

# 添加一个从'action'到'agent'的普通边
# 这意味着在action节点调用完成后，下一个会调用agent节点
workflow.add_edge("action", "agent")

# 最后编译工作流！
# 这会将工作流编译成一个LangChain Runnable对象
# 这意味着你可以像使用其他runnable一样使用它
graph = workflow.compile()
