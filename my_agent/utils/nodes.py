from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode


# 缓存装饰器，最多缓存4个模型实例
@lru_cache(maxsize=4)
def _get_model(model_name: str):
    # 根据模型名称选择不同的LLM模型
    if model_name == "openai":
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "anthropic":
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    # 绑定工具到模型
    model = model.bind_tools(tools)
    return model


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# 系统提示词
system_prompt = """Be a helpful assistant"""


# 定义模型调用函数
def call_model(state, config):
    messages = state["messages"]
    # 添加系统提示词到消息列表开头
    messages = [{"role": "system", "content": system_prompt}] + messages
    # 获取配置中的模型名称
    # model_name = config.get('configurable', {}).get("model_name", "anthropic")
    model_name = "openai"
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode(tools)
