from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from my_agent.utils.tools import tools
from langgraph.prebuilt import ToolNode


# 使用缓存装饰器，最多缓存4个模型实例
# 这样可以避免重复创建相同的模型实例，提高性能
@lru_cache(maxsize=4)
def _get_model(model_name: str):
    # 根据模型名称选择不同的LLM模型
    if model_name == "openai":
        # 使用 OpenAI 的 GPT-4 模型，temperature=0 表示输出更确定性的结果
        model = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    elif model_name == "anthropic":
        # 使用 Anthropic 的 Claude 3 模型
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")
    else:
        # 如果提供了不支持的模型名称，抛出错误
        raise ValueError(f"Unsupported model type: {model_name}")
    
    # 将搜索工具绑定到模型上，使模型能够使用这些工具
    model = model.bind_tools(tools)
    return model


# 定义决策函数：判断是否需要继续执行工具
def should_continue(state):
    # 从状态中获取消息历史
    messages = state["messages"]
    # 获取最后一条消息
    last_message = messages[-1]
    # 如果最后一条消息没有工具调用，则结束对话
    if not last_message.tool_calls:
        return "end"
    # 如果有工具调用，则继续执行工具操作
    else:
        return "continue"


# 定义系统提示词，设置AI助手的行为方式
system_prompt = """Be a helpful assistant"""


# 定义模型调用函数，处理用户输入并生成回复
def call_model(state, config):
    # 获取当前的消息历史
    messages = state["messages"]
    # 在消息列表开头添加系统提示词，设置AI的行为模式
    messages = [{"role": "system", "content": system_prompt}] + messages
    
    # 获取要使用的模型名称
    # 注释掉的是从配置中获取模型名称的代码
    # model_name = config.get('configurable', {}).get("model_name", "anthropic")
    # 当前固定使用 OpenAI 模型
    model_name = "openai"
    
    # 获取模型实例
    model = _get_model(model_name)
    # 调用模型生成回复
    response = model.invoke(messages)
    # 返回包含新消息的字典，这个消息会被添加到现有的消息列表中
    return {"messages": [response]}


# 创建工具节点，用于执行具体的工具操作（如搜索）
tool_node = ToolNode(tools)
