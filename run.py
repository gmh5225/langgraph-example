# 首先加载环境变量
from dotenv import load_dotenv
import os

# 确保在导入其他模块前加载环境变量
load_dotenv()

# 然后再导入其他模块
from my_agent.agent import graph
from langchain_core.messages import HumanMessage

def test_agent():
    # 验证环境变量
    print("Checking API Keys:")
    print(f"TAVILY_API_KEY: {'✓' if os.getenv('TAVILY_API_KEY') else '✗'}")
    print(f"GOOGLE_API_KEY: {'✓' if os.getenv('GOOGLE_API_KEY') else '✗'}")
    print()
    ## return

    # 初始化状态
    initial_state = {
        "messages": [
            HumanMessage(content="2025年春节是什么时候？")
        ]
    }

    # 运行图
    result = graph.invoke(initial_state)

    # 打印对话历史
    print("\n=== 对话历史 ===")
    for message in result["messages"]:
        role = "AI" if message.type == "ai" else "Human"
        print(f"\n{role}: {message.content}")

if __name__ == "__main__":
    print("开始测试 AI 代理...\n")
    test_agent() 