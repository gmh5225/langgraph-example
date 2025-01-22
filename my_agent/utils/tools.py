from langchain_community.tools.tavily_search import TavilySearchResults

# 定义可用的工具列表
# TavilySearchResults 是一个网络搜索工具
# max_results=1 表示每次搜索只返回一个最相关的结果
tools = [TavilySearchResults(max_results=1)]

# 工具使用示例：
# 当 AI 需要搜索信息时，会调用 TavilySearch
# 例如：查询"今天天气怎么样"
# 工具会返回搜索结果供 AI 参考和回答
