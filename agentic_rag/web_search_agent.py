"""互联网搜索Agent - 当本地检索质量不达标时进行网络补充搜索"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


def web_search_tool(query: str, max_results: int = 5) -> str:
    """执行互联网搜索并返回格式化结果"""
    try:
        # 这里可以集成真实的搜索API，如Google Search API, Bing API等
        # 目前返回模拟结果
        
        # 模拟搜索结果
        mock_results = [
            {
                "title": f"法律咨询：{query}相关案例",
                "snippet": f"根据相关法律规定，关于{query}的问题需要考虑具体情况...",
                "url": "https://example-law-site.com/case1"
            },
            {
                "title": f"{query}的法律解释和适用",
                "snippet": f"在司法实践中，{query}通常按照以下原则处理...",
                "url": "https://example-law-site.com/interpretation"
            },
            {
                "title": f"最新司法解释：{query}相关规定",
                "snippet": f"最高人民法院关于{query}的最新司法解释指出...",
                "url": "https://example-court.gov.cn/interpretation"
            }
        ]
        
        # 格式化搜索结果
        formatted_results = []
        for i, result in enumerate(mock_results[:max_results], 1):
            formatted_results.append(f"""【网络搜索结果 {i}】
标题：{result['title']}
内容：{result['snippet']}
来源：{result['url']}""")
        
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"网络搜索失败：{str(e)}"


# 互联网搜索Agent
web_search_agent = LlmAgent(
    name="WebSearchAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是互联网搜索专家。当本地法律条文检索质量不达标时，执行网络搜索补充相关信息。

**重写后的查询：**
{rewritten_query}

**本地检索结果：**
{retrieval_results}

**质量评估结果：**
{quality_evaluation}

**任务：**
1. 分析本地检索的不足之处
2. 基于重写查询执行网络搜索
3. 重点搜索：
   - 最新司法解释
   - 相关案例分析
   - 法律专家观点
   - 实务操作指南

**搜索策略：**
- 使用法律专业术语
- 关注权威法律网站
- 优先获取官方解释
- 补充实务案例

**输出格式：**
## 网络搜索补充

**搜索关键词：**
[优化后的搜索关键词]

**搜索结果：**
[调用web_search_tool获取的结果]

**补充价值：**
- 补充了哪些本地检索缺失的信息
- 提供了哪些最新的法律动态
- 增加了哪些实务操作指导""",
    description="执行互联网搜索补充本地检索不足",
    output_key="web_search_results",
    tools=[web_search_tool],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1536,
    )
)
