"""互联网搜索Agent - 当本地检索质量不达标时进行网络补充搜索"""

import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


def web_search_tool(query: str, max_results: int = 5) -> str:
    """执行互联网搜索并返回格式化结果"""
    try:
        from serpapi import GoogleSearch
        
        # 获取API密钥
        api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            return "错误：未设置SERPAPI_API_KEY环境变量"
        
        # 配置搜索参数 - 基于SerpAPI文档样例
        params = {
            "engine": "google_light",
            "q": query,
            "location": "China",  # 可选：搜索地理位置
            "google_domain": "google.com",
            "hl": "zh-cn",  # 界面语言：中文
            "gl": "cn",     # 搜索地区：中国
            "api_key": api_key
        }
        
        # 执行搜索
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # 解析搜索结果
        organic_results = results.get("organic_results", [])
        
        if not organic_results:
            return "未找到相关搜索结果"
        
        # 格式化搜索结果
        formatted_results = []
        for i, result in enumerate(organic_results[:max_results], 1):
            title = result.get('title', '无标题')
            snippet = result.get('snippet', '无摘要')
            link = result.get('link', '无链接')
            
            formatted_results.append(f"""【网络搜索结果 {i}】
标题：{title}
内容：{snippet}
来源：{link}""")
        
        return "\n\n".join(formatted_results)
        
    except ImportError:
        return "错误：请安装serpapi库 (pip install google-search-results)"
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

**重要限制：**
- 严格基于web_search_tool的实际搜索结果进行分析
- 如果搜索工具返回"未找到相关搜索结果"或错误信息，必须如实报告
- 禁止在无搜索结果时编造或推测任何法律信息
- 禁止基于训练数据生成法律建议，必须依据实际搜索结果

**输出格式：**
## 网络搜索补充

**搜索关键词：**
[优化后的搜索关键词]

**搜索结果：**
[调用web_search_tool获取的结果]

**如果搜索无结果，必须输出：**
## 网络搜索补充

**搜索关键词：**
[优化后的搜索关键词]

**搜索结果：**
互联网查询无结果信息。未能通过网络搜索获取到相关的法律信息补充。

**建议：**
- 尝试调整搜索关键词
- 建议咨询专业法律人士
- 查阅权威法律数据库

**有搜索结果时的补充价值：**
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
