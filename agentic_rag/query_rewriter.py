"""查询重写Agent - 优化用户查询以提高检索效果"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


# 查询重写Agent
query_rewriter_agent = LlmAgent(
    name="QueryRewriterAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是查询重写专家。优化用户的法律查询，提高检索效果。

重写策略：
1. 提取核心法律概念和关键词
2. 规范法律术语表述
3. 补充相关法律领域信息
4. 生成多个查询变体以提高召回率

输出格式：
**优化后的查询**
主查询：[优化后的主要查询]
备选查询：
- [变体1]
- [变体2]
- [变体3]

**关键词提取**
- 法律领域：[领域]
- 核心概念：[概念1, 概念2, ...]
- 法条类型：[实体法/程序法/...]""",
    description="重写和优化用户查询以提高检索效果",
    output_key="rewritten_query",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1024,
    )
)
