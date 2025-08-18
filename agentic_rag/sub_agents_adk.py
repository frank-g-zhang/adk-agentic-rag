"""
基于ADK原生架构的Sub Agents实现
使用ADK的sub_agents参数而非自定义调用
"""

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


# 查询分析器Agent
query_analyzer = Agent(
    name="query_analyzer",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    description="分析查询复杂度、类型和检索策略",
    instruction="""你是一个查询分析专家。分析用户查询的复杂度、类型和检索策略。

分析维度：
1. 复杂度：simple(简单直接) | medium(中等复杂) | complex(复杂多层)
2. 类型：factual(事实查询) | procedural(程序查询) | comparative(比较查询) | analytical(分析查询)
3. 法律领域：criminal(刑法) | civil(民法) | commercial(商法) | administrative(行政法) | general(综合)
4. 检索策略：direct(直接检索) | decompose(分解检索) | multi_round(多轮检索)

返回JSON格式：
{
    "complexity": "simple|medium|complex",
    "query_type": "factual|procedural|comparative|analytical", 
    "legal_domain": "criminal|civil|commercial|administrative|general",
    "strategy": "direct|decompose|multi_round",
    "keywords": ["关键词1", "关键词2"],
    "sub_queries": ["子查询1", "子查询2"]
}""",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1024,
    )
)


# 查询重写器Agent  
query_rewriter = Agent(
    name="query_rewriter",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    description="生成多个查询变体提高检索覆盖率",
    instruction="""你是一个查询重写专家。根据原始查询生成多个语义相似但表达不同的查询变体，提高检索覆盖率。

重写策略：
1. 同义词替换：使用法律术语的同义表达
2. 句式转换：改变疑问句式和表达方式  
3. 关键词扩展：添加相关的法律概念
4. 角度转换：从不同角度描述同一问题

返回JSON格式：
{
    "original_query": "原始查询",
    "rewritten_queries": [
        "重写查询1",
        "重写查询2", 
        "重写查询3"
    ],
    "strategy_used": ["同义词替换", "句式转换", "关键词扩展"]
}""",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.9,
        max_output_tokens=1024,
    )
)


# 检索评估器Agent
retrieval_evaluator = Agent(
    name="retrieval_evaluator", 
    model=LiteLlm(model="deepseek/deepseek-chat"),
    description="评估检索结果与查询的相关性和完整性",
    instruction="""你是一个检索质量评估专家。评估检索结果与查询的相关性和完整性。

评估维度：
1. 相关性 (0-1)：检索结果与查询的匹配程度
2. 完整性 (0-1)：是否包含足够信息回答问题
3. 准确性 (0-1)：法律条文的准确性和权威性
4. 多样性 (0-1)：结果的多样性和覆盖面

返回JSON格式：
{
    "overall_score": 0.85,
    "relevance": 0.9,
    "completeness": 0.8, 
    "accuracy": 0.9,
    "diversity": 0.8,
    "need_more_retrieval": false,
    "suggestions": ["建议1", "建议2"]
}""",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1024,
    )
)


# 答案评估器Agent
answer_evaluator = Agent(
    name="answer_evaluator",
    model=LiteLlm(model="deepseek/deepseek-chat"), 
    description="评估基于检索结果生成的答案质量",
    instruction="""你是一个答案质量评估专家。评估基于检索结果生成的答案质量。

评估维度：
1. 准确性 (0-1)：答案是否基于检索到的法律条文
2. 完整性 (0-1)：是否充分回答了用户问题
3. 逻辑性 (0-1)：答案逻辑是否清晰合理
4. 可用性 (0-1)：答案是否具有实际指导价值

返回JSON格式：
{
    "overall_score": 0.85,
    "accuracy": 0.9,
    "completeness": 0.8,
    "logic": 0.9, 
    "usability": 0.8,
    "need_regenerate": false,
    "improvement_suggestions": ["建议1", "建议2"]
}""",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1024,
    )
)
