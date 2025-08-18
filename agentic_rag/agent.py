"""ADK Web兼容的法律RAG Agent - 优化的4阶段SequentialAgent工作流
查询重写 → 检索 → 评估 → 答案生成
"""

import os
from typing import Optional
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from .config import LawRAGConfig
from .local_retriever import LocalRetriever


# 全局配置和检索器
config = LawRAGConfig()
retriever: Optional[LocalRetriever] = None


def get_retriever():
    """获取全局检索器实例"""
    global retriever
    if retriever is None:
        from .local_retriever import create_local_retriever
        retriever = create_local_retriever()
    return retriever


def execute_retrieval(query: str) -> str:
    """执行检索并返回格式化结果"""
    retriever = get_retriever()
    if retriever.index is None:
        return "⚠️ 法律文本索引尚未建立，请先运行: python init_index.py"
    
    # 执行检索和重排序
    results = retriever.retrieve_and_rerank(query, top_k=10, top_n=5)
    
    if not results:
        return "未找到相关法律条文"
    
    # 格式化检索结果
    formatted_results = []
    for i, result in enumerate(results, 1):
        text = result['text']
        score = result.get('rerank_score', 0)
        metadata = result['metadata']
        
        law_name = metadata.get('law', '未知法律')
        article = metadata.get('article', '未知条款')
        
        formatted_results.append(f"""【检索结果 {i}】(相关性: {score:.3f})
{law_name} {article}
{text}""")
    
    return "\n\n".join(formatted_results)


# 1. 查询重写Agent
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


# 2. 检索执行Agent
retrieval_agent = LlmAgent(
    name="RetrievalAgent", 
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是检索执行专家。基于重写后的查询执行法律条文检索。

**重写后的查询：**
{rewritten_query}

执行策略：
1. 使用主查询进行检索
2. 如果结果不足，尝试备选查询
3. 合并和去重检索结果
4. 按相关性排序

**重要规则：**
- 如果检索工具返回"未找到相关法律条文"，直接输出该结果
- 不要基于空结果生成任何内容

输出检索到的法律条文和统计信息。""",
    description="执行法律条文检索",
    output_key="retrieval_results",
    tools=[execute_retrieval],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=2048,
    )
)


# 3. 质量评估Agent（带阈值判断）
quality_evaluator_agent = LlmAgent(
    name="QualityEvaluatorAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是检索质量评估专家。评估检索结果质量并判断是否达到生成答案的标准。

**重写后的查询：**
{rewritten_query}

**检索结果：**
{retrieval_results}

**重要规则：**
- 如果检索结果是"未找到相关法律条文"，直接输出："质量评估：0分 - 检索未命中，建议用户咨询专业律师"
- 只对有实际内容的检索结果进行评估

评估维度（每项10分）：
1. 相关性：检索结果与查询的匹配程度
2. 完整性：是否包含足够信息回答问题  
3. 准确性：法律条文的准确性和权威性
4. 覆盖面：结果的多样性和全面性

**阈值判断：总分≥8.0分（80%）才能继续生成答案**

输出格式：
**质量评估结果**
- 相关性：[分数]/10 - [评价]
- 完整性：[分数]/10 - [评价]
- 准确性：[分数]/10 - [评价] 
- 覆盖面：[分数]/10 - [评价]
- 总体评分：[平均分]/10

**阈值判断：**
[PASS/FAIL] - [是否达到8.0分阈值的判断和说明]

**处理建议：**
[如果FAIL，给出改进建议；如果PASS，确认可以生成答案]""",
    description="评估检索质量并进行阈值判断",
    output_key="quality_evaluation",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1024,
    )
)


# 4. 答案生成Agent（基于阈值判断）
answer_generator_agent = LlmAgent(
    name="AnswerGeneratorAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是专业的法律咨询专家。基于质量评估结果决定是否生成法律咨询答案。

**重写后的查询：**
{rewritten_query}

**检索结果：**
{retrieval_results}

**质量评估：**
{quality_evaluation}

**关键规则：**
1. 只有当质量评估显示"PASS"且总分≥8.0时，才生成专业法律咨询
2. 如果评估为"FAIL"或检索未命中，输出标准回复："很抱歉，当前检索到的法律条文质量不足以提供准确的法律咨询。建议您：1）重新描述问题；2）咨询专业律师获得准确建议。"
3. 严格基于检索到的法律条文回答，不得编造内容

生成要求：
- 使用专业但易懂的语言
- 提供具体的法律依据和条文引用
- 给出实用的法律建议
- 说明适用条件和注意事项

输出格式：
**法律咨询答案**

[基于检索条文的专业回答]

**法律依据：**
- [具体条文引用1]
- [具体条文引用2]
- ...

**适用条件：**
[说明适用的具体情况]

**注意事项：**
[重要提醒和建议]

**免责声明：**
本咨询基于现有法律条文，具体适用需结合实际情况，建议咨询专业律师。""",
    description="基于质量阈值判断生成专业法律咨询答案",
    output_key="final_answer",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=2048,
    )
)


# 创建优化的4阶段Sequential Workflow Agent
optimized_agentic_rag_workflow = SequentialAgent(
    name="OptimizedAgenticRAGWorkflow",
    sub_agents=[
        query_rewriter_agent,      # 1. 查询重写
        retrieval_agent,           # 2. 检索执行  
        quality_evaluator_agent,   # 3. 质量评估（含阈值判断）
        answer_generator_agent     # 4. 答案生成（基于阈值）
    ],
    description="优化的智能法律咨询工作流：查询重写 → 检索执行 → 质量评估 → 答案生成（80%阈值控制）"
)


def setup_agent(callback_context: CallbackContext) -> None:
    """初始化agent"""
    global retriever
    
    # 设置DeepSeek API密钥
    if config.deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api_key
    
    # 初始化检索器
    retriever = LocalRetriever()
    
    # 检查索引是否已创建
    if retriever.index is None:
        callback_context.state["index_warning"] = "请运行: python init_index.py 建立法律文本索引"
    else:
        callback_context.state["index_ready"] = f"已加载 {len(retriever.texts)} 条法律条文，启用SequentialAgent工作流"


# 导出为root_agent供ADK使用
root_agent = optimized_agentic_rag_workflow

# 导出供ADK使用
__all__ = ['root_agent']
