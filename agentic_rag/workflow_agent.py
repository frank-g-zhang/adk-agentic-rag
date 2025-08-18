"""
基于SequentialAgent的Agentic RAG工作流实现
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from .local_retriever import LocalRetriever


# 全局检索器
retriever = None


def get_retriever():
    """获取全局检索器实例"""
    global retriever
    if retriever is None:
        from .local_retriever import create_local_retriever
        retriever = create_local_retriever()
    return retriever


def analyze_and_retrieve(query: str) -> str:
    """分析查询并执行检索"""
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


# 1. 查询分析Agent
query_analyzer_agent = LlmAgent(
    name="QueryAnalyzerAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是查询分析专家。分析用户的法律查询，提供结构化分析。

分析维度：
1. 查询类型：事实查询、程序查询、比较查询、分析查询
2. 复杂度：简单、中等、复杂
3. 法律领域：刑法、民法、商法、行政法、综合
4. 关键概念：提取核心法律概念

输出格式：
**查询分析结果**
- 查询类型：[类型]
- 复杂度：[复杂度]
- 法律领域：[领域]
- 关键概念：[概念1, 概念2, ...]
- 优化建议：[如何优化查询表述]""",
    description="分析用户查询的类型、复杂度和法律领域",
    output_key="query_analysis",
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
    instruction="""你是检索执行专家。根据查询分析结果，执行法律条文检索。

**查询分析：**
{query_analysis}

基于分析结果，执行检索并返回相关法律条文。如果查询复杂，可以分解为多个子查询进行检索。

**重要规则：**
- 如果检索工具返回"未找到相关法律条文"，你必须直接输出该结果并停止处理
- 不要尝试基于空结果生成任何内容
- 不要调用其他工具或继续分析

输出相关的法律条文和检索统计信息。""",
    description="执行法律条文检索",
    output_key="retrieval_results",
    tools=[analyze_and_retrieve],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=2048,
    )
)


# 3. 检索评估Agent
retrieval_evaluator_agent = LlmAgent(
    name="RetrievalEvaluatorAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是检索质量评估专家。评估检索结果的质量。

**查询分析：**
{query_analysis}

**检索结果：**
{retrieval_results}

**重要规则：**
- 如果检索结果是"未找到相关法律条文"，直接输出："检索未命中，无法提供法律咨询"，不要继续评估
- 只有在有实际检索结果时才进行质量评估

评估维度：
1. 相关性：检索结果与查询的匹配程度 (1-10分)
2. 完整性：是否包含足够信息回答问题 (1-10分)
3. 准确性：法律条文的准确性和权威性 (1-10分)
4. 覆盖面：结果的多样性和覆盖面 (1-10分)

输出格式：
**检索质量评估**
- 相关性：[分数]/10 - [评价]
- 完整性：[分数]/10 - [评价]  
- 准确性：[分数]/10 - [评价]
- 覆盖面：[分数]/10 - [评价]
- 总体评分：[平均分]/10
- 改进建议：[如果需要改进的话]""",
    description="评估检索结果的质量",
    output_key="retrieval_evaluation",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1024,
    )
)


# 4. 答案生成Agent
answer_generator_agent = LlmAgent(
    name="AnswerGeneratorAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是专业的法律咨询专家。基于检索到的法律条文，为用户提供准确、专业的法律咨询。

**查询分析：**
{query_analysis}

**检索结果：**
{retrieval_results}

**检索评估：**
{retrieval_evaluation}

**重要规则：**
- 如果检索结果或评估显示"未找到相关法律条文"或"检索未命中"，直接输出："很抱歉，未能在法律条文库中找到与您问题相关的内容。建议您咨询专业律师获得准确的法律建议。"
- 绝对不要基于空结果或未命中的检索生成任何法律建议

要求：
1. 严格基于检索到的法律条文回答
2. 使用专业但易懂的语言
3. 提供具体的法律依据和条文引用
4. 如果条文不足以完全回答问题，明确说明
5. 不得编造或推测法律条文

输出格式：
**法律咨询答案**

[基于检索条文的专业回答]

**法律依据：**
- [具体条文引用1]
- [具体条文引用2]
- ...

**注意事项：**
[相关注意事项和建议]""",
    description="基于检索结果生成专业法律咨询答案",
    output_key="final_answer",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=2048,
    )
)


# 5. 答案评估Agent
answer_evaluator_agent = LlmAgent(
    name="AnswerEvaluatorAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是答案质量评估专家。评估生成答案的质量。

**生成答案：**
{final_answer}

**检索结果：**
{retrieval_results}

**重要规则：**
- 如果答案显示"未能在法律条文库中找到"或类似未命中信息，直接输出该答案，不要进行评估
- 只对基于实际法律条文的答案进行质量评估

评估维度：
1. 准确性：答案是否基于检索到的法律条文 (1-10分)
2. 完整性：是否充分回答了用户问题 (1-10分)
3. 逻辑性：答案逻辑是否清晰合理 (1-10分)
4. 实用性：答案是否具有实际指导价值 (1-10分)

输出格式：
**答案质量评估**
- 准确性：[分数]/10 - [评价]
- 完整性：[分数]/10 - [评价]
- 逻辑性：[分数]/10 - [评价]
- 实用性：[分数]/10 - [评价]
- 总体评分：[平均分]/10
- 质量总结：[整体评价]

---

{final_answer}""",
    description="评估答案质量并输出最终结果",
    output_key="evaluated_answer",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=2048,
    )
)


# 创建Sequential Workflow Agent
agentic_rag_workflow = SequentialAgent(
    name="AgenticRAGWorkflow",
    sub_agents=[
        query_analyzer_agent,      # 1. 查询分析
        retrieval_agent,          # 2. 检索执行  
        retrieval_evaluator_agent, # 3. 检索评估
        answer_generator_agent,    # 4. 答案生成
        answer_evaluator_agent     # 5. 答案评估
    ],
    description="智能法律咨询工作流：查询分析 → 检索执行 → 质量评估 → 答案生成 → 答案评估"
)


# 导出为root_agent供ADK使用
root_agent = agentic_rag_workflow
