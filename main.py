#!/usr/bin/env python3
"""
ADK Web兼容的法律RAG系统
"""

from google.adk.agents import Agent
from google.adk.tools import Tool
from app.retriever import create_local_retriever

# 初始化检索器
retriever = create_local_retriever()

@Tool
def retrieve_docs(query: str) -> str:
    """检索中国法律相关条文"""
    try:
        if retriever.index is None:
            return "⚠️ 请先运行: python init_index.py 建立法律文本索引"
        
        results = retriever.retrieve_and_rerank(
            query=query,
            top_k=10,
            top_n=5
        )
        
        if not results:
            return "⚠️ 未找到相关法律条文"
            
        formatted_docs = []
        for i, result in enumerate(results, 1):
            text = result['text']
            score = result['rerank_score']
            metadata = result['metadata']
            
            law_name = metadata.get('law', '未知法律')
            article = metadata.get('article', '未知条款')
            
            formatted_docs.append(
                f"""【相关条文 {i}】(置信度: {score:.3f})
{law_name} {article}
{text}
"""
            )
        
        return "\n---\n".join(formatted_docs)
        
    except Exception as e:
        return f"检索错误: {type(e).__name__}: {e}"

# 创建ADK agent
root_agent = Agent(
    name="root_agent",
    model="deepseek-chat",
    instruction="""你是一个专业的中国法律咨询助手。
请基于提供的法律条文准确回答用户问题。
回答时必须引用具体的法律条文，并解释条文含义。
如果提供的条文不足以回答问题，请明确告知。

工具使用说明：
- 使用retrieve_docs工具查找相关法律条文
- 回答格式：先引用条文，再解释，最后给出结论
- 使用简洁明了的语言""",
    tools=[retrieve_docs]
)

# 导出agent供ADK Web使用
__all__ = ['root_agent']