"""
ADK Web兼容的法律RAG Agent
"""

import os
from typing import Optional
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.lite_llm import LiteLlm
from google.genai import types

from .config import LawRAGConfig
from .local_retriever import LocalRetriever

# 全局配置
config = LawRAGConfig()
retriever: Optional[LocalRetriever] = None

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
        callback_context.state["index_ready"] = f"已加载 {len(retriever.texts)} 条法律条文"

def retrieve_docs(query: str) -> str:
    """检索中国法律相关条文"""
    if retriever is None or retriever.index is None:
        return "⚠️ 法律文本索引尚未建立，请先运行: python init_index.py"
    
    try:
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
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="你是一个专业的中国法律咨询助手，基于中国法律条文提供准确咨询。严格遵循：如未检索到相关条文，必须回复'未查询到相关信息'，禁止编造法律条文。",
    tools=[retrieve_docs],
    before_agent_callback=setup_agent,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=2048,
    )
)

# 导出供ADK使用
__all__ = ['root_agent']