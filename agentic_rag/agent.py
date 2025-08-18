"""
ADK Web兼容的法律RAG Agent - SequentialAgent工作流架构
"""

import os
from typing import Optional
from google.adk.agents.callback_context import CallbackContext
from .config import LawRAGConfig
from .local_retriever import LocalRetriever
from .optimized_workflow_agent import optimized_agentic_rag_workflow

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
        callback_context.state["index_ready"] = f"已加载 {len(retriever.texts)} 条法律条文，启用SequentialAgent工作流"

# 使用优化的4阶段SequentialAgent工作流
root_agent = optimized_agentic_rag_workflow

# 导出供ADK使用
__all__ = ['root_agent']