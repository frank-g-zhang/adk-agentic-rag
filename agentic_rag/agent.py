"""ADK Web兼容的法律RAG Agent - 优化的4阶段SequentialAgent工作流
查询重写 → 检索 → 评估 → 答案生成
"""

import os
from typing import Optional
from google.adk.agents import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from .config import LawRAGConfig
from .retriever import LocalRetriever
from .query_rewriter import query_rewriter_agent
from .retriever import get_retrieval_agent
from .quality_evaluator import quality_evaluator_agent
from .answer_generator import answer_generator_agent


# 全局配置和检索器
config = LawRAGConfig()
retriever: Optional[LocalRetriever] = None


# 创建优化的4阶段Sequential Workflow Agent
optimized_agentic_rag_workflow = SequentialAgent(
    name="OptimizedAgenticRAGWorkflow",
    sub_agents=[
        query_rewriter_agent,
        get_retrieval_agent(),
        quality_evaluator_agent,
        answer_generator_agent
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
