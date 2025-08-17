"""
法律RAG系统 - ADK Web兼容
"""

from .agent import root_agent

# 导出agent供ADK使用
agent = root_agent

__all__ = ['agent', 'root_agent']