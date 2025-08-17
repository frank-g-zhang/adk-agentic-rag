"""
ADK Web入口点
"""

from agentic_rag.agent import root_agent

# 导出agent供ADK使用
agent = root_agent

if __name__ == "__main__":
    print("🚀 法律RAG系统已就绪")
    print("运行: adk web")