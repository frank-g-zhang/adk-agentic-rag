#!/usr/bin/env python3
"""测试Agent工作流中的多路查询功能"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agentic_rag.retriever import execute_retrieval

def test_agent_retrieval():
    """测试Agent工作流中的检索功能"""
    
    test_queries = [
        "第十三条规定",
        "什么是个人信息", 
        "个人信息保护原则"
    ]
    
    print("=== Agent工作流检索测试 ===\n")
    
    for query in test_queries:
        print(f"查询: {query}")
        print("-" * 50)
        
        result = execute_retrieval(query)
        print(result)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_agent_retrieval()
