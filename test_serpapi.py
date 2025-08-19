#!/usr/bin/env python3
"""测试SerpAPI集成"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from agentic_rag.web_search_agent import web_search_tool

def test_serpapi_integration():
    """测试SerpAPI集成"""
    
    # 检查环境变量
    api_key = os.getenv('SERPAPI_API_KEY')
    if not api_key:
        print("❌ 未设置SERPAPI_API_KEY环境变量")
        print("请设置API密钥：export SERPAPI_API_KEY='your-api-key'")
        return
    
    print(f"✅ API密钥已设置: {api_key[:10]}...")
    
    # 测试搜索
    test_queries = [
        "刑法第17条",
        "个人信息保护法",
        "合同法违约责任"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试搜索: {query}")
        print("-" * 50)
        
        result = web_search_tool(query, max_results=3)
        print(result)
        print("\n" + "="*80)

if __name__ == "__main__":
    test_serpapi_integration()
