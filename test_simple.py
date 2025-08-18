#!/usr/bin/env python3
"""
简化的多Agent架构测试
直接测试工具函数，避免复杂的ADK Context创建
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_retriever_initialization():
    """测试检索器初始化"""
    print("🔍 测试检索器初始化...")
    
    try:
        from agentic_rag.local_retriever import create_local_retriever
        retriever = create_local_retriever()
        
        if retriever.index is None:
            print("⚠️ 请先运行 python init_index.py 建立索引")
            return False
        
        print(f"✅ 已加载 {len(retriever.texts)} 条法律条文")
        return True
        
    except Exception as e:
        print(f"❌ 检索器初始化失败: {e}")
        return False


def test_tool_functions():
    """测试工具函数"""
    print("\n🔧 测试工具函数...")
    
    try:
        from agentic_rag.agent import retrieve_docs, agentic_legal_consultation
        
        test_queries = [
            "什么是劳动合同？",
            "公司拖欠工资怎么办？",
            "合同违约的法律后果"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- 测试 {i}: {query} ---")
            
            # 测试简单检索
            result1 = retrieve_docs(query)
            if "⚠️" not in result1:
                print("✅ 简单检索成功")
                print(f"结果长度: {len(result1)} 字符")
            else:
                print(f"❌ 简单检索失败: {result1}")
                continue
            
            # 测试智能咨询
            result2 = agentic_legal_consultation(query)
            if "⚠️" not in result2:
                print("✅ 智能咨询成功")
                print(f"结果长度: {len(result2)} 字符")
            else:
                print(f"❌ 智能咨询失败: {result2}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工具函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sub_agents():
    """测试Sub Agents定义"""
    print("\n🤖 测试Sub Agents定义...")
    
    try:
        from agentic_rag.sub_agents_adk import (
            query_analyzer,
            query_rewriter,
            retrieval_evaluator,
            answer_evaluator
        )
        
        agents = [
            ("查询分析器", query_analyzer),
            ("查询重写器", query_rewriter), 
            ("检索评估器", retrieval_evaluator),
            ("答案评估器", answer_evaluator)
        ]
        
        for name, agent in agents:
            print(f"✅ {name}: {agent.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sub Agents加载失败: {e}")
        return False


def test_main_agent():
    """测试主Agent定义"""
    print("\n🎯 测试主Agent定义...")
    
    try:
        from agentic_rag.agent import root_agent
        
        print(f"✅ 主Agent: {root_agent.name}")
        print(f"✅ 工具数量: {len(root_agent.tools) if root_agent.tools else 0}")
        print(f"✅ 子Agent数量: {len(root_agent.sub_agents) if root_agent.sub_agents else 0}")
        
        if root_agent.sub_agents:
            for sub_agent in root_agent.sub_agents:
                print(f"  - {sub_agent.name}: {sub_agent.description}")
        
        return True
        
    except Exception as e:
        print(f"❌ 主Agent加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("🧪 简化多Agent架构测试")
    print("="*40)
    
    tests = [
        ("检索器初始化", test_retriever_initialization),
        ("Sub Agents定义", test_sub_agents),
        ("主Agent定义", test_main_agent),
        ("工具函数", test_tool_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"测试: {name}")
        print('='*40)
        
        if test_func():
            passed += 1
            print(f"✅ {name} 通过")
        else:
            print(f"❌ {name} 失败")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！多Agent架构运行正常")
    else:
        print("⚠️ 部分测试失败，请检查配置")


if __name__ == "__main__":
    main()
