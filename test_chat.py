#!/usr/bin/env python3
"""
简易CLI测试脚本
直接运行法律问答
"""

import os
from app.local_retriever import create_local_retriever

# 初始化检索器
retriever = create_local_retriever()

print("🏛️ 中国法律RAG系统测试")
print("=" * 50)
print("输入问题，输入 'quit' 退出")
print("=" * 50)

def chat():
    """简单的CLI聊天"""
    while True:
        query = input("\n💬 请输入法律问题: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        try:
            results = retriever.retrieve_and_rerank(query, top_k=5, top_n=3)
            
            if not results:
                print("⚠️ 未找到相关法律条文")
                continue
                
            print("\n📋 相关法律条文:")
            print("-" * 30)
            
            for i, result in enumerate(results, 1):
                text = result['text']
                score = result['rerank_score']
                metadata = result['metadata']
                
                law = metadata.get('law', '未知')
                article = metadata.get('article', '未知')
                
                print(f"\n【条文 {i}】(置信度: {score:.3f})")
                print(f"{law} {article}")
                print(text)
                
        except Exception as e:
            print(f"❌ 查询错误: {e}")

if __name__ == "__main__":
    chat()