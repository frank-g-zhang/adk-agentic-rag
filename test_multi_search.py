#!/usr/bin/env python3
"""测试多路查询功能"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agentic_rag.retriever import LocalRetriever

def test_multi_search():
    """测试多路查询功能"""
    
    # 创建测试数据
    test_texts = [
        "《个人信息保护法》第十三条规定，处理个人信息应当具有明确、合理的目的",
        "个人信息是指以电子或者其他方式记录的与已识别或者可识别的自然人有关的各种信息",
        "什么是个人信息保护的基本原则？包括合法、正当、必要和诚信原则",
        "第二十八条规定，敏感个人信息是一旦泄露或者非法使用，容易导致自然人的人格尊严受到侵害",
        "合同法规定当事人应当按照约定全面履行自己的义务",
        "民法典第一编总则编第一章基本规定了民事权利能力和民事行为能力"
    ]
    
    test_metadatas = [
        {"law": "个人信息保护法", "article": "第十三条"},
        {"law": "个人信息保护法", "article": "定义"},
        {"law": "个人信息保护法", "article": "基本原则"},
        {"law": "个人信息保护法", "article": "第二十八条"},
        {"law": "合同法", "article": "履行义务"},
        {"law": "民法典", "article": "第一编"}
    ]
    
    # 创建检索器并添加文档
    retriever = LocalRetriever()
    
    # 检查是否已有重复数据
    print(f"索引中现有文档数量: {len(retriever.texts)}")
    
    # 检查是否有重复的测试数据
    test_exists = any("《个人信息保护法》第十三条规定，处理个人信息应当具有明确、合理的目的" in text for text in retriever.texts)
    
    if not test_exists:
        retriever.add_documents(test_texts, test_metadatas)
        print(f"添加测试数据后文档数量: {len(retriever.texts)}")
    else:
        print("测试数据已存在，跳过添加")
    
    print("=== 多路查询功能测试 ===\n")
    
    # 测试查询
    test_queries = [
        "第十三条规定",  # 精确查询
        "什么是个人信息",  # 语义查询
        "个人信息保护原则"  # 混合查询
    ]
    
    for query in test_queries:
        print(f"查询: {query}")
        print("-" * 50)
        
        # 1. 向量搜索
        print("1. 向量搜索结果:")
        vector_results = retriever.search(query, top_k=3)
        for i, result in enumerate(vector_results, 1):
            print(f"  {i}. [分数: {result['score']:.3f}] {result['text'][:50]}...")
        
        # 2. 关键词搜索
        print("\n2. 关键词搜索结果:")
        keyword_results = retriever.keyword_search(query, top_k=3)
        if keyword_results:
            for i, result in enumerate(keyword_results, 1):
                print(f"  {i}. [分数: {result['score']:.3f}] {result['text'][:50]}...")
        else:
            print("  无关键词搜索结果")
            
        # 显示查询分词结果
        import jieba
        query_tokens = list(jieba.cut(query, cut_all=False))
        print(f"   查询分词: {query_tokens}")
        
        # 3. 混合搜索
        print("\n3. 混合搜索结果:")
        hybrid_results = retriever.hybrid_search(query, top_k=3)
        for i, result in enumerate(hybrid_results, 1):
            print(f"  {i}. [融合分数: {result['score']:.3f}] [向量: {result['vector_score']:.3f}] [关键词: {result['keyword_score']:.3f}]")
            print(f"      {result['text'][:50]}...")
        
        # 4. 智能搜索
        print("\n4. 智能搜索结果:")
        smart_results = retriever.smart_search(query, top_k=3)
        query_type = retriever._analyze_query_type(query)
        print(f"   检测到查询类型: {query_type}")
        for i, result in enumerate(smart_results, 1):
            print(f"  {i}. [融合分数: {result['score']:.3f}] {result['text'][:50]}...")
        
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_multi_search()
