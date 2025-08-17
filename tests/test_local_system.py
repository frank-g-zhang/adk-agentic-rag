#!/usr/bin/env python3
"""
本地RAG系统测试
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.local_retriever import LocalRetriever


class TestLocalRetriever:
    """测试本地检索器"""
    
    def test_initialization(self):
        """测试初始化"""
        retriever = LocalRetriever()
        assert retriever is not None
        assert retriever.embedding_model is not None
        assert retriever.cross_encoder is not None
    
    def test_add_documents(self):
        """测试添加文档"""
        retriever = LocalRetriever()
        
        texts = ["测试文本1", "测试文本2"]
        metadatas = [{"id": 1}, {"id": 2}]
        
        retriever.add_documents(texts, metadatas)
        
        assert len(retriever.texts) == 2
        assert len(retriever.metadatas) == 2
        assert retriever.index is not None
    
    def test_search(self):
        """测试搜索功能"""
        retriever = LocalRetriever()
        
        # 添加测试数据
        texts = [
            "《个人信息保护法》第一条规定：保护个人信息权益",
            "《网络安全法》规定了网络运营者责任",
            "《数据安全法》保障数据安全"
        ]
        metadatas = [
            {"law": "个人信息保护法", "article": "第一条"},
            {"law": "网络安全法", "article": "相关规定"},
            {"law": "数据安全法", "article": "相关规定"}
        ]
        
        retriever.add_documents(texts, metadatas)
        
        # 测试搜索
        results = retriever.search("个人信息保护", top_k=2)
        
        assert len(results) > 0
        assert "个人信息" in results[0]['text']
    
    def test_rerank(self):
        """测试重排序"""
        retriever = LocalRetriever()
        
        documents = [
            {
                'text': '《个人信息保护法》保护个人信息权益',
                'metadata': {'law': '个人信息保护法'},
                'score': 0.8
            },
            {
                'text': '《网络安全法》规定了网络运营者责任',
                'metadata': {'law': '网络安全法'},
                'score': 0.7
            }
        ]
        
        reranked = retriever.rerank("个人信息保护", documents)
        
        assert len(reranked) <= len(documents)
        assert 'rerank_score' in reranked[0]
    
    def test_retrieve_and_rerank(self):
        """测试完整检索流程"""
        retriever = LocalRetriever()
        
        # 添加测试数据
        texts = [
            "《个人信息保护法》第一条规定：保护个人信息权益",
            "《个人信息保护法》第二条规定：个人信息处理原则",
            "《网络安全法》第四十条规定：网络运营者安全义务"
        ]
        metadatas = [
            {"law": "个人信息保护法", "article": "第一条"},
            {"law": "个人信息保护法", "article": "第二条"},
            {"law": "网络安全法", "article": "第四十条"}
        ]
        
        retriever.add_documents(texts, metadatas)
        
        # 测试完整流程
        results = retriever.retrieve_and_rerank("个人信息保护定义", top_k=3, top_n=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        assert 'rerank_score' in results[0]


def test_chinese_law_indexing():
    """测试中国法律文本索引"""
    retriever = LocalRetriever()
    
    # 检查是否自动加载了法律文本
    if retriever.index is not None and len(retriever.texts) > 0:
        print(f"✅ 已加载 {len(retriever.texts)} 条法律条文")
        
        # 测试实际查询
        results = retriever.retrieve_and_rerank("什么是个人信息", top_k=5, top_n=3)
        print(f"✅ 查询结果: {len(results)} 条相关条文")
        
        for result in results:
            print(f"  - {result['metadata'].get('law', '未知')} {result['metadata'].get('article', '未知')}")
    else:
        print("⚠️ 未找到法律文本索引")


if __name__ == "__main__":
    test_chinese_law_indexing()