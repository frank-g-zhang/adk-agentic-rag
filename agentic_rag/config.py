"""
配置管理
"""

import os
from typing import Optional
from pathlib import Path

class LawRAGConfig:
    """法律RAG系统配置"""
    
    def __init__(self):
        self.model_name = os.getenv("LAW_RAG_MODEL", "deepseek-chat")
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # 数据路径
        self.data_dir = Path("data")
        self.law_file = Path("chinese_law.txt")
        self.index_path = self.data_dir / "vectors.index"
        self.texts_path = self.data_dir / "texts.pkl"
        self.metadatas_path = self.data_dir / "metadatas.pkl"
        
        # 缓存路径
        self.cache_dir = Path.home() / ".cache" / "sentence_transformers"
        
    def validate(self) -> bool:
        """验证配置"""
        if not self.deepseek_api_key:
            print("⚠️ 请设置DEEPSEEK_API_KEY环境变量")
            return False
        return True