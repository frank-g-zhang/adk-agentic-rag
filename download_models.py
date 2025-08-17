#!/usr/bin/env python3
"""
预下载模型到本地缓存
运行一次即可，后续可从本地加载
"""

import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

def download_models():
    """预下载模型到本地缓存"""
    
    # 模型列表
    models = {
        "embedding": "BAAI/bge-m3",
        "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
    
    # 设置缓存目录
    cache_dir = Path.home() / ".cache" / "sentence_transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 开始预下载模型...")
    
    # 下载嵌入模型
    print(f"📥 下载嵌入模型: {models['embedding']}")
    embedding_model = SentenceTransformer(models['embedding'])
    embedding_cache = cache_dir / models['embedding'].replace('/', '--')
    embedding_model.save(str(embedding_cache))
    print(f"✅ 嵌入模型已保存到: {embedding_cache}")
    
    # 下载交叉编码器
    print(f"📥 下载交叉编码器: {models['cross_encoder']}")
    cross_encoder = CrossEncoder(models['cross_encoder'])
    cross_encoder_cache = cache_dir / models['cross_encoder'].replace('/', '--')
    cross_encoder.save(str(cross_encoder_cache))
    print(f"✅ 交叉编码器已保存到: {cross_encoder_cache}")
    
    print("\n🎉 模型预下载完成！")
    print(f"所有模型已保存到: {cache_dir}")
    print("下次运行时将从本地缓存加载，无需网络连接")

if __name__ == "__main__":
    download_models()