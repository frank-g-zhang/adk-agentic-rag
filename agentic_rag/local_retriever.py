#!/usr/bin/env python3
"""
Local RAG implementation with BGE-M3, FAISS, and Cross-encoder
Removes all Google Cloud dependencies
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import pickle
import warnings
from pathlib import Path


class LocalRetriever:
    """Local vector retriever using BGE-M3 and FAISS"""
    
    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        index_path: str = "data/vectors.index",
        texts_path: str = "data/texts.pkl",
        metadatas_path: str = "data/metadatas.pkl"
    ):
        # 设置本地缓存目录
        cache_dir = Path.home() / ".cache" / "sentence_transformers"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 禁用在线下载警告
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'  # 优先离线模式
        
        # 检查本地缓存
        self.embedding_model = self._load_model_with_cache(
            embedding_model_name, 
            cache_dir / embedding_model_name.replace('/', '--')
        )
        self.cross_encoder = self._load_model_with_cache(
            cross_encoder_name, 
            cache_dir / cross_encoder_name.replace('/', '--')
        )
        
        # 如果本地没有缓存，允许在线下载
        if self.embedding_model is None:
            warnings.warn(f"本地未找到 {embedding_model_name}，将在线下载...")
            os.environ.pop('HF_HUB_OFFLINE', None)
            self.embedding_model = SentenceTransformer(embedding_model_name)
            
        if self.cross_encoder is None:
            warnings.warn(f"本地未找到 {cross_encoder_name}，将在线下载...")
            os.environ.pop('HF_HUB_OFFLINE', None)
            self.cross_encoder = CrossEncoder(cross_encoder_name)
        
        self.index_path = index_path
        self.texts_path = texts_path
        self.metadatas_path = metadatas_path
        
        self.index = None
        self.texts = []
        self.metadatas = []
        
        self._ensure_data_dir()
        self._load_or_create_index()
    
    def _load_model_with_cache(self, model_name: str, cache_path: Path):
        """
        尝试从本地缓存加载模型
        
        Args:
            model_name: HuggingFace模型名称
            cache_path: 本地缓存路径
            
        Returns:
            加载的模型或None（如果本地不存在）
        """
        try:
            if cache_path.exists() and any(cache_path.iterdir()):
                print(f"✅ 从本地缓存加载模型: {cache_path}")
                if 'cross-encoder' in str(model_name):
                    return CrossEncoder(str(cache_path))
                else:
                    return SentenceTransformer(str(cache_path))
            else:
                print(f"⚠️ 本地缓存未找到: {cache_path}")
                return None
        except Exception as e:
            print(f"⚠️ 本地缓存加载失败: {e}")
            return None

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def _load_or_create_index(self):
        """加载现有索引或创建新索引"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.texts_path, 'rb') as f:
                self.texts = pickle.load(f)
            with open(self.metadatas_path, 'rb') as f:
                self.metadatas = pickle.load(f)
            print(f"Loaded existing index with {len(self.texts)} documents")
        else:
            print("No existing index found, will create new one")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """添加文档到索引"""
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # 生成嵌入
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # 创建或更新索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
        
        self.index.add(embeddings)
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        # 保存到磁盘
        faiss.write_index(self.index, self.index_path)
        with open(self.texts_path, 'wb') as f:
            pickle.dump(self.texts, f)
        with open(self.metadatas_path, 'wb') as f:
            pickle.dump(self.metadatas, f)
        
        print(f"Added {len(texts)} documents to index")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if self.index is None or len(self.texts) == 0:
            return []
        
        # 生成查询向量
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True)
        query_vector = np.array(query_vector).astype('float32')
        
        # FAISS搜索
        scores, indices = self.index.search(query_vector, top_k)
        
        # 构建结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadatas[idx],
                    'score': float(score)
                })
        
        return results
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """使用cross-encoder重排序"""
        if not documents:
            return []
        
        # 构建输入对
        pairs = [[query, doc['text']] for doc in documents]
        
        # 计算重排序分数
        scores = self.cross_encoder.predict(pairs)
        
        # 添加到文档并排序
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # 按重排序分数降序排列
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        return reranked[:top_n]
    
    def retrieve_and_rerank(self, query: str, top_k: int = 10, top_n: int = 5) -> List[Dict[str, Any]]:
        """完整检索流程：搜索+重排序"""
        # 步骤1：向量搜索
        search_results = self.search(query, top_k)
        
        # 步骤2：重排序
        reranked_results = self.rerank(query, search_results, top_n)
        
        return reranked_results


def create_local_retriever() -> LocalRetriever:
    """工厂函数创建本地检索器"""
    retriever = LocalRetriever()
    
    # 如果索引为空，加载法律文本
    if retriever.index is None:
        from pathlib import Path
        law_file = Path(__file__).parent.parent / "chinese_law.txt"
        
        if law_file.exists():
            texts = []
            metadatas = []
            
            with open(law_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        texts.append(line)
                        
                        # 提取元数据
                        metadata = {}
                        if '《' in line and '》' in line:
                            law_start = line.find('《')
                            law_end = line.find('》') + 1
                            metadata['law'] = line[law_start:law_end]
                        
                        if '第' in line and '条规定' in line:
                            article_start = line.find('第')
                            article_end = line.find('条规定') + 2
                            metadata['article'] = line[article_start:article_end]
                        
                        metadata['line_number'] = line_num
                        metadatas.append(metadata)
            
            if texts:
                retriever.add_documents(texts, metadatas)
    
    return retriever