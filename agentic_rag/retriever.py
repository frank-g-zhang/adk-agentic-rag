#!/usr/bin/env python3
"""检索器和检索Agent - 基于BGE-M3和Cross-Encoder的法律条文检索"""

import os
import json
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle


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
        """检索器和检索Agent - 基于BGE-M3和Cross-Encoder的法律条文检索"""
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


# 全局检索器
retriever: Optional[LocalRetriever] = None


def get_retriever():
    """获取全局检索器实例"""
    global retriever
    if retriever is None:
        retriever = create_local_retriever()
    return retriever


def execute_retrieval(query: str) -> str:
    """执行检索并返回格式化结果"""
    retriever = get_retriever()
    if retriever.index is None:
        return "⚠️ 法律文本索引尚未建立，请先运行: python init_index.py"
    
    # 执行检索和重排序
    results = retriever.retrieve_and_rerank(query, top_k=10, top_n=5)
    
    if not results:
        return "未找到相关法律条文"
    
    # 格式化检索结果
    formatted_results = []
    for i, result in enumerate(results, 1):
        text = result['text']
        score = result.get('rerank_score', 0)
        metadata = result['metadata']
        
        law_name = metadata.get('law', '未知法律')
        article = metadata.get('article', '未知条款')
        
        formatted_results.append(f"""【检索结果 {i}】(相关性: {score:.3f})
{law_name} {article}
{text}""")
    
    return "\n\n".join(formatted_results)


# 检索执行Agent - 延迟导入避免循环依赖
def create_retrieval_agent():
    """创建检索执行Agent"""
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.genai import types
    
    return LlmAgent(
        name="RetrievalAgent", 
        model=LiteLlm(model="deepseek/deepseek-chat"),
        instruction="""你是检索执行专家。基于重写后的查询执行法律条文检索。

**重写后的查询：**
{rewritten_query}

**执行步骤：**
1. 解析重写后的查询内容
2. 使用主查询进行检索
3. 如果结果不足，尝试备选查询
4. 合并和去重检索结果
5. 按相关性排序

**重要规则：**
- 如果检索工具返回"未找到相关法律条文"，直接输出该结果
- 不要基于空结果生成任何内容

输出检索到的法律条文和统计信息。""",
        description="执行法律条文检索",
        output_key="retrieval_results",
        tools=[execute_retrieval],
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.8,
            max_output_tokens=2048,
        )
    )

# 懒加载检索Agent
retrieval_agent = None

def get_retrieval_agent():
    """获取检索Agent实例"""
    global retrieval_agent
    if retrieval_agent is None:
        retrieval_agent = create_retrieval_agent()
    return retrieval_agent