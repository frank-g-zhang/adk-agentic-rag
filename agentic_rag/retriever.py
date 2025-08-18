#!/usr/bin/env python3
"""检索器和检索Agent - 基于BGE-M3和Cross-Encoder的法律条文检索，支持多路查询"""

import os
import json
import warnings
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import pickle
import jieba


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
        
        # 加载索引和数据
        self.index = None
        self.texts = []
        self.metadatas = []
        self._load_or_create_index()
        
        # 关键词检索相关
        self.bm25_index = None
        self.term_frequencies = None
        self.doc_frequencies = None
        self.avg_doc_length = 0
        self._build_keyword_index()
    
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
        
        # 生成嵌入向量
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # 如果索引不存在，创建新索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # 内积相似度
        
        # 添加向量到索引
        self.index.add(embeddings.astype('float32'))
        
        # 保存文本和元数据
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        
        # 重建关键词索引
        self._build_keyword_index()
        
        # 保存索引和数据
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
                    'score': float(score),
                    'index': int(idx)  # 添加索引字段以便融合时匹配
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
    
    def _build_keyword_index(self):
        """构建BM25关键词索引"""
        if not self.texts:
            return
            
        # 定义停用词
        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '它', '他', '她', '们', '来', '过', '时', '很', '还', '个', '中', '可以', '这个', '现在', '我们', '所以', '但是', '因为', '如果', '虽然', '然后', '或者', '以及', '等等', '比如', '例如'
        }
        
        # 分词并计算词频
        self.term_frequencies = []
        self.doc_frequencies = defaultdict(int)
        total_length = 0
        
        for text in self.texts:
            # 中文分词，启用精确模式
            tokens = list(jieba.cut(text, cut_all=False))
            
            # 过滤停用词和短词，保留法律术语
            filtered_tokens = []
            for token in tokens:
                token = token.strip()
                if len(token) >= 1 and token not in self.stopwords:
                    # 保留数字、法条编号、法律术语
                    if (token.isdigit() or 
                        re.match(r'第\d+条', token) or 
                        re.match(r'第\d+章', token) or
                        re.match(r'《.*》', token) or
                        len(token) >= 2):
                        filtered_tokens.append(token)
            
            doc_tf = Counter(filtered_tokens)
            self.term_frequencies.append(doc_tf)
            total_length += len(filtered_tokens)
            
            # 计算文档频率
            for term in set(filtered_tokens):
                self.doc_frequencies[term] += 1
        
        self.avg_doc_length = total_length / len(self.texts) if self.texts else 0
        
    def keyword_search(self, query: str, top_k: int = 10, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
        """
        BM25关键词搜索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            k1: BM25参数k1
            b: BM25参数b
            
        Returns:
            搜索结果列表
        """
        if not self.texts or not self.term_frequencies:
            return []
            
        # 对查询进行相同的分词和过滤处理
        query_tokens = list(jieba.cut(query, cut_all=False))
        filtered_query_tokens = []
        for token in query_tokens:
            token = token.strip()
            if len(token) >= 1 and token not in self.stopwords:
                if (token.isdigit() or 
                    re.match(r'第\d+条', token) or 
                    re.match(r'第\d+章', token) or
                    re.match(r'《.*》', token) or
                    len(token) >= 2):
                    filtered_query_tokens.append(token)
        
        if not filtered_query_tokens:
            return []
            
        scores = []
        
        for i, doc_tf in enumerate(self.term_frequencies):
            score = 0
            doc_length = sum(doc_tf.values())
            
            for term in filtered_query_tokens:
                if term in doc_tf:
                    tf = doc_tf[term]
                    df = self.doc_frequencies[term]
                    idf = math.log((len(self.texts) - df + 0.5) / (df + 0.5))
                    
                    # BM25公式
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_length / self.avg_doc_length)
                    score += idf * numerator / denominator
            
            scores.append((i, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 构建结果
        results = []
        for doc_idx, score in scores[:top_k]:
            if score > 0:  # 只返回有匹配的结果
                result = {
                    'text': self.texts[doc_idx],
                    'metadata': self.metadatas[doc_idx] if doc_idx < len(self.metadatas) else {},
                    'score': float(score),
                    'index': doc_idx,
                    'search_type': 'keyword'
                }
                results.append(result)
        
        return results
    
    def hybrid_search(self, query: str, top_k: int = 10, vector_weight: float = 0.6, keyword_weight: float = 0.4) -> List[Dict[str, Any]]:
        """
        混合搜索：向量搜索 + 关键词搜索 + 结果融合
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重
            
        Returns:
            融合后的搜索结果
        """
        print(f"🔄 [混合搜索] 执行并行搜索 - 向量权重:{vector_weight:.1f}, 关键词权重:{keyword_weight:.1f}")
        
        # 并行执行两种搜索
        vector_results = self.search(query, top_k * 2)  # 获取更多候选
        keyword_results = self.keyword_search(query, top_k * 2)
        
        print(f"📊 [搜索结果] 向量搜索: {len(vector_results)}个, 关键词搜索: {len(keyword_results)}个")
        
        # 使用RRF (Reciprocal Rank Fusion) 融合结果
        fused_results = self._fuse_results(vector_results, keyword_results, vector_weight, keyword_weight, top_k)
        
        print(f"🎯 [融合完成] 最终返回: {len(fused_results)}个结果")
        return fused_results
    
    def _fuse_results(self, vector_results: List[Dict], keyword_results: List[Dict], 
                     vector_weight: float, keyword_weight: float, top_k: int) -> List[Dict[str, Any]]:
        """
        使用RRF算法融合多路搜索结果
        
        Args:
            vector_results: 向量搜索结果
            keyword_results: 关键词搜索结果
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重
            top_k: 返回结果数量
            
        Returns:
            融合后的结果列表
        """
        k = 20  # 降低RRF参数，增加排名靠前结果的权重
        doc_scores = defaultdict(float)
        doc_info = {}
        
        print(f"🔀 [RRF融合] 开始融合结果 - RRF参数k={k}")
        
        # 归一化权重
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            vector_weight = vector_weight / total_weight
            keyword_weight = keyword_weight / total_weight
        
        print(f"⚖️ [权重归一化] 向量:{vector_weight:.3f}, 关键词:{keyword_weight:.3f}")
        
        # 处理向量搜索结果
        for rank, result in enumerate(vector_results):
            doc_id = result.get('index', rank)
            rrf_score = vector_weight / (k + rank + 1)
            doc_scores[doc_id] += rrf_score
            
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    'text': result['text'],
                    'metadata': result.get('metadata', {}),
                    'vector_score': result.get('score', 0),
                    'keyword_score': 0,
                    'index': doc_id
                }
        
        # 处理关键词搜索结果
        for rank, result in enumerate(keyword_results):
            doc_id = result.get('index', result.get('text', rank))
            # 对关键词搜索给予更高的基础权重
            base_score = keyword_weight * 2  # 增加关键词搜索的基础权重
            rrf_score = base_score / (k + rank + 1)
            doc_scores[doc_id] += rrf_score
            
            if doc_id not in doc_info:
                doc_info[doc_id] = {
                    'text': result['text'],
                    'metadata': result.get('metadata', {}),
                    'vector_score': 0,
                    'keyword_score': result.get('score', 0),
                    'index': doc_id
                }
            else:
                doc_info[doc_id]['keyword_score'] = result.get('score', 0)
        
        # 按融合分数排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"📈 [融合统计] 处理了{len(doc_info)}个唯一文档")
        
        # 构建最终结果
        final_results = []
        for i, (doc_id, fused_score) in enumerate(sorted_docs[:top_k]):
            info = doc_info[doc_id]
            result = {
                'text': info['text'],
                'metadata': info['metadata'],
                'score': fused_score,
                'vector_score': info['vector_score'],
                'keyword_score': info['keyword_score'],
                'index': info['index'],
                'search_type': 'hybrid'
            }
            final_results.append(result)
            
            # 显示前3个结果的融合详情
            if i < 3:
                print(f"🏆 [Top{i+1}] 融合分数:{fused_score:.4f} = 向量:{info['vector_score']:.3f} + 关键词:{info['keyword_score']:.3f}")
        
        return final_results
    
    def smart_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        智能搜索：根据查询类型自动选择最佳搜索策略
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        print(f"🔍 [智能搜索] 开始分析查询: '{query}'")
        
        # 分析查询类型
        query_type = self._analyze_query_type(query)
        
        if query_type == 'exact':
            # 精确查询：优先关键词搜索
            print(f"📍 [查询类型] 精确查询 - 关键词权重70%, 向量权重30%")
            return self.hybrid_search(query, top_k, vector_weight=0.3, keyword_weight=0.7)
        elif query_type == 'semantic':
            # 语义查询：优先向量搜索
            print(f"🧠 [查询类型] 语义查询 - 向量权重80%, 关键词权重20%")
            return self.hybrid_search(query, top_k, vector_weight=0.8, keyword_weight=0.2)
        else:
            # 混合查询：平衡权重
            print(f"⚖️ [查询类型] 混合查询 - 向量权重60%, 关键词权重40%")
            return self.hybrid_search(query, top_k, vector_weight=0.6, keyword_weight=0.4)
    
    def _analyze_query_type(self, query: str) -> str:
        """
        分析查询类型
        
        Args:
            query: 查询文本
            
        Returns:
            查询类型：'exact', 'semantic', 'mixed'
        """
        exact_score = 0
        semantic_score = 0
        
        # 精确查询特征
        exact_patterns = [
            (r'第\d+条', 3),  # 第XX条 - 高权重
            (r'第\d+章', 2),  # 第XX章
            (r'《[^》]+》', 2),  # 法律名称
            (r'\d+年\d+月\d+日', 2),  # 日期
            (r'[A-Z]{2,}', 1),  # 英文缩写
            (r'规定|条款|法条|条文', 2),  # 法律术语
            (r'第\d+款|第\d+项', 2),  # 具体条款项
        ]
        
        for pattern, weight in exact_patterns:
            if re.search(pattern, query):
                exact_score += weight
        
        # 语义查询特征
        semantic_patterns = [
            (r'什么是|何为|定义', 3),  # 定义类问题
            (r'如何|怎样|怎么', 2),  # 方法类问题
            (r'为什么|原因|意义', 2),  # 原因类问题
            (r'概念|原理|性质', 2),  # 概念类问题
            (r'包括|涉及|范围', 1),  # 范围类问题
            (r'区别|不同|差异', 1),  # 比较类问题
        ]
        
        for pattern, weight in semantic_patterns:
            if re.search(pattern, query):
                semantic_score += weight
        
        # 根据得分决定查询类型
        if exact_score >= 3 and exact_score > semantic_score:
            return 'exact'
        elif semantic_score >= 2 and semantic_score > exact_score:
            return 'semantic'
        else:
            return 'mixed'


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
    
    # 执行智能多路查询
    results = retriever.smart_search(query, top_k=10)
    
    # 对结果进行重排序以进一步优化
    if results:
        results = retriever.rerank(query, results, top_n=5)
    
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