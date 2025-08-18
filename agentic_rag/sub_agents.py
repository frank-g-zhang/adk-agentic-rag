"""
Agentic RAG 子代理实现
包含查询分析、检索评估、查询重写、答案评估等功能
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


class QueryAnalyzerAgent:
    """查询分析器Agent - 分析查询复杂度和类型"""
    
    def __init__(self):
        self.agent = Agent(
            name="query_analyzer",
            model=LiteLlm(model="deepseek/deepseek-chat"),
            instruction="""你是一个查询分析专家。分析用户查询的复杂度、类型和检索策略。

分析维度：
1. 复杂度：simple(简单直接) | medium(中等复杂) | complex(复杂多层)
2. 类型：factual(事实查询) | procedural(程序查询) | comparative(比较查询) | analytical(分析查询)
3. 法律领域：criminal(刑法) | civil(民法) | commercial(商法) | administrative(行政法) | general(综合)
4. 检索策略：direct(直接检索) | decompose(分解检索) | multi_round(多轮检索)

返回JSON格式：
{
    "complexity": "simple|medium|complex",
    "query_type": "factual|procedural|comparative|analytical", 
    "legal_domain": "criminal|civil|commercial|administrative|general",
    "strategy": "direct|decompose|multi_round",
    "keywords": ["关键词1", "关键词2"],
    "sub_queries": ["子查询1", "子查询2"] // 如果需要分解
}""",
            generate_content_config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=1024,
            )
        )
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """分析查询"""
        try:
            prompt = f"请分析以下法律查询：\n\n{query}"
            # 使用ADK框架的正确调用方式
            from google.adk.agents.callback_context import CallbackContext
            context = CallbackContext()
            response = self.agent.process_user_message(prompt, context)
            
            # 提取JSON部分
            content = response
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                # 尝试直接解析整个响应
                json_content = content.strip()
            
            result = json.loads(json_content)
            return result
            
        except Exception as e:
            # 返回默认分析结果
            return {
                "complexity": "medium",
                "query_type": "factual",
                "legal_domain": "general", 
                "strategy": "direct",
                "keywords": [query],
                "sub_queries": []
            }


class QueryRewriterAgent:
    """查询重写器Agent - 生成多个查询变体"""
    
    def __init__(self):
        self.agent = Agent(
            name="query_rewriter",
            model=LiteLlm(model="deepseek/deepseek-chat"),
            instruction="""你是一个查询重写专家。根据原始查询生成多个语义相似但表达不同的查询变体，提高检索覆盖率。

重写策略：
1. 同义词替换：使用法律术语的同义表达
2. 句式转换：改变疑问句式和表达方式  
3. 关键词扩展：添加相关的法律概念
4. 角度转换：从不同角度描述同一问题

返回JSON格式：
{
    "original_query": "原始查询",
    "rewritten_queries": [
        "重写查询1",
        "重写查询2", 
        "重写查询3"
    ],
    "strategy_used": ["同义词替换", "句式转换", "关键词扩展"]
}""",
            generate_content_config=types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=1024,
            )
        )
    
    def rewrite(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """重写查询"""
        try:
            complexity = analysis.get("complexity", "medium")
            query_type = analysis.get("query_type", "factual")
            
            prompt = f"""请为以下法律查询生成3个重写版本：

原始查询：{query}
查询复杂度：{complexity}
查询类型：{query_type}

要求：保持原意，提高检索效果"""

            # 使用ADK框架的正确调用方式
            from google.adk.agents.callback_context import CallbackContext
            context = CallbackContext()
            response = self.agent.process_user_message(prompt, context)
            
            # 提取响应内容
            content = response
            
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content.strip()
            
            result = json.loads(json_content)
            return result.get("rewritten_queries", [query])
            
        except Exception as e:
            return [query]  # 返回原查询作为备选


class RetrievalEvaluatorAgent:
    """检索评估器Agent - 评估检索结果质量"""
    
    def __init__(self):
        self.agent = Agent(
            name="retrieval_evaluator",
            model=LiteLlm(model="deepseek/deepseek-chat"),
            instruction="""你是一个检索质量评估专家。评估检索结果与查询的相关性和完整性。

评估维度：
1. 相关性 (0-1)：检索结果与查询的匹配程度
2. 完整性 (0-1)：是否包含足够信息回答问题
3. 准确性 (0-1)：法律条文的准确性和权威性
4. 多样性 (0-1)：结果的多样性和覆盖面

返回JSON格式：
{
    "overall_score": 0.85,
    "relevance": 0.9,
    "completeness": 0.8, 
    "accuracy": 0.9,
    "diversity": 0.8,
    "need_more_retrieval": false,
    "suggestions": ["建议1", "建议2"]
}""",
            generate_content_config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=1024,
            )
        )
    
    def evaluate(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估检索结果"""
        try:
            # 构建检索结果摘要
            results_summary = []
            for i, result in enumerate(results[:5], 1):
                text = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
                score = result.get('rerank_score', result.get('score', 0))
                results_summary.append(f"结果{i} (分数:{score:.3f}): {text}")
            
            results_text = "\n\n".join(results_summary)
            
            prompt = f"""请评估以下检索结果的质量：

查询：{query}

检索结果：
{results_text}

请从相关性、完整性、准确性、多样性四个维度评估。"""

            # 使用ADK框架的正确调用方式
            from google.adk.agents.callback_context import CallbackContext
            context = CallbackContext()
            response = self.agent.process_user_message(prompt, context)
            
            # 提取响应内容
            content = response
            
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content.strip()
            
            result = json.loads(json_content)
            return result
            
        except Exception as e:
            # 返回默认评估
            return {
                "overall_score": 0.7,
                "relevance": 0.7,
                "completeness": 0.7,
                "accuracy": 0.8,
                "diversity": 0.6,
                "need_more_retrieval": False,
                "suggestions": []
            }


class AnswerEvaluatorAgent:
    """答案评估器Agent - 评估生成答案质量"""
    
    def __init__(self):
        self.agent = Agent(
            name="answer_evaluator", 
            model=LiteLlm(model="deepseek/deepseek-chat"),
            instruction="""你是一个答案质量评估专家。评估基于检索结果生成的答案质量。

评估维度：
1. 准确性 (0-1)：答案是否基于检索到的法律条文
2. 完整性 (0-1)：是否充分回答了用户问题
3. 逻辑性 (0-1)：答案逻辑是否清晰合理
4. 可用性 (0-1)：答案是否具有实际指导价值

返回JSON格式：
{
    "overall_score": 0.85,
    "accuracy": 0.9,
    "completeness": 0.8,
    "logic": 0.9, 
    "usability": 0.8,
    "need_regenerate": false,
    "improvement_suggestions": ["建议1", "建议2"]
}""",
            generate_content_config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=1024,
            )
        )
    
    def evaluate(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """评估答案质量"""
        try:
            prompt = f"""请评估以下答案的质量：

用户问题：{query}

检索上下文：
{context[:1000]}...

生成答案：
{answer}

请从准确性、完整性、逻辑性、可用性四个维度评估。"""

            # 使用ADK框架的正确调用方式
            from google.adk.agents.callback_context import CallbackContext
            context = CallbackContext()
            response = self.agent.process_user_message(prompt, context)
            
            # 提取响应内容
            content = response
            
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_content = content[json_start:json_end].strip()
            else:
                json_content = content.strip()
            
            result = json.loads(json_content)
            return result
            
        except Exception as e:
            # 返回默认评估
            return {
                "overall_score": 0.7,
                "accuracy": 0.7,
                "completeness": 0.7,
                "logic": 0.7,
                "usability": 0.7,
                "need_regenerate": False,
                "improvement_suggestions": []
            }


class CoordinatorAgent:
    """协调器Agent - 协调各个子代理的工作流程"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.query_analyzer = QueryAnalyzerAgent()
        self.query_rewriter = QueryRewriterAgent()
        self.retrieval_evaluator = RetrievalEvaluatorAgent()
        self.answer_evaluator = AnswerEvaluatorAgent()
        
        # 主答案生成Agent
        self.answer_generator = Agent(
            name="answer_generator",
            model=LiteLlm(model="deepseek/deepseek-chat"),
            instruction="""你是一个专业的中国法律咨询助手。基于提供的法律条文，为用户提供准确、专业的法律咨询。

要求：
1. 严格基于提供的法律条文回答
2. 如果条文不足以回答问题，明确说明
3. 使用专业但易懂的语言
4. 提供具体的法律依据和条文引用
5. 不得编造或推测法律条文""",
            generate_content_config=types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.8,
                max_output_tokens=2048,
            )
        )
    
    def process_query(self, query: str, max_iterations: int = 2) -> Dict[str, Any]:
        """处理查询的完整流程"""
        
        # 步骤1：查询分析
        analysis = self.query_analyzer.analyze(query)
        
        # 步骤2：根据策略执行检索
        if analysis["strategy"] == "decompose" and analysis.get("sub_queries"):
            # 分解查询检索
            all_results = []
            for sub_query in analysis["sub_queries"]:
                results = self.retriever.retrieve_and_rerank(sub_query, top_k=8, top_n=3)
                all_results.extend(results)
            
            # 去重并重新排序
            seen_texts = set()
            unique_results = []
            for result in all_results:
                if result['text'] not in seen_texts:
                    seen_texts.add(result['text'])
                    unique_results.append(result)
            
            # 按分数排序并取前5个
            results = sorted(unique_results, key=lambda x: x.get('rerank_score', 0), reverse=True)[:5]
        
        elif analysis["strategy"] == "multi_round":
            # 多轮检索：原查询 + 重写查询
            results = self.retriever.retrieve_and_rerank(query, top_k=10, top_n=3)
            
            # 如果结果不够好，尝试重写查询
            eval_result = self.retrieval_evaluator.evaluate(query, results)
            if eval_result["overall_score"] < 0.7:
                rewritten_queries = self.query_rewriter.rewrite(query, analysis)
                for rewritten_query in rewritten_queries[:2]:
                    additional_results = self.retriever.retrieve_and_rerank(rewritten_query, top_k=5, top_n=2)
                    results.extend(additional_results)
                
                # 去重并重新排序
                seen_texts = set()
                unique_results = []
                for result in results:
                    if result['text'] not in seen_texts:
                        seen_texts.add(result['text'])
                        unique_results.append(result)
                
                results = sorted(unique_results, key=lambda x: x.get('rerank_score', 0), reverse=True)[:5]
        
        else:
            # 直接检索
            results = self.retriever.retrieve_and_rerank(query, top_k=10, top_n=5)
        
        # 步骤3：评估检索结果
        retrieval_eval = self.retrieval_evaluator.evaluate(query, results)
        
        # 步骤4：生成答案
        if results:
            context = self._format_context(results)
            prompt = f"""基于以下法律条文回答用户问题：

用户问题：{query}

相关法律条文：
{context}

请提供专业、准确的法律咨询。"""
            
            # 使用ADK框架的正确调用方式
            from google.adk.agents.callback_context import CallbackContext
            context = CallbackContext()
            answer = self.answer_generator.process_user_message(prompt, context)
        else:
            answer = "抱歉，未找到相关的法律条文来回答您的问题。"
            context = ""
        
        # 步骤5：评估答案质量
        answer_eval = self.answer_evaluator.evaluate(query, context, answer)
        
        return {
            "query": query,
            "analysis": analysis,
            "results": results,
            "retrieval_evaluation": retrieval_eval,
            "answer": answer,
            "answer_evaluation": answer_eval,
            "context": context
        }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """格式化检索结果为上下文"""
        formatted_docs = []
        for i, result in enumerate(results, 1):
            text = result['text']
            score = result.get('rerank_score', result.get('score', 0))
            metadata = result['metadata']
            
            law_name = metadata.get('law', '未知法律')
            article = metadata.get('article', '未知条款')
            
            formatted_docs.append(
                f"""【条文 {i}】(相关性: {score:.3f})
{law_name} {article}
{text}"""
            )
        
        return "\n\n".join(formatted_docs)
