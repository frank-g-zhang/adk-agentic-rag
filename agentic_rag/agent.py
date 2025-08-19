"""ADK Web兼容的法律RAG Agent - 支持条件分支的智能法律咨询工作流
条件分支工作流：查询重写 → 检索 → 评估 → 条件分支（直接答案 vs 互联网搜索补救）
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.models.lite_llm import LiteLlm
from typing_extensions import override
from .config import LawRAGConfig
from .retriever import LocalRetriever
from .query_rewriter import query_rewriter_agent
from .retriever import get_retrieval_agent
from .quality_evaluator import quality_evaluator_agent
from .answer_generator import answer_generator_agent
from .web_search_agent import web_search_agent

# 配置日志
logger = logging.getLogger(__name__)


# 全局配置和检索器
config = LawRAGConfig()
retriever: Optional[LocalRetriever] = None


class ConditionalWorkflowAgent(BaseAgent):
    """
    条件工作流Agent - 实现带有条件分支的法律RAG工作流
    
    工作流程：
    1. 查询重写
    2. 本地检索
    3. 质量评估
    4. 条件分支：
       - 质量达标 → 直接生成答案
       - 质量不达标 → 触发互联网搜索 → 生成增强答案
    """
    
    def __init__(self, 
                 name: str = "ConditionalWorkflowAgent",
                 description: str = "支持条件分支的智能法律咨询工作流",
                 **kwargs):
        
        # 初始化所有子Agent
        query_rewriter = query_rewriter_agent
        retrieval_agent = get_retrieval_agent()
        quality_evaluator = quality_evaluator_agent
        answer_generator = answer_generator_agent
        web_search_agent_instance = web_search_agent
        
        
        # 构建sub_agents列表
        sub_agents = [
            query_rewriter,
            retrieval_agent,
            quality_evaluator,
            answer_generator,
            web_search_agent_instance
        ]
        
        super().__init__(
            name=name,
            description=description,
            sub_agents=sub_agents,
            **kwargs
        )
        
        # 使用object.__setattr__绕过Pydantic的字段验证
        object.__setattr__(self, 'query_rewriter', query_rewriter)
        object.__setattr__(self, 'retrieval_agent', retrieval_agent)
        object.__setattr__(self, 'quality_evaluator', quality_evaluator)
        object.__setattr__(self, 'answer_generator', answer_generator)
        object.__setattr__(self, 'web_search_agent', web_search_agent_instance)

    def _parse_quality_evaluation(self, evaluation_text: str) -> tuple[int, bool]:
        """
        解析质量评估结果，提取分数和通过状态
        
        Args:
            evaluation_text: 质量评估Agent的输出文本
            
        Returns:
            tuple: (quality_score, quality_passed)
        """
        import re
        
        quality_score = 0
        quality_passed = False
        
        try:
            # 优先解析总分 - 查找 "总分：[X]/40分" 或 "总分：X/40分"
            score_pattern = r'总分：\[?(\d+)\]?/40分'
            score_match = re.search(score_pattern, evaluation_text)
            if score_match:
                quality_score = int(score_match.group(1))
            else:
                # 如果没有找到总分，才使用百分比 - 查找 "(百分比: X.X%)" 支持小数
                percentage_pattern = r'\(百分比:\s*(\d+(?:\.\d+)?)%\)'
                percentage_match = re.search(percentage_pattern, evaluation_text)
                if percentage_match:
                    percentage = float(percentage_match.group(1))
                    quality_score = int(percentage)  # 转换为整数分数
            
            # 解析判断结果 - 查找 "判断结果：[PASS]" 或 "判断结果：PASS"
            result_pattern = r'判断结果：\[?(PASS|FAIL)\]?'
            result_match = re.search(result_pattern, evaluation_text)
            if result_match:
                quality_passed = result_match.group(1) == 'PASS'
            
            logger.info(f"解析质量评估结果: 分数={quality_score}, 通过={quality_passed}")
            
        except Exception as e:
            logger.error(f"解析质量评估结果失败: {str(e)}")
            logger.debug(f"评估文本: {evaluation_text}")
            
        return quality_score, quality_passed

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """执行条件工作流的核心逻辑"""
        
        logger.info(f"[{self.name}] 开始执行条件分支法律RAG工作流")
        
        try:
            # 阶段1: 查询重写
            logger.info(f"[{self.name}] 阶段1: 执行查询重写...")
            async for event in self.query_rewriter.run_async(ctx):
                logger.debug(f"[{self.name}] 查询重写事件: {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
            
            rewritten_query = ctx.session.state.get("rewritten_query", "")
            if not rewritten_query:
                logger.error(f"[{self.name}] 查询重写失败，中止工作流")
                return
            
            logger.info(f"[{self.name}] 查询重写完成: {rewritten_query}")
            
            # 阶段2: 本地检索
            logger.info(f"[{self.name}] 阶段2: 执行本地检索...")
            async for event in self.retrieval_agent.run_async(ctx):
                logger.debug(f"[{self.name}] 检索事件: {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
            
            retrieval_results = ctx.session.state.get("retrieval_results", [])
            if not retrieval_results:
                logger.error(f"[{self.name}] 本地检索失败，中止工作流")
                return
            
            logger.info(f"[{self.name}] 本地检索完成，获得 {len(retrieval_results)} 个结果")
            
            # 阶段3: 质量评估
            logger.info(f"[{self.name}] 阶段3: 执行质量评估...")
            async for event in self.quality_evaluator.run_async(ctx):
                logger.debug(f"[{self.name}] 质量评估事件: {event.model_dump_json(indent=2, exclude_none=True)}")
                yield event
            
            # 解析质量评估结果
            quality_evaluation = ctx.session.state.get("quality_evaluation", "")
            quality_score, quality_passed = self._parse_quality_evaluation(quality_evaluation)
            
            # 将解析结果存储到session state中
            ctx.session.state["quality_score"] = quality_score
            ctx.session.state["quality_passed"] = quality_passed
            
            logger.info(f"[{self.name}] 质量评估完成: 分数={quality_score}, 通过={quality_passed}")
            
            # 阶段4: 条件分支决策
            # 使用quality_passed作为主要判断依据，它已经基于80%阈值进行了判断
            if quality_passed:
                # 直接路径：质量达标，直接生成答案
                logger.info(f"[{self.name}] 选择直接路径: 质量达标，直接生成答案")
                
                # 设置空的网络搜索结果，确保模板变量存在
                ctx.session.state["web_search_results"] = "无网络搜索结果"
                
                async for event in self.answer_generator.run_async(ctx):
                    logger.debug(f"[{self.name}] 答案生成事件: {event.model_dump_json(indent=2, exclude_none=True)}")
                    yield event
                
                # 确保有最终答案 - answer_generator的output_key是"final_answer"
                final_answer = ctx.session.state.get("final_answer", "")
                if final_answer:
                    logger.info(f"[{self.name}] 直接路径完成，生成最终答案")
                else:
                    ctx.session.state["final_answer"] = "抱歉，无法生成满意的答案。"
                    logger.warning(f"[{self.name}] 直接路径答案生成失败，使用默认回复")
                    
            else:
                # 补救路径：质量不达标，触发互联网搜索
                logger.info(f"[{self.name}] 选择补救路径: 质量不达标，触发互联网搜索")
                
                # 4a. 执行网络搜索
                logger.info(f"[{self.name}] 阶段4a: 执行网络搜索...")
                async for event in self.web_search_agent.run_async(ctx):
                    logger.debug(f"[{self.name}] 网络搜索事件: {event.model_dump_json(indent=2, exclude_none=True)}")
                    yield event
                
                web_results = ctx.session.state.get("web_search_results", [])
                logger.info(f"[{self.name}] 网络搜索完成，获得 {len(web_results)} 个结果")
                
                # 4b. 直接使用增强答案Agent处理本地+网络结果
                logger.info(f"[{self.name}] 阶段4b: 使用本地和网络结果生成最终答案...")
                
                # 设置双数据源输入供answer_generator使用
                ctx.session.state["web_search_results"] = web_results if web_results else "无网络搜索结果"
                
                # 使用answer_generator处理本地+网络结果
                async for event in self.answer_generator.run_async(ctx):
                    logger.debug(f"[{self.name}] 答案生成事件: {event.model_dump_json(indent=2, exclude_none=True)}")
                    yield event
                
                # 确保有最终答案
                if not ctx.session.state.get("final_answer"):
                    ctx.session.state["final_answer"] = "基于现有信息，我尽力为您提供法律建议，但建议您咨询专业律师获取更准确的意见。"
                    logger.warning(f"[{self.name}] 答案生成失败，使用默认回复")
                else:
                    logger.info(f"[{self.name}] 补救路径完成，生成最终答案")
            
            logger.info(f"[{self.name}] 条件分支工作流执行完成")
            
        except Exception as e:
            # 错误处理：确保总是有回复
            error_message = f"处理过程中出现错误：{str(e)}。建议您重新提问或咨询专业律师。"
            ctx.session.state["final_answer"] = error_message
            logger.error(f"[{self.name}] 工作流执行出错: {str(e)}", exc_info=True)
            
            # 生成错误事件 - 使用兼容的事件类型
            try:
                from google.adk.events import TextEvent
                error_event = TextEvent(text=error_message)
                yield error_event
            except ImportError:
                # 如果TextEvent不可用，使用其他事件类型
                from google.adk.events import Event
                error_event = Event(type="text", data={"text": error_message})
                yield error_event


# 创建条件工作流实例
conditional_workflow_agent = ConditionalWorkflowAgent(
    name="ConditionalWorkflowAgent",
    description="支持条件分支的智能法律咨询工作流：查询重写 → 检索执行 → 质量评估 → 条件分支（直接答案 vs 互联网搜索补救）"
)

# 支持条件分支的增强工作流
enhanced_agentic_rag_workflow = conditional_workflow_agent


async def execute_conditional_workflow(user_query: str) -> str:
    """
    执行条件工作流的便捷函数
    
    Args:
        user_query: 用户查询
        
    Returns:
        最终答案字符串
    """
    from google.adk.agents.session import Session
    from google.adk.agents.invocation_context import InvocationContext
    
    # 创建会话和上下文
    session = Session()
    session.state["user_query"] = user_query
    
    ctx = InvocationContext(
        invocation_id="conditional_workflow_test",
        agent=conditional_workflow_agent,
        session=session
    )
    
    # 执行工作流
    final_answer = ""
    async for event in conditional_workflow_agent.run_async(ctx):
        # 处理事件，提取最终结果
        pass
    
    # 获取最终答案
    final_answer = session.state.get("final_answer", "执行工作流时出现错误")
    
    return final_answer



def setup_agent(callback_context: CallbackContext) -> None:
    """初始化agent"""
    global retriever
    
    # 设置DeepSeek API密钥
    if config.deepseek_api_key:
        os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api_key
    
    # 初始化检索器
    retriever = LocalRetriever()
    
    # 检查索引是否已创建
    if retriever.index is None:
        callback_context.state["index_warning"] = "请运行: python init_index.py 建立法律文本索引"
    else:
        callback_context.state["index_ready"] = f"已加载 {len(retriever.texts)} 条法律条文，启用条件分支工作流"


# 导出为root_agent供ADK使用
root_agent = conditional_workflow_agent

# 导出供ADK使用
__all__ = ['root_agent', 'conditional_workflow_agent', 'execute_conditional_workflow', 'ConditionalWorkflowAgent']
