"""
测试基于SequentialAgent的Agentic RAG工作流
"""

import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agentic_rag.workflow_agent import (
    agentic_rag_workflow,
    query_analyzer_agent,
    retrieval_agent,
    retrieval_evaluator_agent,
    answer_generator_agent,
    answer_evaluator_agent,
    analyze_and_retrieve,
    get_retriever
)


def test_retriever_initialization():
    """测试检索器初始化"""
    retriever = get_retriever()
    assert retriever is not None
    print("✅ 检索器初始化成功")


def test_analyze_and_retrieve_tool():
    """测试检索工具函数"""
    query = "什么是合同违约？"
    result = analyze_and_retrieve(query)
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"✅ 检索工具测试通过，结果长度: {len(result)}")


def test_individual_agents():
    """测试各个子Agent的配置"""
    agents = [
        query_analyzer_agent,
        retrieval_agent,
        retrieval_evaluator_agent,
        answer_generator_agent,
        answer_evaluator_agent
    ]
    
    for agent in agents:
        assert agent.name is not None
        assert agent.model is not None
        assert agent.instruction is not None
        assert agent.output_key is not None
        print(f"✅ {agent.name} 配置正确")


def test_sequential_workflow_configuration():
    """测试SequentialAgent工作流配置"""
    assert agentic_rag_workflow.name == "AgenticRAGWorkflow"
    assert len(agentic_rag_workflow.sub_agents) == 5
    
    # 验证子Agent顺序
    expected_names = [
        "QueryAnalyzerAgent",
        "RetrievalAgent", 
        "RetrievalEvaluatorAgent",
        "AnswerGeneratorAgent",
        "AnswerEvaluatorAgent"
    ]
    
    actual_names = [agent.name for agent in agentic_rag_workflow.sub_agents]
    assert actual_names == expected_names
    print("✅ SequentialAgent工作流配置正确")


def test_output_key_chain():
    """测试输出键链接"""
    expected_keys = [
        "query_analysis",
        "retrieval_results", 
        "retrieval_evaluation",
        "final_answer",
        "evaluated_answer"
    ]
    
    actual_keys = [agent.output_key for agent in agentic_rag_workflow.sub_agents]
    assert actual_keys == expected_keys
    print("✅ 输出键链接正确")


def test_instruction_templates():
    """测试指令模板中的状态变量引用"""
    # 检查retrieval_agent是否引用了query_analysis
    retrieval_instruction = retrieval_agent.instruction
    assert "{query_analysis}" in retrieval_instruction
    assert "{user_query}" in retrieval_instruction
    
    # 检查retrieval_evaluator_agent是否引用了前面的输出
    evaluator_instruction = retrieval_evaluator_agent.instruction
    assert "{user_query}" in evaluator_instruction
    assert "{query_analysis}" in evaluator_instruction
    assert "{retrieval_results}" in evaluator_instruction
    
    # 检查answer_generator_agent是否引用了所有前面的输出
    generator_instruction = answer_generator_agent.instruction
    assert "{user_query}" in generator_instruction
    assert "{query_analysis}" in generator_instruction
    assert "{retrieval_results}" in generator_instruction
    assert "{retrieval_evaluation}" in generator_instruction
    
    print("✅ 指令模板状态变量引用正确")


if __name__ == "__main__":
    print("🚀 开始测试基于SequentialAgent的Agentic RAG工作流...")
    
    try:
        test_retriever_initialization()
        test_analyze_and_retrieve_tool()
        test_individual_agents()
        test_sequential_workflow_configuration()
        test_output_key_chain()
        test_instruction_templates()
        
        print("\n🎉 所有测试通过！SequentialAgent工作流配置正确")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
