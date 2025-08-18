"""
测试优化的4阶段SequentialAgent工作流
"""

import pytest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agentic_rag.agent import (
    optimized_agentic_rag_workflow,
    query_rewriter_agent,
    retrieval_agent,
    quality_evaluator_agent,
    answer_generator_agent,
    execute_retrieval,
    get_retriever
)


def test_optimized_workflow_configuration():
    """测试优化工作流配置"""
    assert optimized_agentic_rag_workflow.name == "OptimizedAgenticRAGWorkflow"
    assert len(optimized_agentic_rag_workflow.sub_agents) == 4
    
    # 验证4阶段Agent顺序
    expected_names = [
        "QueryRewriterAgent",
        "RetrievalAgent", 
        "QualityEvaluatorAgent",
        "AnswerGeneratorAgent"
    ]
    
    actual_names = [agent.name for agent in optimized_agentic_rag_workflow.sub_agents]
    assert actual_names == expected_names
    print("✅ 优化工作流配置正确")


def test_output_key_chain():
    """测试输出键链接"""
    expected_keys = [
        "rewritten_query",
        "retrieval_results", 
        "quality_evaluation",
        "final_answer"
    ]
    
    actual_keys = [agent.output_key for agent in optimized_agentic_rag_workflow.sub_agents]
    assert actual_keys == expected_keys
    print("✅ 输出键链接正确")


def test_retrieval_tool():
    """测试检索工具"""
    query = "什么是合同违约？"
    result = execute_retrieval(query)
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"✅ 检索工具测试通过，结果长度: {len(result)}")


def test_agent_instructions():
    """测试Agent指令中的关键要素"""
    
    # 查询重写Agent
    rewriter_instruction = query_rewriter_agent.instruction
    assert "重写策略" in rewriter_instruction
    assert "多个查询变体" in rewriter_instruction
    
    # 检索Agent
    retrieval_instruction = retrieval_agent.instruction
    assert "{rewritten_query}" in retrieval_instruction
    assert "未找到相关法律条文" in retrieval_instruction
    
    # 质量评估Agent
    evaluator_instruction = quality_evaluator_agent.instruction
    assert "阈值判断" in evaluator_instruction
    assert "8.0分" in evaluator_instruction
    assert "PASS/FAIL" in evaluator_instruction
    
    # 答案生成Agent
    generator_instruction = answer_generator_agent.instruction
    assert "PASS" in generator_instruction
    assert "FAIL" in generator_instruction
    assert "质量评估" in generator_instruction
    
    print("✅ Agent指令关键要素检查通过")


def test_threshold_mechanism():
    """测试阈值机制相关配置"""
    evaluator_instruction = quality_evaluator_agent.instruction
    generator_instruction = answer_generator_agent.instruction
    
    # 检查阈值设置
    assert "≥8.0分" in evaluator_instruction or ">=8.0" in evaluator_instruction
    assert "80%" in evaluator_instruction
    
    # 检查答案生成的阈值判断
    assert "总分≥8.0" in generator_instruction or "PASS" in generator_instruction
    
    print("✅ 阈值机制配置正确")


def test_state_variable_references():
    """测试状态变量引用"""
    # 检索Agent引用rewritten_query
    assert "{rewritten_query}" in retrieval_agent.instruction
    
    # 质量评估Agent引用前面的输出
    evaluator_instruction = quality_evaluator_agent.instruction
    assert "{rewritten_query}" in evaluator_instruction
    assert "{retrieval_results}" in evaluator_instruction
    
    # 答案生成Agent引用所有前面的输出
    generator_instruction = answer_generator_agent.instruction
    assert "{rewritten_query}" in generator_instruction
    assert "{retrieval_results}" in generator_instruction
    assert "{quality_evaluation}" in generator_instruction
    
    print("✅ 状态变量引用正确")


if __name__ == "__main__":
    print("🚀 开始测试优化的4阶段SequentialAgent工作流...")
    
    try:
        test_optimized_workflow_configuration()
        test_output_key_chain()
        test_retrieval_tool()
        test_agent_instructions()
        test_threshold_mechanism()
        test_state_variable_references()
        
        print("\n🎉 所有测试通过！优化的4阶段工作流配置正确")
        print("\n📊 工作流程：")
        print("1. 查询重写 → 优化用户查询表述")
        print("2. 检索执行 → 基于重写查询检索法律条文")
        print("3. 质量评估 → 评估检索质量，80%阈值判断")
        print("4. 答案生成 → 仅在质量达标时生成专业咨询")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
