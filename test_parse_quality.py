#!/usr/bin/env python3
"""测试质量评估结果解析功能"""

from agentic_rag.agent import ConditionalWorkflowAgent

def test_parse_quality_evaluation():
    """测试解析质量评估结果"""
    
    # 创建测试实例
    agent = ConditionalWorkflowAgent()
    
    # 测试用例1：带方括号的PASS结果
    evaluation_text_1 = """质量评估报告
评分详情：
相关性：[9]/10分 - 检索结果直接回答了用户关于个人信息合法使用情形的问题
完整性：[9]/10分 - 涵盖了《个人信息保护法》中所有合法使用个人信息的法定情形
准确性：[10]/10分 - 所有引用法条内容与官方文本完全一致
覆盖面：[8]/10分 - 除核心法条外，还包含关联条款

总分：[36]/40分 (百分比: 90%)

阈值判断：
判断结果：[PASS]
判断依据：总分36分＞32分阈值（80%）"""

    score, passed = agent._parse_quality_evaluation(evaluation_text_1)
    print(f"测试1 - 分数: {score}, 通过: {passed}")
    assert score == 90, f"期望分数90，实际{score}"
    assert passed == True, f"期望通过True，实际{passed}"
    
    # 测试用例2：不带方括号的FAIL结果
    evaluation_text_2 = """质量评估报告
总分：25/40分 (百分比: 62%)
阈值判断：
判断结果：FAIL
判断依据：总分低于32分阈值"""

    score, passed = agent._parse_quality_evaluation(evaluation_text_2)
    print(f"测试2 - 分数: {score}, 通过: {passed}")
    assert score == 62, f"期望分数62，实际{score}"
    assert passed == False, f"期望通过False，实际{passed}"
    
    print("所有测试通过！")

if __name__ == "__main__":
    test_parse_quality_evaluation()
