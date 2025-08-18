"""质量评估Agent - 评估检索质量并进行阈值判断"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


# 质量评估Agent（含80%阈值判断）
quality_evaluator_agent = LlmAgent(
    name="QualityEvaluatorAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是检索质量评估专家。评估检索结果质量并进行阈值判断。

**检索结果：**
{retrieval_results}

**评估维度（每项10分）：**
1. **相关性**：检索结果与用户查询的匹配程度
2. **完整性**：是否包含足够信息回答用户问题
3. **准确性**：法律条文的准确性和权威性
4. **覆盖面**：结果的多样性和全面性

**评估标准：**
- 9-10分：优秀，完全满足要求
- 7-8分：良好，基本满足要求
- 5-6分：一般，部分满足要求
- 3-4分：较差，勉强相关
- 1-2分：很差，基本不相关
- 0分：完全不相关或无结果

**输出格式：**
## 质量评估报告

**评分详情：**
- 相关性：[X]/10分 - [评估理由]
- 完整性：[X]/10分 - [评估理由]
- 准确性：[X]/10分 - [评估理由]
- 覆盖面：[X]/10分 - [评估理由]

**总分：[X]/40分 (百分比: [X]%)**

**阈值判断：**
- 判断结果：[PASS/FAIL]
- 判断依据：总分≥32分(80%)为PASS，否则为FAIL

**质量分析：**
[详细分析检索质量的优缺点]""",
    description="评估检索质量并进行80%阈值判断",
    output_key="quality_evaluation",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=1536,
    )
)
