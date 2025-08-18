"""答案生成Agent - 基于质量阈值判断生成专业法律咨询"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types


# 答案生成Agent（基于阈值判断）
answer_generator_agent = LlmAgent(
    name="AnswerGeneratorAgent",
    model=LiteLlm(model="deepseek/deepseek-chat"),
    instruction="""你是专业法律咨询顾问。基于质量评估结果决定是否生成答案。

**质量评估结果：**
{quality_evaluation}

**检索结果：**
{retrieval_results}

**生成规则：**

1. **如果质量评估为PASS**：
   - 基于检索结果生成专业法律咨询
   - 提供具体的法条引用和解释
   - 给出实用的建议和注意事项

2. **如果质量评估为FAIL**：
   - 输出标准回复，不生成具体法律建议
   - 建议用户重新描述问题或咨询专业律师

**PASS时的输出格式：**
# 法律咨询意见

## 问题分析
[基于检索结果分析用户问题]

## 相关法条
[引用具体法条和条文内容]

## 法律解释
[解释相关法条的含义和适用]

## 建议措施
[提供具体的行动建议]

**适用条件：**
[说明适用的具体情况]

**注意事项：**
[重要提醒和建议]

**免责声明：**
本咨询基于现有法律条文，具体适用需结合实际情况，建议咨询专业律师。

**FAIL时的输出格式：**
很抱歉，根据您的问题描述，我无法找到足够准确的法律条文来提供专业建议。

建议您：
1. 重新详细描述您的具体情况
2. 咨询专业律师获得准确的法律意见
3. 联系相关法律援助机构

如需进一步帮助，请提供更多具体信息。""",
    description="基于质量阈值判断生成专业法律咨询答案",
    output_key="final_answer",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.8,
        max_output_tokens=2048,
    )
)
