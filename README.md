# 智能法律咨询RAG系统

基于Google ADK框架的智能法律咨询系统，采用优化的4阶段Sequential工作流，提供专业的中文法律咨询服务。

## 🏗️ 系统架构

### 4阶段Sequential工作流

```
用户查询 → 查询重写 → 检索执行 → 质量评估 → 答案生成
```

1. **查询重写阶段** - 优化用户查询，提取法律关键词，生成多个查询变体
2. **检索执行阶段** - 基于BGE-M3和Cross-Encoder的混合检索，返回相关法律条文
3. **质量评估阶段** - 评估检索质量，进行80%阈值判断（PASS/FAIL）
4. **答案生成阶段** - 仅在质量达标时生成专业法律咨询，否则建议重新描述问题

### 增强版条件分支工作流（推荐）

项目现在支持两种工作流模式：

#### 1. ConditionalWorkflowAgent - 智能条件分支工作流

**核心特性：**
- 支持条件分支逻辑，根据质量评估结果智能选择执行路径
- 质量评估失败时自动触发互联网搜索补救机制
- 确保用户始终能获得回复，提升用户体验

**工作流程：**

1. **查询重写阶段** (`QueryRewriterAgent`)
   - 分析用户查询的法律领域和复杂度
   - 优化查询关键词，提高检索精度

2. **本地检索阶段** (`RetrievalAgent`)
   - 使用BGE-M3模型进行向量检索
   - 应用Cross-Encoder重排序优化结果

3. **质量评估阶段** (`QualityEvaluatorAgent`)
   - 4维度评估：相关性、完整性、权威性、时效性
   - 80%质量阈值判断

4. **条件分支决策**
   - **直接路径**（质量≥80%）：直接生成专业法律答案
   - **补救路径**（质量<80%）：触发以下流程
     - 4a. 互联网搜索补充信息
     - 4b. 本地+网络结果智能融合
     - 4c. 二次质量评估
     - 4d. 生成最终答案（无论二次评估结果如何）

#### 2. SequentialAgent - 传统4阶段工作流（保留）

1. **查询重写阶段** (`QueryRewriterAgent`)
2. **检索执行阶段** (`RetrievalAgent`)  
3. **质量评估阶段** (`QualityEvaluatorAgent`)
4. **答案生成阶段** (`AnswerGeneratorAgent`)

### 核心组件

#### 基础组件
- **LocalRetriever**: BGE-M3嵌入 + FAISS索引 + Cross-Encoder重排序
- **QueryRewriterAgent**: 查询优化和关键词提取
- **RetrievalAgent**: 检索执行和结果格式化
- **QualityEvaluatorAgent**: 4维度质量评估（相关性、完整性、准确性、覆盖面）
- **AnswerGeneratorAgent**: 基于阈值的智能答案生成

#### 增强组件（条件分支工作流）
- **ConditionalWorkflowAgent**: 自定义条件分支控制器，支持智能路径选择
- **WebSearchAgent**: 互联网搜索补救机制
- **ResultMergerAgent**: 本地和网络搜索结果智能融合
- **SecondaryEvaluatorAgent**: 融合结果的二次质量评估
- **FinalAnswerAgent**: 多路径最终答案生成

## ✨ 核心特性

- **🧠 智能工作流**: 支持条件分支的增强工作流 + 传统Sequential架构
- **🔍 混合检索**: BGE-M3语义检索 + Cross-Encoder精确重排序
- **📊 质量控制**: 80%阈值机制 + 互联网搜索补救机制
- **🌐 智能补救**: 质量评估失败时自动触发网络搜索，确保用户始终获得回复
- **🎯 专业领域**: 专注中文法律咨询，支持74条法律条文
- **⚡ 高效缓存**: 本地模型缓存，快速响应
- **🔧 ADK兼容**: 完全兼容Google ADK框架

## 📁 项目结构

```
agentic_rag/
├── agentic_rag/                    # 核心包
│   ├── __init__.py                 # ADK导出接口
│   ├── agent.py                    # 主工作流Agent定义
│   ├── conditional_workflow_agent.py # 条件分支工作流Agent（推荐）
│   ├── query_rewriter.py            # 查询重写Agent
│   ├── retriever.py                 # 检索器和检索Agent
│   ├── quality_evaluator.py         # 质量评估Agent
│   ├── answer_generator.py          # 答案生成Agent
│   ├── web_search_agent.py          # 互联网搜索Agent
│   ├── result_merger.py             # 结果融合和二次评估Agent
│   └── config.py                    # 配置管理
├── chinese_law.txt                 # 法律条文数据
├── init_index.py                   # 索引初始化脚本
├── main.py                         # 本地运行入口
├── test_optimized_workflow.py      # 工作流测试
├── download_models.py              # 模型下载脚本
└── requirements.txt                # 依赖包
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd agentic_rag

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置 DEEPSEEK_API_KEY
```

### 2. 模型下载

```bash
# 下载BGE-M3和Cross-Encoder模型
python download_models.py
```

### 3. 索引构建

```bash
# 构建法律条文索引
python init_index.py
```

### 4. 运行测试

```bash
# 测试工作流
python test_optimized_workflow.py

# 本地运行
python main.py
```

## 🔧 配置说明

### 环境变量

```bash
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 配置文件 (config.py)

```python
class LawRAGConfig:
    deepseek_api_key: str           # DeepSeek API密钥
    embedding_model: str            # 嵌入模型名称
    cross_encoder_model: str        # 重排序模型名称
    index_path: str                 # 索引文件路径
    quality_threshold: float        # 质量阈值 (默认0.8)
```

## 📊 工作流详解

### 阶段1: 查询重写
- **输入**: 用户原始查询
- **处理**: 提取法律关键词，规范术语，生成查询变体
- **输出**: 优化后的主查询和备选查询

### 阶段2: 检索执行
- **输入**: 重写后的查询
- **处理**: BGE-M3语义检索 → Cross-Encoder重排序
- **输出**: 格式化的相关法律条文

### 阶段3: 质量评估
- **输入**: 检索结果
- **评估维度**: 相关性、完整性、准确性、覆盖面 (各10分)
- **输出**: 总分和PASS/FAIL判断 (≥32分为PASS)

### 阶段4: 答案生成
- **PASS**: 生成专业法律咨询意见
- **FAIL**: 建议重新描述问题或咨询专业律师

## 🧪 测试验证

系统包含完整的测试套件：

```bash
python test_optimized_workflow.py
```

测试覆盖：
- ✅ 工作流配置验证
- ✅ Agent指令检查
- ✅ 质量阈值机制
- ✅ 状态变量引用
- ✅ 检索工具功能

## 🔄 ADK集成

### 导出接口

```python
from agentic_rag import agent

# agent 是 OptimizedAgenticRAGWorkflow 的实例
# 可直接用于ADK Web部署
```

### Web部署

1. 确保索引已构建
2. 配置环境变量
3. 通过ADK Web界面部署

## 📈 性能特点

- **检索精度**: BGE-M3 + Cross-Encoder双重保障
- **响应速度**: 本地模型缓存，平均响应 < 3秒
- **质量控制**: 80%阈值过滤，确保专业性
- **扩展性**: 模块化设计，易于添加新功能

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 发起Pull Request

## 📄 许可证

本项目采用Apache 2.0许可证。
