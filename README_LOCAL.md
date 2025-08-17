# 本地RAG系统使用指南

## 🚀 快速开始

### 1. 预下载模型（仅需一次）
```bash
# 运行预下载脚本，将模型下载到本地缓存
python download_models.py
```

### 2. 配置环境
```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，设置DeepSeek API密钥
DEEPSEEK_API_KEY=your-api-key-here
```

### 3. 运行系统
```bash
# 启动本地RAG系统
adk run root_agent

# 或使用Python直接运行
python -m app.agent
```

## 📁 文件结构
```
├── app/
│   ├── local_retriever.py  # 本地检索器（支持缓存）
│   └── agent.py           # 主agent（使用DeepSeek）
├── data/
│   ├── vectors.index      # FAISS向量索引
│   ├── texts.pkl          # 法律文本
│   └── metadatas.pkl      # 元数据
├── chinese_law.txt        # 法律文本源文件
├── download_models.py     # 模型预下载脚本
└── .env                   # 环境变量配置
```

## 🔧 本地缓存机制

### 缓存位置
- **嵌入模型**: `~/.cache/sentence_transformers/BAAI--bge-m3`
- **交叉编码器**: `~/.cache/sentence_transformer/cross-encoder--ms-marco-MiniLM-L-6-v2`

### 缓存优势
- ✅ **零网络延迟** - 首次下载后无需网络连接
- ✅ **隐私保护** - 所有计算在本地完成
- ✅ **速度提升** - 避免重复下载模型
- ✅ **离线运行** - 完全脱离互联网环境

## 📋 使用场景

### 场景1：完全离线运行
1. 预先运行 `python download_models.py`
2. 配置 `.env` 文件（可离线配置）
3. 运行系统，无需任何网络连接

### 场景2：首次使用
1. 运行 `python download_models.py` 下载模型
2. 系统会自动加载法律文本并建立索引
3. 后续使用完全本地运行

## 🎯 测试查询示例
```python
# 测试查询
from app.local_retriever import create_local_retriever

retriever = create_local_retriever()
results = retriever.retrieve_and_rerank("个人信息保护法", top_k=3, top_n=2)
for result in results:
    print(result['text'])
```

## ⚙️ 环境变量配置
```bash
# 模型配置（可选）
EMBEDDING_MODEL=BAAI/bge-m3
CROSS_ENCODER=cross-encoder/ms-marco-MiniLM-L-6-v2

# API配置（必需）
DEEPSEEK_API_KEY=your-api-key

# 存储路径（可选）
DATA_DIR=./data
```

## 🛠️ 故障排除

### 模型加载失败
- 检查 `~/.cache/sentence_transformers/` 目录是否存在模型文件
- 重新运行 `python download_models.py`

### 索引建立失败
- 确认 `chinese_law.txt` 文件存在
- 检查数据目录权限

### API连接失败
- 确认 `DEEPSEEK_API_KEY` 配置正确
- 检查网络连接

## 📊 性能指标
- **索引大小**: < 1MB
- **内存占用**: < 500MB
- **查询延迟**: < 500ms（首次加载模型后）
- **冷启动**: < 30秒（模型已缓存）