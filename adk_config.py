"""
ADK配置和初始化
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 确保DeepSeek API密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请设置DEEPSEEK_API_KEY环境变量")

# 配置模型
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY