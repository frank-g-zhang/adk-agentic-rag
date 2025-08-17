#!/usr/bin/env python3
"""
法律文本索引初始化脚本
一次性运行，永久使用
"""

import os
import sys
from pathlib import Path
import warnings
import time

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from app.local_retriever import LocalRetriever

def init_law_index():
    """初始化法律文本索引"""
    
    print("🚀 开始初始化法律文本索引...")
    print("=" * 50)
    
    # 检查法律文本文件
    law_file = Path("chinese_law.txt")
    if not law_file.exists():
        print(f"❌ 未找到法律文本文件: {law_file.absolute()}")
        return False
    
    # 检查索引是否已存在
    index_file = Path("data/vectors.index")
    if index_file.exists():
        print("✅ 索引已存在，跳过创建")
        return True
    
    try:
        # 创建检索器（会自动加载模型）
        print("📥 加载模型...")
        retriever = LocalRetriever()
        
        # 读取法律文本
        print("📖 读取法律文本...")
        texts = []
        metadatas = []
        
        with open(law_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"📊 共读取 {len(lines)} 行法律文本")
        
        # 处理每一行
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            texts.append(line)
            
            # 提取元数据
            metadata = {}
            
            # 提取法律名称
            if '《' in line and '》' in line:
                law_start = line.find('《')
                law_end = line.find('》') + 1
                metadata['law'] = line[law_start:law_end]
            
            # 提取条款号
            if '第' in line and '条规定' in line:
                article_start = line.find('第')
                article_end = line.find('条规定') + 2
                metadata['article'] = line[article_start:article_end]
            
            metadata['line_number'] = line_num
            metadatas.append(metadata)
            
            # 显示进度
            if line_num % 10 == 0:
                print(f"   处理中... {line_num}/{len(lines)} 行")
        
        if not texts:
            print("❌ 没有找到有效的法律文本")
            return False
        
        print(f"✅ 处理完成，共 {len(texts)} 条有效法律条文")
        
        # 创建索引
        print("🔍 创建向量索引...")
        start_time = time.time()
        retriever.add_documents(texts, metadatas)
        elapsed = time.time() - start_time
        
        print(f"✅ 索引创建完成！耗时: {elapsed:.2f}秒")
        print(f"📊 索引包含: {len(texts)} 个文档向量")
        print(f"📁 索引文件: {index_file.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 索引创建失败: {e}")
        return False

def verify_index():
    """验证索引是否可用"""
    try:
        retriever = LocalRetriever()
        
        if retriever.index is None:
            print("❌ 索引验证失败")
            return False
            
        print(f"✅ 索引验证成功！包含 {len(retriever.texts)} 条法律条文")
        
        # 测试查询
        if retriever.texts:
            test_query = "个人信息保护法"
            results = retriever.search(test_query, top_k=3)
            print(f"✅ 测试查询成功！找到 {len(results)} 条相关条文")
            
            if results:
                print("📋 前3条结果:")
                for i, result in enumerate(results[:3], 1):
                    law = result['metadata'].get('law', '未知')
                    article = result['metadata'].get('article', '未知')
                    print(f"   {i}. {law} {article}")
        
        return True
        
    except Exception as e:
        print(f"❌ 索引验证失败: {e}")
        return False

def main():
    """主函数"""
    print("🏛️ 中国法律RAG系统 - 索引初始化")
    print("=" * 50)
    
    # 运行初始化
    success = init_law_index()
    
    if success:
        print("\n🔍 验证索引...")
        verify_success = verify_index()
        
        if verify_success:
            print("\n🎉 索引初始化完成！")
            print("\n✅ 系统已就绪，可以开始使用了：")
            print("   adk run root_agent")
            print("   或")
            print("   python -m app.agent")
        else:
            print("\n❌ 索引验证失败")
    else:
        print("\n❌ 索引初始化失败")
        print("请检查 chinese_law.txt 是否存在且包含有效内容")

if __name__ == "__main__":
    main()