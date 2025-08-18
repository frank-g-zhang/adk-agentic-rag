#!/usr/bin/env python3
"""清理和重建索引工具"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agentic_rag.retriever import LocalRetriever

def clean_and_rebuild_index():
    """清理重复数据并重建索引"""
    
    # 创建检索器
    retriever = LocalRetriever()
    
    print(f"当前索引文档数量: {len(retriever.texts)}")
    
    if len(retriever.texts) == 0:
        print("索引为空，无需清理")
        return
    
    # 检查重复文档
    unique_texts = []
    unique_metadatas = []
    duplicates_count = 0
    
    for i, text in enumerate(retriever.texts):
        if text not in unique_texts:
            unique_texts.append(text)
            unique_metadatas.append(retriever.metadatas[i] if i < len(retriever.metadatas) else {})
        else:
            duplicates_count += 1
    
    print(f"发现重复文档: {duplicates_count} 个")
    print(f"去重后文档数量: {len(unique_texts)}")
    
    if duplicates_count > 0:
        # 删除现有索引文件
        index_files = [
            "data/vectors.index",
            "data/texts.pkl", 
            "data/metadatas.pkl"
        ]
        
        for file_path in index_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"删除文件: {file_path}")
        
        # 重新创建检索器并添加去重后的文档
        new_retriever = LocalRetriever()
        new_retriever.add_documents(unique_texts, unique_metadatas)
        
        print(f"重建索引完成，最终文档数量: {len(new_retriever.texts)}")
    else:
        print("无重复文档，索引正常")

def show_index_stats():
    """显示索引统计信息"""
    retriever = LocalRetriever()
    
    print("=== 索引统计信息 ===")
    print(f"文档总数: {len(retriever.texts)}")
    
    # 统计法律类型
    law_types = {}
    for metadata in retriever.metadatas:
        law = metadata.get('law', '未知')
        law_types[law] = law_types.get(law, 0) + 1
    
    print("\n法律文档分布:")
    for law, count in sorted(law_types.items()):
        print(f"  {law}: {count} 个文档")
    
    # 显示前5个文档示例
    print("\n前5个文档示例:")
    for i in range(min(5, len(retriever.texts))):
        text_preview = retriever.texts[i][:80] + "..." if len(retriever.texts[i]) > 80 else retriever.texts[i]
        print(f"  {i+1}. {text_preview}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="索引管理工具")
    parser.add_argument("--clean", action="store_true", help="清理重复数据并重建索引")
    parser.add_argument("--stats", action="store_true", help="显示索引统计信息")
    
    args = parser.parse_args()
    
    if args.clean:
        clean_and_rebuild_index()
    elif args.stats:
        show_index_stats()
    else:
        print("使用 --clean 清理索引，或 --stats 查看统计信息")
