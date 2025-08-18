"""
测试SequentialAgent工作流在ADK环境中的完整执行
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from agentic_rag.agent import root_agent


async def test_sequential_workflow():
    """测试完整的SequentialAgent工作流"""
    
    print("🚀 开始测试SequentialAgent工作流...")
    
    # 测试查询
    test_query = "个人信息使用者在什么情况下可以使用个人信息"
    
    print(f"📝 测试查询: {test_query}")
    print("=" * 60)
    
    try:
        # 执行工作流
        result = await root_agent.run_async(test_query)
        
        print("✅ 工作流执行完成")
        print("📊 执行结果:")
        print("-" * 40)
        print(result.content)
        print("-" * 40)
        
        # 检查状态信息
        if hasattr(result, 'state') and result.state:
            print("\n📋 工作流状态信息:")
            for key, value in result.state.items():
                print(f"  {key}: {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ 工作流执行失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_configuration():
    """测试Agent配置"""
    print("🔍 检查Agent配置...")
    
    # 检查root_agent类型
    print(f"Agent类型: {type(root_agent).__name__}")
    print(f"Agent名称: {root_agent.name}")
    print(f"子Agent数量: {len(root_agent.sub_agents)}")
    
    # 列出所有子Agent
    print("子Agent列表:")
    for i, sub_agent in enumerate(root_agent.sub_agents, 1):
        print(f"  {i}. {sub_agent.name} -> {sub_agent.output_key}")
    
    print("✅ Agent配置检查完成")


if __name__ == "__main__":
    print("🧪 SequentialAgent工作流测试")
    print("=" * 60)
    
    # 配置检查
    test_agent_configuration()
    print()
    
    # 异步工作流测试
    success = asyncio.run(test_sequential_workflow())
    
    if success:
        print("\n🎉 测试成功！SequentialAgent工作流运行正常")
    else:
        print("\n💥 测试失败！请检查配置和错误信息")
