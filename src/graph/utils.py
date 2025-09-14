"""
Graph实用工具函数

提供图形节点和工作流使用的通用工具函数
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def show_agent_reasoning(content: Dict[str, Any], agent_name: str) -> None:
    """
    显示代理推理过程

    Args:
        content: 要显示的内容
        agent_name: 代理名称
    """
    try:
        print(f"\n{'='*60}")
        print(f"🤖 {agent_name}")
        print(f"{'='*60}")

        # 显示核心信息
        if isinstance(content, dict):
            for key, value in content.items():
                if key in ['futures_risk_analysis', 'portfolio_changes', 'execution_summary']:
                    print(f"\n📊 {key}:")
                    _print_dict_content(value, indent=1)
                elif key == 'comprehensive_decision':
                    print(f"\n🎯 综合决策:")
                    _print_dict_content(value, indent=1)
                elif key.endswith('_summary'):
                    print(f"\n📋 {key}:")
                    _print_dict_content(value, indent=1)

        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"显示代理推理失败: {e}")


def _print_dict_content(content: Any, indent: int = 0) -> None:
    """
    递归打印字典内容

    Args:
        content: 要打印的内容
        indent: 缩进级别
    """
    indent_str = "  " * indent

    if isinstance(content, dict):
        for key, value in content.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                print(f"{indent_str}{key}: [复杂数据结构]")
            else:
                print(f"{indent_str}{key}: {value}")
    elif isinstance(content, list):
        for i, item in enumerate(content):
            print(f"{indent_str}[{i}]: {item}")
    else:
        print(f"{indent_str}{content}")