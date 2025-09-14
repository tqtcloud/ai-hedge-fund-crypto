"""
期货交易工作流

专门为期货交易设计的工作流，集成期货信号系统和风险管理
"""

from typing import List
from langgraph.graph import END, StateGraph

from src.graph import AgentState, StartNode, DataNode, EmptyNode
from src.graph.futures_risk_management_node import FuturesRiskManagementNode
from src.graph.futures_portfolio_management_node import FuturesPortfolioManagementNode
from src.strategies.futures_macd_strategy import FuturesMacdStrategy
from src.strategies.futures_rsi_strategy import FuturesRSIStrategy
from src.utils import Interval


class FuturesWorkflow:
    """
    期货交易工作流管理器

    构建专门的期货交易决策流程：
    1. 数据收集和预处理
    2. 期货策略信号生成 (long/short/neutral)
    3. 期货风险管理分析
    4. 期货投资组合管理和执行
    """

    @staticmethod
    def create_futures_workflow(intervals: List[Interval]) -> StateGraph:
        """
        创建期货交易工作流

        Args:
            intervals: 时间间隔列表

        Returns:
            配置好的StateGraph工作流
        """
        workflow = StateGraph(AgentState)

        # 1. 开始节点
        start_node = StartNode()
        workflow.add_node("start_node", start_node)

        # 2. 数据合并节点
        merged_data_node = EmptyNode()
        workflow.add_node("merge_data_node", merged_data_node)

        # 3. 为每个时间间隔添加数据节点
        for interval in intervals:
            node_name = f"{interval.value}_node"
            data_node = DataNode(interval)
            workflow.add_node(node_name, data_node)
            workflow.add_edge("start_node", node_name)
            workflow.add_edge(node_name, "merge_data_node")

        # 4. 期货策略节点
        futures_strategies = {
            "futures_macd_strategy": FuturesMacdStrategy(),
            "futures_rsi_strategy": FuturesRSIStrategy()
        }

        for strategy_name, strategy_instance in futures_strategies.items():
            workflow.add_node(strategy_name, strategy_instance)
            workflow.add_edge("merge_data_node", strategy_name)

        # 5. 期货风险管理节点
        futures_risk_management_node = FuturesRiskManagementNode()
        workflow.add_node("futures_risk_management_node", futures_risk_management_node)

        # 连接所有策略到风险管理
        for strategy_name in futures_strategies.keys():
            workflow.add_edge(strategy_name, "futures_risk_management_node")

        # 6. 期货投资组合管理节点
        futures_portfolio_management_node = FuturesPortfolioManagementNode()
        workflow.add_node("futures_portfolio_management_node", futures_portfolio_management_node)

        # 连接风险管理到投资组合管理
        workflow.add_edge("futures_risk_management_node", "futures_portfolio_management_node")

        # 7. 结束
        workflow.add_edge("futures_portfolio_management_node", END)

        # 8. 设置入口点
        workflow.set_entry_point("start_node")

        return workflow


class FuturesWorkflowFactory:
    """
    期货工作流工厂

    提供不同配置的期货交易工作流
    """

    @staticmethod
    def create_standard_futures_workflow(intervals: List[Interval]) -> StateGraph:
        """创建标准期货交易工作流"""
        return FuturesWorkflow.create_futures_workflow(intervals)

    @staticmethod
    def create_conservative_futures_workflow(intervals: List[Interval]) -> StateGraph:
        """
        创建保守的期货交易工作流

        特点：
        - 只使用长时间框架
        - 更严格的风险控制
        - 较低的杠杆限制
        """
        # 过滤出长时间框架
        conservative_intervals = [interval for interval in intervals
                                if interval.value in ['1h', '4h', '1d']]

        if not conservative_intervals:
            conservative_intervals = intervals  # 如果没有长时间框架，使用全部

        return FuturesWorkflow.create_futures_workflow(conservative_intervals)

    @staticmethod
    def create_aggressive_futures_workflow(intervals: List[Interval]) -> StateGraph:
        """
        创建积极的期货交易工作流

        特点：
        - 使用所有时间框架
        - 包含短时间框架策略
        - 允许更高的杠杆
        """
        return FuturesWorkflow.create_futures_workflow(intervals)

    @staticmethod
    def get_workflow_by_risk_level(risk_level: str, intervals: List[Interval]) -> StateGraph:
        """
        根据风险等级获取相应的工作流

        Args:
            risk_level: 风险等级 ('conservative', 'moderate', 'aggressive')
            intervals: 时间间隔列表

        Returns:
            对应风险等级的工作流
        """
        if risk_level.lower() == 'conservative':
            return FuturesWorkflowFactory.create_conservative_futures_workflow(intervals)
        elif risk_level.lower() == 'aggressive':
            return FuturesWorkflowFactory.create_aggressive_futures_workflow(intervals)
        else:  # moderate or default
            return FuturesWorkflowFactory.create_standard_futures_workflow(intervals)