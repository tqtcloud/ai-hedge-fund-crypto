"""
期货交易代理

专门为期货交易设计的智能代理，集成期货信号系统、风险管理和投资组合管理
"""

from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage
from datetime import datetime
import logging

from utils import Interval, save_graph_as_png, parse_str_to_json
from .futures_workflow import FuturesWorkflowFactory

# 导入期货系统组件
from src.futures.signals.futures_signal_system import FuturesSignalSystem
from src.futures.margin.margin_manager import MarginManager
from src.futures.market.market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)


class FuturesAgent:
    """
    期货交易代理

    核心功能：
    1. 集成期货信号系统生成 long/short/neutral 信号
    2. 基于保证金的风险管理
    3. 杠杆控制和强平保护
    4. 专门的期货投资组合管理
    5. 与传统Agent兼容的接口
    """

    def __init__(
        self,
        intervals: List[Interval],
        show_agent_graph: bool = True,
        risk_level: str = "moderate",
        futures_signal_system: Optional[FuturesSignalSystem] = None,
        margin_manager: Optional[MarginManager] = None,
        market_analyzer: Optional[MarketAnalyzer] = None
    ):
        """
        初始化期货交易代理

        Args:
            intervals: 时间间隔列表
            show_agent_graph: 是否显示代理图
            risk_level: 风险等级 ('conservative', 'moderate', 'aggressive')
            futures_signal_system: 期货信号系统实例
            margin_manager: 保证金管理器实例
            market_analyzer: 市场分析器实例
        """
        self.intervals = intervals
        self.risk_level = risk_level
        self.futures_signal_system = futures_signal_system
        self.margin_manager = margin_manager
        self.market_analyzer = market_analyzer

        # 创建期货工作流
        workflow = FuturesWorkflowFactory.get_workflow_by_risk_level(risk_level, intervals)
        self.agent = workflow.compile()

        # 保存图像（如果启用）
        if show_agent_graph:
            file_path = f"futures_{risk_level}_workflow_graph.png"
            try:
                save_graph_as_png(self.agent, file_path)
                logger.info(f"期货代理工作流图已保存: {file_path}")
            except Exception as e:
                logger.warning(f"保存期货代理图失败: {e}")

        logger.info(f"期货交易代理初始化完成 - 风险等级: {risk_level}, 时间框架: {len(intervals)}个")

    def run(
        self,
        primary_interval: Interval,
        tickers: List[str],
        end_date: datetime,
        portfolio: Dict,
        show_reasoning: bool = False,
        model_name: str = "gpt-4o",
        model_provider: str = "openai",
        model_base_url: Optional[str] = None
    ) -> Dict:
        """
        执行期货交易工作流

        Args:
            primary_interval: 主要时间间隔
            tickers: 资产符号列表
            end_date: 结束日期
            portfolio: 初始投资组合状态
            show_reasoning: 是否显示推理过程
            model_name: LLM模型名称
            model_provider: LLM提供商
            model_base_url: LLM基础URL

        Returns:
            包含交易决策和分析师信号的字典
        """
        try:
            logger.info("🚀 开始执行期货交易工作流...")

            # 验证期货系统组件
            self._validate_futures_components(portfolio)

            # 准备期货交易输入数据
            futures_input_data = self._prepare_futures_input_data(
                primary_interval, tickers, portfolio, end_date
            )

            # 执行期货工作流
            final_state = self.agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            content="基于期货交易语义(long/short/neutral)做出交易决策，"
                                  "考虑保证金要求、杠杆控制和强平风险。",
                        )
                    ],
                    "data": futures_input_data,
                    "metadata": {
                        "show_reasoning": show_reasoning,
                        "model_name": model_name,
                        "model_provider": model_provider,
                        "model_base_url": model_base_url,
                        "agent_type": "futures_trading"
                    },
                },
            )

            # 处理和返回结果
            result = self._process_futures_result(final_state)

            logger.info("✅ 期货交易工作流执行完成")
            return result

        except Exception as e:
            logger.error(f"❌ 期货交易工作流执行失败: {e}")
            return {
                "error": str(e),
                "decisions": {"error": "期货交易工作流执行失败"},
                "analyst_signals": {},
                "timestamp": datetime.now().isoformat()
            }

    def _validate_futures_components(self, portfolio: Dict) -> None:
        """验证期货系统组件"""
        try:
            # 检查投资组合中的期货组件
            if "futures_signal_system" not in portfolio and not self.futures_signal_system:
                logger.warning("未找到期货信号系统，某些功能可能受限")

            if "margin_manager" not in portfolio and not self.margin_manager:
                logger.warning("未找到保证金管理器，风险管理功能可能受限")

            if "market_analyzer" not in portfolio and not self.market_analyzer:
                logger.warning("未找到市场分析器，分析功能可能受限")

            # 检查投资组合结构
            required_keys = ["cash", "margin_requirement", "margin_used", "positions", "realized_gains"]
            for key in required_keys:
                if key not in portfolio:
                    logger.warning(f"投资组合缺少必要键: {key}")

            logger.info("期货系统组件验证完成")

        except Exception as e:
            logger.error(f"期货系统组件验证失败: {e}")
            raise

    def _prepare_futures_input_data(
        self,
        primary_interval: Interval,
        tickers: List[str],
        portfolio: Dict,
        end_date: datetime
    ) -> Dict:
        """准备期货交易输入数据"""
        try:
            # 确保portfolio包含期货组件
            enhanced_portfolio = portfolio.copy()

            # 添加期货系统组件到portfolio（如果可用）
            if self.futures_signal_system and "futures_signal_system" not in enhanced_portfolio:
                enhanced_portfolio["futures_signal_system"] = self.futures_signal_system

            if self.margin_manager and "margin_manager" not in enhanced_portfolio:
                enhanced_portfolio["margin_manager"] = self.margin_manager

            if self.market_analyzer and "market_analyzer" not in enhanced_portfolio:
                enhanced_portfolio["market_analyzer"] = self.market_analyzer

            # 准备输入数据结构
            input_data = {
                "primary_interval": primary_interval,
                "intervals": self.intervals,
                "tickers": tickers,
                "portfolio": enhanced_portfolio,
                "end_date": end_date,
                "analyst_signals": {},
                "trading_mode": "futures",
                "risk_level": self.risk_level,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"期货交易输入数据准备完成: {len(tickers)}个交易对, {len(self.intervals)}个时间框架")
            return input_data

        except Exception as e:
            logger.error(f"期货交易输入数据准备失败: {e}")
            raise

    def _process_futures_result(self, final_state: Dict) -> Dict:
        """处理期货交易结果"""
        try:
            # 获取最终消息内容
            messages = final_state.get("messages", [])
            if not messages:
                logger.warning("没有收到工作流消息")
                return {
                    "decisions": {"status": "no_messages"},
                    "analyst_signals": {}
                }

            # 解析最终决策消息
            final_message = messages[-1]
            final_content = final_message.content

            # 尝试解析JSON内容
            try:
                decisions = parse_str_to_json(final_content)
            except Exception as e:
                logger.warning(f"决策消息解析失败: {e}")
                decisions = {"raw_content": final_content, "parse_error": str(e)}

            # 获取分析师信号
            data = final_state.get("data", {})
            analyst_signals = data.get("analyst_signals", {})

            # 获取期货特定的数据
            futures_risk_analysis = data.get("futures_risk_analysis", {})
            portfolio_changes = data.get("portfolio_changes", {})
            execution_summary = data.get("execution_summary", {})

            # 构建结果
            result = {
                "decisions": decisions,
                "analyst_signals": analyst_signals,
                "futures_risk_analysis": futures_risk_analysis,
                "portfolio_changes": portfolio_changes,
                "execution_summary": execution_summary,
                "trading_mode": "futures",
                "risk_level": self.risk_level,
                "timestamp": datetime.now().isoformat(),
                "total_messages": len(messages)
            }

            # 添加期货交易摘要
            if portfolio_changes:
                result["futures_trading_summary"] = self._create_futures_trading_summary(
                    portfolio_changes, execution_summary
                )

            return result

        except Exception as e:
            logger.error(f"期货交易结果处理失败: {e}")
            return {
                "error": str(e),
                "decisions": {"error": "结果处理失败"},
                "analyst_signals": {},
                "timestamp": datetime.now().isoformat()
            }

    def _create_futures_trading_summary(
        self,
        portfolio_changes: Dict,
        execution_summary: Dict
    ) -> Dict:
        """创建期货交易摘要"""
        try:
            summary = {
                "total_actions": len(portfolio_changes),
                "successful_trades": 0,
                "rejected_trades": 0,
                "long_positions_opened": 0,
                "short_positions_opened": 0,
                "positions_closed": 0,
                "total_margin_used": 0.0,
                "actions_breakdown": {}
            }

            for ticker, change in portfolio_changes.items():
                action = change.get("action", "UNKNOWN")
                summary["actions_breakdown"][ticker] = action

                if action == "OPEN_LONG":
                    summary["successful_trades"] += 1
                    summary["long_positions_opened"] += 1
                    summary["total_margin_used"] += change.get("required_margin", 0.0)

                elif action == "OPEN_SHORT":
                    summary["successful_trades"] += 1
                    summary["short_positions_opened"] += 1
                    summary["total_margin_used"] += change.get("required_margin", 0.0)

                elif action == "CLOSE_POSITION":
                    summary["successful_trades"] += 1
                    summary["positions_closed"] += 1

                elif action in ["REJECTED", "INSUFFICIENT_FUNDS", "INSUFFICIENT_MARGIN"]:
                    summary["rejected_trades"] += 1

            # 添加执行摘要中的信息
            if execution_summary:
                summary["portfolio_impact"] = execution_summary.get("portfolio_impact", {})
                summary["recommendations"] = execution_summary.get("recommendations", [])

            return summary

        except Exception as e:
            logger.error(f"期货交易摘要创建失败: {e}")
            return {"error": str(e)}

    def get_agent_info(self) -> Dict:
        """获取代理信息"""
        return {
            "agent_type": "futures_trading",
            "risk_level": self.risk_level,
            "intervals": [interval.value for interval in self.intervals],
            "has_futures_signal_system": self.futures_signal_system is not None,
            "has_margin_manager": self.margin_manager is not None,
            "has_market_analyzer": self.market_analyzer is not None,
            "timestamp": datetime.now().isoformat()
        }

    def update_risk_level(self, new_risk_level: str) -> bool:
        """
        更新风险等级并重新编译工作流

        Args:
            new_risk_level: 新的风险等级

        Returns:
            是否成功更新
        """
        try:
            if new_risk_level not in ['conservative', 'moderate', 'aggressive']:
                logger.error(f"无效的风险等级: {new_risk_level}")
                return False

            self.risk_level = new_risk_level

            # 重新创建工作流
            workflow = FuturesWorkflowFactory.get_workflow_by_risk_level(new_risk_level, self.intervals)
            self.agent = workflow.compile()

            logger.info(f"风险等级已更新为: {new_risk_level}")
            return True

        except Exception as e:
            logger.error(f"风险等级更新失败: {e}")
            return False

    async def initialize_futures_components(self) -> bool:
        """
        初始化期货系统组件

        Returns:
            是否成功初始化
        """
        try:
            # 初始化保证金管理器
            if self.margin_manager:
                await self.margin_manager.initialize()
                logger.info("保证金管理器初始化完成")

            # 初始化期货信号系统
            if self.futures_signal_system:
                self.futures_signal_system.initialize()
                logger.info("期货信号系统初始化完成")

            logger.info("所有期货系统组件初始化完成")
            return True

        except Exception as e:
            logger.error(f"期货系统组件初始化失败: {e}")
            return False

    async def cleanup_futures_components(self) -> None:
        """清理期货系统组件资源"""
        try:
            if self.margin_manager:
                await self.margin_manager.cleanup()
                logger.info("保证金管理器资源已清理")

            if self.futures_signal_system:
                self.futures_signal_system.clear_cache()
                logger.info("期货信号系统缓存已清理")

            logger.info("期货系统组件资源清理完成")

        except Exception as e:
            logger.error(f"期货系统组件清理失败: {e}")