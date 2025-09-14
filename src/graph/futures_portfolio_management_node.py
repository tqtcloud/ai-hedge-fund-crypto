"""
期货投资组合管理节点

专门为期货交易设计的投资组合管理组件：
- 处理 long/short/neutral 交易决策
- 保证金基础的仓位计算
- 杠杆风险控制
- 期货特有的盈亏计算
- 强平风险监控
"""

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from langchain_core.messages import HumanMessage

from src.graph.base_node import BaseNode
from src.graph.state import AgentState
from src.graph.utils import show_agent_reasoning

# 导入期货系统组件
from src.futures.models.data_models import TradingDirection, OperationType, RiskLevel
from src.futures.margin.margin_manager import MarginManager
from src.futures.constants import RiskParameters

logger = logging.getLogger(__name__)


class FuturesPortfolioManagementNode(BaseNode):
    """
    期货投资组合管理节点

    核心功能：
    1. 基于期货风险管理决策执行交易
    2. 保证金基础的仓位大小计算
    3. 期货仓位的开仓、平仓、加仓、减仓
    4. 盈亏计算和强平风险监控
    5. 投资组合风险分散管理
    """

    def __init__(self):
        super().__init__()
        self.node_name = "futures_portfolio_management"
        logger.info("期货投资组合管理节点初始化")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        执行期货投资组合管理决策

        Args:
            state: Agent状态，包含风险管理决策和市场数据

        Returns:
            更新的状态字典，包含投资组合变更和交易决策
        """
        try:
            logger.info("💼 开始期货投资组合管理...")

            data = state.get("data", {})
            metadata = state.get("metadata", {})

            # 获取风险管理决策
            risk_management_decision = data.get("risk_management_decision", {})
            futures_risk_analysis = data.get("futures_risk_analysis", {})

            # 获取当前投资组合状态
            portfolio = data.get("portfolio", {})
            margin_manager = portfolio.get("margin_manager")

            # 检查是否应该暂停交易
            overall_action = risk_management_decision.get("overall_action", "NORMAL_TRADING")
            if overall_action in ["SUSPEND_TRADING"]:
                logger.warning(f"⚠️ 交易已暂停: {overall_action}")
                return self._create_suspension_response(state, overall_action)

            # 执行期货交易决策
            portfolio_changes = {}
            execution_summary = {}

            trading_decisions = risk_management_decision.get("trading_decisions", {})

            for ticker, decision in trading_decisions.items():
                try:
                    change_result = self._execute_futures_trading_decision(
                        ticker=ticker,
                        decision=decision,
                        portfolio=portfolio,
                        margin_manager=margin_manager,
                        market_data=data
                    )

                    portfolio_changes[ticker] = change_result
                    logger.info(f"✅ {ticker} 交易执行完成: {change_result.get('action', 'UNKNOWN')}")

                except Exception as e:
                    logger.error(f"❌ {ticker} 交易执行失败: {e}")
                    portfolio_changes[ticker] = {
                        "action": "ERROR",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }

            # 更新投资组合状态
            updated_portfolio = self._update_portfolio_state(
                portfolio, portfolio_changes, margin_manager
            )

            # 计算投资组合统计
            portfolio_stats = self._calculate_portfolio_statistics(
                updated_portfolio, portfolio_changes
            )

            # 生成执行摘要
            execution_summary = self._generate_execution_summary(
                portfolio_changes, portfolio_stats, overall_action
            )

            # 构建投资组合管理消息
            portfolio_message_content = {
                "portfolio_changes": portfolio_changes,
                "execution_summary": execution_summary,
                "updated_portfolio": self._sanitize_portfolio_for_output(updated_portfolio),
                "portfolio_statistics": portfolio_stats,
                "overall_action": overall_action,
                "timestamp": datetime.now().isoformat(),
                "node_type": "futures_portfolio_management"
            }

            message = HumanMessage(
                content=json.dumps(portfolio_message_content, ensure_ascii=False, indent=2),
                name="futures_portfolio_management_agent"
            )

            # 显示推理过程（如果启用）
            if metadata.get("show_reasoning", False):
                show_agent_reasoning(
                    portfolio_message_content,
                    "期货投资组合管理分析师 (Futures Portfolio Management)"
                )

            # 更新状态
            new_messages = state.get("messages", []) + [message]

            # 更新数据中的投资组合状态
            data["portfolio"] = updated_portfolio
            data["portfolio_changes"] = portfolio_changes
            data["execution_summary"] = execution_summary

            return {
                "messages": new_messages,
                "data": data
            }

        except Exception as e:
            logger.error(f"💥 期货投资组合管理节点执行失败: {e}")
            return self._create_error_response(state, str(e))

    def _execute_futures_trading_decision(
        self,
        ticker: str,
        decision: Dict[str, Any],
        portfolio: Dict[str, Any],
        margin_manager: Optional[MarginManager],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行单个交易对的期货交易决策

        Args:
            ticker: 交易对符号
            decision: 风险管理决策
            portfolio: 当前投资组合状态
            margin_manager: 保证金管理器
            market_data: 市场数据

        Returns:
            交易执行结果
        """
        try:
            action = decision.get("action", "HOLD")
            risk_adjusted_size = decision.get("risk_adjusted_size", 0.0)
            max_leverage = decision.get("max_leverage", 1.0)
            confidence = decision.get("confidence", 0.0)

            # 获取当前价格
            current_price = self._get_current_price(ticker, market_data)
            if current_price is None or current_price <= 0:
                raise ValueError(f"无效的当前价格: {current_price}")

            # 获取当前仓位
            current_position = self._get_current_position(ticker, portfolio)

            # 根据动作类型执行交易
            if action == "REJECT":
                return self._handle_rejected_trade(ticker, decision)

            elif action == "HOLD":
                return self._handle_hold_position(ticker, current_position, decision)

            elif action == "CLOSE":
                return self._handle_close_position(
                    ticker, current_position, current_price, decision, portfolio
                )

            elif action in ["OPEN_LONG", "OPEN_SHORT"]:
                return self._handle_open_position(
                    ticker, action, risk_adjusted_size, max_leverage,
                    current_price, current_position, decision, portfolio, margin_manager
                )

            else:
                logger.warning(f"未知的交易动作: {action}")
                return {
                    "action": "UNKNOWN",
                    "ticker": ticker,
                    "reason": f"未知的交易动作: {action}",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"交易决策执行失败 {ticker}: {e}")
            return {
                "action": "ERROR",
                "ticker": ticker,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _get_current_price(self, ticker: str, market_data: Dict[str, Any]) -> Optional[float]:
        """获取当前价格"""
        try:
            # 尝试从不同时间框架获取最新价格
            for interval_key in [f"{ticker}_1m", f"{ticker}_5m", f"{ticker}_15m", f"{ticker}_1h"]:
                if interval_key in market_data:
                    df = market_data[interval_key]
                    if not df.empty and 'close' in df.columns:
                        return float(df['close'].iloc[-1])

            logger.warning(f"无法获取 {ticker} 的当前价格")
            return None

        except Exception as e:
            logger.error(f"获取当前价格失败 {ticker}: {e}")
            return None

    def _get_current_position(self, ticker: str, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """获取当前持仓信息"""
        try:
            positions = portfolio.get("positions", {})
            if ticker in positions:
                position = positions[ticker]
                return {
                    "long": position.get("long", 0.0),
                    "short": position.get("short", 0.0),
                    "long_cost_basis": position.get("long_cost_basis", 0.0),
                    "short_cost_basis": position.get("short_cost_basis", 0.0),
                    "short_margin_used": position.get("short_margin_used", 0.0)
                }
            else:
                return {
                    "long": 0.0,
                    "short": 0.0,
                    "long_cost_basis": 0.0,
                    "short_cost_basis": 0.0,
                    "short_margin_used": 0.0
                }

        except Exception as e:
            logger.error(f"获取持仓信息失败 {ticker}: {e}")
            return {"long": 0.0, "short": 0.0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0}

    def _handle_rejected_trade(self, ticker: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        """处理被拒绝的交易"""
        return {
            "action": "REJECTED",
            "ticker": ticker,
            "reason": decision.get("reasons", ["交易被风险管理系统拒绝"]),
            "warnings": decision.get("warnings", []),
            "position_change": 0.0,
            "cost_impact": 0.0,
            "margin_impact": 0.0,
            "timestamp": datetime.now().isoformat()
        }

    def _handle_hold_position(
        self,
        ticker: str,
        current_position: Dict[str, Any],
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理保持仓位"""
        return {
            "action": "HOLD",
            "ticker": ticker,
            "reason": decision.get("reasons", ["保持当前仓位"]),
            "current_long": current_position["long"],
            "current_short": current_position["short"],
            "position_change": 0.0,
            "cost_impact": 0.0,
            "margin_impact": 0.0,
            "timestamp": datetime.now().isoformat()
        }

    def _handle_close_position(
        self,
        ticker: str,
        current_position: Dict[str, Any],
        current_price: float,
        decision: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """处理平仓操作"""
        try:
            long_position = current_position["long"]
            short_position = current_position["short"]

            if long_position == 0 and short_position == 0:
                return {
                    "action": "NO_POSITION_TO_CLOSE",
                    "ticker": ticker,
                    "reason": "没有持仓需要平仓",
                    "position_change": 0.0,
                    "cost_impact": 0.0,
                    "margin_impact": 0.0,
                    "timestamp": datetime.now().isoformat()
                }

            total_pnl = 0.0
            margin_released = 0.0

            # 平多仓
            if long_position > 0:
                long_cost_basis = current_position["long_cost_basis"]
                long_pnl = (current_price - long_cost_basis) * long_position
                total_pnl += long_pnl

            # 平空仓
            if short_position > 0:
                short_cost_basis = current_position["short_cost_basis"]
                short_pnl = (short_cost_basis - current_price) * short_position
                total_pnl += short_pnl
                margin_released = current_position["short_margin_used"]

            return {
                "action": "CLOSE_POSITION",
                "ticker": ticker,
                "reason": decision.get("reasons", ["执行平仓操作"]),
                "closed_long": long_position,
                "closed_short": short_position,
                "realized_pnl": total_pnl,
                "margin_released": margin_released,
                "position_change": -(long_position + short_position),
                "cost_impact": total_pnl,
                "margin_impact": margin_released,
                "new_long": 0.0,
                "new_short": 0.0,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"平仓处理失败 {ticker}: {e}")
            return {
                "action": "ERROR",
                "ticker": ticker,
                "error": f"平仓失败: {e}",
                "timestamp": datetime.now().isoformat()
            }

    def _handle_open_position(
        self,
        ticker: str,
        action: str,
        risk_adjusted_size: float,
        max_leverage: float,
        current_price: float,
        current_position: Dict[str, Any],
        decision: Dict[str, Any],
        portfolio: Dict[str, Any],
        margin_manager: Optional[MarginManager]
    ) -> Dict[str, Any]:
        """处理开仓操作"""
        try:
            if risk_adjusted_size <= 0:
                return {
                    "action": "INVALID_SIZE",
                    "ticker": ticker,
                    "reason": f"无效的仓位大小: {risk_adjusted_size}",
                    "timestamp": datetime.now().isoformat()
                }

            # 限制杠杆
            effective_leverage = min(max_leverage, RiskParameters.MAX_LEVERAGE)

            # 计算实际仓位大小（基于保证金）
            available_cash = portfolio.get("cash", 0.0)
            max_position_ratio = RiskParameters.MAX_POSITION_RATIO

            # 计算可用于该交易的资金
            available_for_trade = available_cash * max_position_ratio
            actual_position_size = min(risk_adjusted_size, available_for_trade)

            if actual_position_size < 10.0:  # 最小交易金额
                return {
                    "action": "INSUFFICIENT_FUNDS",
                    "ticker": ticker,
                    "reason": f"可用资金不足，需要至少10 USDT，实际可用: {actual_position_size:.2f}",
                    "timestamp": datetime.now().isoformat()
                }

            # 计算所需保证金
            required_margin = actual_position_size / effective_leverage
            notional_value = actual_position_size

            # 检查保证金充足性
            if required_margin > available_cash:
                return {
                    "action": "INSUFFICIENT_MARGIN",
                    "ticker": ticker,
                    "reason": f"保证金不足，需要: {required_margin:.2f}, 可用: {available_cash:.2f}",
                    "timestamp": datetime.now().isoformat()
                }

            # 计算合约数量（以币为单位）
            if action == "OPEN_LONG":
                # 做多：买入合约
                contract_quantity = notional_value / current_price
                new_long = current_position["long"] + contract_quantity
                new_short = current_position["short"]

                # 计算新的成本基础
                total_cost = (current_position["long"] * current_position["long_cost_basis"] +
                             contract_quantity * current_price)
                new_long_cost_basis = total_cost / new_long if new_long > 0 else current_price

                return {
                    "action": "OPEN_LONG",
                    "ticker": ticker,
                    "reason": decision.get("reasons", ["开多仓"]),
                    "contract_quantity": contract_quantity,
                    "notional_value": notional_value,
                    "entry_price": current_price,
                    "leverage": effective_leverage,
                    "required_margin": required_margin,
                    "new_long": new_long,
                    "new_short": new_short,
                    "new_long_cost_basis": new_long_cost_basis,
                    "position_change": contract_quantity,
                    "cost_impact": -required_margin,
                    "margin_impact": required_margin,
                    "timestamp": datetime.now().isoformat()
                }

            elif action == "OPEN_SHORT":
                # 做空：卖出合约
                contract_quantity = notional_value / current_price
                new_long = current_position["long"]
                new_short = current_position["short"] + contract_quantity

                # 计算新的成本基础
                total_cost = (current_position["short"] * current_position["short_cost_basis"] +
                             contract_quantity * current_price)
                new_short_cost_basis = total_cost / new_short if new_short > 0 else current_price

                # 计算空仓保证金
                new_short_margin_used = current_position["short_margin_used"] + required_margin

                return {
                    "action": "OPEN_SHORT",
                    "ticker": ticker,
                    "reason": decision.get("reasons", ["开空仓"]),
                    "contract_quantity": contract_quantity,
                    "notional_value": notional_value,
                    "entry_price": current_price,
                    "leverage": effective_leverage,
                    "required_margin": required_margin,
                    "new_long": new_long,
                    "new_short": new_short,
                    "new_short_cost_basis": new_short_cost_basis,
                    "new_short_margin_used": new_short_margin_used,
                    "position_change": contract_quantity,
                    "cost_impact": -required_margin,
                    "margin_impact": required_margin,
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            logger.error(f"开仓处理失败 {ticker}: {e}")
            return {
                "action": "ERROR",
                "ticker": ticker,
                "error": f"开仓失败: {e}",
                "timestamp": datetime.now().isoformat()
            }

    def _update_portfolio_state(
        self,
        portfolio: Dict[str, Any],
        portfolio_changes: Dict[str, Any],
        margin_manager: Optional[MarginManager]
    ) -> Dict[str, Any]:
        """更新投资组合状态"""
        try:
            updated_portfolio = portfolio.copy()

            # 更新现金余额
            total_cash_impact = 0.0
            total_margin_impact = 0.0

            # 处理每个交易对的变更
            for ticker, change in portfolio_changes.items():
                if change.get("action") in ["ERROR", "REJECTED", "UNKNOWN"]:
                    continue

                # 更新现金和保证金
                cash_impact = change.get("cost_impact", 0.0)
                margin_impact = change.get("margin_impact", 0.0)

                total_cash_impact += cash_impact
                total_margin_impact += margin_impact

                # 更新仓位信息
                if ticker not in updated_portfolio["positions"]:
                    updated_portfolio["positions"][ticker] = {
                        "long": 0.0,
                        "short": 0.0,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                        "short_margin_used": 0.0
                    }

                position = updated_portfolio["positions"][ticker]

                # 根据交易类型更新仓位
                action = change.get("action", "")

                if action == "CLOSE_POSITION":
                    # 平仓
                    position["long"] = 0.0
                    position["short"] = 0.0
                    position["long_cost_basis"] = 0.0
                    position["short_cost_basis"] = 0.0
                    position["short_margin_used"] = 0.0

                    # 更新已实现损益
                    if ticker not in updated_portfolio["realized_gains"]:
                        updated_portfolio["realized_gains"][ticker] = {"long": 0.0, "short": 0.0}

                    realized_pnl = change.get("realized_pnl", 0.0)
                    if realized_pnl != 0:
                        # 简单分配给long（实际应该更精细）
                        updated_portfolio["realized_gains"][ticker]["long"] += realized_pnl

                elif action == "OPEN_LONG":
                    # 开多仓
                    position["long"] = change.get("new_long", position["long"])
                    position["long_cost_basis"] = change.get("new_long_cost_basis", position["long_cost_basis"])

                elif action == "OPEN_SHORT":
                    # 开空仓
                    position["short"] = change.get("new_short", position["short"])
                    position["short_cost_basis"] = change.get("new_short_cost_basis", position["short_cost_basis"])
                    position["short_margin_used"] = change.get("new_short_margin_used", position["short_margin_used"])

            # 更新总体资金状态
            updated_portfolio["cash"] = max(0.0, portfolio.get("cash", 0.0) + total_cash_impact)
            updated_portfolio["margin_used"] = max(0.0, portfolio.get("margin_used", 0.0) + total_margin_impact)

            logger.info(f"投资组合状态已更新: 现金={updated_portfolio['cash']:.2f}, 保证金使用={updated_portfolio['margin_used']:.2f}")

            return updated_portfolio

        except Exception as e:
            logger.error(f"投资组合状态更新失败: {e}")
            return portfolio

    def _calculate_portfolio_statistics(
        self,
        portfolio: Dict[str, Any],
        portfolio_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算投资组合统计数据"""
        try:
            stats = {
                "total_cash": portfolio.get("cash", 0.0),
                "total_margin_used": portfolio.get("margin_used", 0.0),
                "available_margin": 0.0,
                "margin_utilization_ratio": 0.0,
                "active_positions": 0,
                "total_long_positions": 0,
                "total_short_positions": 0,
                "total_notional_value": 0.0,
                "executed_trades": 0,
                "rejected_trades": 0,
                "errors": 0
            }

            # 计算可用保证金
            initial_cash = portfolio.get("cash", 0.0) + portfolio.get("margin_used", 0.0)
            stats["available_margin"] = initial_cash - portfolio.get("margin_used", 0.0)

            # 计算保证金利用率
            if initial_cash > 0:
                stats["margin_utilization_ratio"] = portfolio.get("margin_used", 0.0) / initial_cash

            # 统计仓位
            positions = portfolio.get("positions", {})
            for ticker, position in positions.items():
                long_pos = position.get("long", 0.0)
                short_pos = position.get("short", 0.0)

                if long_pos > 0 or short_pos > 0:
                    stats["active_positions"] += 1

                if long_pos > 0:
                    stats["total_long_positions"] += 1
                    # 简化的名义价值计算（需要当前价格）
                    stats["total_notional_value"] += long_pos * position.get("long_cost_basis", 0.0)

                if short_pos > 0:
                    stats["total_short_positions"] += 1
                    stats["total_notional_value"] += short_pos * position.get("short_cost_basis", 0.0)

            # 统计交易执行情况
            for ticker, change in portfolio_changes.items():
                action = change.get("action", "")
                if action in ["OPEN_LONG", "OPEN_SHORT", "CLOSE_POSITION"]:
                    stats["executed_trades"] += 1
                elif action in ["REJECTED", "INSUFFICIENT_FUNDS", "INSUFFICIENT_MARGIN"]:
                    stats["rejected_trades"] += 1
                elif action == "ERROR":
                    stats["errors"] += 1

            return stats

        except Exception as e:
            logger.error(f"投资组合统计计算失败: {e}")
            return {"error": str(e)}

    def _generate_execution_summary(
        self,
        portfolio_changes: Dict[str, Any],
        portfolio_stats: Dict[str, Any],
        overall_action: str
    ) -> Dict[str, Any]:
        """生成执行摘要"""
        try:
            summary = {
                "overall_action": overall_action,
                "total_transactions": len(portfolio_changes),
                "successful_trades": portfolio_stats.get("executed_trades", 0),
                "rejected_trades": portfolio_stats.get("rejected_trades", 0),
                "errors": portfolio_stats.get("errors", 0),
                "portfolio_impact": {
                    "cash_balance": portfolio_stats.get("total_cash", 0.0),
                    "margin_used": portfolio_stats.get("total_margin_used", 0.0),
                    "active_positions": portfolio_stats.get("active_positions", 0)
                },
                "recommendations": []
            }

            # 生成建议
            margin_utilization = portfolio_stats.get("margin_utilization_ratio", 0.0)
            if margin_utilization > 0.8:
                summary["recommendations"].append("保证金利用率过高，建议降低杠杆或减少仓位")
            elif margin_utilization > 0.6:
                summary["recommendations"].append("保证金利用率偏高，需要谨慎管理风险")

            if portfolio_stats.get("errors", 0) > 0:
                summary["recommendations"].append("存在交易执行错误，建议检查系统状态")

            if portfolio_stats.get("rejected_trades", 0) > portfolio_stats.get("executed_trades", 0):
                summary["recommendations"].append("大量交易被拒绝，建议检查风险参数设置")

            return summary

        except Exception as e:
            logger.error(f"执行摘要生成失败: {e}")
            return {"error": str(e)}

    def _sanitize_portfolio_for_output(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """清理投资组合数据以便输出"""
        try:
            sanitized = {
                "cash": round(portfolio.get("cash", 0.0), 2),
                "margin_used": round(portfolio.get("margin_used", 0.0), 2),
                "margin_requirement": portfolio.get("margin_requirement", 0.0),
                "positions": {},
                "realized_gains": {}
            }

            # 清理仓位数据
            positions = portfolio.get("positions", {})
            for ticker, position in positions.items():
                if position.get("long", 0.0) > 0 or position.get("short", 0.0) > 0:
                    sanitized["positions"][ticker] = {
                        "long": round(position.get("long", 0.0), 6),
                        "short": round(position.get("short", 0.0), 6),
                        "long_cost_basis": round(position.get("long_cost_basis", 0.0), 2),
                        "short_cost_basis": round(position.get("short_cost_basis", 0.0), 2),
                        "short_margin_used": round(position.get("short_margin_used", 0.0), 2)
                    }

            # 清理已实现损益数据
            realized_gains = portfolio.get("realized_gains", {})
            for ticker, gains in realized_gains.items():
                if gains.get("long", 0.0) != 0 or gains.get("short", 0.0) != 0:
                    sanitized["realized_gains"][ticker] = {
                        "long": round(gains.get("long", 0.0), 2),
                        "short": round(gains.get("short", 0.0), 2)
                    }

            return sanitized

        except Exception as e:
            logger.error(f"投资组合数据清理失败: {e}")
            return portfolio

    def _create_suspension_response(self, state: AgentState, overall_action: str) -> Dict[str, Any]:
        """创建交易暂停响应"""
        suspension_message = HumanMessage(
            content=json.dumps({
                "status": "trading_suspended",
                "overall_action": overall_action,
                "reason": "风险管理系统暂停交易",
                "timestamp": datetime.now().isoformat(),
                "node_type": "futures_portfolio_management"
            }),
            name="futures_portfolio_suspension"
        )

        return {
            "messages": state.get("messages", []) + [suspension_message],
            "data": state.get("data", {})
        }

    def _create_error_response(self, state: AgentState, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        error_message_obj = HumanMessage(
            content=json.dumps({
                "error": error_message,
                "portfolio_management_status": "failed",
                "timestamp": datetime.now().isoformat(),
                "node_type": "futures_portfolio_management"
            }),
            name="futures_portfolio_management_error"
        )

        return {
            "messages": state.get("messages", []) + [error_message_obj],
            "data": state.get("data", {})
        }