"""
投资组合计算工具类

该模块包含用于计算合约交易基础参数的辅助方法，
将复杂的计算逻辑从 PortfolioManagementNode 中分离出来，
提高代码的可维护性和可测试性。
"""

from typing import Dict, Any, Tuple
import logging
from src.utils.exceptions import (
    MarginInsufficientError,
    LeverageExceedsLimitError,
    PositionSizeError,
    RiskLimitExceededError
)

logger = logging.getLogger(__name__)


class PortfolioCalculator:
    """投资组合参数计算器"""
    
    @staticmethod
    def determine_direction_and_operation(
        ticker: str, 
        technical_data: Dict[str, Any], 
        current_positions: Dict[str, Any]
    ) -> Tuple[str, str]:
        """确定交易方向和操作类型"""
        # 获取当前持仓情况
        position_data = current_positions.get(ticker, {})
        if isinstance(position_data, dict):
            long_position = position_data.get("long", 0.0)
            short_position = position_data.get("short", 0.0)
            # 计算净持仓（多头为正，空头为负）
            current_position = long_position - short_position
        else:
            # 向后兼容，如果是数值类型
            current_position = float(position_data) if position_data else 0.0
        
        # 从技术分析获取信号
        signal = "neutral"
        confidence = 0
        
        # 检查是否有跨时间框架分析
        if "cross_timeframe_analysis" in technical_data:
            cross_analysis = technical_data["cross_timeframe_analysis"]
            trend_alignment = cross_analysis.get("trend_alignment", "mixed")
            signal_strength = cross_analysis.get("overall_signal_strength", "weak")
            
            # 根据趋势对齐情况确定信号
            if trend_alignment == "aligned" and signal_strength in ["moderate", "strong"]:
                # 需要确定具体方向，检查主导时间框架的信号
                dominant_timeframe = cross_analysis.get("dominant_timeframe", "1h")
                timeframe_data = technical_data.get(dominant_timeframe, {})
                signal = timeframe_data.get("signal", "neutral")
                confidence = timeframe_data.get("confidence", 0)
        else:
            # 回退到单一时间框架分析
            for timeframe in ["4h", "1h", "30m", "15m", "5m"]:
                if timeframe in technical_data:
                    timeframe_data = technical_data[timeframe]
                    signal = timeframe_data.get("signal", "neutral")
                    confidence = timeframe_data.get("confidence", 0)
                    break
        
        # 确定方向
        if signal == "bullish" and confidence >= 60:
            direction = "long"
        elif signal == "bearish" and confidence >= 60:
            direction = "short"
        else:
            direction = "long"  # 默认多头，保守策略
        
        # 确定操作类型
        if current_position == 0:
            operation = "open"
        elif (current_position > 0 and direction == "long") or (current_position < 0 and direction == "short"):
            operation = "add"
        elif (current_position > 0 and direction == "short") or (current_position < 0 and direction == "long"):
            operation = "close"
        else:
            operation = "open"
            
        return direction, operation
    
    @staticmethod
    def calculate_leverage(
        risk_data: Dict[str, Any], 
        technical_data: Dict[str, Any],
        ticker: str = "UNKNOWN"
    ) -> int:
        """计算杠杆倍数"""
        try:
            # 从风险管理获取推荐杠杆
            leverage_analysis = risk_data.get("leverage_analysis", {})
            recommended_leverage = leverage_analysis.get("recommended_leverage", 3)
            max_safe_leverage = leverage_analysis.get("max_safe_leverage", 5)
            volatility_adjusted_leverage = leverage_analysis.get("volatility_adjusted_leverage", 3)
            
            # 交易所杠杆限制 (这里可以根据不同交易对设置)
            exchange_max_leverage = 20  # 默认20x
            if ticker.startswith("BTC"):
                exchange_max_leverage = 125
            elif ticker.startswith("ETH"):
                exchange_max_leverage = 50
            
            # 从技术分析获取波动率信息
            volatility_risk = "moderate"
            if "cross_timeframe_analysis" in technical_data:
                signal_strength = technical_data["cross_timeframe_analysis"].get("overall_signal_strength", "weak")
                if signal_strength == "strong":
                    volatility_risk = "low"
                elif signal_strength == "weak":
                    volatility_risk = "high"
            
            # 根据波动率调整杠杆
            if volatility_risk == "high":
                leverage = min(recommended_leverage, 2)
            elif volatility_risk == "low":
                leverage = min(max_safe_leverage, recommended_leverage + 1)
            else:
                leverage = recommended_leverage
            
            # 使用波动率调整后的杠杆作为参考
            leverage = min(leverage, volatility_adjusted_leverage)
            
            # 确保杠杆在合理范围内
            leverage = max(1, min(leverage, 10))
            
            # 检查是否超过交易所限制
            if leverage > exchange_max_leverage:
                logger.warning(f"杠杆倍数 {leverage} 超过交易所限制 {exchange_max_leverage}")
                raise LeverageExceedsLimitError(
                    requested_leverage=leverage,
                    max_allowed_leverage=exchange_max_leverage,
                    ticker=ticker,
                    reason="exchange_limit"
                )
            
            # 检查是否超过风险管理限制
            risk_max_leverage = max_safe_leverage + 2  # 风险限制稍微宽松
            if leverage > risk_max_leverage:
                logger.warning(f"杠杆倍数 {leverage} 超过风险管理限制 {risk_max_leverage}")
                raise LeverageExceedsLimitError(
                    requested_leverage=leverage,
                    max_allowed_leverage=risk_max_leverage,
                    ticker=ticker,
                    reason="risk_limit"
                )
            
            return leverage
            
        except LeverageExceedsLimitError:
            # 重新抛出杠杆异常
            raise
        except Exception as e:
            logger.error(f"计算杠杆时发生错误: {e}")
            # 返回保守的默认值
            return 2
    
    @staticmethod
    def calculate_position_size(
        portfolio_cash: float,
        risk_data: Dict[str, Any],
        current_price: float,
        leverage: int,
        margin_requirement: float,
        ticker: str = "UNKNOWN"
    ) -> Tuple[float, float]:
        """计算仓位大小和比例"""
        try:
            # 获取风险控制参数
            position_risk_control = risk_data.get("position_risk_control", {})
            max_position_size = position_risk_control.get("max_position_size", portfolio_cash * 0.1)
            position_sizing_factor = position_risk_control.get("position_sizing_factor", 0.02)
            risk_per_trade = position_risk_control.get("risk_per_trade", portfolio_cash * 0.02)
            
            # 获取保证金管理参数
            margin_management = risk_data.get("margin_management", {})
            available_margin = margin_management.get("available_margin", portfolio_cash * 0.8)
            margin_utilization = margin_management.get("margin_utilization", 0.3)
            
            # 基于风险计算的仓位大小
            risk_based_size = risk_per_trade * leverage / position_sizing_factor
            
            # 基于可用资金计算的仓位大小
            available_funds = min(portfolio_cash, available_margin)
            fund_based_size = available_funds * margin_utilization * leverage
            
            # 基于最大仓位限制
            max_allowed_size = min(max_position_size, available_funds * 0.5)
            
            # 选择最保守的仓位大小
            position_size = min(risk_based_size, fund_based_size, max_allowed_size)
            
            # 确保仓位大小为正数且不超过限制
            position_size = max(10.0, min(position_size, portfolio_cash * leverage * 0.3))
            
            # 计算所需保证金
            required_margin = position_size / leverage
            
            # 检查保证金是否足够
            if required_margin > available_margin:
                logger.warning(f"保证金不足: 需要 {required_margin:.2f}, 可用 {available_margin:.2f}")
                raise MarginInsufficientError(
                    required_margin=required_margin,
                    available_margin=available_margin,
                    ticker=ticker,
                    position_size=position_size,
                    leverage=leverage
                )
            
            # 设置仓位大小限制
            min_position_size = 10.0  # 最小仓位
            max_position_size_limit = available_margin * leverage  # 最大仓位限制
            
            # 检查仓位大小是否合理
            if position_size < min_position_size:
                raise PositionSizeError(
                    position_size=position_size,
                    issue_type="too_small",
                    ticker=ticker,
                    min_size=min_position_size
                )
            
            if position_size > max_position_size_limit:
                raise PositionSizeError(
                    position_size=position_size,
                    issue_type="too_large",
                    ticker=ticker,
                    max_size=max_position_size_limit
                )
            
            # 计算仓位比例
            total_portfolio_value = portfolio_cash  # 简化计算，实际应包含所有资产
            position_ratio = position_size / (total_portfolio_value * leverage) if total_portfolio_value > 0 else 0.0
            position_ratio = max(0.001, min(position_ratio, 0.1))  # 限制在0.1%-10%之间
            
            # 检查风险暴露限制
            exposure_limit = 0.2  # 20%最大暴露
            if position_ratio > exposure_limit:
                raise RiskLimitExceededError(
                    risk_type="exposure",
                    current_value=position_ratio,
                    limit_value=exposure_limit,
                    ticker=ticker
                )
            
            return position_size, position_ratio
            
        except (MarginInsufficientError, PositionSizeError, RiskLimitExceededError):
            # 重新抛出特定异常
            raise
        except Exception as e:
            logger.error(f"计算仓位大小时发生错误: {e}")
            # 返回保守的默认值
            return 10.0, 0.001
    
    @staticmethod
    def determine_entry_strategy(
        current_price: float,
        technical_data: Dict[str, Any],
        direction: str
    ) -> Tuple[float, str]:
        """确定入场策略和价格目标"""
        # 默认使用市价单和当前价格
        entry_price_target = current_price
        order_type = "market"
        
        # 检查是否有价格水平信息
        price_levels = {}
        atr_values = {}
        
        # 从任何时间框架获取价格水平和ATR信息
        for timeframe in ["4h", "1h", "30m", "15m", "5m"]:
            if timeframe in technical_data:
                timeframe_data = technical_data[timeframe]
                if "price_levels" in timeframe_data:
                    price_levels = timeframe_data["price_levels"]
                if "atr_values" in timeframe_data:
                    atr_values = timeframe_data["atr_values"]
                break
        
        # 如果有价格水平信息，尝试优化入场点
        if price_levels:
            support_levels = price_levels.get("support_levels", [])
            resistance_levels = price_levels.get("resistance_levels", [])
            
            if direction == "long" and support_levels:
                # 对于多头，尝试在支撑位附近入场
                nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                if abs(nearest_support - current_price) / current_price <= 0.02:  # 2%以内
                    entry_price_target = nearest_support * 1.001  # 稍高于支撑位
                    order_type = "limit"
            elif direction == "short" and resistance_levels:
                # 对于空头，尝试在阻力位附近入场
                nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                if abs(nearest_resistance - current_price) / current_price <= 0.02:  # 2%以内
                    entry_price_target = nearest_resistance * 0.999  # 稍低于阻力位
                    order_type = "limit"
        
        # 如果有ATR信息，可以基于波动率调整入场策略
        if atr_values:
            atr_14 = atr_values.get("atr_14", 0)
            if atr_14 > 0:
                atr_percentage = atr_14 / current_price
                # 如果波动率较高，使用限价单等待更好的入场点
                if atr_percentage > 0.03:  # 日波动率超过3%
                    if direction == "long":
                        entry_price_target = current_price * (1 - atr_percentage * 0.5)
                    else:
                        entry_price_target = current_price * (1 + atr_percentage * 0.5)
                    order_type = "limit"
        
        return entry_price_target, order_type
    
    @staticmethod
    def validate_basic_params(basic_params: Dict[str, Any]) -> bool:
        """验证basic_params的数据完整性和合理性"""
        required_fields = [
            "direction", "operation", "leverage", "position_size", 
            "position_ratio", "current_price", "contract_value", 
            "contract_quantity", "entry_price_target", "order_type"
        ]
        
        # 检查必需字段
        for field in required_fields:
            if field not in basic_params:
                return False
        
        # 检查数值合理性
        if basic_params["leverage"] < 1 or basic_params["leverage"] > 125:
            return False
            
        if basic_params["position_size"] <= 0:
            return False
            
        if basic_params["position_ratio"] < 0 or basic_params["position_ratio"] > 1:
            return False
            
        if basic_params["direction"] not in ["long", "short"]:
            return False
            
        if basic_params["operation"] not in ["open", "close", "add", "reduce"]:
            return False
            
        if basic_params["order_type"] not in ["market", "limit", "stop_limit"]:
            return False
            
        return True