import os
from typing import Dict, Any, List, Optional
import json
import logging
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from .base_node import BaseNode, AgentState
from .state import show_agent_reasoning
from .portfolio_calculator import PortfolioCalculator
from src.llm import get_llm, json_parser
from src.utils.exceptions import (
    ContractTradingError,
    MarginInsufficientError,
    LeverageExceedsLimitError,
    LiquidationRiskError,
    PositionSizeError,
    RiskLimitExceededError
)
from src.utils.error_recovery import ErrorRecoveryManager
from src.utils.validators import create_validator, ValidationSeverity

logger = logging.getLogger(__name__)


class PortfolioManagementNode(BaseNode):
    """
    投资组合管理节点
    
    负责生成完整的交易决策，包括数据验证、风险评估和策略制定
    """
    
    def __init__(self, validation_level: str = "moderate"):
        """
        初始化投资组合管理节点
        
        Args:
            validation_level: 验证级别 (strict, moderate, lenient)
        """
        super().__init__()
        self.validation_level = validation_level
        self._validator = None
        self._error_recovery = ErrorRecoveryManager()
    
    def _get_validator(self):
        """获取或创建验证器实例"""
        if self._validator is None:
            self._validator = create_validator(validation_level=self.validation_level)
        return self._validator
    
    def _validate_trading_decision(self, ticker: str, decision_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        验证交易决策数据
        
        Args:
            ticker: 交易对
            decision_data: 待验证的决策数据
            context: 验证上下文
            
        Returns:
            包含验证结果和可能修正后数据的字典
        """
        try:
            validator = self._get_validator()
            
            # 执行验证
            is_valid, validation_results, corrected_data = validator.validate(decision_data, context)
            
            # 统计验证结果
            critical_issues = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
            error_issues = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
            warning_issues = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
            
            # 记录验证日志
            if critical_issues or error_issues:
                logger.warning(f"{ticker} 验证发现问题: 严重 {len(critical_issues)}, 错误 {len(error_issues)}, 警告 {len(warning_issues)}")
                for issue in critical_issues + error_issues:
                    logger.warning(f"  - {issue.field_name}: {issue.message}")
            
            # 生成验证报告
            validation_report = validator.format_validation_report(validation_results)
            
            return {
                "is_valid": is_valid,
                "validation_results": validation_results,
                "corrected_data": corrected_data,
                "validation_report": validation_report,
                "critical_count": len(critical_issues),
                "error_count": len(error_issues),
                "warning_count": len(warning_issues),
                "suggestions": validator.get_suggestions(validation_results)
            }
            
        except Exception as e:
            logger.error(f"{ticker} 数据验证失败: {str(e)}")
            return {
                "is_valid": False,
                "validation_results": [],
                "corrected_data": decision_data,
                "validation_report": f"验证过程异常: {str(e)}",
                "critical_count": 1,
                "error_count": 0,
                "warning_count": 0,
                "suggestions": ["请检查数据格式和验证器配置"]
            }
    
    def _extract_volatility_from_signals(self, analyst_signals: Dict[str, Any], ticker: str) -> Optional[float]:
        """
        从分析信号中提取波动率信息
        
        Args:
            analyst_signals: 分析师信号数据
            ticker: 交易对
            
        Returns:
            波动率值，如果未找到则返回None
        """
        try:
            # 尝试从技术分析信号中获取波动率
            tech_signals = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
            
            for timeframe, signals in tech_signals.items():
                if isinstance(signals, dict):
                    # 查找波动率策略信号
                    volatility_signals = signals.get("strategy_signals", {}).get("volatility", {})
                    if volatility_signals and "metrics" in volatility_signals:
                        metrics = volatility_signals["metrics"]
                        
                        # 尝试获取历史波动率
                        if "historical_volatility" in metrics:
                            return float(metrics["historical_volatility"])
                        
                        # 尝试获取ATR比率作为波动率代理
                        if "atr_ratio" in metrics:
                            return float(metrics["atr_ratio"])
            
            # 如果没有找到明确的波动率指标，返回默认值
            return 0.03  # 3%作为默认波动率
            
        except Exception as e:
            logger.warning(f"提取 {ticker} 波动率信息失败: {e}")
            return 0.03
    
    def calculate_basic_params(
        self,
        ticker: str,
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any],
        current_price: float
    ) -> Dict[str, Any]:
        """
        计算基础交易参数，根据futures_trading_output_specification.md规范
        增加了完整的异常处理和错误恢复机制
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            analyst_signals: 分析师信号数据，包含技术分析和风险管理信息
            portfolio: 投资组合数据，包含现金、持仓、保证金等信息
            current_price: 当前市场价格
            
        Returns:
            包含basic_params字段的字典，符合规范要求
        """
        # 初始化错误恢复管理器
        recovery_manager = ErrorRecoveryManager()
        
        # 获取风险管理数据和技术分析数据
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        
        # 获取投资组合基础信息
        portfolio_cash = portfolio.get("cash", 0.0)
        current_positions = portfolio.get("positions", {})
        margin_requirement = portfolio.get("margin_requirement", 0.5)  # 默认50%保证金要求
        
        # 使用PortfolioCalculator进行计算
        calculator = PortfolioCalculator()
        
        # 存储计算过程中的异常信息
        calculation_errors = []
        recovery_attempts = 0
        max_recovery_attempts = 3
        
        while recovery_attempts <= max_recovery_attempts:
            try:
                # 确定交易方向和操作类型
                direction, operation = calculator.determine_direction_and_operation(
                    ticker, technical_data, current_positions
                )
                
                # 计算杠杆倍数（带异常处理）
                try:
                    leverage = calculator.calculate_leverage(risk_data, technical_data, ticker)
                except LeverageExceedsLimitError as e:
                    logger.warning(f"杠杆超限异常: {e}")
                    calculation_errors.append(e)
                    # 自动调整到最大允许杠杆
                    leverage = e.max_allowed_leverage
                    logger.info(f"杠杆已自动调整至: {leverage}")
                
                # 计算仓位大小和比例（带异常处理）
                try:
                    position_size, position_ratio = calculator.calculate_position_size(
                        portfolio_cash, risk_data, current_price, leverage, margin_requirement, ticker
                    )
                except (MarginInsufficientError, PositionSizeError, RiskLimitExceededError) as e:
                    logger.warning(f"仓位计算异常: {e}")
                    calculation_errors.append(e)
                    
                    # 尝试错误恢复
                    basic_params_temp = {
                        "direction": direction,
                        "operation": operation,
                        "leverage": leverage,
                        "position_size": portfolio_cash * 0.01,  # 临时值用于恢复计算
                        "position_ratio": 0.001,
                        "current_price": current_price,
                        "contract_value": 0,
                        "contract_quantity": 0,
                        "entry_price_target": current_price,
                        "order_type": "market"
                    }
                    
                    recovered, adjusted_params = recovery_manager.apply_recovery_strategy(e, basic_params_temp)
                    
                    if recovered:
                        logger.info(f"错误恢复成功: {type(e).__name__}")
                        position_size = adjusted_params["position_size"]
                        leverage = adjusted_params["leverage"]
                        position_ratio = adjusted_params.get("position_ratio", 0.001)
                        
                        # 如果有紧急操作指示，记录下来
                        if "operation" in adjusted_params:
                            operation = adjusted_params["operation"]
                        if "urgency" in adjusted_params:
                            # 记录紧急程度，后续在metadata中使用
                            pass
                    else:
                        logger.error(f"错误恢复失败: {type(e).__name__}")
                        if recovery_attempts >= max_recovery_attempts:
                            raise e
                        recovery_attempts += 1
                        continue
                
                # 计算合约价值和数量
                contract_value = position_size / leverage
                contract_quantity = position_size / current_price
                
                # 确定入场价格目标和订单类型
                entry_price_target, order_type = calculator.determine_entry_strategy(
                    current_price, technical_data, direction
                )
                
                # 构建基础参数字典
                basic_params = {
                    "direction": direction,
                    "operation": operation,
                    "leverage": leverage,
                    "position_size": round(position_size, 2),
                    "position_ratio": round(position_ratio, 4),
                    "current_price": current_price,
                    "contract_value": round(contract_value, 2),
                    "contract_quantity": round(contract_quantity, 6),
                    "entry_price_target": round(entry_price_target, 2),
                    "order_type": order_type
                }
                
                # 验证参数有效性
                if not calculator.validate_basic_params(basic_params):
                    error_msg = f"计算的basic_params不符合规范要求: {basic_params}"
                    logger.error(error_msg)
                    
                    if recovery_attempts >= max_recovery_attempts:
                        raise ValueError(error_msg)
                    
                    recovery_attempts += 1
                    continue
                
                # 添加异常处理相关的元数据
                if calculation_errors:
                    basic_params["_calculation_errors"] = [
                        {
                            "error_type": type(e).__name__,
                            "error_code": e.error_code if hasattr(e, 'error_code') else "UNKNOWN",
                            "recovery_applied": True
                        }
                        for e in calculation_errors
                    ]
                    basic_params["_recovery_attempts"] = recovery_attempts
                
                logger.info(f"基础参数计算完成: {ticker}")
                return basic_params
                
            except ContractTradingError as e:
                logger.error(f"合约交易异常: {e}")
                calculation_errors.append(e)
                
                # 尝试通用恢复策略
                if recovery_attempts < max_recovery_attempts:
                    recovery_attempts += 1
                    logger.info(f"尝试恢复计算 (第 {recovery_attempts} 次)")
                    continue
                else:
                    # 返回保守的默认参数
                    logger.warning("达到最大恢复尝试次数，返回保守参数")
                    return self._get_conservative_params(ticker, current_price, portfolio_cash, calculation_errors)
                    
            except Exception as e:
                logger.error(f"计算基础参数时发生未预期错误: {e}")
                if recovery_attempts < max_recovery_attempts:
                    recovery_attempts += 1
                    continue
                else:
                    raise e
        
        # 如果所有恢复尝试都失败，返回保守参数
        logger.warning("所有恢复尝试都失败，返回保守参数")
        return self._get_conservative_params(ticker, current_price, portfolio_cash, calculation_errors)

    
    def _get_conservative_params(
        self,
        ticker: str,
        current_price: float,
        portfolio_cash: float,
        calculation_errors: List[ContractTradingError]
    ) -> Dict[str, Any]:
        """
        生成保守的默认交易参数，用于异常恢复失败的情况
        
        Args:
            ticker: 交易对符号
            current_price: 当前价格
            portfolio_cash: 可用现金
            calculation_errors: 计算过程中遇到的异常列表
            
        Returns:
            保守的交易参数字典
        """
        logger.info(f"生成保守参数: {ticker}")
        
        # 使用最保守的设置
        leverage = 1  # 最小杠杆
        position_size = min(portfolio_cash * 0.01, 50.0)  # 1%资金或50 USDT，取较小值
        position_ratio = 0.001  # 0.1%仓位比例
        
        contract_value = position_size / leverage
        contract_quantity = position_size / current_price
        
        basic_params = {
            "direction": "long",  # 默认多头方向
            "operation": "open",  # 默认开仓操作
            "leverage": leverage,
            "position_size": round(position_size, 2),
            "position_ratio": round(position_ratio, 4),
            "current_price": current_price,
            "contract_value": round(contract_value, 2),
            "contract_quantity": round(contract_quantity, 6),
            "entry_price_target": round(current_price, 2),
            "order_type": "market"
        }
        
        # 添加异常信息到元数据
        basic_params["_is_conservative_fallback"] = True
        basic_params["_calculation_errors"] = [
            {
                "error_type": type(e).__name__,
                "error_code": e.error_code if hasattr(e, 'error_code') else "UNKNOWN",
                "recovery_applied": False,
                "message": str(e)
            }
            for e in calculation_errors
        ]
        
        logger.warning(f"使用保守参数: leverage={leverage}, position_size={position_size}")
        return basic_params

    def design_risk_management(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        设计风险管理策略，根据futures_trading_output_specification.md规范
        增加了完整的异常处理和强平风险管理
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            basic_params: 基础交易参数，包含方向、杠杆、仓位等
            analyst_signals: 分析师信号数据，包含ATR数据和价格水平
            portfolio: 投资组合数据
            
        Returns:
            包含risk_management字段的字典，符合规范要求
        """
        try:
            # 初始化错误恢复管理器
            recovery_manager = ErrorRecoveryManager()
            
            # 获取基础参数
            direction = basic_params.get("direction", "long")
            leverage = basic_params.get("leverage", 1)
            position_size = basic_params.get("position_size", 0.0)
            current_price = basic_params.get("current_price", 0.0)
            entry_price_target = basic_params.get("entry_price_target", current_price)
            
            # 获取技术分析数据
            technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
            risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
            
            # 获取ATR数据
            atr_values = {}
            price_levels = {}
            volatility_analysis = {}
            
            # 从技术分析中提取关键数据
            for timeframe in ["4h", "1h", "30m", "15m", "5m"]:
                if timeframe in technical_data:
                    timeframe_data = technical_data[timeframe]
                    if "atr_values" in timeframe_data:
                        atr_values = timeframe_data["atr_values"]
                    if "price_levels" in timeframe_data:
                        price_levels = timeframe_data["price_levels"]
                    if "volatility_analysis" in timeframe_data:
                        volatility_analysis = timeframe_data["volatility_analysis"]
                    break
            
            # 获取风险管理相关数据
            margin_management = risk_data.get("margin_management", {})
            liquidation_analysis = risk_data.get("liquidation_analysis", {})
            position_risk_control = risk_data.get("position_risk_control", {})
            
            # 计算ATR基础值
            atr_14 = atr_values.get("atr_14", current_price * 0.02)  # 默认2%波动
            atr_28 = atr_values.get("atr_28", current_price * 0.015)  # 默认1.5%波动
            atr_percentile = atr_values.get("atr_percentile", 0.5)
            
            # 确保ATR值是数值类型
            if not isinstance(atr_14, (int, float)):
                atr_14 = current_price * 0.02
                
            if not isinstance(atr_28, (int, float)):
                atr_28 = current_price * 0.015
                
            if not isinstance(atr_percentile, (int, float)):
                atr_percentile = 0.5
            
            # 1. 计算止损价格
            # 基于ATR的止损距离
            atr_multiplier = 2.0 if direction == "long" else 2.0
            if atr_percentile > 0.8:  # 高波动环境
                atr_multiplier = 2.5
            elif atr_percentile < 0.2:  # 低波动环境
                atr_multiplier = 1.5
                
            atr_stop_distance = atr_14 * atr_multiplier
            
            if direction == "long":
                stop_loss = entry_price_target - atr_stop_distance
                atr_based_stop = current_price - atr_stop_distance
            else:
                stop_loss = entry_price_target + atr_stop_distance
                atr_based_stop = current_price + atr_stop_distance
                
            # 考虑支撑阻力位调整止损
            if price_levels:
                support_levels = price_levels.get("support_levels", [])
                resistance_levels = price_levels.get("resistance_levels", [])
                
                if direction == "long" and support_levels:
                    # 寻找最近的支撑位作为止损参考
                    nearby_support = [s for s in support_levels if s < current_price and s > stop_loss]
                    if nearby_support:
                        stop_loss = max(stop_loss, min(nearby_support) * 0.995)  # 稍低于支撑位
                        
                elif direction == "short" and resistance_levels:
                    # 寻找最近的阻力位作为止损参考
                    nearby_resistance = [r for r in resistance_levels if r > current_price and r < stop_loss]
                    if nearby_resistance:
                        stop_loss = min(stop_loss, max(nearby_resistance) * 1.005)  # 稍高于阻力位
            
            # 2. 计算强平价格和风险检测
            liquidation_price = liquidation_analysis.get("liquidation_price", 0.0)
            if liquidation_price <= 0:
                # 简化的强平价格计算
                margin_ratio = 1.0 / leverage  # 保证金比例
                if direction == "long":
                    liquidation_price = entry_price_target * (1 - margin_ratio * 0.9)
                else:
                    liquidation_price = entry_price_target * (1 + margin_ratio * 0.9)
            
            # 检查强平风险
            distance_to_liquidation = abs(current_price - liquidation_price) / current_price * 100
            
            if distance_to_liquidation <= 15.0:  # 距离强平15%以内
                risk_level = "emergency" if distance_to_liquidation <= 5.0 else "critical" if distance_to_liquidation <= 10.0 else "high"
                
                logger.warning(f"检测到强平风险: 距离强平 {distance_to_liquidation:.2f}%")
                
                liquidation_error = LiquidationRiskError(
                    current_price=current_price,
                    liquidation_price=liquidation_price,
                    distance_to_liquidation=distance_to_liquidation,
                    risk_level=risk_level,
                    ticker=ticker,
                    position_info={
                        "direction": direction,
                        "leverage": leverage,
                        "position_size": position_size
                    }
                )
                
                # 应用强平风险恢复策略
                recovered, adjusted_params = recovery_manager.handle_liquidation_risk_error(
                    liquidation_error, basic_params
                )
                
                if recovered:
                    if "stop_loss" in adjusted_params:
                        stop_loss = adjusted_params["stop_loss"]
                        logger.info(f"强平风险恢复: 设置紧急止损 {stop_loss}")
                    
                    # 如果建议平仓，在metadata中标记
                    if adjusted_params.get("operation") == "close":
                        basic_params["_emergency_close_recommended"] = True
                        logger.critical("建议立即平仓以避免强平")
            
            # 3. 计算止盈价格
            # 使用风险收益比计算止盈
            risk_distance = abs(entry_price_target - stop_loss)
            risk_reward_ratio = 2.0  # 默认1:2风险收益比
            
            # 根据信号强度调整风险收益比
            overall_confidence = 50  # 默认置信度
            if "cross_timeframe_analysis" in technical_data:
                cross_analysis = technical_data["cross_timeframe_analysis"]
                signal_strength = cross_analysis.get("overall_signal_strength", "moderate")
                if signal_strength == "strong":
                    risk_reward_ratio = 2.5
                    overall_confidence = 75
                elif signal_strength == "weak":
                    risk_reward_ratio = 1.5
                    overall_confidence = 35
                    
            profit_distance = risk_distance * risk_reward_ratio
            
            if direction == "long":
                take_profit = entry_price_target + profit_distance
            else:
                take_profit = entry_price_target - profit_distance
                
            # 考虑阻力支撑位调整止盈
            if price_levels:
                if direction == "long" and resistance_levels:
                    nearby_resistance = [r for r in resistance_levels if r > entry_price_target]
                    if nearby_resistance:
                        take_profit = min(take_profit, min(nearby_resistance) * 0.995)
                elif direction == "short" and support_levels:
                    nearby_support = [s for s in support_levels if s < entry_price_target]
                    if nearby_support:
                        take_profit = max(take_profit, max(nearby_support) * 1.005)
            
            # 4. 计算跟踪止损
            trailing_stop_distance = atr_14 * 1.5  # 相对保守的跟踪距离
            if direction == "long":
                trailing_stop = current_price - trailing_stop_distance
            else:
                trailing_stop = current_price + trailing_stop_distance
                
            # 5. 计算保证金占用
            contract_value = position_size / leverage  # 实际投入资金
            margin_required = contract_value  # 初始保证金
            maintenance_margin = margin_management.get("maintenance_margin", contract_value * 0.5)
            
            # 6. 计算风险百分比
            # 正确计算投资组合总价值：现金 + 所有持仓的市值
            portfolio_cash = portfolio.get("cash", 0.0)
            positions_value = 0.0
            
            # 遍历所有持仓，计算总市值
            positions = portfolio.get("positions", {})
            for ticker_pos in positions.values():
                if isinstance(ticker_pos, dict):
                    long_shares = ticker_pos.get("long", 0.0)
                    short_shares = ticker_pos.get("short", 0.0)
                    long_cost_basis = ticker_pos.get("long_cost_basis", 0.0)
                    short_cost_basis = ticker_pos.get("short_cost_basis", 0.0)
                    
                    # 计算多头持仓价值（使用成本价）
                    if long_shares > 0 and long_cost_basis > 0:
                        positions_value += long_shares * long_cost_basis
                    
                    # 计算空头持仓价值（空头减少总价值）
                    if short_shares > 0 and short_cost_basis > 0:
                        positions_value -= short_shares * short_cost_basis
            
            portfolio_value = portfolio_cash + positions_value
            risk_amount = abs(entry_price_target - stop_loss) * (position_size / current_price)
            risk_percentage = (risk_amount / portfolio_value * 100) if portfolio_value > 0 else 0.0
            
            # 检查风险百分比是否超限
            max_risk_percentage = position_risk_control.get("max_risk_per_trade", 2.0)  # 默认2%
            if risk_percentage > max_risk_percentage:
                logger.warning(f"风险百分比超限: {risk_percentage:.2f}% > {max_risk_percentage}%")
                
                risk_error = RiskLimitExceededError(
                    risk_type="drawdown",
                    current_value=risk_percentage,
                    limit_value=max_risk_percentage,
                    ticker=ticker
                )
                
                # 应用风险恢复策略（调整仓位大小）
                recovered, adjusted_params = recovery_manager.handle_risk_limit_exceeded_error(
                    risk_error, basic_params
                )
                
                if recovered:
                    # 重新计算相关参数
                    adjusted_position_size = adjusted_params["position_size"]
                    risk_amount = abs(entry_price_target - stop_loss) * (adjusted_position_size / current_price)
                    risk_percentage = (risk_amount / portfolio_value * 100) if portfolio_value > 0 else 0.0
                    logger.info(f"风险控制恢复: 调整后风险百分比 {risk_percentage:.2f}%")
            
            # 7. 波动率调整仓位
            volatility_forecast = volatility_analysis.get("volatility_forecast", atr_14 / current_price)
            base_position_ratio = basic_params.get("position_ratio", 0.02)
            
            # 确保volatility_forecast是数值类型
            if not isinstance(volatility_forecast, (int, float)):
                volatility_forecast = atr_14 / current_price if current_price > 0 else 0.05
            
            # 根据波动率调整仓位
            if volatility_forecast > 0.05:  # 高波动
                volatility_adjustment = 0.7
            elif volatility_forecast < 0.02:  # 低波动
                volatility_adjustment = 1.3
            else:
                volatility_adjustment = 1.0
                
            volatility_adjusted_size = position_size * volatility_adjustment
            
            # 8. 计算最大亏损金额
            max_loss_amount = abs(entry_price_target - stop_loss) * (position_size / current_price)
            
            # 9. 预期持仓时间（基于信号强度和波动率）
            base_holding_hours = 24  # 基础持仓时间24小时
            
            # 确保overall_confidence是数值类型
            if not isinstance(overall_confidence, (int, float)):
                overall_confidence = 50  # 默认值
                
            if overall_confidence > 70:
                position_hold_time = int(base_holding_hours * 1.5)  # 高置信度延长持仓
            elif overall_confidence < 40:
                position_hold_time = int(base_holding_hours * 0.5)  # 低置信度缩短持仓
            else:
                position_hold_time = base_holding_hours
                
            # 根据波动率调整持仓时间
            if volatility_forecast > 0.04:
                position_hold_time = int(position_hold_time * 0.7)  # 高波动缩短持仓
            
            # 重新计算实际的风险收益比（基于最终的止损止盈价格）
            actual_risk_distance = abs(entry_price_target - stop_loss)
            actual_reward_distance = abs(take_profit - entry_price_target)
            actual_risk_reward_ratio = actual_reward_distance / actual_risk_distance if actual_risk_distance > 0 else risk_reward_ratio
            
            # 构建风险管理字典
            risk_management = {
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "trailing_stop": round(trailing_stop, 2),
                "liquidation_price": round(liquidation_price, 2),
                "margin_required": round(margin_required, 2),
                "risk_percentage": round(risk_percentage, 2),
                "risk_reward_ratio": round(actual_risk_reward_ratio, 2),
                "atr_based_stop": round(atr_based_stop, 2),
                "volatility_adjusted_size": round(volatility_adjusted_size, 2),
                "max_loss_amount": round(max_loss_amount, 2),
                "position_hold_time": position_hold_time
            }
            
            # 添加风险监控信息
            risk_management["_risk_monitoring"] = {
                "distance_to_liquidation": round(distance_to_liquidation, 2),
                "liquidation_risk_level": "low" if distance_to_liquidation > 15.0 else "high",
                "risk_percentage_status": "normal" if risk_percentage <= max_risk_percentage else "exceeded",
                "volatility_level": "high" if volatility_forecast > 0.05 else "low" if volatility_forecast < 0.02 else "normal"
            }
            
            logger.info(f"风险管理设计完成: {ticker}")
            return risk_management
            
        except (LiquidationRiskError, RiskLimitExceededError) as e:
            logger.error(f"风险管理设计异常: {e}")
            # 返回极保守的风险管理参数
            return self._get_conservative_risk_management(
                ticker, basic_params, current_price, [e]
            )
            
        except Exception as e:
            logger.error(f"风险管理设计时发生未预期错误: {e}")
            # 返回默认风险管理参数
            return self._get_conservative_risk_management(
                ticker, basic_params, basic_params.get("current_price", 0.0), [e]
            )

    def _get_conservative_risk_management(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        current_price: float,
        errors: List[Exception]
    ) -> Dict[str, Any]:
        """
        生成保守的风险管理参数，用于异常恢复失败的情况
        
        Args:
            ticker: 交易对符号
            basic_params: 基础交易参数
            current_price: 当前价格
            errors: 计算过程中遇到的异常列表
            
        Returns:
            保守的风险管理参数字典
        """
        logger.info(f"生成保守风险管理参数: {ticker}")
        
        direction = basic_params.get("direction", "long")
        entry_price = basic_params.get("entry_price_target", current_price)
        position_size = basic_params.get("position_size", 10.0)
        leverage = basic_params.get("leverage", 1)
        
        # 使用非常保守的止损距离（1%）
        stop_distance = current_price * 0.01
        
        if direction == "long":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * 1.5  # 1.5:1 收益风险比
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * 1.5
        
        # 计算保守的强平价格（留足缓冲）
        margin_ratio = 1.0 / leverage
        if direction == "long":
            liquidation_price = entry_price * (1 - margin_ratio * 0.5)  # 更保守的缓冲
        else:
            liquidation_price = entry_price * (1 + margin_ratio * 0.5)
        
        risk_management = {
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "trailing_stop": round(stop_loss, 2),  # 与固定止损相同
            "liquidation_price": round(liquidation_price, 2),
            "margin_required": round(position_size / leverage, 2),
            "risk_percentage": 1.0,  # 固定1%风险
            "risk_reward_ratio": 1.5,  # 保守的收益风险比
            "atr_based_stop": round(stop_loss, 2),
            "volatility_adjusted_size": round(position_size * 0.5, 2),  # 减半仓位
            "max_loss_amount": round(position_size * 0.01, 2),  # 预期最大亏损1%
            "position_hold_time": 12  # 短期持仓12小时
        }
        
        # 添加异常信息
        risk_management["_is_conservative_fallback"] = True
        risk_management["_errors"] = [
            {
                "error_type": type(e).__name__,
                "message": str(e)
            }
            for e in errors
        ]
        
        risk_management["_risk_monitoring"] = {
            "distance_to_liquidation": 50.0,  # 假设50%安全距离
            "liquidation_risk_level": "low",
            "risk_percentage_status": "conservative",
            "volatility_level": "unknown"
        }
        
        logger.warning(f"使用保守风险管理参数: {ticker}")
        return risk_management

    def analyze_timeframes(
        self,
        ticker: str,
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析多时间框架信号一致性，根据futures_trading_output_specification.md规范
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            analyst_signals: 技术分析师信号数据
            portfolio: 投资组合数据
            
        Returns:
            包含timeframe_analysis字段的字典，包含：
            - consensus_score: float (共识评分 0-1)
            - dominant_timeframe: "5m|15m|30m|1h|4h"
            - signal_alignment: "strong|moderate|weak"
            - conflicting_signals: int (冲突信号数量)
            - timeframe_weights: 各时间框架权重
            - overall_direction_confidence: float
        """
        # 获取技术分析数据
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        
        # 定义时间框架优先级权重（较大时间框架权重更高）
        timeframe_priorities = {
            "5m": 1.0,
            "15m": 1.5, 
            "30m": 2.0,
            "1h": 3.0,
            "4h": 4.0
        }
        
        # 收集各时间框架的信号数据
        timeframe_signals = {}
        total_weight = 0.0
        
        for timeframe in ["5m", "15m", "30m", "1h", "4h"]:
            if timeframe in technical_data:
                timeframe_data = technical_data[timeframe]
                signal = timeframe_data.get("signal", "neutral")
                confidence = timeframe_data.get("confidence", 0) / 100.0  # 转换为0-1范围
                
                # 将信号转换为数值：bullish=1, neutral=0, bearish=-1
                signal_value = 1.0 if signal == "bullish" else (-1.0 if signal == "bearish" else 0.0)
                
                # 计算加权信号强度
                weight = timeframe_priorities.get(timeframe, 1.0)
                weighted_signal = signal_value * confidence * weight
                
                timeframe_signals[timeframe] = {
                    "signal": signal,
                    "confidence": confidence,
                    "weight": weight,
                    "weighted_signal": weighted_signal,
                    "signal_value": signal_value
                }
                
                total_weight += weight
        
        # 如果没有获取到任何信号数据，返回默认值
        if not timeframe_signals:
            return {
                "consensus_score": 0.0,
                "dominant_timeframe": "1h",
                "signal_alignment": "weak", 
                "conflicting_signals": 0,
                "timeframe_weights": {
                    "5m": 0.2, "15m": 0.2, "30m": 0.2, "1h": 0.2, "4h": 0.2
                },
                "overall_direction_confidence": 0.0
            }
        
        # 计算标准化权重
        normalized_weights = {}
        for timeframe, data in timeframe_signals.items():
            normalized_weights[timeframe] = data["weight"] / total_weight if total_weight > 0 else 0.0
            
        # 补充缺失的时间框架权重为0
        for tf in ["5m", "15m", "30m", "1h", "4h"]:
            if tf not in normalized_weights:
                normalized_weights[tf] = 0.0
        
        # 计算共识评分 - 衡量信号方向的一致性
        signals_list = []
        confidences_list = []
        weights_list = []
        
        for data in timeframe_signals.values():
            if data["signal"] != "neutral":  # 只考虑有明确方向的信号
                signals_list.append(data["signal_value"])
                confidences_list.append(data["confidence"])
                weights_list.append(data["weight"])
        
        if not signals_list:
            consensus_score = 0.0
            overall_direction = 0.0
        else:
            # 计算加权平均信号方向
            weighted_direction = sum(s * c * w for s, c, w in zip(signals_list, confidences_list, weights_list))
            total_weighted = sum(c * w for c, w in zip(confidences_list, weights_list))
            overall_direction = weighted_direction / total_weighted if total_weighted > 0 else 0.0
            
            # 计算一致性分数 - 基于信号方向的标准差
            if len(signals_list) > 1:
                avg_signal = sum(signals_list) / len(signals_list)
                variance = sum((s - avg_signal) ** 2 for s in signals_list) / len(signals_list)
                std_deviation = variance ** 0.5
                consensus_score = max(0.0, 1.0 - std_deviation / 2.0)  # 标准差越小，一致性越高
            else:
                consensus_score = confidences_list[0]  # 只有一个信号时，使用其置信度
        
        # 识别冲突信号
        bullish_count = sum(1 for data in timeframe_signals.values() if data["signal"] == "bullish")
        bearish_count = sum(1 for data in timeframe_signals.values() if data["signal"] == "bearish")
        conflicting_signals = min(bullish_count, bearish_count)
        
        # 确定主导时间框架 - 权重最高且有明确信号的时间框架
        dominant_timeframe = "1h"  # 默认值
        max_weighted_signal = 0.0
        
        for timeframe, data in timeframe_signals.items():
            if data["signal"] != "neutral":
                weighted_strength = abs(data["weighted_signal"])
                if weighted_strength > max_weighted_signal:
                    max_weighted_signal = weighted_strength
                    dominant_timeframe = timeframe
        
        # 评估信号对齐强度
        if consensus_score >= 0.7 and conflicting_signals <= 1:
            signal_alignment = "strong"
        elif consensus_score >= 0.5 and conflicting_signals <= 2:
            signal_alignment = "moderate"
        else:
            signal_alignment = "weak"
        
        # 计算整体方向置信度
        overall_direction_confidence = abs(overall_direction) * consensus_score
        
        # 检查cross_timeframe_analysis数据（如果可用）
        cross_analysis = technical_data.get("cross_timeframe_analysis", {})
        if cross_analysis:
            # 如果存在跨时间框架分析数据，可以用来调整结果
            existing_consensus = cross_analysis.get("timeframe_consensus", consensus_score)
            existing_dominant = cross_analysis.get("dominant_timeframe", dominant_timeframe)
            existing_trend_alignment = cross_analysis.get("trend_alignment", "mixed")
            
            # 加权平均现有数据和计算数据
            consensus_score = (consensus_score * 0.7 + existing_consensus * 0.3)
            
            # 如果现有数据指定了主导时间框架，优先使用
            if existing_dominant in ["5m", "15m", "30m", "1h", "4h"]:
                dominant_timeframe = existing_dominant
                
            # 根据趋势对齐情况调整信号对齐强度
            if existing_trend_alignment == "aligned" and signal_alignment != "strong":
                signal_alignment = "moderate"
            elif existing_trend_alignment == "divergent":
                signal_alignment = "weak"
        
        # 构建结果字典
        timeframe_analysis = {
            "consensus_score": round(float(consensus_score), 3),
            "dominant_timeframe": dominant_timeframe,
            "signal_alignment": signal_alignment,
            "conflicting_signals": int(conflicting_signals),
            "timeframe_weights": {
                timeframe: round(float(weight), 3) 
                for timeframe, weight in normalized_weights.items()
            },
            "overall_direction_confidence": round(float(overall_direction_confidence), 3)
        }
        
        return timeframe_analysis

    def calculate_cost_benefit(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算成本收益分析，根据futures_trading_output_specification.md规范
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            basic_params: 基础交易参数，包含杠杆、仓位等信息
            risk_management: 风险管理参数，包含止损止盈价格
            analyst_signals: 技术分析师信号数据
            portfolio: 投资组合数据
            
        Returns:
            包含cost_benefit_analysis字段的字典，包含12个必需字段：
            - estimated_trading_fee: float (预计交易手续费)
            - funding_rate: float (当前资金费率)
            - funding_cost_8h: float (8小时资金费用)
            - funding_cost_daily: float (日资金费用)
            - holding_cost_total: float (总持仓成本)
            - break_even_price: float (盈亏平衡价)
            - target_profit: float (目标利润)
            - expected_return: float (期望收益率)
            - profit_probability: float (获利概率)
            - loss_probability: float (亏损概率)  
            - expected_value: float (期望值)
            - roi_annualized: float (年化收益率)
        """
        # 获取基础参数
        direction = basic_params.get("direction", "long")
        leverage = basic_params.get("leverage", 1)
        position_size = basic_params.get("position_size", 0.0)
        current_price = basic_params.get("current_price", 0.0)
        entry_price_target = basic_params.get("entry_price_target", current_price)
        contract_quantity = basic_params.get("contract_quantity", 0.0)
        contract_value = basic_params.get("contract_value", 0.0)
        
        # 获取风险管理参数
        stop_loss = risk_management.get("stop_loss", 0.0)
        take_profit = risk_management.get("take_profit", 0.0)
        position_hold_time = risk_management.get("position_hold_time", 24)  # 小时
        
        # 交易所费率配置（基于主流交易所如币安的费率）
        maker_fee_rate = 0.0002  # 0.02% maker费率
        taker_fee_rate = 0.0004  # 0.04% taker费率
        funding_rate_base = 0.0001  # 0.01% 基础资金费率（每8小时）
        
        # 1. 计算预计交易手续费
        # 开仓费用 + 平仓费用（假设使用taker费率）
        estimated_trading_fee = position_size * taker_fee_rate * 2  # 开仓和平仓
        
        # 2. 获取当前资金费率（从技术分析数据中获取，如果没有则使用默认值）
        funding_rate = funding_rate_base
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        
        # 尝试从不同时间框架获取市场情况来调整资金费率
        market_sentiment = 0.0  # 中性
        for timeframe in ["4h", "1h", "30m"]:
            if timeframe in technical_data:
                tf_data = technical_data[timeframe]
                signal = tf_data.get("signal", "neutral")
                confidence = tf_data.get("confidence", 0) / 100.0
                
                if signal == "bullish":
                    market_sentiment += confidence * 0.3
                elif signal == "bearish":
                    market_sentiment -= confidence * 0.3
                break
                
        # 根据市场情绪调整资金费率（强烈看涨时做多成本高，看跌时做空成本高）
        if direction == "long" and market_sentiment > 0.5:
            funding_rate = funding_rate_base * (1 + market_sentiment)
        elif direction == "short" and market_sentiment < -0.5:
            funding_rate = funding_rate_base * (1 + abs(market_sentiment))
        else:
            funding_rate = funding_rate_base
            
        # 限制资金费率在合理范围内
        funding_rate = max(-0.0075, min(funding_rate, 0.0075))  # -0.75% to 0.75%
        
        # 3. 计算8小时资金费用
        funding_cost_8h = position_size * abs(funding_rate)
        if direction == "short":
            funding_cost_8h *= -1 if funding_rate > 0 else 1
        else:
            funding_cost_8h *= 1 if funding_rate > 0 else -1
            
        # 4. 计算日资金费用（3次8小时）
        funding_cost_daily = funding_cost_8h * 3
        
        # 5. 计算总持仓成本
        # 包括交易费用和预期持仓期间的资金费用
        holding_periods_8h = max(1, position_hold_time / 8)  # 8小时周期数
        total_funding_cost = funding_cost_8h * holding_periods_8h
        holding_cost_total = estimated_trading_fee + abs(total_funding_cost)
        
        # 6. 计算盈亏平衡价格
        # 需要覆盖所有持仓成本的价格
        cost_per_unit = holding_cost_total / contract_quantity if contract_quantity > 0 else 0
        
        if direction == "long":
            break_even_price = entry_price_target + cost_per_unit
        else:
            break_even_price = entry_price_target - cost_per_unit
            
        # 7. 计算目标利润
        if take_profit > 0:
            if direction == "long":
                target_profit = (take_profit - entry_price_target) * contract_quantity - holding_cost_total
            else:
                target_profit = (entry_price_target - take_profit) * contract_quantity - holding_cost_total
        else:
            # 如果没有设置止盈，使用2:1风险收益比估算
            risk_distance = abs(entry_price_target - stop_loss) if stop_loss > 0 else entry_price_target * 0.02
            target_profit = risk_distance * contract_quantity * 2 - holding_cost_total
            
        target_profit = max(0, target_profit)  # 确保非负
        
        # 8. 计算期望收益率
        expected_return = target_profit / contract_value if contract_value > 0 else 0.0
        expected_return = max(-1.0, min(expected_return, 10.0))  # 限制在合理范围
        
        # 9. 计算获利概率和亏损概率
        # 基于技术分析信号强度、距离支撑阻力位等因素
        base_probability = 0.5  # 基础概率
        
        # 从技术分析获取信号强度调整概率
        signal_adjustment = 0.0
        overall_confidence = 0.5
        
        # 检查跨时间框架分析
        cross_analysis = technical_data.get("cross_timeframe_analysis", {})
        if cross_analysis:
            consensus_score = cross_analysis.get("timeframe_consensus", 0.5)
            signal_strength = cross_analysis.get("overall_signal_strength", "moderate")
            
            if signal_strength == "strong" and consensus_score > 0.7:
                signal_adjustment = 0.2
                overall_confidence = 0.75
            elif signal_strength == "weak" or consensus_score < 0.3:
                signal_adjustment = -0.2
                overall_confidence = 0.35
            else:
                overall_confidence = 0.5 + (consensus_score - 0.5) * 0.5
        
        # 基于价格位置调整概率
        price_adjustment = 0.0
        for timeframe in ["4h", "1h", "30m"]:
            if timeframe in technical_data:
                tf_data = technical_data[timeframe]
                if "price_levels" in tf_data:
                    price_levels = tf_data["price_levels"]
                    support_levels = price_levels.get("support_levels", [])
                    resistance_levels = price_levels.get("resistance_levels", [])
                    
                    # 检查是否接近关键价位
                    if direction == "long" and support_levels:
                        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))
                        if abs(nearest_support - current_price) / current_price < 0.02:
                            price_adjustment = 0.1  # 在支撑位附近做多增加成功概率
                    elif direction == "short" and resistance_levels:
                        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
                        if abs(nearest_resistance - current_price) / current_price < 0.02:
                            price_adjustment = 0.1  # 在阻力位附近做空增加成功概率
                    break
        
        # 基于止损止盈比例调整概率
        risk_reward_adjustment = 0.0
        if stop_loss > 0 and take_profit > 0:
            if direction == "long":
                risk = abs(entry_price_target - stop_loss)
                reward = abs(take_profit - entry_price_target)
            else:
                risk = abs(entry_price_target - stop_loss) 
                reward = abs(entry_price_target - take_profit)
                
            risk_reward_ratio = reward / risk if risk > 0 else 1.0
            if risk_reward_ratio > 2.0:
                risk_reward_adjustment = 0.1
            elif risk_reward_ratio < 1.0:
                risk_reward_adjustment = -0.1
        
        # 计算最终概率
        profit_probability = base_probability + signal_adjustment + price_adjustment + risk_reward_adjustment
        profit_probability = max(0.1, min(profit_probability, 0.9))  # 限制在10%-90%
        loss_probability = 1.0 - profit_probability
        
        # 10. 计算期望值
        # 考虑目标利润和最大亏损
        max_loss = abs(entry_price_target - stop_loss) * contract_quantity + holding_cost_total if stop_loss > 0 else contract_value * 0.5
        expected_value = profit_probability * target_profit - loss_probability * max_loss
        
        # 11. 计算年化收益率
        # 基于预期持仓时间
        holding_days = position_hold_time / 24
        if holding_days > 0 and contract_value > 0:
            roi_annualized = (expected_return * (365 / holding_days))
        else:
            roi_annualized = 0.0
            
        roi_annualized = max(-3.0, min(roi_annualized, 10.0))  # 限制在合理范围
        
        # 构建成本收益分析结果
        cost_benefit_analysis = {
            "estimated_trading_fee": round(float(estimated_trading_fee), 4),
            "funding_rate": round(float(funding_rate), 6),
            "funding_cost_8h": round(float(funding_cost_8h), 4),
            "funding_cost_daily": round(float(funding_cost_daily), 4),
            "holding_cost_total": round(float(holding_cost_total), 4),
            "break_even_price": round(float(break_even_price), 2),
            "target_profit": round(float(target_profit), 2),
            "expected_return": round(float(expected_return), 4),
            "profit_probability": round(float(profit_probability), 3),
            "loss_probability": round(float(loss_probability), 3),
            "expected_value": round(float(expected_value), 2),
            "roi_annualized": round(float(roi_annualized), 4)
        }
        
        return cost_benefit_analysis

    def _assess_technical_risk_with_error_handling(
        self,
        ticker: str,
        analyst_signals: Dict[str, Any],
        basic_params: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        带异常处理的技术风险评估包装器
        """
        try:
            return self.assess_technical_risk(ticker, analyst_signals, basic_params, portfolio)
        except Exception as e:
            logger.error(f"技术风险评估异常: {e}")
            return self._get_conservative_technical_risk_assessment(ticker, basic_params, [e])
    
    def _get_conservative_technical_risk_assessment(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        errors: List[Exception]
    ) -> Dict[str, Any]:
        """
        生成保守的技术风险评估，用于异常恢复
        
        Args:
            ticker: 交易对符号
            basic_params: 基础交易参数
            errors: 计算过程中遇到的异常列表
            
        Returns:
            保守的技术风险评估字典
        """
        logger.info(f"生成保守技术风险评估: {ticker}")
        
        # 使用最保守的风险评估
        technical_risk_assessment = {
            "volatility_risk": "high",  # 假设高波动风险
            "trend_strength": "weak",   # 假设弱趋势
            "mean_reversion_risk": "high",  # 假设高均值回归风险
            "statistical_edge": "weak",     # 假设弱统计优势
            "momentum_alignment": False,    # 假设动量不对齐
            "support_resistance_proximity": "at_level",  # 假设在关键位置
            "breakout_probability": 0.3,    # 低突破概率
            "false_breakout_risk": 0.7      # 高假突破风险
        }
        
        # 添加异常信息
        technical_risk_assessment["_is_conservative_fallback"] = True
        technical_risk_assessment["_errors"] = [
            {
                "error_type": type(e).__name__,
                "message": str(e)
            }
            for e in errors
        ]
        
        logger.warning(f"使用保守技术风险评估: {ticker}")
        return technical_risk_assessment

    def assess_technical_risk(
        self,
        ticker: str,
        analyst_signals: Dict[str, Any],
        basic_params: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估技术风险，根据futures_trading_output_specification.md规范
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            analyst_signals: 技术分析师信号数据，包含各时间框架的技术指标
            basic_params: 基础交易参数，包含方向、杠杆、仓位等信息
            portfolio: 投资组合数据
            
        Returns:
            包含technical_risk_assessment字段的字典，包含8个必需字段：
            - volatility_risk: "low|moderate|high|extreme" 
            - trend_strength: "weak|moderate|strong"
            - mean_reversion_risk: "low|moderate|high"
            - statistical_edge: "weak|moderate|strong"
            - momentum_alignment: boolean
            - support_resistance_proximity: "far|near|at_level"
            - breakout_probability: float
            - false_breakout_risk: float
        """
        # 获取技术分析数据
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        
        # 获取当前价格和交易方向
        current_price = basic_params.get("current_price", 0.0)
        direction = basic_params.get("direction", "long")
        
        # 初始化评估结果
        volatility_risk = "moderate"
        trend_strength = "moderate" 
        mean_reversion_risk = "moderate"
        statistical_edge = "moderate"
        momentum_alignment = False
        support_resistance_proximity = "far"
        breakout_probability = 0.5
        false_breakout_risk = 0.3
        
        # 收集各时间框架的数据进行综合分析
        timeframe_data_collection = {}
        dominant_timeframe = "1h"  # 默认主导时间框架
        
        # 按权重优先级收集时间框架数据
        timeframe_priorities = ["4h", "1h", "30m", "15m", "5m"]
        
        for tf in timeframe_priorities:
            if tf in technical_data:
                timeframe_data_collection[tf] = technical_data[tf]
                if not dominant_timeframe or len(timeframe_data_collection) == 1:
                    dominant_timeframe = tf
        
        # 如果没有数据，返回保守的默认评估
        if not timeframe_data_collection:
            return {
                "volatility_risk": "moderate",
                "trend_strength": "weak",
                "mean_reversion_risk": "moderate", 
                "statistical_edge": "weak",
                "momentum_alignment": False,
                "support_resistance_proximity": "far",
                "breakout_probability": 0.3,
                "false_breakout_risk": 0.5
            }
        
        # 1. 评估波动率风险
        volatility_metrics = []
        atr_values = []
        volatility_percentiles = []
        
        for tf, tf_data in timeframe_data_collection.items():
            # 收集波动率指标
            if "volatility_analysis" in tf_data:
                vol_analysis = tf_data["volatility_analysis"]
                vol_percentile = vol_analysis.get("volatility_percentile", 0.5)
                volatility_percentiles.append(vol_percentile)
                
            if "atr_values" in tf_data:
                atr_data = tf_data["atr_values"]
                atr_14 = atr_data.get("atr_14", 0.0)
                if atr_14 > 0 and current_price > 0:
                    atr_ratio = atr_14 / current_price
                    atr_values.append(atr_ratio)
                    
            # 收集策略信号中的波动率数据
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                if "volatility" in strategies:
                    vol_metrics = strategies["volatility"].get("metrics", {})
                    vol_z_score = vol_metrics.get("volatility_z_score", 0.0)
                    volatility_metrics.append(abs(vol_z_score))
        
        # 计算综合波动率风险
        if volatility_percentiles:
            avg_vol_percentile = sum(volatility_percentiles) / len(volatility_percentiles)
            if avg_vol_percentile > 0.8:
                volatility_risk = "extreme"
            elif avg_vol_percentile > 0.6:
                volatility_risk = "high"
            elif avg_vol_percentile > 0.3:
                volatility_risk = "moderate"
            else:
                volatility_risk = "low"
        elif atr_values:
            avg_atr_ratio = sum(atr_values) / len(atr_values)
            if avg_atr_ratio > 0.05:
                volatility_risk = "extreme"
            elif avg_atr_ratio > 0.03:
                volatility_risk = "high" 
            elif avg_atr_ratio > 0.015:
                volatility_risk = "moderate"
            else:
                volatility_risk = "low"
                
        # 2. 评估趋势强度
        trend_signals = []
        adx_values = []
        
        for tf, tf_data in timeframe_data_collection.items():
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                
                # 趋势跟随策略信号
                if "trend_following" in strategies:
                    trend_data = strategies["trend_following"]
                    confidence = trend_data.get("confidence", 0) / 100.0
                    signal = trend_data.get("signal", "neutral")
                    
                    if signal != "neutral":
                        trend_signals.append(confidence)
                        
                    # ADX值
                    metrics = trend_data.get("metrics", {})
                    adx = metrics.get("adx", 0.0)
                    if adx > 0:
                        adx_values.append(adx)
        
        # 计算趋势强度
        if adx_values:
            avg_adx = sum(adx_values) / len(adx_values)
            if avg_adx > 40:
                trend_strength = "strong"
            elif avg_adx > 25:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
        elif trend_signals:
            avg_trend_confidence = sum(trend_signals) / len(trend_signals)
            if avg_trend_confidence > 0.7:
                trend_strength = "strong"
            elif avg_trend_confidence > 0.5:
                trend_strength = "moderate"
            else:
                trend_strength = "weak"
                
        # 3. 评估均值回归风险
        mean_reversion_signals = []
        rsi_values = []
        z_scores = []
        
        for tf, tf_data in timeframe_data_collection.items():
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                
                if "mean_reversion" in strategies:
                    mr_data = strategies["mean_reversion"]
                    confidence = mr_data.get("confidence", 0) / 100.0
                    mean_reversion_signals.append(confidence)
                    
                    metrics = mr_data.get("metrics", {})
                    z_score = metrics.get("z_score", 0.0)
                    rsi_14 = metrics.get("rsi_14", 50.0)
                    
                    if abs(z_score) > 0:
                        z_scores.append(abs(z_score))
                    if rsi_14 > 0:
                        rsi_values.append(abs(rsi_14 - 50))  # 距离中性水平的偏差
        
        # 计算均值回归风险
        high_mr_risk_factors = 0
        
        if z_scores:
            avg_z_score = sum(z_scores) / len(z_scores)
            if avg_z_score > 2.0:
                high_mr_risk_factors += 1
                
        if rsi_values:
            avg_rsi_deviation = sum(rsi_values) / len(rsi_values)
            if avg_rsi_deviation > 30:  # RSI > 80 或 < 20
                high_mr_risk_factors += 1
                
        if high_mr_risk_factors >= 2:
            mean_reversion_risk = "high"
        elif high_mr_risk_factors >= 1:
            mean_reversion_risk = "moderate"
        else:
            mean_reversion_risk = "low"
            
        # 4. 评估统计优势
        statistical_signals = []
        hurst_exponents = []
        
        for tf, tf_data in timeframe_data_collection.items():
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                
                if "statistical_arbitrage" in strategies:
                    stat_data = strategies["statistical_arbitrage"]
                    confidence = stat_data.get("confidence", 0) / 100.0
                    statistical_signals.append(confidence)
                    
                    metrics = stat_data.get("metrics", {})
                    hurst = metrics.get("hurst_exponent", 0.5)
                    if hurst > 0:
                        hurst_exponents.append(hurst)
        
        # 计算统计优势
        if statistical_signals:
            avg_stat_confidence = sum(statistical_signals) / len(statistical_signals)
            if avg_stat_confidence > 0.7:
                statistical_edge = "strong"
            elif avg_stat_confidence > 0.4:
                statistical_edge = "moderate"
            else:
                statistical_edge = "weak"
        
        # 通过Hurst指数调整统计优势
        if hurst_exponents:
            avg_hurst = sum(hurst_exponents) / len(hurst_exponents)
            # Hurst > 0.5 表示趋势持续性，< 0.5 表示均值回归
            if abs(avg_hurst - 0.5) > 0.2:  # 明显偏离随机游走
                if statistical_edge == "weak":
                    statistical_edge = "moderate"
                elif statistical_edge == "moderate":
                    statistical_edge = "strong"
                    
        # 5. 评估动量对齐
        momentum_signals = []
        momentum_directions = []
        
        for tf, tf_data in timeframe_data_collection.items():
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                
                if "momentum" in strategies:
                    mom_data = strategies["momentum"]
                    signal = mom_data.get("signal", "neutral")
                    confidence = mom_data.get("confidence", 0) / 100.0
                    
                    if signal != "neutral" and confidence > 0.3:
                        momentum_signals.append(confidence)
                        momentum_directions.append(1 if signal == "bullish" else -1)
        
        # 检查动量对齐
        if momentum_directions:
            # 检查方向一致性
            direction_consistency = all(d > 0 for d in momentum_directions) or all(d < 0 for d in momentum_directions)
            avg_momentum_confidence = sum(momentum_signals) / len(momentum_signals) if momentum_signals else 0
            
            # 检查与交易方向的对齐
            expected_direction = 1 if direction == "long" else -1
            momentum_aligned_with_trade = all(d * expected_direction > 0 for d in momentum_directions)
            
            momentum_alignment = (direction_consistency and 
                                avg_momentum_confidence > 0.5 and 
                                momentum_aligned_with_trade)
        
        # 6. 评估支撑阻力位接近程度
        price_levels_data = {}
        for tf, tf_data in timeframe_data_collection.items():
            if "price_levels" in tf_data:
                price_levels_data = tf_data["price_levels"]
                break
                
        if price_levels_data and current_price > 0:
            support_levels = price_levels_data.get("support_levels", [])
            resistance_levels = price_levels_data.get("resistance_levels", [])
            
            # 计算距离最近支撑阻力位的距离
            min_distance = float('inf')
            
            for level in support_levels + resistance_levels:
                if level > 0:
                    distance_ratio = abs(current_price - level) / current_price
                    min_distance = min(min_distance, distance_ratio)
            
            if min_distance != float('inf'):
                if min_distance < 0.01:  # 1%以内
                    support_resistance_proximity = "at_level"
                elif min_distance < 0.03:  # 3%以内
                    support_resistance_proximity = "near"
                else:
                    support_resistance_proximity = "far"
        
        # 7. 评估突破概率
        breakout_factors = 0
        
        # 基于波动率评估突破概率
        if volatility_risk in ["high", "extreme"]:
            breakout_factors += 1
            
        # 基于趋势强度
        if trend_strength == "strong":
            breakout_factors += 1
            
        # 基于价格位置
        if support_resistance_proximity == "at_level":
            breakout_factors += 1
            
        # 基于动量对齐
        if momentum_alignment:
            breakout_factors += 1
            
        # 计算突破概率
        breakout_probability = min(0.9, 0.2 + breakout_factors * 0.15)
        
        # 8. 评估假突破风险
        false_breakout_factors = 0
        
        # 高均值回归风险增加假突破概率
        if mean_reversion_risk == "high":
            false_breakout_factors += 1
            
        # 弱统计优势增加假突破风险
        if statistical_edge == "weak":
            false_breakout_factors += 1
            
        # 波动率过高也增加假突破风险
        if volatility_risk == "extreme":
            false_breakout_factors += 1
            
        # 计算假突破风险
        false_breakout_risk = min(0.8, 0.2 + false_breakout_factors * 0.15)
        
        # 构建技术风险评估结果
        technical_risk_assessment = {
            "volatility_risk": volatility_risk,
            "trend_strength": trend_strength,
            "mean_reversion_risk": mean_reversion_risk,
            "statistical_edge": statistical_edge,
            "momentum_alignment": momentum_alignment,
            "support_resistance_proximity": support_resistance_proximity,
            "breakout_probability": round(float(breakout_probability), 3),
            "false_breakout_risk": round(float(false_breakout_risk), 3)
        }
        
        return technical_risk_assessment

    def _design_execution_strategy_with_error_handling(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        technical_risk: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        带异常处理的执行策略设计包装器
        """
        try:
            return self.design_execution_strategy(
                ticker, basic_params, risk_management, technical_risk, analyst_signals, portfolio
            )
        except Exception as e:
            logger.error(f"执行策略设计异常: {e}")
            return self._get_conservative_execution_strategy(ticker, basic_params, [e])
    
    def _get_conservative_execution_strategy(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        errors: List[Exception]
    ) -> Dict[str, Any]:
        """
        生成保守的执行策略，用于异常恢复
        
        Args:
            ticker: 交易对符号
            basic_params: 基础交易参数
            errors: 计算过程中遇到的异常列表
            
        Returns:
            保守的执行策略字典
        """
        logger.info(f"生成保守执行策略: {ticker}")
        
        # 使用最保守的执行策略
        execution_strategy = {
            "entry_timing": "immediate",    # 立即执行，避免复杂策略
            "order_splitting": False,       # 不分割订单
            "slippage_tolerance": 0.001,    # 低滑点容忍度
            "max_execution_time": 300,      # 5分钟最大执行时间
            "fill_probability": 0.8,        # 保守的成交概率
            "market_impact": "low",         # 假设低市场影响
            "liquidity_assessment": "sufficient",  # 假设充足流动性
            "execution_complexity": "simple"       # 简单执行策略
        }
        
        # 添加异常信息
        execution_strategy["_is_conservative_fallback"] = True
        execution_strategy["_errors"] = [
            {
                "error_type": type(e).__name__,
                "message": str(e)
            }
            for e in errors
        ]
        
        logger.warning(f"使用保守执行策略: {ticker}")
        return execution_strategy

    def design_execution_strategy(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        technical_risk_assessment: Dict[str, Any],
        market_environment: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        设计执行策略，根据futures_trading_output_specification.md规范
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            basic_params: 基础交易参数，包含方向、杠杆、仓位等信息
            risk_management: 风险管理参数，包含止损止盈等信息
            technical_risk_assessment: 技术风险评估结果
            market_environment: 市场环境评估结果
            analyst_signals: 技术分析师信号数据，包含各时间框架的技术分析
            portfolio: 投资组合数据
            
        Returns:
            包含execution_strategy字段的字典，包含8个必需字段：
            - entry_strategy: "immediate|gradual|wait_for_dip"
            - entry_timing: "now|wait_5m|wait_15m|wait_pullback"  
            - order_placement: "aggressive|passive|hidden"
            - position_building: "single_entry|scale_in|dca"
            - exit_strategy: "target_based|signal_based|time_based"
            - partial_profit_taking: boolean
            - scale_out_levels: [float, float, float]
            - emergency_exit_conditions: [string, string]
        """
        # 获取基础参数
        direction = basic_params.get("direction", "long")
        current_price = basic_params.get("current_price", 0.0)
        entry_price_target = basic_params.get("entry_price_target", current_price)
        leverage = basic_params.get("leverage", 1)
        position_size = basic_params.get("position_size", 0.0)
        
        # 获取风险管理参数
        stop_loss = risk_management.get("stop_loss", 0.0)
        take_profit = risk_management.get("take_profit", 0.0)
        risk_percentage = risk_management.get("risk_percentage", 0.0)
        position_hold_time = risk_management.get("position_hold_time", 24)
        
        # 获取技术风险评估
        volatility_risk = technical_risk_assessment.get("volatility_risk", "moderate")
        trend_strength = technical_risk_assessment.get("trend_strength", "moderate")
        support_resistance_proximity = technical_risk_assessment.get("support_resistance_proximity", "far")
        breakout_probability = technical_risk_assessment.get("breakout_probability", 0.5)
        momentum_alignment = technical_risk_assessment.get("momentum_alignment", False)
        
        # 获取市场环境
        trend_regime = market_environment.get("trend_regime", "ranging")
        volatility_regime = market_environment.get("volatility_regime", "normal")
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        market_structure = market_environment.get("market_structure", "neutral")
        sentiment_indicator = market_environment.get("sentiment_indicator", 0.0)
        
        # 获取技术分析数据
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        
        # 初始化执行策略参数
        entry_strategy = "gradual"
        entry_timing = "wait_15m"
        order_placement = "passive"
        position_building = "scale_in"
        exit_strategy = "target_based"
        partial_profit_taking = True
        scale_out_levels = []
        emergency_exit_conditions = []
        
        # 1. 确定入场策略 (entry_strategy)
        # 基于市场环境、波动率风险和信号强度
        strong_signal_factors = 0
        
        # 检查强信号因子
        if trend_strength == "strong":
            strong_signal_factors += 1
        if momentum_alignment:
            strong_signal_factors += 1
        if breakout_probability > 0.7:
            strong_signal_factors += 1
        if abs(sentiment_indicator) > 0.6:
            strong_signal_factors += 1
        if trend_regime == "trending":
            strong_signal_factors += 1
            
        # 检查低波动率和良好流动性
        stable_conditions = (volatility_risk in ["low", "moderate"] and 
                           liquidity_assessment in ["good", "excellent"])
        
        if strong_signal_factors >= 3 and stable_conditions:
            entry_strategy = "immediate"  # 强信号且市场条件稳定
        elif volatility_risk == "extreme" or liquidity_assessment == "poor":
            entry_strategy = "wait_for_dip"  # 等待更好的入场机会
        elif strong_signal_factors >= 2:
            entry_strategy = "gradual"  # 逐步建仓
        else:
            entry_strategy = "wait_for_dip"  # 信号不够强，等待机会
            
        # 2. 确定入场时机 (entry_timing)
        if entry_strategy == "immediate":
            if support_resistance_proximity == "at_level":
                entry_timing = "now"  # 在关键价位立即入场
            else:
                entry_timing = "wait_5m"  # 等待短暂确认
        elif entry_strategy == "gradual":
            if volatility_regime in ["elevated", "extreme"]:
                entry_timing = "wait_15m"  # 高波动环境等待稳定
            elif trend_regime == "trending" and momentum_alignment:
                entry_timing = "wait_5m"  # 趋势环境快速跟进
            else:
                entry_timing = "wait_15m"  # 标准等待时间
        else:  # wait_for_dip
            if support_resistance_proximity == "near":
                entry_timing = "wait_pullback"  # 等待回调到支撑位
            else:
                entry_timing = "wait_pullback"  # 等待更好的入场价格
                
        # 3. 确定订单放置方式 (order_placement)
        if liquidity_assessment == "excellent" and volatility_risk == "low":
            if entry_strategy == "immediate":
                order_placement = "aggressive"  # 市价单快速成交
            else:
                order_placement = "passive"  # 限价单降低成本
        elif liquidity_assessment in ["poor", "fair"] or volatility_risk == "extreme":
            order_placement = "hidden"  # 隐藏订单避免冲击
        elif position_size > current_price * 10000:  # 大额订单
            order_placement = "hidden"  # 大额订单使用隐藏方式
        else:
            order_placement = "passive"  # 默认被动下单
            
        # 4. 确定仓位建立方式 (position_building)
        if entry_strategy == "immediate" and risk_percentage <= 2.0:
            position_building = "single_entry"  # 低风险一次性建仓
        elif volatility_risk in ["high", "extreme"] or risk_percentage > 3.0:
            position_building = "dca"  # 高风险使用定投策略
        elif trend_regime == "ranging" or market_structure == "neutral":
            position_building = "scale_in"  # 震荡市场分批建仓
        else:
            position_building = "scale_in"  # 默认分批建仓
            
        # 5. 确定退出策略 (exit_strategy)
        if position_hold_time <= 4:  # 短期持仓
            exit_strategy = "target_based"  # 基于目标价退出
        elif trend_strength == "strong" and momentum_alignment:
            exit_strategy = "signal_based"  # 基于信号变化退出
        elif position_hold_time >= 48:  # 长期持仓
            exit_strategy = "time_based"  # 基于时间退出
        else:
            exit_strategy = "target_based"  # 默认目标价退出
            
        # 6. 确定分批止盈 (partial_profit_taking)
        if leverage >= 10 or risk_percentage > 3.0:
            partial_profit_taking = True  # 高杠杆必须分批止盈
        elif position_hold_time <= 2:  # 超短期持仓
            partial_profit_taking = False  # 短期交易一次性平仓
        elif trend_strength == "weak":
            partial_profit_taking = True  # 趋势较弱需分批退出
        else:
            partial_profit_taking = True  # 默认分批止盈
            
        # 7. 计算分批平仓价位 (scale_out_levels)
        if partial_profit_taking and take_profit > 0:
            if direction == "long":
                # 计算三个阶段的止盈价位
                profit_range = take_profit - entry_price_target
                level_1 = entry_price_target + profit_range * 0.33  # 33%利润
                level_2 = entry_price_target + profit_range * 0.66  # 66%利润
                level_3 = take_profit  # 100%利润
            else:  # short
                profit_range = entry_price_target - take_profit
                level_1 = entry_price_target - profit_range * 0.33
                level_2 = entry_price_target - profit_range * 0.66
                level_3 = take_profit
                
            scale_out_levels = [
                round(float(level_1), 2),
                round(float(level_2), 2), 
                round(float(level_3), 2)
            ]
        else:
            # 不分批止盈时返回目标价作为单一平仓价位
            if take_profit > 0:
                scale_out_levels = [round(float(take_profit), 2), 0.0, 0.0]
            else:
                scale_out_levels = [0.0, 0.0, 0.0]
                
        # 8. 设定紧急退出条件 (emergency_exit_conditions)
        emergency_conditions = []
        
        # 基于风险水平设定紧急条件
        if leverage >= 20:
            emergency_conditions.append(f"保证金率低于20%时强制平仓")
        elif leverage >= 10:
            emergency_conditions.append(f"保证金率低于30%时强制平仓")
        else:
            emergency_conditions.append(f"保证金率低于50%时强制平仓")
            
        # 基于波动率设定条件
        if volatility_risk == "extreme":
            emergency_conditions.append("波动率超过历史90%分位数时减仓50%")
        elif volatility_risk == "high":
            emergency_conditions.append("波动率超过历史85%分位数时减仓30%")
        else:
            emergency_conditions.append("波动率超过历史95%分位数时减仓50%")
            
        # 基于技术信号设定条件
        if trend_strength == "strong" and momentum_alignment:
            emergency_conditions.append("主要技术指标同时反转时立即平仓")
        else:
            emergency_conditions.append("止损位被突破且成交量放大时立即平仓")
            
        # 基于市场环境设定条件
        if market_structure != "neutral":
            emergency_conditions.append("市场结构发生反转且超过2个时间框架确认时平仓")
        
        # 限制紧急条件数量为2个最重要的
        emergency_exit_conditions = emergency_conditions[:2]
        
        # 根据跨时间框架分析进行最终调整
        cross_analysis = technical_data.get("cross_timeframe_analysis", {})
        if cross_analysis:
            timeframe_consensus = cross_analysis.get("timeframe_consensus", 0.5)
            overall_signal_strength = cross_analysis.get("overall_signal_strength", "moderate")
            dominant_timeframe = cross_analysis.get("dominant_timeframe", "1h")
            
            # 基于时间框架一致性调整策略
            if timeframe_consensus > 0.8 and overall_signal_strength == "strong":
                # 高度一致的强信号，可以更激进
                if entry_strategy == "gradual":
                    entry_strategy = "immediate"
                if entry_timing == "wait_15m":
                    entry_timing = "wait_5m"
                if order_placement == "passive":
                    order_placement = "aggressive"
                    
            elif timeframe_consensus < 0.3 or overall_signal_strength == "weak":
                # 信号分歧或偏弱，更加保守
                if entry_strategy == "immediate":
                    entry_strategy = "gradual"
                if entry_timing in ["now", "wait_5m"]:
                    entry_timing = "wait_pullback"
                if order_placement == "aggressive":
                    order_placement = "passive"
                partial_profit_taking = True  # 强制分批止盈
                
            # 基于主导时间框架调整持仓策略
            if dominant_timeframe in ["5m", "15m"]:
                # 短周期主导，适合短线交易
                if position_building == "dca":
                    position_building = "scale_in"
                exit_strategy = "signal_based"  # 快速响应信号变化
            elif dominant_timeframe == "4h":
                # 长周期主导，适合中长线
                if position_building == "single_entry" and volatility_risk != "low":
                    position_building = "scale_in"
                if exit_strategy == "signal_based":
                    exit_strategy = "target_based"  # 更有耐心等待目标价
        
        # 构建执行策略结果
        execution_strategy = {
            "entry_strategy": entry_strategy,
            "entry_timing": entry_timing,
            "order_placement": order_placement,
            "position_building": position_building,
            "exit_strategy": exit_strategy,
            "partial_profit_taking": partial_profit_taking,
            "scale_out_levels": scale_out_levels,
            "emergency_exit_conditions": emergency_exit_conditions
        }
        
        return execution_strategy

    def evaluate_market_environment(
        self,
        ticker: str,
        analyst_signals: Dict[str, Any],
        basic_params: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        评估市场环境，根据futures_trading_output_specification.md规范
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            analyst_signals: 技术分析师信号数据，包含各时间框架的技术分析和策略信号
            basic_params: 基础交易参数，包含方向、杠杆、当前价格等信息
            portfolio: 投资组合数据
            
        Returns:
            包含market_environment字段的字典，包含9个必需字段：
            - trend_regime: "trending|ranging|transitional"
            - volatility_regime: "low|normal|elevated|extreme"  
            - liquidity_assessment: "poor|fair|good|excellent"
            - market_structure: "bullish|bearish|neutral"
            - market_phase: "accumulation|markup|distribution|decline"
            - sentiment_indicator: float (情绪指标)
            - fear_greed_index: int (恐慌贪婪指数)
            - funding_rate_trend: "increasing|decreasing|stable"
            - open_interest_trend: "increasing|decreasing|stable"
        """
        # 获取技术分析数据
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        
        # 获取当前价格和交易方向
        current_price = basic_params.get("current_price", 0.0)
        direction = basic_params.get("direction", "long")
        
        # 初始化评估结果
        trend_regime = "ranging"
        volatility_regime = "normal"
        liquidity_assessment = "fair"
        market_structure = "neutral"
        market_phase = "accumulation"
        sentiment_indicator = 0.0
        fear_greed_index = 50
        funding_rate_trend = "stable"
        open_interest_trend = "stable"
        
        # 收集多时间框架数据进行综合分析
        timeframe_data = {}
        trend_signals = []
        volatility_metrics = []
        momentum_signals = []
        volume_data = []
        
        # 优先使用较长时间框架的数据（权重更高）
        timeframe_priorities = ["4h", "1h", "30m", "15m", "5m"]
        
        for tf in timeframe_priorities:
            if tf in technical_data:
                timeframe_data[tf] = technical_data[tf]
        
        # 如果没有数据，返回中性的默认评估
        if not timeframe_data:
            return {
                "trend_regime": "ranging",
                "volatility_regime": "normal",
                "liquidity_assessment": "fair",
                "market_structure": "neutral",
                "market_phase": "accumulation",
                "sentiment_indicator": 0.0,
                "fear_greed_index": 50,
                "funding_rate_trend": "stable",
                "open_interest_trend": "stable"
            }
        
        # 1. 评估趋势状态 (trend_regime)
        adx_values = []
        trend_strength_values = []
        trend_signals_count = {"bullish": 0, "bearish": 0, "neutral": 0}
        
        for tf, tf_data in timeframe_data.items():
            # 收集趋势跟踪策略信号
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                if "trend_following" in strategies:
                    trend_data = strategies["trend_following"]
                    signal = trend_data.get("signal", "neutral")
                    confidence = trend_data.get("confidence", 0) / 100.0
                    
                    trend_signals_count[signal] += 1
                    trend_signals.append({"signal": signal, "confidence": confidence, "timeframe": tf})
                    
                    # 收集ADX值
                    metrics = trend_data.get("metrics", {})
                    adx = metrics.get("adx", 0.0)
                    trend_strength = metrics.get("trend_strength", 0.0)
                    
                    if adx > 0:
                        adx_values.append(adx)
                    if trend_strength > 0:
                        trend_strength_values.append(trend_strength)
        
        # 判断趋势状态
        total_signals = sum(trend_signals_count.values())
        if total_signals > 0:
            # 计算趋势一致性
            max_direction = max(trend_signals_count.values())
            trend_consistency = max_direction / total_signals
            
            # 计算平均ADX（趋势强度指标）
            avg_adx = sum(adx_values) / len(adx_values) if adx_values else 20
            avg_trend_strength = sum(trend_strength_values) / len(trend_strength_values) if trend_strength_values else 0.3
            
            if trend_consistency > 0.7 and avg_adx > 30:
                trend_regime = "trending"
            elif trend_consistency < 0.4 or avg_adx < 20:
                trend_regime = "ranging"
            else:
                trend_regime = "transitional"
        
        # 2. 评估波动率状态 (volatility_regime)
        volatility_percentiles = []
        atr_ratios = []
        vol_z_scores = []
        
        for tf, tf_data in timeframe_data.items():
            # 收集波动率分析数据
            if "volatility_analysis" in tf_data:
                vol_analysis = tf_data["volatility_analysis"]
                vol_percentile = vol_analysis.get("volatility_percentile", 0.5)
                volatility_percentiles.append(vol_percentile)
            
            # 收集ATR数据
            if "atr_values" in tf_data:
                atr_data = tf_data["atr_values"]
                atr_14 = atr_data.get("atr_14", 0.0)
                if atr_14 > 0 and current_price > 0:
                    atr_ratio = atr_14 / current_price
                    atr_ratios.append(atr_ratio)
            
            # 收集波动率策略信号
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                if "volatility" in strategies:
                    vol_data = strategies["volatility"]
                    metrics = vol_data.get("metrics", {})
                    vol_z_score = metrics.get("volatility_z_score", 0.0)
                    if vol_z_score != 0:
                        vol_z_scores.append(abs(vol_z_score))
        
        # 判断波动率状态
        if volatility_percentiles:
            avg_vol_percentile = sum(volatility_percentiles) / len(volatility_percentiles)
        elif atr_ratios:
            avg_atr_ratio = sum(atr_ratios) / len(atr_ratios)
            # 将ATR比例转换为百分位估计
            if avg_atr_ratio > 0.05:
                avg_vol_percentile = 0.9
            elif avg_atr_ratio > 0.03:
                avg_vol_percentile = 0.7
            elif avg_atr_ratio > 0.015:
                avg_vol_percentile = 0.5
            else:
                avg_vol_percentile = 0.3
        else:
            avg_vol_percentile = 0.5
        
        if avg_vol_percentile > 0.85:
            volatility_regime = "extreme"
        elif avg_vol_percentile > 0.7:
            volatility_regime = "elevated"
        elif avg_vol_percentile > 0.3:
            volatility_regime = "normal"
        else:
            volatility_regime = "low"
        
        # 3. 评估流动性 (liquidity_assessment)
        # 基于交易量和价差分析（简化实现）
        volume_signals = []
        volume_momentum_values = []
        
        for tf, tf_data in timeframe_data.items():
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                if "momentum" in strategies:
                    momentum_data = strategies["momentum"]
                    metrics = momentum_data.get("metrics", {})
                    volume_momentum = metrics.get("volume_momentum", 0.0)
                    if volume_momentum != 0:
                        volume_momentum_values.append(abs(volume_momentum))
        
        # 基于成交量动量评估流动性
        if volume_momentum_values:
            avg_volume_momentum = sum(volume_momentum_values) / len(volume_momentum_values)
            if avg_volume_momentum > 0.8:
                liquidity_assessment = "excellent"
            elif avg_volume_momentum > 0.5:
                liquidity_assessment = "good"
            elif avg_volume_momentum > 0.2:
                liquidity_assessment = "fair"
            else:
                liquidity_assessment = "poor"
        else:
            # 默认基于波动率反向评估（高波动通常意味着较好的流动性）
            if volatility_regime == "extreme":
                liquidity_assessment = "good"
            elif volatility_regime == "elevated":
                liquidity_assessment = "fair"
            else:
                liquidity_assessment = "fair"
        
        # 4. 评估市场结构 (market_structure)
        bullish_signals = trend_signals_count.get("bullish", 0)
        bearish_signals = trend_signals_count.get("bearish", 0)
        
        # 综合价格动量和趋势信号
        price_momentum_sum = 0.0
        momentum_count = 0
        
        for tf, tf_data in timeframe_data.items():
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                if "momentum" in strategies:
                    momentum_data = strategies["momentum"]
                    signal = momentum_data.get("signal", "neutral")
                    confidence = momentum_data.get("confidence", 0) / 100.0
                    
                    if signal == "bullish":
                        price_momentum_sum += confidence
                    elif signal == "bearish":
                        price_momentum_sum -= confidence
                    momentum_count += 1
        
        # 综合判断市场结构
        if bullish_signals > bearish_signals * 1.5 or price_momentum_sum > 0.3:
            market_structure = "bullish"
        elif bearish_signals > bullish_signals * 1.5 or price_momentum_sum < -0.3:
            market_structure = "bearish"
        else:
            market_structure = "neutral"
        
        # 5. 评估市场阶段 (market_phase)
        # 基于趋势状态、动量和波动率的综合判断
        rsi_values = []
        z_scores = []
        
        for tf, tf_data in timeframe_data.items():
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                # 收集均值回归指标
                if "mean_reversion" in strategies:
                    mr_data = strategies["mean_reversion"]
                    metrics = mr_data.get("metrics", {})
                    rsi_14 = metrics.get("rsi_14", 50.0)
                    z_score = metrics.get("z_score", 0.0)
                    
                    if rsi_14 > 0:
                        rsi_values.append(rsi_14)
                    if z_score != 0:
                        z_scores.append(z_score)
        
        # 计算市场阶段
        avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 50.0
        avg_z_score = sum(z_scores) / len(z_scores) if z_scores else 0.0
        
        if trend_regime == "trending":
            if market_structure == "bullish" and avg_rsi < 70:
                market_phase = "markup"  # 上升趋势
            elif market_structure == "bearish" and avg_rsi > 30:
                market_phase = "decline"  # 下降趋势
            else:
                market_phase = "distribution"  # 趋势后期
        else:  # ranging or transitional
            if avg_rsi < 40 or avg_z_score < -1.5:
                market_phase = "accumulation"  # 底部区域
            elif avg_rsi > 60 or avg_z_score > 1.5:
                market_phase = "distribution"  # 顶部区域
            else:
                market_phase = "accumulation"  # 默认积累阶段
        
        # 6. 计算情绪指标 (sentiment_indicator)
        # 综合各策略信号的置信度和方向
        sentiment_scores = []
        
        for tf, tf_data in timeframe_data.items():
            tf_weight = {"4h": 4.0, "1h": 3.0, "30m": 2.0, "15m": 1.5, "5m": 1.0}.get(tf, 1.0)
            
            if "strategy_signals" in tf_data:
                strategies = tf_data["strategy_signals"]
                for strategy_name, strategy_data in strategies.items():
                    signal = strategy_data.get("signal", "neutral")
                    confidence = strategy_data.get("confidence", 0) / 100.0
                    
                    if signal == "bullish":
                        sentiment_scores.append(confidence * tf_weight)
                    elif signal == "bearish":
                        sentiment_scores.append(-confidence * tf_weight)
        
        # 标准化情绪指标到-1到1的范围
        if sentiment_scores:
            raw_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            sentiment_indicator = max(-1.0, min(1.0, raw_sentiment))
        else:
            sentiment_indicator = 0.0
        
        # 7. 计算恐慌贪婪指数 (fear_greed_index)
        # 基于多个技术指标的综合评估
        fear_greed_factors = []
        
        # RSI贡献（超买超卖状态）
        if rsi_values:
            avg_rsi = sum(rsi_values) / len(rsi_values)
            rsi_contribution = (avg_rsi - 50) / 50 * 50  # 转换为-50到50的范围
            fear_greed_factors.append(rsi_contribution)
        
        # 波动率贡献（高波动率表示恐慌）
        vol_contribution = -(avg_vol_percentile - 0.5) * 40  # 高波动率降低指数
        fear_greed_factors.append(vol_contribution)
        
        # 趋势强度贡献
        if adx_values:
            avg_adx = sum(adx_values) / len(adx_values)
            trend_contribution = (avg_adx - 25) / 25 * 20  # ADX>25表示趋势强劲
            fear_greed_factors.append(trend_contribution)
        
        # 动量贡献
        momentum_contribution = sentiment_indicator * 30
        fear_greed_factors.append(momentum_contribution)
        
        # 计算最终恐慌贪婪指数
        if fear_greed_factors:
            fg_score = sum(fear_greed_factors) / len(fear_greed_factors)
            fear_greed_index = int(max(0, min(100, 50 + fg_score)))
        else:
            fear_greed_index = 50
        
        # 8. 评估资金费率趋势 (funding_rate_trend)
        # 由于缺少历史资金费率数据，基于市场情绪和结构进行估算
        if market_structure == "bullish" and sentiment_indicator > 0.3:
            funding_rate_trend = "increasing"  # 多头情绪推高资金费率
        elif market_structure == "bearish" and sentiment_indicator < -0.3:
            funding_rate_trend = "decreasing"  # 空头情绪降低资金费率
        elif abs(sentiment_indicator) < 0.2:
            funding_rate_trend = "stable"
        else:
            # 基于波动率判断
            if volatility_regime in ["elevated", "extreme"]:
                funding_rate_trend = "increasing"  # 高波动通常伴随高资金费率
            else:
                funding_rate_trend = "stable"
        
        # 9. 评估持仓量趋势 (open_interest_trend)
        # 基于趋势强度和市场阶段进行推断
        if trend_regime == "trending" and volatility_regime in ["normal", "elevated"]:
            if market_phase in ["markup", "decline"]:
                open_interest_trend = "increasing"  # 趋势阶段通常伴随持仓量增加
            else:
                open_interest_trend = "stable"
        elif trend_regime == "ranging":
            if volatility_regime == "low":
                open_interest_trend = "decreasing"  # 横盘低波动通常伴随持仓量减少
            else:
                open_interest_trend = "stable"
        else:  # transitional
            if market_phase == "accumulation":
                open_interest_trend = "increasing"  # 积累阶段通常增仓
            else:
                open_interest_trend = "stable"
        
        # 基于跨时间框架分析进行最终调整
        cross_analysis = technical_data.get("cross_timeframe_analysis", {})
        if cross_analysis:
            # 如果存在跨时间框架分析数据，用其调整评估结果
            timeframe_consensus = cross_analysis.get("timeframe_consensus", 0.5)
            trend_alignment = cross_analysis.get("trend_alignment", "mixed")
            overall_signal_strength = cross_analysis.get("overall_signal_strength", "moderate")
            
            # 基于时间框架一致性调整情绪指标
            if timeframe_consensus > 0.8 and trend_alignment == "aligned":
                sentiment_indicator *= 1.2  # 增强信号
                sentiment_indicator = max(-1.0, min(1.0, sentiment_indicator))
            elif timeframe_consensus < 0.3 or trend_alignment == "divergent":
                sentiment_indicator *= 0.7  # 削弱信号
                
            # 基于信号强度调整趋势判断
            if overall_signal_strength == "strong" and trend_regime == "transitional":
                if sentiment_indicator > 0.3:
                    trend_regime = "trending"
            elif overall_signal_strength == "weak" and trend_regime == "trending":
                trend_regime = "transitional"
        
        # 构建市场环境评估结果
        market_environment = {
            "trend_regime": trend_regime,
            "volatility_regime": volatility_regime,
            "liquidity_assessment": liquidity_assessment,
            "market_structure": market_structure,
            "market_phase": market_phase,
            "sentiment_indicator": round(float(sentiment_indicator), 3),
            "fear_greed_index": int(fear_greed_index),
            "funding_rate_trend": funding_rate_trend,
            "open_interest_trend": open_interest_trend
        }
        
        return market_environment

    def generate_scenario_analysis(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        technical_risk_assessment: Dict[str, Any],
        market_environment: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成情景分析，根据futures_trading_output_specification.md规范
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            basic_params: 基础交易参数，包含方向、杠杆、仓位等
            risk_management: 风险管理参数，包含止损、止盈等
            technical_risk_assessment: 技术风险评估结果
            market_environment: 市场环境评估结果
            analyst_signals: 分析师信号数据
            portfolio: 投资组合数据
            
        Returns:
            包含scenario_analysis字段的字典，包含4个情景：
            - best_case: 最佳情况
            - base_case: 基础情况  
            - worst_case: 最坏情况
            - black_swan: 黑天鹅事件
        """
        # 获取基础数据
        current_price = basic_params.get("current_price", 0.0)
        direction = basic_params.get("direction", "long")
        position_size = basic_params.get("position_size", 0.0)
        leverage = basic_params.get("leverage", 1)
        
        # 获取技术分析数据
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        
        # 获取ATR值用于价格目标计算
        atr_values = {}
        for timeframe in ["5m", "15m", "30m", "1h", "4h"]:
            if timeframe in technical_data:
                tf_data = technical_data[timeframe]
                if "atr_values" in tf_data:
                    atr_values[timeframe] = tf_data["atr_values"].get("atr_14", current_price * 0.02)
        
        # 使用主要时间框架的ATR，默认为1h
        primary_atr = atr_values.get("1h", current_price * 0.02)
        
        # 获取价格水平
        price_levels = {}
        for timeframe in ["1h", "4h"]:
            if timeframe in technical_data and "price_levels" in technical_data[timeframe]:
                price_levels = technical_data[timeframe]["price_levels"]
                break
        
        support_levels = price_levels.get("support_levels", [current_price * 0.95])
        resistance_levels = price_levels.get("resistance_levels", [current_price * 1.05])
        
        # 获取波动率分析
        volatility_analysis = {}
        for timeframe in ["1h", "4h"]:
            if timeframe in technical_data and "volatility_analysis" in technical_data[timeframe]:
                volatility_analysis = technical_data[timeframe]["volatility_analysis"]
                break
                
        volatility_percentile = volatility_analysis.get("volatility_percentile", 50.0)
        volatility_forecast = volatility_analysis.get("volatility_forecast", 0.02)
        
        # 获取市场环境信息
        trend_regime = market_environment.get("trend_regime", "ranging")
        volatility_regime = market_environment.get("volatility_regime", "normal")
        market_structure = market_environment.get("market_structure", "neutral")
        
        # 计算基础概率权重
        trend_weight = 1.2 if trend_regime == "trending" else 0.8
        volatility_weight = 1.3 if volatility_regime in ["elevated", "extreme"] else 1.0
        structure_weight = 1.1 if market_structure != "neutral" else 1.0
        
        # 1. 最佳情况分析
        if direction == "long":
            # 多头最佳情况：价格突破阻力位
            best_target = max(resistance_levels) if resistance_levels else current_price * 1.08
            best_target += primary_atr * 2  # 额外突破空间
        else:
            # 空头最佳情况：价格跌破支撑位
            best_target = min(support_levels) if support_levels else current_price * 0.92
            best_target -= primary_atr * 2  # 额外下跌空间
        
        # 计算最佳情况收益
        price_change_pct = abs(best_target - current_price) / current_price
        best_profit = position_size * price_change_pct * leverage
        
        # 最佳情况概率：基于趋势强度和技术指标一致性
        technical_strength = technical_risk_assessment.get("trend_strength", "moderate")
        momentum_alignment = technical_risk_assessment.get("momentum_alignment", False)
        
        best_probability = 0.25  # 基础概率
        if technical_strength == "strong":
            best_probability += 0.10
        if momentum_alignment:
            best_probability += 0.05
        if trend_regime == "trending":
            best_probability += 0.10
            
        best_probability = min(best_probability * trend_weight * structure_weight, 0.60)
        
        best_case = {
            "price_target": round(best_target, 2),
            "profit_potential": round(best_profit, 2),
            "probability": round(best_probability, 3),
            "timeframe": 24 if trend_regime == "trending" else 48  # 小时
        }
        
        # 2. 基础情况分析
        if direction == "long":
            base_target = current_price + primary_atr * 1.5
        else:
            base_target = current_price - primary_atr * 1.5
            
        price_change_pct = abs(base_target - current_price) / current_price
        base_profit = position_size * price_change_pct * leverage
        
        # 基础情况概率：较高的概率
        base_probability = 0.45
        if technical_strength in ["moderate", "strong"]:
            base_probability += 0.10
        if volatility_regime in ["normal", "low"]:
            base_probability += 0.05
            
        base_probability = min(base_probability * trend_weight, 0.75)
        
        base_case = {
            "price_target": round(base_target, 2),
            "profit_potential": round(base_profit, 2),
            "probability": round(base_probability, 3),
            "timeframe": 12 if trend_regime == "trending" else 24  # 小时
        }
        
        # 3. 最坏情况分析
        stop_loss = risk_management.get("stop_loss", current_price)
        if direction == "long":
            worst_target = min(stop_loss, current_price - primary_atr * 2)
        else:
            worst_target = max(stop_loss, current_price + primary_atr * 2)
            
        price_change_pct = abs(worst_target - current_price) / current_price
        worst_loss = position_size * price_change_pct * leverage
        
        # 最坏情况概率
        worst_probability = 0.20
        if technical_risk_assessment.get("false_breakout_risk", 0.0) > 0.3:
            worst_probability += 0.10
        if volatility_regime == "extreme":
            worst_probability += 0.05
        if market_structure == "neutral" and trend_regime == "ranging":
            worst_probability += 0.05
            
        worst_probability = min(worst_probability * volatility_weight, 0.45)
        
        worst_case = {
            "price_target": round(worst_target, 2),
            "loss_potential": round(worst_loss, 2),
            "probability": round(worst_probability, 3),
            "timeframe": 6 if volatility_regime == "extreme" else 12  # 小时
        }
        
        # 4. 黑天鹅事件分析
        liquidation_price = risk_management.get("liquidation_price", 0.0)
        max_loss = risk_management.get("max_loss_amount", position_size)
        
        # 黑天鹅触发条件
        trigger_conditions = []
        if volatility_regime == "extreme":
            trigger_conditions.append("市场波动率达到极端水平")
        if trend_regime == "transitional":
            trigger_conditions.append("市场趋势发生急剧转换")
            
        # 默认触发条件
        if not trigger_conditions:
            trigger_conditions = [
                f"价格突然反向移动超过{int(primary_atr/current_price*100*3)}%",
                "市场流动性急剧下降或重大消息事件"
            ]
        
        # 黑天鹅概率
        black_swan_probability = 0.05  # 基础5%概率
        if volatility_regime == "extreme":
            black_swan_probability = 0.10
        elif volatility_regime == "elevated":
            black_swan_probability = 0.07
            
        # 缓解策略
        liquidation_distance = abs(liquidation_price - current_price) / current_price if liquidation_price > 0 else 0.20
        if liquidation_distance < 0.10:
            mitigation_strategy = "立即降低杠杆或减仓，增加止损保护"
        elif liquidation_distance < 0.20:
            mitigation_strategy = "密切监控价格走势，准备快速止损退出"
        else:
            mitigation_strategy = "保持跟踪止损，设置价格预警机制"
            
        black_swan = {
            "trigger_conditions": trigger_conditions,
            "max_loss": round(max_loss, 2),
            "probability": round(black_swan_probability, 3),
            "mitigation_strategy": mitigation_strategy
        }
        
        # 验证概率总和合理性（允许一定误差）
        total_probability = best_probability + base_probability + worst_probability
        if total_probability > 1.05:  # 允许5%误差
            # 按比例调整概率
            adjustment_factor = 1.0 / total_probability
            best_case["probability"] = round(best_probability * adjustment_factor, 3)
            base_case["probability"] = round(base_probability * adjustment_factor, 3)
            worst_case["probability"] = round(worst_probability * adjustment_factor, 3)
        
        scenario_analysis = {
            "best_case": best_case,
            "base_case": base_case,
            "worst_case": worst_case,
            "black_swan": black_swan
        }
        
        return scenario_analysis

    def create_decision_metadata(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        technical_risk_assessment: Dict[str, Any],
        market_environment: Dict[str, Any],
        cost_benefit_analysis: Dict[str, Any],
        execution_strategy: Dict[str, Any],
        scenario_analysis: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建决策元数据，记录LLM实际推理过程的完整元数据
        根据futures_trading_output_specification.md规范
        
        Args:
            ticker: 交易对符号，如"BTCUSDT"
            basic_params: 基础交易参数，包含方向、杠杆、仓位等
            risk_management: 风险管理参数，包含止损、止盈等
            technical_risk_assessment: 技术风险评估结果
            market_environment: 市场环境评估结果
            cost_benefit_analysis: 成本效益分析结果
            execution_strategy: 执行策略建议
            scenario_analysis: 情景分析结果
            analyst_signals: 分析师信号数据
            portfolio: 投资组合数据
            
        Returns:
            包含decision_metadata字段的字典，符合规范要求
        """
        # 获取基础数据
        direction = basic_params.get("direction", "long")
        leverage = basic_params.get("leverage", 1)
        position_size = basic_params.get("position_size", 0.0)
        current_price = basic_params.get("current_price", 0.0)
        
        # 获取技术分析和风险管理数据
        technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
        risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
        
        # === 1. 计算置信度分解 ===
        # 技术分析置信度：基于多时间框架信号强度和一致性
        tech_confidence = self._calculate_technical_confidence(technical_data, technical_risk_assessment)
        
        # 风险评估置信度：基于风险控制措施的完整性和合理性
        risk_confidence = self._calculate_risk_confidence(risk_management, risk_data)
        
        # 市场条件置信度：基于市场环境的清晰度和稳定性
        market_confidence = self._calculate_market_confidence(market_environment)
        
        # 成本效益置信度：基于预期收益和成本的合理性
        cost_benefit_confidence = self._calculate_cost_benefit_confidence(cost_benefit_analysis)
        
        # 执行可行性置信度：基于流动性和执行复杂度
        execution_confidence = self._calculate_execution_confidence(execution_strategy, market_environment)
        
        # 总体置信度：加权平均
        total_confidence = int(
            tech_confidence * 0.30 +
            risk_confidence * 0.25 +
            market_confidence * 0.20 +
            cost_benefit_confidence * 0.15 +
            execution_confidence * 0.10
        )
        
        confidence_breakdown = {
            "technical_analysis": round(tech_confidence, 2),
            "risk_assessment": round(risk_confidence, 2),
            "market_conditions": round(market_confidence, 2),
            "cost_benefit": round(cost_benefit_confidence, 2),
            "execution_feasibility": round(execution_confidence, 2)
        }
        
        # === 2. 提取决策因子 ===
        decision_factors = self._extract_decision_factors(
            technical_data, risk_management, market_environment, 
            cost_benefit_analysis, basic_params
        )
        
        # === 3. 生成替代场景 ===
        alternative_scenarios = self._generate_alternative_scenarios(
            scenario_analysis, market_environment, technical_risk_assessment
        )
        
        # === 4. 构建决策树路径 ===
        decision_tree_path = self._build_decision_tree_path(
            direction, technical_risk_assessment, market_environment, risk_management
        )
        
        # === 5. 生成推理链条 ===
        reasoning_chain = self._generate_reasoning_chain(
            technical_data, risk_management, market_environment, basic_params
        )
        
        # === 6. 收集支持和反对证据 ===
        supporting_evidence, contrary_evidence = self._collect_evidence(
            technical_data, market_environment, cost_benefit_analysis, direction
        )
        
        # 构建决策元数据
        decision_metadata = {
            "confidence": total_confidence,
            "confidence_breakdown": confidence_breakdown,
            "decision_factors": decision_factors,
            "alternative_scenarios": alternative_scenarios,
            "decision_tree_path": decision_tree_path,
            "reasoning_chain": reasoning_chain,
            "supporting_evidence": supporting_evidence,
            "contrary_evidence": contrary_evidence
        }
        
        return decision_metadata
    
    def _calculate_technical_confidence(self, technical_data: Dict[str, Any], 
                                       technical_risk_assessment: Dict[str, Any]) -> float:
        """计算技术分析置信度"""
        confidence = 50.0  # 基础置信度
        
        # 检查信号一致性
        signal_alignment = technical_risk_assessment.get("momentum_alignment", False)
        if signal_alignment:
            confidence += 15.0
        
        # 检查趋势强度
        trend_strength = technical_risk_assessment.get("trend_strength", "weak")
        if trend_strength == "strong":
            confidence += 20.0
        elif trend_strength == "moderate":
            confidence += 10.0
        
        # 检查支撑阻力位距离
        sr_proximity = technical_risk_assessment.get("support_resistance_proximity", "far")
        if sr_proximity == "far":
            confidence += 10.0
        elif sr_proximity == "at_level":
            confidence -= 5.0
        
        # 检查突破概率
        breakout_prob = technical_risk_assessment.get("breakout_probability", 0.5)
        confidence += (breakout_prob - 0.5) * 20.0
        
        return min(max(confidence, 0.0), 100.0)
    
    def _calculate_risk_confidence(self, risk_management: Dict[str, Any], 
                                  risk_data: Dict[str, Any]) -> float:
        """计算风险评估置信度"""
        confidence = 50.0  # 基础置信度
        
        # 检查风险收益比
        risk_reward_ratio = risk_management.get("risk_reward_ratio", 0.0)
        if risk_reward_ratio >= 2.0:
            confidence += 20.0
        elif risk_reward_ratio >= 1.5:
            confidence += 10.0
        elif risk_reward_ratio < 1.0:
            confidence -= 15.0
        
        # 检查强平距离
        liquidation_distance = risk_management.get("liquidation_price", 0.0)
        if liquidation_distance > 0:  # 有合理的强平距离
            confidence += 15.0
        
        # 检查保证金使用率
        margin_required = risk_management.get("margin_required", 0.0)
        portfolio_cash = 10000.0  # 假设现金值，实际应从portfolio获取
        if margin_required > 0 and margin_required < portfolio_cash * 0.3:
            confidence += 10.0
        elif margin_required > portfolio_cash * 0.5:
            confidence -= 20.0
        
        return min(max(confidence, 0.0), 100.0)
    
    def _calculate_market_confidence(self, market_environment: Dict[str, Any]) -> float:
        """计算市场条件置信度"""
        confidence = 50.0  # 基础置信度
        
        # 检查趋势明确性
        trend_regime = market_environment.get("trend_regime", "ranging")
        if trend_regime == "trending":
            confidence += 20.0
        elif trend_regime == "ranging":
            confidence += 5.0
        else:  # transitional
            confidence -= 10.0
        
        # 检查波动率状态
        volatility_regime = market_environment.get("volatility_regime", "normal")
        if volatility_regime == "normal":
            confidence += 15.0
        elif volatility_regime == "low":
            confidence += 10.0
        elif volatility_regime == "elevated":
            confidence -= 5.0
        else:  # extreme
            confidence -= 20.0
        
        # 检查流动性
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        if liquidity_assessment == "excellent":
            confidence += 15.0
        elif liquidity_assessment == "good":
            confidence += 10.0
        elif liquidity_assessment == "poor":
            confidence -= 15.0
        
        return min(max(confidence, 0.0), 100.0)
    
    def _calculate_cost_benefit_confidence(self, cost_benefit_analysis: Dict[str, Any]) -> float:
        """计算成本效益置信度"""
        confidence = 50.0  # 基础置信度
        
        # 检查期望收益率
        expected_return = cost_benefit_analysis.get("expected_return", 0.0)
        if expected_return > 0.05:  # 5%以上收益
            confidence += 20.0
        elif expected_return > 0.02:  # 2%以上收益
            confidence += 10.0
        elif expected_return < 0:
            confidence -= 30.0
        
        # 检查获利概率
        profit_probability = cost_benefit_analysis.get("profit_probability", 0.5)
        confidence += (profit_probability - 0.5) * 30.0
        
        # 检查费用合理性
        trading_fee = cost_benefit_analysis.get("estimated_trading_fee", 0.0)
        expected_profit = cost_benefit_analysis.get("target_profit", 0.0)
        if expected_profit > 0 and trading_fee / expected_profit < 0.1:  # 费用占预期利润10%以内
            confidence += 10.0
        
        return min(max(confidence, 0.0), 100.0)
    
    def _calculate_execution_confidence(self, execution_strategy: Dict[str, Any], 
                                       market_environment: Dict[str, Any]) -> float:
        """计算执行可行性置信度"""
        confidence = 50.0  # 基础置信度
        
        # 检查执行复杂度
        execution_complexity = execution_strategy.get("execution_complexity", "moderate")
        if execution_complexity == "simple":
            confidence += 20.0
        elif execution_complexity == "complex":
            confidence -= 15.0
        
        # 检查入场时机
        entry_timing = execution_strategy.get("entry_timing", "now")
        if entry_timing == "now":
            confidence += 15.0
        elif entry_timing.startswith("wait"):
            confidence += 5.0
        
        # 检查流动性影响
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        if liquidity_assessment in ["excellent", "good"]:
            confidence += 15.0
        elif liquidity_assessment == "poor":
            confidence -= 20.0
        
        return min(max(confidence, 0.0), 100.0)
    
    def _extract_decision_factors(self, technical_data: Dict[str, Any], 
                                 risk_management: Dict[str, Any],
                                 market_environment: Dict[str, Any],
                                 cost_benefit_analysis: Dict[str, Any],
                                 basic_params: Dict[str, Any]) -> Dict[str, List[str]]:
        """提取决策因子"""
        primary_drivers = []
        supporting_factors = []
        risk_factors = []
        uncertainty_factors = []
        
        direction = basic_params.get("direction", "long")
        
        # 主要驱动因子
        trend_regime = market_environment.get("trend_regime", "ranging")
        if trend_regime == "trending":
            primary_drivers.append(f"市场处于{trend_regime}趋势状态，支持{direction}方向")
        
        expected_return = cost_benefit_analysis.get("expected_return", 0.0)
        if expected_return > 0.03:
            primary_drivers.append(f"预期收益率{expected_return:.2%}，具有良好盈利潜力")
        
        risk_reward_ratio = risk_management.get("risk_reward_ratio", 0.0)
        if risk_reward_ratio >= 1.5:
            primary_drivers.append(f"风险收益比{risk_reward_ratio:.2f}，风险可控")
        
        # 支持因子
        volatility_regime = market_environment.get("volatility_regime", "normal")
        if volatility_regime == "normal":
            supporting_factors.append("波动率处于正常水平，利于交易执行")
        
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        if liquidity_assessment in ["excellent", "good"]:
            supporting_factors.append(f"市场流动性{liquidity_assessment}，执行成本较低")
        
        # 风险因子
        if volatility_regime in ["elevated", "extreme"]:
            risk_factors.append(f"波动率{volatility_regime}，增加交易风险")
        
        margin_required = risk_management.get("margin_required", 0.0)
        if margin_required > 5000:  # 假设阈值
            risk_factors.append(f"保证金需求{margin_required}较高，增加资金压力")
        
        # 不确定性因子
        if trend_regime == "transitional":
            uncertainty_factors.append("市场处于趋势转换期，方向不确定性较高")
        
        funding_rate = cost_benefit_analysis.get("funding_rate", 0.0)
        if abs(funding_rate) > 0.001:  # 0.1%以上资金费率
            uncertainty_factors.append(f"资金费率{funding_rate:.4f}可能影响持仓成本")
        
        # 确保每个列表都有合理数量的因子
        if len(primary_drivers) == 0:
            primary_drivers.append("基于综合分析的交易机会")
        if len(supporting_factors) == 0:
            supporting_factors.append("技术指标支持当前方向")
        if len(risk_factors) == 0:
            risk_factors.append("市场波动风险")
        if len(uncertainty_factors) == 0:
            uncertainty_factors.append("价格走势不确定性")
        
        return {
            "primary_drivers": primary_drivers[:3],  # 最多3个
            "supporting_factors": supporting_factors[:2],  # 最多2个
            "risk_factors": risk_factors[:2],  # 最多2个
            "uncertainty_factors": uncertainty_factors[:2]  # 最多2个
        }
    
    def _generate_alternative_scenarios(self, scenario_analysis: Dict[str, Any],
                                       market_environment: Dict[str, Any],
                                       technical_risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成替代场景"""
        alternatives = []
        
        # 场景1：市场趋势反转
        trend_regime = market_environment.get("trend_regime", "ranging")
        if trend_regime == "trending":
            alternatives.append({
                "condition": "市场趋势发生反转，技术指标背离",
                "alternative_action": "平仓止损或减仓观望",
                "probability": 0.20,
                "impact": "moderate"
            })
        
        # 场景2：波动率突然放大
        volatility_regime = market_environment.get("volatility_regime", "normal")
        if volatility_regime in ["low", "normal"]:
            alternatives.append({
                "condition": "市场波动率突然升高，价格大幅震荡",
                "alternative_action": "降低杠杆或缩小仓位",
                "probability": 0.15,
                "impact": "high"
            })
        
        # 场景3：流动性枯竭
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        if liquidity_assessment in ["excellent", "good"]:
            alternatives.append({
                "condition": "市场流动性突然枯竭，买卖价差扩大",
                "alternative_action": "暂停交易或使用限价单",
                "probability": 0.10,
                "impact": "moderate"
            })
        
        # 场景4：技术突破失败
        breakout_probability = technical_risk_assessment.get("breakout_probability", 0.5)
        if breakout_probability > 0.6:
            alternatives.append({
                "condition": "关键价位突破失败，形成假突破",
                "alternative_action": "反向操作或快速止损",
                "probability": 0.25,
                "impact": "high"
            })
        
        # 如果没有生成任何场景，添加默认场景
        if not alternatives:
            alternatives.append({
                "condition": "市场环境发生重大变化",
                "alternative_action": "重新评估交易策略",
                "probability": 0.20,
                "impact": "moderate"
            })
        
        return alternatives[:4]  # 最多返回4个场景
    
    def _build_decision_tree_path(self, direction: str, 
                                 technical_risk_assessment: Dict[str, Any],
                                 market_environment: Dict[str, Any],
                                 risk_management: Dict[str, Any]) -> List[str]:
        """构建决策树路径"""
        path = []
        
        # 第一层：市场环境判断
        trend_regime = market_environment.get("trend_regime", "ranging")
        path.append(f"市场环境评估: {trend_regime}")
        
        # 第二层：技术信号确认
        trend_strength = technical_risk_assessment.get("trend_strength", "weak")
        path.append(f"技术信号强度: {trend_strength}")
        
        # 第三层：风险评估
        risk_reward_ratio = risk_management.get("risk_reward_ratio", 0.0)
        if risk_reward_ratio >= 1.5:
            path.append("风险评估: 可接受")
        else:
            path.append("风险评估: 需谨慎")
        
        # 第四层：最终决策
        path.append(f"决策结果: {direction}方向交易")
        
        return path[:3]  # 返回前3个路径节点
    
    def _generate_reasoning_chain(self, technical_data: Dict[str, Any],
                                 risk_management: Dict[str, Any],
                                 market_environment: Dict[str, Any],
                                 basic_params: Dict[str, Any]) -> List[str]:
        """生成推理链条"""
        chain = []
        
        direction = basic_params.get("direction", "long")
        leverage = basic_params.get("leverage", 1)
        
        # 推理步骤1：市场分析
        trend_regime = market_environment.get("trend_regime", "ranging")
        volatility_regime = market_environment.get("volatility_regime", "normal")
        chain.append(f"市场分析显示趋势为{trend_regime}，波动率{volatility_regime}，支持{direction}操作")
        
        # 推理步骤2：技术确认
        timeframe_data = {}
        for tf in ["1h", "4h"]:  # 主要时间框架
            if tf in technical_data:
                signal = technical_data[tf].get("signal", "neutral")
                confidence = technical_data[tf].get("confidence", 50)
                timeframe_data[tf] = {"signal": signal, "confidence": confidence}
        
        if timeframe_data:
            chain.append(f"技术分析确认{direction}信号，多时间框架支持交易方向")
        else:
            chain.append("基于现有技术指标确认交易方向")
        
        # 推理步骤3：风险控制
        risk_reward_ratio = risk_management.get("risk_reward_ratio", 0.0)
        stop_loss = risk_management.get("stop_loss", 0.0)
        chain.append(f"设置{leverage}倍杠杆，风险收益比{risk_reward_ratio:.2f}，止损位{stop_loss}")
        
        return chain[:3]  # 返回前3个推理步骤
    
    def _collect_evidence(self, technical_data: Dict[str, Any],
                         market_environment: Dict[str, Any],
                         cost_benefit_analysis: Dict[str, Any],
                         direction: str) -> tuple[List[str], List[str]]:
        """收集支持和反对证据"""
        supporting_evidence = []
        contrary_evidence = []
        
        # 支持证据
        trend_regime = market_environment.get("trend_regime", "ranging")
        if trend_regime == "trending":
            supporting_evidence.append(f"市场趋势明确，支持{direction}方向交易")
        
        expected_return = cost_benefit_analysis.get("expected_return", 0.0)
        if expected_return > 0.02:
            supporting_evidence.append(f"预期收益率{expected_return:.2%}具有吸引力")
        
        profit_probability = cost_benefit_analysis.get("profit_probability", 0.5)
        if profit_probability > 0.6:
            supporting_evidence.append(f"获利概率{profit_probability:.1%}相对较高")
        
        # 反对证据
        volatility_regime = market_environment.get("volatility_regime", "normal")
        if volatility_regime in ["elevated", "extreme"]:
            contrary_evidence.append(f"波动率{volatility_regime}增加交易风险")
        
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        if liquidity_assessment == "poor":
            contrary_evidence.append("市场流动性不佳，可能影响交易执行")
        
        funding_rate = cost_benefit_analysis.get("funding_rate", 0.0)
        if abs(funding_rate) > 0.002:  # 0.2%以上
            contrary_evidence.append(f"资金费率{funding_rate:.4f}较高，增加持仓成本")
        
        # 确保每个列表至少有一个证据
        if not supporting_evidence:
            supporting_evidence.append("综合技术指标支持交易决策")
        if not contrary_evidence:
            contrary_evidence.append("市场不确定性带来潜在风险")
        
        return supporting_evidence[:2], contrary_evidence[:2]

    def setup_monitoring_alerts(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        technical_risk_assessment: Dict[str, Any],
        market_environment: Dict[str, Any],
        analyst_signals: Dict[str, Any],
        portfolio: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        设置监控告警参数，基于实际价格和真实仓位配置告警阈值
        
        Args:
            ticker: 交易对符号
            basic_params: 基础交易参数
            risk_management: 风险管理参数
            technical_risk_assessment: 技术风险评估
            market_environment: 市场环境评估
            analyst_signals: 分析师信号
            portfolio: 投资组合数据
            
        Returns:
            Dict: 包含四类告警配置的字典
        """
        try:
            # 获取基础数据
            current_price = basic_params.get("current_price", 0.0)
            leverage = basic_params.get("leverage", 1)
            direction = basic_params.get("direction", "long")
            position_size = basic_params.get("position_size", 0.0)
            
            # 获取风险管理数据
            stop_loss = risk_management.get("stop_loss", 0.0)
            take_profit = risk_management.get("take_profit", 0.0)
            liquidation_price = risk_management.get("liquidation_price", 0.0)
            margin_required = risk_management.get("margin_required", 0.0)
            
            # 获取技术分析数据
            risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
            technical_data = analyst_signals.get("technical_analyst_agent", {}).get(ticker, {})
            
            # 获取ATR数据用于动态阈值设置
            atr_values = {}
            for timeframe, data in technical_data.items():
                if isinstance(data, dict) and "atr_values" in data:
                    atr_values = data["atr_values"]
                    break
            
            atr_14 = atr_values.get("atr_14", current_price * 0.02)  # 默认2%
            volatility_percentile = 0.5  # 默认值
            
            # 从波动率分析获取数据
            for timeframe, data in technical_data.items():
                if isinstance(data, dict) and "volatility_analysis" in data:
                    volatility_percentile = data["volatility_analysis"].get("volatility_percentile", 0.5)
                    break
            
            # 1. 价格告警配置
            price_alerts = {
                "stop_loss_alert": self._calculate_stop_loss_alert(
                    current_price, stop_loss, direction, atr_14
                ),
                "take_profit_alert": self._calculate_take_profit_alert(
                    current_price, take_profit, direction, atr_14
                ),
                "liquidation_warning": self._calculate_liquidation_warning(
                    current_price, liquidation_price, direction, leverage
                ),
                "margin_call_warning": self._calculate_margin_call_warning(
                    current_price, liquidation_price, direction, leverage
                )
            }
            
            # 2. 风险告警配置
            risk_alerts = {
                "max_drawdown_alert": self._calculate_max_drawdown_alert(
                    portfolio, position_size, leverage
                ),
                "volatility_spike_alert": self._calculate_volatility_spike_alert(
                    volatility_percentile, atr_14, current_price
                ),
                "correlation_breakdown_alert": self._calculate_correlation_breakdown_alert(
                    market_environment, technical_risk_assessment
                ),
                "funding_rate_spike_alert": self._calculate_funding_rate_spike_alert(
                    market_environment
                )
            }
            
            # 3. 信号告警配置
            signal_alerts = {
                "signal_reversal_alert": self._should_enable_signal_reversal_alert(
                    technical_risk_assessment, market_environment
                ),
                "trend_change_alert": self._should_enable_trend_change_alert(
                    technical_risk_assessment, market_environment
                ),
                "momentum_divergence_alert": self._should_enable_momentum_divergence_alert(
                    technical_risk_assessment, technical_data
                ),
                "volume_anomaly_alert": self._should_enable_volume_anomaly_alert(
                    technical_data, market_environment
                )
            }
            
            # 4. 系统告警配置
            system_alerts = {
                "api_latency_warning": self._should_enable_api_latency_warning(
                    market_environment
                ),
                "data_freshness_warning": self._should_enable_data_freshness_warning(),
                "execution_slippage_warning": self._calculate_execution_slippage_warning(
                    market_environment, position_size, current_price
                ),
                "liquidity_warning": self._should_enable_liquidity_warning(
                    market_environment
                )
            }
            
            return {
                "price_alerts": price_alerts,
                "risk_alerts": risk_alerts,
                "signal_alerts": signal_alerts,
                "system_alerts": system_alerts
            }
            
        except Exception as e:
            print(f"Warning: Failed to setup monitoring alerts for {ticker}: {str(e)}")
            # 返回默认的告警配置
            return self._get_default_monitoring_alerts()
    
    def _calculate_stop_loss_alert(self, current_price: float, stop_loss: float, 
                                  direction: str, atr: float) -> float:
        """计算止损告警价格"""
        if stop_loss > 0:
            # 在止损价格基础上增加缓冲区
            buffer = atr * 0.5  # 0.5个ATR作为缓冲
            if direction == "long":
                return stop_loss + buffer
            else:
                return stop_loss - buffer
        else:
            # 如果没有设置止损，使用ATR设置默认告警
            if direction == "long":
                return current_price - atr * 1.5
            else:
                return current_price + atr * 1.5
    
    def _calculate_take_profit_alert(self, current_price: float, take_profit: float,
                                   direction: str, atr: float) -> float:
        """计算止盈告警价格"""
        if take_profit > 0:
            # 在止盈价格附近设置告警
            buffer = atr * 0.3  # 0.3个ATR作为缓冲
            if direction == "long":
                return take_profit - buffer
            else:
                return take_profit + buffer
        else:
            # 如果没有设置止盈，使用ATR设置默认目标
            if direction == "long":
                return current_price + atr * 2.0
            else:
                return current_price - atr * 2.0
    
    def _calculate_liquidation_warning(self, current_price: float, liquidation_price: float,
                                     direction: str, leverage: int) -> float:
        """计算强平预警价格"""
        if liquidation_price > 0:
            # 在强平价格前设置预警，预警距离基于杠杆倍数
            safety_margin_pct = max(0.1, 1.0 / leverage)  # 至少10%，高杠杆时更小
            
            if direction == "long":
                return liquidation_price + abs(liquidation_price - current_price) * safety_margin_pct
            else:
                return liquidation_price - abs(liquidation_price - current_price) * safety_margin_pct
        else:
            # 估算强平价格并设置预警
            estimated_liquidation_distance = current_price * 0.8 / leverage
            if direction == "long":
                return current_price - estimated_liquidation_distance * 0.7
            else:
                return current_price + estimated_liquidation_distance * 0.7
    
    def _calculate_margin_call_warning(self, current_price: float, liquidation_price: float,
                                     direction: str, leverage: int) -> float:
        """计算保证金追加预警价格"""
        if liquidation_price > 0:
            # 在强平价格前更早设置追保预警
            margin_call_ratio = 0.3  # 在30%距离时预警
            
            if direction == "long":
                return current_price - (current_price - liquidation_price) * margin_call_ratio
            else:
                return current_price + (liquidation_price - current_price) * margin_call_ratio
        else:
            # 估算追保价格
            estimated_distance = current_price * 0.5 / leverage
            if direction == "long":
                return current_price - estimated_distance
            else:
                return current_price + estimated_distance
    
    def _calculate_max_drawdown_alert(self, portfolio: Dict[str, Any], 
                                    position_size: float, leverage: int) -> float:
        """计算最大回撤告警阈值"""
        portfolio_value = portfolio.get("total_value", 1000000.0)  # 默认100万
        position_risk = position_size * leverage / portfolio_value
        
        # 基于仓位风险设置回撤告警，风险越高阈值越低
        base_threshold = 0.1  # 基础10%回撤
        risk_adjustment = position_risk * 0.5  # 风险调整
        
        return max(0.05, base_threshold - risk_adjustment)  # 最低5%
    
    def _calculate_volatility_spike_alert(self, volatility_percentile: float,
                                        atr: float, current_price: float) -> float:
        """计算波动率飙升告警阈值"""
        # 基于当前波动率位置设置告警阈值
        if volatility_percentile > 0.8:
            # 高波动率环境，设置较低的告警阈值
            return atr / current_price * 1.5  # 1.5倍ATR百分比
        elif volatility_percentile > 0.5:
            # 正常波动率环境
            return atr / current_price * 2.0  # 2倍ATR百分比
        else:
            # 低波动率环境，设置较高的告警阈值
            return atr / current_price * 2.5  # 2.5倍ATR百分比
    
    def _calculate_correlation_breakdown_alert(self, market_environment: Dict[str, Any],
                                             technical_risk: Dict[str, Any]) -> float:
        """计算相关性破坏告警阈值"""
        # 基于市场环境和技术风险设置相关性告警
        market_structure = market_environment.get("market_structure", "neutral")
        trend_strength = technical_risk.get("trend_strength", "moderate")
        
        if market_structure == "neutral" or trend_strength == "weak":
            return 0.3  # 在不确定环境中降低告警阈值
        else:
            return 0.5  # 正常环境的告警阈值
    
    def _calculate_funding_rate_spike_alert(self, market_environment: Dict[str, Any]) -> float:
        """计算资金费率飙升告警阈值"""
        funding_rate_trend = market_environment.get("funding_rate_trend", "stable")
        
        if funding_rate_trend == "increasing":
            return 0.002  # 0.2%，在上升趋势中设置较低告警
        else:
            return 0.005  # 0.5%，正常情况下的告警阈值
    
    def _should_enable_signal_reversal_alert(self, technical_risk: Dict[str, Any],
                                           market_environment: Dict[str, Any]) -> bool:
        """判断是否启用信号反转告警"""
        trend_strength = technical_risk.get("trend_strength", "moderate")
        market_phase = market_environment.get("market_phase", "neutral")
        
        # 在趋势较弱或市场转换期启用
        return trend_strength == "weak" or market_phase in ["distribution", "accumulation"]
    
    def _should_enable_trend_change_alert(self, technical_risk: Dict[str, Any],
                                        market_environment: Dict[str, Any]) -> bool:
        """判断是否启用趋势变化告警"""
        trend_regime = market_environment.get("trend_regime", "ranging")
        
        # 在趋势市场中启用趋势变化告警
        return trend_regime == "trending"
    
    def _should_enable_momentum_divergence_alert(self, technical_risk: Dict[str, Any],
                                               technical_data: Dict[str, Any]) -> bool:
        """判断是否启用动量背离告警"""
        momentum_alignment = technical_risk.get("momentum_alignment", False)
        
        # 当动量不一致时启用告警
        return not momentum_alignment
    
    def _should_enable_volume_anomaly_alert(self, technical_data: Dict[str, Any],
                                          market_environment: Dict[str, Any]) -> bool:
        """判断是否启用成交量异常告警"""
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        
        # 在流动性较差时启用成交量异常告警
        return liquidity_assessment in ["poor", "fair"]
    
    def _should_enable_api_latency_warning(self, market_environment: Dict[str, Any]) -> bool:
        """判断是否启用API延迟告警"""
        volatility_regime = market_environment.get("volatility_regime", "normal")
        
        # 在高波动率环境中启用API延迟告警
        return volatility_regime in ["elevated", "extreme"]
    
    def _should_enable_data_freshness_warning(self) -> bool:
        """判断是否启用数据新鲜度告警"""
        # 始终启用数据新鲜度告警
        return True
    
    def _calculate_execution_slippage_warning(self, market_environment: Dict[str, Any],
                                            position_size: float, current_price: float) -> float:
        """计算执行滑点告警阈值"""
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        volatility_regime = market_environment.get("volatility_regime", "normal")
        
        # 计算仓位相对大小
        position_impact = min(0.01, position_size / (current_price * 1000000))  # 假设1M流动性基准
        
        base_slippage = 0.001  # 基础0.1%滑点
        
        # 根据流动性调整
        if liquidity_assessment == "poor":
            base_slippage *= 3
        elif liquidity_assessment == "fair":
            base_slippage *= 1.5
        
        # 根据波动率调整
        if volatility_regime == "extreme":
            base_slippage *= 2
        elif volatility_regime == "elevated":
            base_slippage *= 1.5
        
        # 加上仓位冲击
        return base_slippage + position_impact
    
    def _should_enable_liquidity_warning(self, market_environment: Dict[str, Any]) -> bool:
        """判断是否启用流动性告警"""
        liquidity_assessment = market_environment.get("liquidity_assessment", "fair")
        
        # 在流动性较差时启用告警
        return liquidity_assessment in ["poor", "fair"]
    
    def _get_default_monitoring_alerts(self) -> Dict[str, Any]:
        """获取默认的告警配置"""
        return {
            "price_alerts": {
                "stop_loss_alert": 0.0,
                "take_profit_alert": 0.0,
                "liquidation_warning": 0.0,
                "margin_call_warning": 0.0
            },
            "risk_alerts": {
                "max_drawdown_alert": 0.1,
                "volatility_spike_alert": 0.05,
                "correlation_breakdown_alert": 0.5,
                "funding_rate_spike_alert": 0.005
            },
            "signal_alerts": {
                "signal_reversal_alert": True,
                "trend_change_alert": True,
                "momentum_divergence_alert": True,
                "volume_anomaly_alert": True
            },
            "system_alerts": {
                "api_latency_warning": True,
                "data_freshness_warning": True,
                "execution_slippage_warning": 0.002,
                "liquidity_warning": True
            }
        }

    def generate_decision_summary(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        technical_risk_assessment: Dict[str, Any],
        market_environment: Dict[str, Any],
        execution_strategy: Dict[str, Any],
        scenario_analysis: Dict[str, Any],
        decision_metadata: Dict[str, Any],
        cost_benefit_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        生成最终决策摘要，基于实际市场状况和真实交易参数
        
        Args:
            ticker: 交易对符号
            basic_params: 基础交易参数
            risk_management: 风险管理数据
            technical_risk_assessment: 技术风险评估
            market_environment: 市场环境
            execution_strategy: 执行策略
            scenario_analysis: 情景分析
            decision_metadata: 决策元数据
            cost_benefit_analysis: 成本收益分析
            
        Returns:
            包含 decision_summary 字段的字典，符合文档规范要求
        """
        try:
            # 1. 确定行动类型
            action_type = self._determine_action_type(basic_params)
            
            # 2. 评估紧急程度
            urgency = self._assess_urgency(
                technical_risk_assessment, 
                market_environment, 
                decision_metadata,
                execution_strategy
            )
            
            # 3. 确定预期持仓期
            expected_holding_period = self._determine_holding_period(
                technical_risk_assessment,
                execution_strategy,
                market_environment
            )
            
            # 4. 确定策略类别
            strategy_category = self._determine_strategy_category(
                technical_risk_assessment,
                market_environment,
                decision_metadata
            )
            
            # 5. 评估风险类别
            risk_category = self._assess_risk_category(
                risk_management,
                basic_params,
                technical_risk_assessment
            )
            
            # 6. 评估执行复杂性
            execution_complexity = self._assess_execution_complexity(
                execution_strategy,
                market_environment,
                basic_params
            )
            
            # 7. 生成详细推理说明
            reasoning = self._generate_detailed_reasoning(
                ticker,
                basic_params,
                risk_management,
                technical_risk_assessment,
                market_environment,
                cost_benefit_analysis,
                scenario_analysis,
                decision_metadata
            )
            
            return {
                "action_type": action_type,
                "urgency": urgency,
                "expected_holding_period": expected_holding_period,
                "strategy_category": strategy_category,
                "risk_category": risk_category,
                "execution_complexity": execution_complexity,
                "reasoning": reasoning
            }
            
        except Exception as e:
            # 如果生成摘要失败，返回默认摘要
            return {
                "action_type": "hold",
                "urgency": "wait",
                "expected_holding_period": "short_term",
                "strategy_category": "mean_reversion",
                "risk_category": "conservative",
                "execution_complexity": "simple",
                "reasoning": f"由于数据处理错误，采用保守策略：{str(e)}"
            }

    def _determine_action_type(self, basic_params: Dict[str, Any]) -> str:
        """确定行动类型"""
        direction = basic_params.get("direction", "")
        operation = basic_params.get("operation", "")
        
        if operation == "open":
            return "open_long" if direction == "long" else "open_short"
        elif operation == "close":
            return "close_long" if direction == "long" else "close_short"
        elif operation == "add":
            return "add"
        elif operation == "reduce":
            return "reduce"
        else:
            return "hold"

    def _assess_urgency(
        self,
        technical_risk: Dict[str, Any],
        market_env: Dict[str, Any],
        decision_meta: Dict[str, Any],
        execution_strategy: Dict[str, Any]
    ) -> str:
        """评估紧急程度"""
        # 获取关键指标
        breakout_probability = technical_risk.get("breakout_probability", 0.0)
        volatility_risk = technical_risk.get("volatility_risk", "moderate")
        market_structure = market_env.get("market_structure", "neutral")
        confidence = decision_meta.get("confidence", 50)
        entry_timing = execution_strategy.get("entry_timing", "wait_15m")
        
        # 立即执行条件
        if (breakout_probability > 0.8 or 
            volatility_risk == "extreme" or 
            confidence > 90 or 
            entry_timing == "now"):
            return "immediate"
        
        # 高优先级条件
        if (breakout_probability > 0.6 or 
            volatility_risk == "high" or 
            confidence > 75 or 
            market_structure in ["bullish", "bearish"]):
            return "high"
        
        # 中等优先级条件
        if (breakout_probability > 0.4 or 
            confidence > 60 or 
            entry_timing in ["wait_5m", "wait_pullback"]):
            return "medium"
        
        # 低优先级条件
        if confidence > 40:
            return "low"
        
        return "wait"

    def _determine_holding_period(
        self,
        technical_risk: Dict[str, Any],
        execution_strategy: Dict[str, Any],
        market_env: Dict[str, Any]
    ) -> str:
        """确定预期持仓期"""
        trend_strength = technical_risk.get("trend_strength", "moderate")
        exit_strategy = execution_strategy.get("exit_strategy", "signal_based")
        volatility_regime = market_env.get("volatility_regime", "normal")
        
        # 剥头皮交易条件
        if (volatility_regime == "extreme" or 
            exit_strategy == "time_based" or 
            trend_strength == "weak"):
            return "scalp"
        
        # 短期交易条件
        if (volatility_regime == "elevated" or 
            trend_strength == "moderate" or 
            exit_strategy == "target_based"):
            return "short_term"
        
        # 中期交易条件
        if (trend_strength == "strong" or 
            volatility_regime == "low"):
            return "medium_term"
        
        # 长期交易条件
        if trend_strength == "strong" and volatility_regime == "low":
            return "long_term"
        
        return "short_term"

    def _determine_strategy_category(
        self,
        technical_risk: Dict[str, Any],
        market_env: Dict[str, Any],
        decision_meta: Dict[str, Any]
    ) -> str:
        """确定策略类别"""
        trend_strength = technical_risk.get("trend_strength", "moderate")
        mean_reversion_risk = technical_risk.get("mean_reversion_risk", "moderate")
        breakout_probability = technical_risk.get("breakout_probability", 0.0)
        market_structure = market_env.get("market_structure", "neutral")
        
        primary_drivers = decision_meta.get("decision_factors", {}).get("primary_drivers", [])
        
        # 突破策略
        if (breakout_probability > 0.6 or 
            "breakout" in primary_drivers or 
            "突破" in str(primary_drivers)):
            return "breakout"
        
        # 趋势跟随策略
        if (trend_strength == "strong" or 
            market_structure in ["bullish", "bearish"] or 
            "trend" in primary_drivers or 
            "趋势" in str(primary_drivers)):
            return "trend_following"
        
        # 均值回归策略
        if (mean_reversion_risk == "high" or 
            market_structure == "neutral" or 
            "reversion" in primary_drivers or 
            "回归" in str(primary_drivers)):
            return "mean_reversion"
        
        # 套利策略（默认情况）
        return "arbitrage"

    def _assess_risk_category(
        self,
        risk_management: Dict[str, Any],
        basic_params: Dict[str, Any],
        technical_risk: Dict[str, Any]
    ) -> str:
        """评估风险类别"""
        leverage = basic_params.get("leverage", 1)
        risk_percentage = risk_management.get("risk_percentage", 0.02)
        position_ratio = basic_params.get("position_ratio", 0.1)
        volatility_risk = technical_risk.get("volatility_risk", "moderate")
        
        # 投机性风险
        if (leverage > 20 or 
            risk_percentage > 0.05 or 
            position_ratio > 0.5 or 
            volatility_risk == "extreme"):
            return "speculative"
        
        # 激进风险
        if (leverage > 10 or 
            risk_percentage > 0.03 or 
            position_ratio > 0.3 or 
            volatility_risk == "high"):
            return "aggressive"
        
        # 中等风险
        if (leverage > 5 or 
            risk_percentage > 0.02 or 
            position_ratio > 0.2 or 
            volatility_risk == "moderate"):
            return "moderate"
        
        # 保守风险
        return "conservative"

    def _assess_execution_complexity(
        self,
        execution_strategy: Dict[str, Any],
        market_env: Dict[str, Any],
        basic_params: Dict[str, Any]
    ) -> str:
        """评估执行复杂性"""
        entry_strategy = execution_strategy.get("entry_strategy", "immediate")
        position_building = execution_strategy.get("position_building", "single_entry")
        order_type = basic_params.get("order_type", "market")
        liquidity = market_env.get("liquidity_assessment", "good")
        partial_profit = execution_strategy.get("partial_profit_taking", False)
        
        # 复杂执行条件
        if (position_building in ["scale_in", "dca"] or 
            entry_strategy == "gradual" or 
            order_type == "stop_limit" or 
            partial_profit or 
            liquidity == "poor"):
            return "complex"
        
        # 中等复杂度条件
        if (entry_strategy == "wait_for_dip" or 
            order_type == "limit" or 
            liquidity == "fair"):
            return "moderate"
        
        # 简单执行
        return "simple"

    def _generate_detailed_reasoning(
        self,
        ticker: str,
        basic_params: Dict[str, Any],
        risk_management: Dict[str, Any],
        technical_risk: Dict[str, Any],
        market_env: Dict[str, Any],
        cost_benefit: Dict[str, Any],
        scenario_analysis: Dict[str, Any],
        decision_meta: Dict[str, Any]
    ) -> str:
        """生成详细的中文推理说明"""
        
        # 基础信息
        direction = basic_params.get("direction", "")
        operation = basic_params.get("operation", "")
        leverage = basic_params.get("leverage", 1)
        position_size = basic_params.get("position_size", 0)
        
        # 风险指标
        stop_loss = risk_management.get("stop_loss", 0)
        take_profit = risk_management.get("take_profit", 0)
        risk_reward_ratio = risk_management.get("risk_reward_ratio", 0)
        
        # 技术指标
        trend_strength = technical_risk.get("trend_strength", "moderate")
        breakout_probability = technical_risk.get("breakout_probability", 0)
        volatility_risk = technical_risk.get("volatility_risk", "moderate")
        
        # 市场环境
        market_structure = market_env.get("market_structure", "neutral")
        volatility_regime = market_env.get("volatility_regime", "normal")
        liquidity = market_env.get("liquidity_assessment", "good")
        
        # 成本收益
        expected_return = cost_benefit.get("expected_return", 0)
        profit_probability = cost_benefit.get("profit_probability", 0.5)
        trading_fee = cost_benefit.get("estimated_trading_fee", 0)
        
        # 置信度
        confidence = decision_meta.get("confidence", 50)
        
        reasoning_parts = []
        
        # 1. 交易决策概述
        reasoning_parts.append(f"对于{ticker}，基于当前市场分析，决定{operation}{direction}仓位")
        
        # 2. 技术分析要点
        if trend_strength == "strong":
            reasoning_parts.append(f"技术面显示{trend_strength}趋势，支持{direction}方向交易")
        
        if breakout_probability > 0.6:
            reasoning_parts.append(f"突破概率达到{breakout_probability:.1%}，技术形态良好")
        
        if volatility_risk == "high":
            reasoning_parts.append("当前波动率较高，需要密切关注风险控制")
        
        # 3. 风险管理分析
        reasoning_parts.append(f"采用{leverage}倍杠杆，仓位规模{position_size:.0f}USDT")
        
        if risk_reward_ratio > 1.5:
            reasoning_parts.append(f"风险收益比{risk_reward_ratio:.2f}，风险回报比例合理")
        
        reasoning_parts.append(f"设置止损{stop_loss:.2f}，止盈{take_profit:.2f}，严格控制下行风险")
        
        # 4. 市场环境评估
        if market_structure != "neutral":
            reasoning_parts.append(f"市场结构呈现{market_structure}特征，符合交易方向")
        
        if liquidity != "excellent":
            reasoning_parts.append(f"流动性{liquidity}，执行时需注意滑点控制")
        
        # 5. 成本分析
        if expected_return > 0:
            reasoning_parts.append(f"预期收益率{expected_return:.2%}，获利概率{profit_probability:.1%}")
        
        if trading_fee > 0:
            reasoning_parts.append(f"预计交易费用{trading_fee:.2f}USDT，已纳入成本考量")
        
        # 6. 执行建议
        if confidence > 80:
            reasoning_parts.append("高置信度信号，建议积极执行")
        elif confidence > 60:
            reasoning_parts.append("中等置信度信号，建议谨慎执行")
        else:
            reasoning_parts.append("低置信度信号，建议观望或小仓位试探")
        
        # 7. 风险提示
        if volatility_regime == "extreme":
            reasoning_parts.append("极端波动环境下，请严格遵守止损纪律")
        
        reasoning_parts.append("请密切关注市场变化，及时调整策略")
        
        return "；".join(reasoning_parts) + "。"

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Makes final trading decisions and generates orders for multiple tickers"""

        data = state.get('data', {})
        data['name'] = "PortfolioManagementNode"
        
        # Get the portfolio and analyst signals
        portfolio = data.get("portfolio", {})
        analyst_signals = data.get("analyst_signals", {})
        tickers = data.get("tickers", [])

        # Get position limits, current prices, and signals for every ticker
        position_limits = {}
        current_prices = {}
        max_shares = {}
        signals_by_ticker = {}
        
        for ticker in tickers:
            # Get position limits and current prices for the ticker
            risk_data = analyst_signals.get("risk_management_agent", {}).get(ticker, {})
            position_limits[ticker] = risk_data.get("remaining_position_limit", 0.0)
            current_prices[ticker] = risk_data.get("current_price", 0.0)

            # Calculate maximum shares allowed based on position limit and price
            if current_prices[ticker] > 0.0:
                max_shares[ticker] = float(position_limits[ticker] / current_prices[ticker])
            else:
                max_shares[ticker] = 0.0

            # Get signals for the ticker
            ticker_signals = {}
            for agent, signals in analyst_signals.items():
                if agent == "technical_analyst_agent" and ticker in signals:
                    ticker_signals[agent] = signals[ticker]

            signals_by_ticker[ticker] = ticker_signals

        # Generate the basic trading decision using LLM
        basic_decision_result = generate_trading_decision(
            tickers=tickers,
            signals_by_ticker=signals_by_ticker,
            current_prices=current_prices,
            max_shares=max_shares,
            portfolio=portfolio,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
            model_base_url=state["metadata"]["model_base_url"],
        )

        # 为每个ticker构建完整的75+字段决策输出
        enhanced_decisions = {}
        for ticker in tickers:
            ticker_decision = basic_decision_result.get("decisions", {}).get(ticker, {})
            
            # 开始构建完整的决策输出结构
            enhanced_decision = {}
            
            try:
                # 1. 信号整合 - 计算basic_params (基础交易参数)
                basic_params = self.calculate_basic_params(
                    ticker=ticker,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio,
                    current_price=current_prices.get(ticker, 0.0)
                )
                enhanced_decision["basic_params"] = basic_params
                
                # 2. 参数计算 - 设计risk_management (风险管理参数)
                risk_management = self.design_risk_management(
                    ticker=ticker,
                    basic_params=basic_params,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio
                )
                enhanced_decision["risk_management"] = risk_management
                
                # 3. 风险评估 - 分析timeframes (多时间框架信号强度)
                timeframe_analysis = self.analyze_timeframes(
                    ticker=ticker,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio
                )
                enhanced_decision["timeframe_analysis"] = timeframe_analysis
                
                # 4. 技术风险评估 - assess_technical_risk
                technical_risk_assessment = self.assess_technical_risk(
                    ticker=ticker,
                    analyst_signals=analyst_signals,
                    basic_params=basic_params,
                    portfolio=portfolio
                )
                enhanced_decision["technical_risk_assessment"] = technical_risk_assessment
                
                # 5. 成本分析 - calculate_cost_benefit (成本收益分析)
                cost_benefit_analysis = self.calculate_cost_benefit(
                    ticker=ticker,
                    basic_params=basic_params,
                    risk_management=risk_management,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio
                )
                enhanced_decision["cost_benefit_analysis"] = cost_benefit_analysis
                
                # 6. 环境评估 - evaluate_market_environment (市场环境评估)
                market_environment = self.evaluate_market_environment(
                    ticker=ticker,
                    analyst_signals=analyst_signals,
                    basic_params=basic_params,
                    portfolio=portfolio
                )
                enhanced_decision["market_environment"] = market_environment
                
                # 7. 策略设计 - design_execution_strategy (执行策略建议)
                execution_strategy = self.design_execution_strategy(
                    ticker=ticker,
                    basic_params=basic_params,
                    risk_management=risk_management,
                    technical_risk_assessment=technical_risk_assessment,
                    market_environment=market_environment,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio
                )
                enhanced_decision["execution_strategy"] = execution_strategy
                
                # 8. 情景分析 - generate_scenario_analysis
                scenario_analysis = self.generate_scenario_analysis(
                    ticker=ticker,
                    basic_params=basic_params,
                    risk_management=risk_management,
                    technical_risk_assessment=technical_risk_assessment,
                    market_environment=market_environment,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio
                )
                enhanced_decision["scenario_analysis"] = scenario_analysis
                
                # 9. 元数据 - create_decision_metadata (AI决策元数据)
                decision_metadata = self.create_decision_metadata(
                    ticker=ticker,
                    basic_params=basic_params,
                    risk_management=risk_management,
                    technical_risk_assessment=technical_risk_assessment,
                    market_environment=market_environment,
                    cost_benefit_analysis=cost_benefit_analysis,
                    execution_strategy=execution_strategy,
                    scenario_analysis=scenario_analysis,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio
                )
                enhanced_decision["decision_metadata"] = decision_metadata
                
                # 10. 告警 - setup_monitoring_alerts (监控和告警)
                monitoring_alerts = self.setup_monitoring_alerts(
                    ticker=ticker,
                    basic_params=basic_params,
                    risk_management=risk_management,
                    technical_risk_assessment=technical_risk_assessment,
                    market_environment=market_environment,
                    analyst_signals=analyst_signals,
                    portfolio=portfolio
                )
                enhanced_decision["monitoring_alerts"] = monitoring_alerts
                
                # 11. 摘要 - generate_decision_summary (最终决策摘要)
                decision_summary = self.generate_decision_summary(
                    ticker=ticker,
                    basic_params=basic_params,
                    risk_management=risk_management,
                    technical_risk_assessment=technical_risk_assessment,
                    market_environment=market_environment,
                    execution_strategy=execution_strategy,
                    scenario_analysis=scenario_analysis,
                    decision_metadata=decision_metadata,
                    cost_benefit_analysis=cost_benefit_analysis
                )
                enhanced_decision["decision_summary"] = decision_summary
                
                # 12. 保持兼容性：添加传统的基础字段
                # 从decision_summary中提取基础字段以保持向后兼容性
                enhanced_decision["action"] = decision_summary.get("action_type", "hold")
                enhanced_decision["quantity"] = basic_params.get("contract_quantity", 0.0)
                enhanced_decision["confidence"] = decision_metadata.get("confidence", 50.0)
                enhanced_decision["reasoning"] = decision_summary.get("reasoning", "")
                
                # 13. 数据验证 - 验证完整的交易决策数据
                logger.info(f"开始验证 {ticker} 的交易决策数据")
                
                # 构建验证上下文
                validation_context = {
                    "ticker": ticker,
                    "account_balance": portfolio.get("total_value", 100000.0),
                    "portfolio_value": portfolio.get("total_value", 100000.0),
                    "volatility": self._extract_volatility_from_signals(analyst_signals, ticker),
                    "current_drawdown": portfolio.get("current_drawdown", 0.0),
                    "correlations": portfolio.get("correlations", {}),
                    "top_positions": portfolio.get("top_positions", [])
                }
                
                # 执行验证
                validation_result = self._validate_trading_decision(
                    ticker=ticker,
                    decision_data=enhanced_decision,
                    context=validation_context
                )
                
                # 将验证结果添加到决策中
                enhanced_decision["validation"] = {
                    "is_valid": validation_result["is_valid"],
                    "critical_count": validation_result["critical_count"],
                    "error_count": validation_result["error_count"],
                    "warning_count": validation_result["warning_count"],
                    "suggestions": validation_result["suggestions"][:5],  # 只保留前5个建议
                    "validation_level": self.validation_level
                }
                
                # 如果验证发现可修正的数据，使用修正后的数据
                if validation_result["corrected_data"] != enhanced_decision:
                    logger.info(f"{ticker} 应用数据验证修正")
                    # 保留验证信息，更新其他数据
                    validation_info = enhanced_decision["validation"]
                    enhanced_decision = validation_result["corrected_data"]
                    enhanced_decision["validation"] = validation_info
                
                # 如果有严重错误，将动作改为hold
                if validation_result["critical_count"] > 0:
                    logger.warning(f"{ticker} 发现严重验证错误，强制设置为持仓")
                    enhanced_decision["action"] = "hold"
                    enhanced_decision["quantity"] = 0.0
                    enhanced_decision["validation"]["forced_hold"] = True
                    enhanced_decision["validation"]["reason"] = "严重验证错误"
                
                # 记录验证报告（在调试模式下）
                if state.get("metadata", {}).get("show_reasoning", False):
                    logger.info(f"{ticker} 验证报告:\n{validation_result['validation_report']}")
                
            except Exception as e:
                # 如果计算失败，记录错误但不影响基本功能
                print(f"Warning: Failed to calculate enhanced params for {ticker}: {str(e)}")
                
                # 创建空的结构以保持一致性，同时保持基础字段
                enhanced_decision = {
                    "action": ticker_decision.get("action", "hold"),
                    "quantity": ticker_decision.get("quantity", 0.0),
                    "confidence": ticker_decision.get("confidence", 50.0),
                    "reasoning": ticker_decision.get("reasoning", ""),
                    "basic_params": {},
                    "risk_management": {},
                    "timeframe_analysis": {},
                    "technical_risk_assessment": {},
                    "cost_benefit_analysis": {},
                    "market_environment": {},
                    "execution_strategy": {},
                    "scenario_analysis": {},
                    "decision_metadata": {},
                    "monitoring_alerts": {},
                    "decision_summary": {}
                }
            
            enhanced_decisions[ticker] = enhanced_decision

        # 创建增强后的结果
        enhanced_result = {
            "decisions": enhanced_decisions
        }

        # Create the portfolio management message
        message = HumanMessage(
            content=json.dumps(enhanced_result.get("decisions", {})),
            name="portfolio_management",
        )

        # Print the decision if the flag is set
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(enhanced_result.get("decisions"),
                                 "Portfolio Management Agent")

        return {
            "messages": [message],
            "data": state["data"],
        }


def generate_trading_decision(
        tickers: List[str],
        signals_by_ticker: Dict[str, Dict[str, Any]],
        current_prices: Dict[str, float],
        max_shares: Dict[str, float],
        portfolio: Dict[str, float],
        model_name: str,
        model_provider: str,
        model_base_url: Optional[str] = None
):
    """Attempts to get a decision from the LLM with retry logic"""
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions based on multiple tickers.
  
                Trading Rules:
                - For long positions:
                  * Only buy if you have available cash
                  * Only sell if you currently hold long shares of that ticker
                  * Sell quantity must be ≤ current long position shares
                  * Buy quantity must be ≤ max_shares for that ticker
  
                - For short positions:
                  * Only short if you have available margin (position value × margin requirement)
                  * Only cover if you currently have short shares of that ticker
                  * Cover quantity must be ≤ current short position shares
                  * Short quantity must respect margin requirements
  
                - The max_shares values are pre-calculated to respect position limits
                - Consider both long and short opportunities based on signals
                - Maintain appropriate risk management with both long and short exposure
  
                Available Actions:
                - "buy": Open or add to long position
                - "sell": Close or reduce long position
                - "short": Open or add to short position
                - "cover": Close or reduce short position
                - "hold": No action
  
                Inputs:
                - signals_by_ticker: dictionary of ticker → signals
                - max_shares: maximum shares allowed per ticker
                - portfolio_cash: current cash in portfolio
                - portfolio_positions: current positions (both long and short)
                - current_prices: current prices for each ticker
                - margin_requirement: current margin requirement for short positions (e.g., 0.5 means 50%)
                - total_margin_used: total margin currently in use
                """,
            ),
            (
                "human",
                """Based on the team's analysis, make your trading decisions for each ticker.
  
                Here are the signals by ticker:
                {signals_by_ticker}
  
                Current Prices:
                {current_prices}
  
                Maximum Shares Allowed For Purchases:
                {max_shares}
  
                Portfolio Cash: {portfolio_cash}
                Current Positions: {portfolio_positions}
                Current Margin Requirement: {margin_requirement}
                Total Margin Used: {total_margin_used}
  
                Output strictly in JSON with the following structure:
                {{
                  "decisions": {{
                    "TICKER1": {{
                      "action": "buy/sell/short/cover/hold",
                      "quantity": float,
                      "confidence": float between 0 and 100,
                      "reasoning": "string"
                    }},
                    "TICKER2": {{
                      ...
                    }},
                    ...
                  }}
                }}
                """,
            ),
        ]
    )

    llm = get_llm(provider=model_provider, model=model_name, base_url=model_base_url)

    chain = prompt | llm | json_parser
    result = chain.invoke(
        {
            "signals_by_ticker": json.dumps(signals_by_ticker, indent=2),
            "current_prices": json.dumps(current_prices, indent=2),
            "max_shares": json.dumps(max_shares, indent=2),
            "portfolio_cash": f"{portfolio.get('cash', 0.0):.2f}",
            "portfolio_positions": json.dumps(portfolio.get('positions', {}), indent=2),
            "margin_requirement": f"{portfolio.get('margin_requirement', 0.0):.2f}",
            "total_margin_used": f"{portfolio.get('margin_used', 0.0):.2f}",
        }
    )
    # print("the return result :", result)
    return result
