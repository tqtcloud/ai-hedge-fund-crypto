"""
逻辑一致性验证器

验证交易参数之间的逻辑一致性，包括：
- 交易方向与止损止盈的关系
- 杠杆与波动率的匹配
- 仓位规模与账户平衡
- 时间框架信号一致性
"""

from typing import Dict, Any, List, Optional
import math
from .base_validator import BaseValidator, ValidationResult, ValidationSeverity, ValidationMixin


class LogicalConsistencyValidator(BaseValidator, ValidationMixin):
    """
    逻辑一致性验证器
    
    验证交易参数之间是否存在逻辑冲突和不一致
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化逻辑一致性验证器
        
        Args:
            config: 验证器配置，包含逻辑约束规则
        """
        super().__init__("LogicalConsistencyValidator", config)
        
        # 从配置中提取逻辑约束
        self.constraints = config.get("logical_constraints", {})
    
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """
        执行逻辑一致性验证
        
        Args:
            data: 待验证的交易数据
            context: 验证上下文信息
            
        Returns:
            验证结果列表
        """
        if not self.is_enabled():
            return []
        
        results = []
        
        # 验证交易方向与止损止盈逻辑
        results.extend(self._validate_direction_stop_logic(data, context))
        
        # 验证杠杆与波动率关系
        results.extend(self._validate_leverage_volatility(data, context))
        
        # 验证仓位与账户平衡
        results.extend(self._validate_position_account_balance(data, context))
        
        # 验证时间框架一致性
        results.extend(self._validate_timeframe_consistency(data, context))
        
        # 验证风险收益比一致性
        results.extend(self._validate_risk_reward_consistency(data, context))
        
        # 验证价格关系逻辑
        results.extend(self._validate_price_relationships(data, context))
        
        # 验证信号强度与置信度一致性
        results.extend(self._validate_signal_confidence_consistency(data, context))
        
        # 记录验证结果
        for result in results:
            self.log_validation_result(result)
        
        return results
    
    def _validate_direction_stop_logic(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证交易方向与止损止盈的逻辑关系"""
        results = []
        
        direction = self._get_nested_value(data, "basic_params.direction")
        current_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.current_price"))
        stop_loss = self._safe_float_conversion(self._get_nested_value(data, "risk_management.stop_loss"))
        take_profit = self._safe_float_conversion(self._get_nested_value(data, "risk_management.take_profit"))
        entry_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.entry_price_target", current_price))
        
        if not all([direction, current_price > 0, stop_loss > 0, take_profit > 0]):
            return results  # 缺少必要数据，跳过验证
        
        direction_config = self.constraints.get("direction_stop_logic", {})
        
        if direction == "long":
            # 多头逻辑检查
            if stop_loss >= entry_price:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="risk_management.stop_loss",
                        message=f"多头止损价格错误: 止损 {stop_loss} >= 入场价 {entry_price}",
                        current_value=stop_loss,
                        suggestion=f"多头止损应低于入场价，建议设置为 {entry_price * 0.95:.4f}",
                        corrected_value=entry_price * 0.95,
                        context={"direction": direction, "entry_price": entry_price}
                    )
                )
            
            if take_profit <= entry_price:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="risk_management.take_profit",
                        message=f"多头止盈价格错误: 止盈 {take_profit} <= 入场价 {entry_price}",
                        current_value=take_profit,
                        suggestion=f"多头止盈应高于入场价，建议设置为 {entry_price * 1.05:.4f}",
                        corrected_value=entry_price * 1.05,
                        context={"direction": direction, "entry_price": entry_price}
                    )
                )
        
        elif direction == "short":
            # 空头逻辑检查
            if stop_loss <= entry_price:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="risk_management.stop_loss",
                        message=f"空头止损价格错误: 止损 {stop_loss} <= 入场价 {entry_price}",
                        current_value=stop_loss,
                        suggestion=f"空头止损应高于入场价，建议设置为 {entry_price * 1.05:.4f}",
                        corrected_value=entry_price * 1.05,
                        context={"direction": direction, "entry_price": entry_price}
                    )
                )
            
            if take_profit >= entry_price:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="risk_management.take_profit",
                        message=f"空头止盈价格错误: 止盈 {take_profit} >= 入场价 {entry_price}",
                        current_value=take_profit,
                        suggestion=f"空头止盈应低于入场价，建议设置为 {entry_price * 0.95:.4f}",
                        corrected_value=entry_price * 0.95,
                        context={"direction": direction, "entry_price": entry_price}
                    )
                )
        
        return results
    
    def _validate_leverage_volatility(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证杠杆与波动率的匹配关系"""
        results = []
        
        leverage = self._safe_float_conversion(self._get_nested_value(data, "basic_params.leverage"))
        
        # 从技术分析数据或上下文获取波动率
        volatility = None
        if context and "volatility" in context:
            volatility = self._safe_float_conversion(context["volatility"])
        else:
            # 尝试从数据中获取波动率信息
            volatility_data = self._get_nested_value(data, "technical_risk_assessment.volatility_risk")
            if volatility_data == "high":
                volatility = 0.08
            elif volatility_data == "moderate":
                volatility = 0.04
            elif volatility_data == "low":
                volatility = 0.02
            elif volatility_data == "extreme":
                volatility = 0.15
        
        if leverage <= 0 or volatility is None:
            return results
        
        leverage_config = self.constraints.get("leverage_volatility", {})
        high_vol_max_leverage = leverage_config.get("high_volatility_max_leverage", 10)
        volatility_threshold = leverage_config.get("volatility_threshold", 0.05)
        
        if volatility >= volatility_threshold and leverage > high_vol_max_leverage:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="basic_params.leverage",
                    message=f"高波动率环境下杠杆过高: 杠杆 {leverage}x，波动率 {volatility:.1%}",
                    current_value=leverage,
                    expected_range={"max_safe_leverage": high_vol_max_leverage},
                    suggestion=f"高波动率 ({volatility:.1%}) 时建议杠杆不超过 {high_vol_max_leverage}x",
                    corrected_value=high_vol_max_leverage,
                    context={"volatility": volatility, "threshold": volatility_threshold}
                )
            )
        
        # 根据波动率给出杠杆建议
        if volatility > 0:
            recommended_leverage = min(leverage, max(1, int(0.02 / volatility)))
            if leverage > recommended_leverage * 2:
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name="basic_params.leverage",
                        message=f"当前波动率下杠杆偏高: 杠杆 {leverage}x，波动率 {volatility:.1%}",
                        current_value=leverage,
                        suggestion=f"基于当前波动率，建议杠杆 {recommended_leverage}x 以下",
                        corrected_value=recommended_leverage,
                        context={"volatility": volatility}
                    )
                )
        
        return results
    
    def _validate_position_account_balance(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证仓位规模与账户平衡"""
        results = []
        
        position_size = self._safe_float_conversion(self._get_nested_value(data, "basic_params.position_size"))
        position_ratio = self._safe_float_conversion(self._get_nested_value(data, "basic_params.position_ratio"))
        
        # 从上下文获取账户信息
        account_balance = None
        if context:
            account_balance = self._safe_float_conversion(context.get("account_balance"))
            # 或从风险管理数据获取
            portfolio_value = self._safe_float_conversion(context.get("portfolio_value"))
            if portfolio_value > 0:
                account_balance = portfolio_value
        
        if position_size <= 0 or position_ratio <= 0:
            return results
        
        position_config = self.constraints.get("position_account_balance", {})
        max_single_ratio = position_config.get("max_single_position_ratio", 0.3)
        total_exposure_ratio = position_config.get("total_exposure_ratio", 0.8)
        
        # 检查单笔交易占账户比例
        if position_ratio > max_single_ratio:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="basic_params.position_ratio",
                    message=f"单笔交易仓位过大: {position_ratio:.1%}，超过最大比例 {max_single_ratio:.1%}",
                    current_value=position_ratio,
                    expected_range={"max_ratio": max_single_ratio},
                    suggestion=f"建议降低仓位至 {max_single_ratio:.1%} 以下",
                    corrected_value=max_single_ratio,
                    context={"max_single_ratio": max_single_ratio}
                )
            )
        
        # 检查仓位大小与账户余额的一致性
        if account_balance and account_balance > 0:
            calculated_ratio = position_size / account_balance
            ratio_difference = abs(calculated_ratio - position_ratio)
            
            if ratio_difference > 0.01:  # 1%的误差容忍度
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="basic_params.position_ratio",
                        message=f"仓位比例计算不一致: 声明 {position_ratio:.1%}，实际 {calculated_ratio:.1%}",
                        current_value=position_ratio,
                        suggestion="请检查仓位大小和账户余额的计算",
                        corrected_value=calculated_ratio,
                        context={
                            "position_size": position_size,
                            "account_balance": account_balance,
                            "calculated_ratio": calculated_ratio
                        }
                    )
                )
        
        return results
    
    def _validate_timeframe_consistency(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证时间框架一致性"""
        results = []
        
        # 获取时间框架分析数据
        consensus_score = self._safe_float_conversion(
            self._get_nested_value(data, "timeframe_analysis.consensus_score")
        )
        conflicting_signals = self._safe_int_conversion(
            self._get_nested_value(data, "timeframe_analysis.conflicting_signals")
        )
        signal_alignment = self._get_nested_value(data, "timeframe_analysis.signal_alignment")
        
        timeframe_config = self.constraints.get("timeframe_consistency", {})
        min_consensus_score = timeframe_config.get("min_consensus_score", 0.6)
        max_conflicting_signals = timeframe_config.get("conflicting_signals_limit", 2)
        
        # 检查共识分数
        if consensus_score > 0 and consensus_score < min_consensus_score:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="timeframe_analysis.consensus_score",
                    message=f"时间框架共识度偏低: {consensus_score:.1%}，低于建议值 {min_consensus_score:.1%}",
                    current_value=consensus_score,
                    expected_range={"min_consensus": min_consensus_score},
                    suggestion="低共识度信号建议谨慎交易，等待更强一致性",
                    context={"min_consensus_score": min_consensus_score}
                )
            )
        
        # 检查冲突信号数量
        if conflicting_signals > max_conflicting_signals:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="timeframe_analysis.conflicting_signals",
                    message=f"冲突信号过多: {conflicting_signals} 个，超过建议上限 {max_conflicting_signals}",
                    current_value=conflicting_signals,
                    expected_range={"max_conflicts": max_conflicting_signals},
                    suggestion="多重冲突信号表明市场方向不明确，建议等待明确信号",
                    context={"max_conflicting_signals": max_conflicting_signals}
                )
            )
        
        # 检查信号对齐程度
        if signal_alignment == "weak":
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="timeframe_analysis.signal_alignment",
                    message="时间框架信号对齐程度较弱",
                    current_value=signal_alignment,
                    suggestion="弱信号对齐建议降低仓位或等待更强信号",
                    context={"signal_alignment": signal_alignment}
                )
            )
        
        return results
    
    def _validate_risk_reward_consistency(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证风险收益比一致性"""
        results = []
        
        current_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.current_price"))
        stop_loss = self._safe_float_conversion(self._get_nested_value(data, "risk_management.stop_loss"))
        take_profit = self._safe_float_conversion(self._get_nested_value(data, "risk_management.take_profit"))
        stated_rrr = self._safe_float_conversion(self._get_nested_value(data, "risk_management.risk_reward_ratio"))
        direction = self._get_nested_value(data, "basic_params.direction")
        
        if not all([current_price > 0, stop_loss > 0, take_profit > 0, stated_rrr > 0, direction]):
            return results
        
        # 根据方向计算实际风险收益比
        if direction == "long":
            risk = current_price - stop_loss
            reward = take_profit - current_price
        elif direction == "short":
            risk = stop_loss - current_price
            reward = current_price - take_profit
        else:
            return results
        
        if risk <= 0 or reward <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="risk_management.risk_reward_ratio",
                    message=f"风险或收益计算异常: 风险 {risk:.4f}，收益 {reward:.4f}",
                    current_value=stated_rrr,
                    suggestion="请检查止损止盈设置是否正确",
                    context={
                        "calculated_risk": risk,
                        "calculated_reward": reward,
                        "direction": direction
                    }
                )
            )
            return results
        
        calculated_rrr = reward / risk
        rrr_difference = abs(calculated_rrr - stated_rrr)
        
        # 允许5%的误差容忍度
        if rrr_difference > stated_rrr * 0.05:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="risk_management.risk_reward_ratio",
                    message=f"风险收益比计算不一致: 声明 {stated_rrr:.2f}，实际 {calculated_rrr:.2f}",
                    current_value=stated_rrr,
                    suggestion="请检查风险收益比计算是否准确",
                    corrected_value=calculated_rrr,
                    context={
                        "calculated_rrr": calculated_rrr,
                        "risk": risk,
                        "reward": reward,
                        "price_levels": {
                            "current": current_price,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit
                        }
                    }
                )
            )
        
        return results
    
    def _validate_price_relationships(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证价格关系的逻辑性"""
        results = []
        
        current_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.current_price"))
        entry_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.entry_price_target"))
        liquidation_price = self._safe_float_conversion(self._get_nested_value(data, "risk_management.liquidation_price"))
        break_even_price = self._safe_float_conversion(self._get_nested_value(data, "cost_benefit_analysis.break_even_price"))
        
        if current_price <= 0:
            return results
        
        # 入场价与当前价的合理性检查
        if entry_price > 0:
            price_diff_pct = abs(entry_price - current_price) / current_price
            if price_diff_pct > 0.05:  # 超过5%差异给出警告
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name="basic_params.entry_price_target",
                        message=f"入场价与当前价差异较大: {price_diff_pct:.1%}",
                        current_value=entry_price,
                        suggestion="较大的价格差异可能导致订单难以成交",
                        context={
                            "current_price": current_price,
                            "price_difference_pct": price_diff_pct
                        }
                    )
                )
        
        # 强平价格合理性检查
        if liquidation_price > 0:
            direction = self._get_nested_value(data, "basic_params.direction")
            if direction == "long" and liquidation_price >= current_price:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="risk_management.liquidation_price",
                        message=f"多头强平价格异常: {liquidation_price} >= 当前价格 {current_price}",
                        current_value=liquidation_price,
                        suggestion="多头强平价格应低于当前价格",
                        context={"direction": direction, "current_price": current_price}
                    )
                )
            elif direction == "short" and liquidation_price <= current_price:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="risk_management.liquidation_price",
                        message=f"空头强平价格异常: {liquidation_price} <= 当前价格 {current_price}",
                        current_value=liquidation_price,
                        suggestion="空头强平价格应高于当前价格",
                        context={"direction": direction, "current_price": current_price}
                    )
                )
        
        # 盈亏平衡价格合理性检查
        if break_even_price > 0:
            be_diff_pct = abs(break_even_price - current_price) / current_price
            if be_diff_pct > 0.1:  # 超过10%差异可能有问题
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.break_even_price",
                        message=f"盈亏平衡价格与当前价差异较大: {be_diff_pct:.1%}",
                        current_value=break_even_price,
                        suggestion="盈亏平衡价格差异过大可能表明成本过高",
                        context={
                            "current_price": current_price,
                            "difference_pct": be_diff_pct
                        }
                    )
                )
        
        return results
    
    def _validate_signal_confidence_consistency(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证信号强度与置信度的一致性"""
        results = []
        
        overall_confidence = self._safe_float_conversion(
            self._get_nested_value(data, "decision_metadata.confidence")
        )
        signal_alignment = self._get_nested_value(data, "timeframe_analysis.signal_alignment")
        consensus_score = self._safe_float_conversion(
            self._get_nested_value(data, "timeframe_analysis.consensus_score")
        )
        
        if overall_confidence <= 0:
            return results
        
        # 检查置信度与信号对齐程度的一致性
        if signal_alignment == "strong" and overall_confidence < 70:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="decision_metadata.confidence",
                    message=f"信号对齐强烈但置信度偏低: 对齐 {signal_alignment}，置信度 {overall_confidence}",
                    current_value=overall_confidence,
                    suggestion="强信号对齐通常对应更高置信度",
                    context={"signal_alignment": signal_alignment}
                )
            )
        elif signal_alignment == "weak" and overall_confidence > 80:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="decision_metadata.confidence",
                    message=f"信号对齐较弱但置信度过高: 对齐 {signal_alignment}，置信度 {overall_confidence}",
                    current_value=overall_confidence,
                    suggestion="弱信号对齐情况下置信度不应过高",
                    context={"signal_alignment": signal_alignment}
                )
            )
        
        # 检查置信度与共识分数的一致性
        if consensus_score > 0:
            expected_confidence_range = (consensus_score * 60, consensus_score * 100)  # 基于共识分数的期望置信度范围
            if overall_confidence < expected_confidence_range[0]:
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        field_name="decision_metadata.confidence",
                        message=f"置信度低于共识分数预期: 置信度 {overall_confidence}，共识 {consensus_score:.1%}",
                        current_value=overall_confidence,
                        suggestion="考虑根据时间框架共识调整置信度评估",
                        context={
                            "consensus_score": consensus_score,
                            "expected_range": expected_confidence_range
                        }
                    )
                )
        
        return results