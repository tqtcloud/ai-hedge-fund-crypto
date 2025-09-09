"""
风险约束验证器

验证交易是否符合风险管理要求，包括：
- 保证金使用率控制
- 强平距离安全性
- VaR限制
- 最大回撤约束
- 集中度风险控制
"""

from typing import Dict, Any, List, Optional
import math
from .base_validator import BaseValidator, ValidationResult, ValidationSeverity, ValidationMixin


class RiskConstraintValidator(BaseValidator, ValidationMixin):
    """
    风险约束验证器
    
    验证各种风险指标是否在安全范围内
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化风险约束验证器
        
        Args:
            config: 验证器配置，包含风险约束规则
        """
        super().__init__("RiskConstraintValidator", config)
        
        # 从配置中提取风险约束
        self.constraints = config.get("risk_constraints", {})
    
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """
        执行风险约束验证
        
        Args:
            data: 待验证的交易数据
            context: 验证上下文信息
            
        Returns:
            验证结果列表
        """
        if not self.is_enabled():
            return []
        
        results = []
        
        # 验证保证金使用率
        results.extend(self._validate_margin_utilization(data, context))
        
        # 验证强平距离
        results.extend(self._validate_liquidation_distance(data, context))
        
        # 验证VaR限制
        results.extend(self._validate_var_limits(data, context))
        
        # 验证最大回撤
        results.extend(self._validate_drawdown_limits(data, context))
        
        # 验证相关性风险
        results.extend(self._validate_correlation_risk(data, context))
        
        # 验证集中度风险
        results.extend(self._validate_concentration_risk(data, context))
        
        # 验证仓位风险控制
        results.extend(self._validate_position_risk_control(data, context))
        
        # 验证动态风险指标
        results.extend(self._validate_dynamic_risk_metrics(data, context))
        
        # 记录验证结果
        for result in results:
            self.log_validation_result(result)
        
        return results
    
    def _validate_margin_utilization(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证保证金使用率"""
        results = []
        
        margin_utilization = self._safe_float_conversion(
            self._get_nested_value(data, "margin_management.margin_utilization")
        )
        available_margin = self._safe_float_conversion(
            self._get_nested_value(data, "margin_management.available_margin")
        )
        initial_margin = self._safe_float_conversion(
            self._get_nested_value(data, "margin_management.initial_margin")
        )
        
        margin_config = self.constraints.get("margin_utilization", {})
        warning_level = margin_config.get("warning_level", 0.7)
        critical_level = margin_config.get("critical_level", 0.85)
        emergency_level = margin_config.get("emergency_level", 0.95)
        
        if margin_utilization <= 0:
            return results
        
        # 检查保证金使用率等级
        if margin_utilization >= emergency_level:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    field_name="margin_management.margin_utilization",
                    message=f"保证金使用率极高: {margin_utilization:.1%}，面临强平风险",
                    current_value=margin_utilization,
                    expected_range={
                        "warning": warning_level,
                        "critical": critical_level,
                        "emergency": emergency_level
                    },
                    suggestion="立即降低仓位或增加保证金，避免强制平仓",
                    corrected_value=warning_level
                )
            )
        elif margin_utilization >= critical_level:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="margin_management.margin_utilization",
                    message=f"保证金使用率过高: {margin_utilization:.1%}，存在强平风险",
                    current_value=margin_utilization,
                    expected_range={
                        "warning": warning_level,
                        "critical": critical_level
                    },
                    suggestion="建议降低仓位或增加保证金至安全水平",
                    corrected_value=warning_level
                )
            )
        elif margin_utilization >= warning_level:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="margin_management.margin_utilization",
                    message=f"保证金使用率偏高: {margin_utilization:.1%}",
                    current_value=margin_utilization,
                    suggestion="建议密切监控保证金水平，准备追加保证金"
                )
            )
        
        # 检查可用保证金充足性
        if available_margin <= initial_margin * 0.2:  # 可用保证金低于初始保证金的20%
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="margin_management.available_margin",
                    message=f"可用保证金不足: {available_margin:.2f}，建议保留更多缓冲",
                    current_value=available_margin,
                    suggestion="增加账户资金或减少仓位以提高可用保证金",
                    context={
                        "initial_margin": initial_margin,
                        "buffer_ratio": available_margin / initial_margin if initial_margin > 0 else 0
                    }
                )
            )
        
        return results
    
    def _validate_liquidation_distance(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证强平距离"""
        results = []
        
        current_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.current_price"))
        liquidation_price = self._safe_float_conversion(self._get_nested_value(data, "risk_management.liquidation_price"))
        direction = self._get_nested_value(data, "basic_params.direction")
        
        distance_config = self.constraints.get("liquidation_distance", {})
        min_safe_distance = distance_config.get("min_safe_distance", 15.0)
        warning_distance = distance_config.get("warning_distance", 10.0)
        critical_distance = distance_config.get("critical_distance", 5.0)
        
        if not all([current_price > 0, liquidation_price > 0, direction]):
            return results
        
        # 计算强平距离
        if direction == "long":
            distance_pct = (current_price - liquidation_price) / current_price * 100
        elif direction == "short":
            distance_pct = (liquidation_price - current_price) / current_price * 100
        else:
            return results
        
        if distance_pct <= critical_distance:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    field_name="risk_management.liquidation_price",
                    message=f"强平距离极近: {distance_pct:.1f}%，面临立即强平风险",
                    current_value=liquidation_price,
                    expected_range={
                        "min_safe": min_safe_distance,
                        "warning": warning_distance,
                        "critical": critical_distance
                    },
                    suggestion="立即降低杠杆或平仓避免强制平仓",
                    context={
                        "distance_pct": distance_pct,
                        "current_price": current_price,
                        "direction": direction
                    }
                )
            )
        elif distance_pct <= warning_distance:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="risk_management.liquidation_price",
                    message=f"强平距离过近: {distance_pct:.1f}%，存在强平风险",
                    current_value=liquidation_price,
                    expected_range={
                        "min_safe": min_safe_distance,
                        "warning": warning_distance
                    },
                    suggestion="建议降低杠杆或增加保证金以增加强平距离",
                    context={
                        "distance_pct": distance_pct,
                        "current_price": current_price,
                        "direction": direction
                    }
                )
            )
        elif distance_pct < min_safe_distance:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="risk_management.liquidation_price",
                    message=f"强平距离偏近: {distance_pct:.1f}%，低于安全水平",
                    current_value=liquidation_price,
                    suggestion="考虑适当降低杠杆以增加安全边际",
                    context={
                        "distance_pct": distance_pct,
                        "min_safe_distance": min_safe_distance
                    }
                )
            )
        
        return results
    
    def _validate_var_limits(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证VaR限制"""
        results = []
        
        daily_var = self._safe_float_conversion(
            self._get_nested_value(data, "dynamic_risk_metrics.var_1day")
        )
        weekly_var = self._safe_float_conversion(
            self._get_nested_value(data, "dynamic_risk_metrics.var_7day")
        )
        
        # 从上下文获取账户价值
        account_value = None
        if context:
            account_value = self._safe_float_conversion(context.get("account_balance", 0))
            if account_value <= 0:
                account_value = self._safe_float_conversion(context.get("portfolio_value", 0))
        
        var_config = self.constraints.get("var_limits", {})
        daily_var_ratio = var_config.get("daily_var_ratio", 0.02)
        weekly_var_ratio = var_config.get("weekly_var_ratio", 0.05)
        
        if account_value and account_value > 0:
            # 验证日VaR
            if daily_var > 0:
                daily_var_pct = daily_var / account_value
                if daily_var_pct > daily_var_ratio:
                    results.append(
                        self._create_result(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field_name="dynamic_risk_metrics.var_1day",
                            message=f"日VaR超限: {daily_var_pct:.1%}，超过限制 {daily_var_ratio:.1%}",
                            current_value=daily_var,
                            expected_range={"max_ratio": daily_var_ratio},
                            suggestion="降低仓位或杠杆以控制日风险价值",
                            context={
                                "account_value": account_value,
                                "var_ratio": daily_var_pct
                            }
                        )
                    )
            
            # 验证周VaR
            if weekly_var > 0:
                weekly_var_pct = weekly_var / account_value
                if weekly_var_pct > weekly_var_ratio:
                    results.append(
                        self._create_result(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field_name="dynamic_risk_metrics.var_7day",
                            message=f"周VaR超限: {weekly_var_pct:.1%}，超过限制 {weekly_var_ratio:.1%}",
                            current_value=weekly_var,
                            expected_range={"max_ratio": weekly_var_ratio},
                            suggestion="调整投资组合配置以控制周风险价值",
                            context={
                                "account_value": account_value,
                                "var_ratio": weekly_var_pct
                            }
                        )
                    )
        
        return results
    
    def _validate_drawdown_limits(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证最大回撤限制"""
        results = []
        
        max_drawdown = self._safe_float_conversion(
            self._get_nested_value(data, "dynamic_risk_metrics.maximum_drawdown")
        )
        
        # 从上下文获取当前回撤情况
        current_drawdown = None
        if context:
            current_drawdown = self._safe_float_conversion(context.get("current_drawdown", 0))
        
        drawdown_config = self.constraints.get("drawdown_limits", {})
        warning_drawdown = drawdown_config.get("warning_drawdown", 0.05)
        max_drawdown_limit = drawdown_config.get("max_drawdown", 0.15)
        
        # 验证预期最大回撤
        if max_drawdown > max_drawdown_limit:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="dynamic_risk_metrics.maximum_drawdown",
                    message=f"预期最大回撤过高: {max_drawdown:.1%}，超过限制 {max_drawdown_limit:.1%}",
                    current_value=max_drawdown,
                    expected_range={"max_limit": max_drawdown_limit},
                    suggestion="降低仓位或风险敞口以控制最大回撤",
                    corrected_value=max_drawdown_limit
                )
            )
        elif max_drawdown > warning_drawdown:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="dynamic_risk_metrics.maximum_drawdown",
                    message=f"预期最大回撤偏高: {max_drawdown:.1%}",
                    current_value=max_drawdown,
                    suggestion="考虑优化风险管理策略以降低回撤风险"
                )
            )
        
        # 验证当前回撤情况
        if current_drawdown is not None and current_drawdown > 0:
            if current_drawdown >= max_drawdown_limit:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.CRITICAL,
                        field_name="portfolio.current_drawdown",
                        message=f"当前回撤达到限制: {current_drawdown:.1%}",
                        current_value=current_drawdown,
                        suggestion="立即评估并调整策略，考虑暂停交易",
                        context={"max_limit": max_drawdown_limit}
                    )
                )
            elif current_drawdown >= warning_drawdown:
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name="portfolio.current_drawdown",
                        message=f"当前回撤偏高: {current_drawdown:.1%}",
                        current_value=current_drawdown,
                        suggestion="密切监控回撤情况，准备调整策略"
                    )
                )
        
        return results
    
    def _validate_correlation_risk(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证相关性风险"""
        results = []
        
        diversification_score = self._safe_float_conversion(
            self._get_nested_value(data, "position_risk_control.diversification_score")
        )
        
        correlation_config = self.constraints.get("correlation_limits", {})
        max_correlation = correlation_config.get("max_correlation", 0.8)
        min_diversification = correlation_config.get("diversification_min_score", 0.3)
        
        # 验证分散化评分
        if diversification_score > 0 and diversification_score < min_diversification:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="position_risk_control.diversification_score",
                    message=f"投资组合分散化不足: {diversification_score:.2f}，低于最小要求 {min_diversification:.2f}",
                    current_value=diversification_score,
                    expected_range={"min_score": min_diversification},
                    suggestion="增加不同资产或策略的配置以提高分散化",
                    corrected_value=min_diversification
                )
            )
        
        # 从上下文获取相关性信息
        if context and "correlations" in context:
            correlations = context["correlations"]
            high_correlations = [
                (pair, corr) for pair, corr in correlations.items() 
                if abs(corr) > max_correlation
            ]
            
            if high_correlations:
                correlation_pairs = ", ".join([f"{pair}: {corr:.2f}" for pair, corr in high_correlations[:3]])
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="portfolio.correlations",
                        message=f"发现高相关性资产对: {correlation_pairs}",
                        suggestion="考虑减少高相关性资产的同时持仓",
                        context={"high_correlations": high_correlations}
                    )
                )
        
        return results
    
    def _validate_concentration_risk(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证集中度风险"""
        results = []
        
        position_concentration = self._safe_float_conversion(
            self._get_nested_value(data, "position_risk_control.position_concentration")
        )
        position_ratio = self._safe_float_conversion(
            self._get_nested_value(data, "basic_params.position_ratio")
        )
        
        concentration_config = self.constraints.get("concentration_limits", {})
        single_asset_max = concentration_config.get("single_asset_max_weight", 0.4)
        top3_assets_max = concentration_config.get("top3_assets_max_weight", 0.7)
        
        # 验证单一资产集中度
        if position_ratio > single_asset_max:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="basic_params.position_ratio",
                    message=f"单一资产权重过高: {position_ratio:.1%}，超过限制 {single_asset_max:.1%}",
                    current_value=position_ratio,
                    expected_range={"max_weight": single_asset_max},
                    suggestion="降低单一资产的权重以减少集中度风险",
                    corrected_value=single_asset_max
                )
            )
        
        # 验证整体集中度
        if position_concentration > 0 and position_concentration > single_asset_max:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="position_risk_control.position_concentration",
                    message=f"投资组合集中度过高: {position_concentration:.1%}",
                    current_value=position_concentration,
                    suggestion="增加资产配置的多样性以降低集中度风险"
                )
            )
        
        # 从上下文获取前几大持仓信息
        if context and "top_positions" in context:
            top_positions = context["top_positions"]
            if isinstance(top_positions, list) and len(top_positions) >= 3:
                top3_weight = sum(pos.get("weight", 0) for pos in top_positions[:3])
                if top3_weight > top3_assets_max:
                    results.append(
                        self._create_result(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field_name="portfolio.top3_concentration",
                            message=f"前3大资产权重过高: {top3_weight:.1%}，超过限制 {top3_assets_max:.1%}",
                            current_value=top3_weight,
                            expected_range={"max_weight": top3_assets_max},
                            suggestion="平衡前几大持仓的权重分配",
                            context={"top3_positions": top_positions[:3]}
                        )
                    )
        
        return results
    
    def _validate_position_risk_control(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证仓位风险控制"""
        results = []
        
        max_position_size = self._safe_float_conversion(
            self._get_nested_value(data, "position_risk_control.max_position_size")
        )
        risk_per_trade = self._safe_float_conversion(
            self._get_nested_value(data, "position_risk_control.risk_per_trade")
        )
        max_daily_risk = self._safe_float_conversion(
            self._get_nested_value(data, "position_risk_control.max_daily_risk")
        )
        current_position_size = self._safe_float_conversion(
            self._get_nested_value(data, "basic_params.position_size")
        )
        
        # 验证仓位大小是否超过限制
        if max_position_size > 0 and current_position_size > max_position_size:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="basic_params.position_size",
                    message=f"仓位大小超过限制: {current_position_size:.2f}，最大允许 {max_position_size:.2f}",
                    current_value=current_position_size,
                    expected_range={"max_size": max_position_size},
                    suggestion="降低仓位大小至允许范围内",
                    corrected_value=max_position_size
                )
            )
        
        # 验证单笔交易风险
        if risk_per_trade > 0.05:  # 单笔交易风险超过5%
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="position_risk_control.risk_per_trade",
                    message=f"单笔交易风险过高: {risk_per_trade:.1%}，建议不超过5%",
                    current_value=risk_per_trade,
                    suggestion="降低仓位大小或调整止损以减少单笔交易风险",
                    corrected_value=0.03  # 建议3%风险
                )
            )
        elif risk_per_trade > 0.03:  # 3-5%之间给出警告
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="position_risk_control.risk_per_trade",
                    message=f"单笔交易风险偏高: {risk_per_trade:.1%}",
                    current_value=risk_per_trade,
                    suggestion="建议将单笔交易风险控制在2-3%以内"
                )
            )
        
        # 验证日最大风险
        if max_daily_risk > 0.1:  # 日最大风险超过10%
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="position_risk_control.max_daily_risk",
                    message=f"日最大风险过高: {max_daily_risk:.1%}，建议不超过10%",
                    current_value=max_daily_risk,
                    suggestion="限制日内交易频率和单笔风险",
                    corrected_value=0.08
                )
            )
        
        return results
    
    def _validate_dynamic_risk_metrics(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证动态风险指标"""
        results = []
        
        expected_shortfall = self._safe_float_conversion(
            self._get_nested_value(data, "dynamic_risk_metrics.expected_shortfall")
        )
        sharpe_impact = self._safe_float_conversion(
            self._get_nested_value(data, "dynamic_risk_metrics.sharpe_ratio_impact")
        )
        risk_adjusted_return = self._safe_float_conversion(
            self._get_nested_value(data, "dynamic_risk_metrics.risk_adjusted_return")
        )
        
        # 验证期望损失
        if expected_shortfall > 0.15:  # 期望损失超过15%
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="dynamic_risk_metrics.expected_shortfall",
                    message=f"期望损失较高: {expected_shortfall:.1%}",
                    current_value=expected_shortfall,
                    suggestion="考虑优化风险管理策略以降低尾部风险"
                )
            )
        
        # 验证夏普比率影响
        if sharpe_impact < -0.1:  # 对夏普比率的负面影响超过-0.1
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="dynamic_risk_metrics.sharpe_ratio_impact",
                    message=f"对夏普比率有较大负面影响: {sharpe_impact:.3f}",
                    current_value=sharpe_impact,
                    suggestion="该交易可能降低组合的风险调整收益，请谨慎考虑"
                )
            )
        
        # 验证风险调整收益
        if risk_adjusted_return < 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="dynamic_risk_metrics.risk_adjusted_return",
                    message=f"风险调整收益为负: {risk_adjusted_return:.1%}",
                    current_value=risk_adjusted_return,
                    suggestion="负的风险调整收益表明该交易不划算，建议重新评估"
                )
            )
        elif risk_adjusted_return < 0.05:  # 风险调整收益低于5%
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="dynamic_risk_metrics.risk_adjusted_return",
                    message=f"风险调整收益偏低: {risk_adjusted_return:.1%}",
                    current_value=risk_adjusted_return,
                    suggestion="考虑寻找风险调整收益更高的交易机会"
                )
            )
        
        return results