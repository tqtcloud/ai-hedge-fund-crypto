"""
数值范围验证器

验证数值字段是否在合理范围内，包括杠杆倍数、置信度、价格等。
支持自动修正和详细的错误提示。
"""

from typing import Dict, Any, List, Optional, Tuple
import math
from .base_validator import BaseValidator, ValidationResult, ValidationSeverity, ValidationMixin


class NumericalRangeValidator(BaseValidator, ValidationMixin):
    """
    数值范围验证器
    
    验证各种数值字段是否在预定义的合理范围内
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数值范围验证器
        
        Args:
            config: 验证器配置，包含各字段的数值约束
        """
        super().__init__("NumericalRangeValidator", config)
        
        # 从配置中提取数值约束
        self.constraints = config.get("numerical_constraints", {})
        
        # 字段验证映射表
        self.field_validators = {
            "basic_params.leverage": self._validate_leverage,
            "decision_metadata.confidence": self._validate_confidence,
            "basic_params.current_price": self._validate_price,
            "basic_params.position_size": self._validate_position_size,
            "basic_params.position_ratio": self._validate_position_ratio,
            "risk_management.risk_reward_ratio": self._validate_risk_reward_ratio,
            "risk_management.stop_loss": self._validate_stop_loss,
            "risk_management.take_profit": self._validate_take_profit,
            "risk_management.risk_percentage": self._validate_risk_percentage,
            "margin_management.margin_utilization": self._validate_margin_utilization,
            "cost_benefit_analysis.expected_return": self._validate_expected_return,
            "cost_benefit_analysis.profit_probability": self._validate_probability,
            "cost_benefit_analysis.loss_probability": self._validate_probability,
        }
    
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """
        执行数值范围验证
        
        Args:
            data: 待验证的交易数据
            context: 验证上下文信息
            
        Returns:
            验证结果列表
        """
        if not self.is_enabled():
            return []
        
        results = []
        
        # 遍历所有字段验证器
        for field_path, validator_func in self.field_validators.items():
            value = self._get_nested_value(data, field_path)
            if value is not None:
                try:
                    field_results = validator_func(field_path, value, data, context)
                    results.extend(field_results)
                except Exception as e:
                    results.append(
                        self._create_result(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            field_name=field_path,
                            message=f"验证过程中发生异常: {str(e)}",
                            current_value=value,
                            suggestion="请检查数据格式是否正确"
                        )
                    )
        
        # 记录验证结果
        for result in results:
            self.log_validation_result(result)
        
        return results
    
    def _validate_leverage(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证杠杆倍数"""
        leverage_config = self.constraints.get("leverage", {})
        min_leverage = leverage_config.get("min", 1)
        max_leverage = leverage_config.get("max", 125)
        warning_threshold = leverage_config.get("warning_threshold", 50)
        conservative_max = leverage_config.get("conservative_max", 20)
        
        leverage = self._safe_float_conversion(value)
        results = []
        
        # 基本范围检查
        if leverage < min_leverage:
            corrected_value = min_leverage
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"杠杆倍数过低: {leverage}，最小值为 {min_leverage}",
                    current_value=leverage,
                    expected_range={"min": min_leverage, "max": max_leverage},
                    suggestion=f"建议将杠杆倍数调整为 {corrected_value}",
                    corrected_value=corrected_value
                )
            )
        elif leverage > max_leverage:
            corrected_value = max_leverage
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"杠杆倍数超限: {leverage}，最大值为 {max_leverage}",
                    current_value=leverage,
                    expected_range={"min": min_leverage, "max": max_leverage},
                    suggestion=f"建议将杠杆倍数调整为 {corrected_value}",
                    corrected_value=corrected_value
                )
            )
        
        # 高风险警告
        elif leverage > warning_threshold:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"杠杆倍数偏高: {leverage}，建议谨慎使用",
                    current_value=leverage,
                    suggestion=f"考虑降低到 {conservative_max} 以下以减少风险"
                )
            )
        
        # 波动率调整建议
        if context and "volatility" in context:
            volatility = context["volatility"]
            if volatility > 0.05 and leverage > 10:  # 高波动率时杠杆建议
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name=field_name,
                        message=f"高波动率环境下杠杆倍数建议降低: 当前 {leverage}，波动率 {volatility:.2%}",
                        current_value=leverage,
                        suggestion="在高波动率时期考虑使用更低的杠杆倍数"
                    )
                )
        
        return results
    
    def _validate_confidence(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证置信度"""
        confidence_config = self.constraints.get("confidence", {})
        min_confidence = confidence_config.get("min", 0)
        max_confidence = confidence_config.get("max", 100)
        reliable_threshold = confidence_config.get("reliable_threshold", 65)
        high_confidence = confidence_config.get("high_confidence", 85)
        
        confidence = self._safe_float_conversion(value)
        results = []
        
        # 基本范围检查
        if confidence < min_confidence:
            corrected_value = min_confidence
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"置信度过低: {confidence}，最小值为 {min_confidence}",
                    current_value=confidence,
                    expected_range={"min": min_confidence, "max": max_confidence},
                    suggestion=f"建议将置信度调整为 {corrected_value}",
                    corrected_value=corrected_value
                )
            )
        elif confidence > max_confidence:
            corrected_value = max_confidence
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"置信度超限: {confidence}，最大值为 {max_confidence}",
                    current_value=confidence,
                    expected_range={"min": min_confidence, "max": max_confidence},
                    suggestion=f"建议将置信度调整为 {corrected_value}",
                    corrected_value=corrected_value
                )
            )
        
        # 可信度评估
        elif confidence < reliable_threshold:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"置信度偏低: {confidence}，低于可信阈值 {reliable_threshold}",
                    current_value=confidence,
                    suggestion="低置信度信号建议谨慎交易或等待更强信号"
                )
            )
        elif confidence >= high_confidence:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    field_name=field_name,
                    message=f"高置信度信号: {confidence}",
                    current_value=confidence,
                    suggestion="高置信度信号，适合积极交易"
                )
            )
        
        return results
    
    def _validate_price(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证价格"""
        price_config = self.constraints.get("price", {})
        min_price = price_config.get("min", 0.000001)
        max_price = price_config.get("max", 1000000)
        
        price = self._safe_float_conversion(value)
        results = []
        
        if price <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"价格必须为正数: {price}",
                    current_value=price,
                    suggestion="请提供有效的正数价格"
                )
            )
        elif price < min_price:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"价格过低: {price}，最小值为 {min_price}",
                    current_value=price,
                    expected_range={"min": min_price, "max": max_price},
                    suggestion=f"请检查价格是否正确"
                )
            )
        elif price > max_price:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"价格异常高: {price}，超过常见范围 {max_price}",
                    current_value=price,
                    suggestion="请确认价格是否正确"
                )
            )
        
        return results
    
    def _validate_position_size(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证仓位大小"""
        position_config = self.constraints.get("position_size", {})
        min_value = position_config.get("min_value", 10.0)
        max_value = position_config.get("max_value", 1000000.0)
        
        position_size = self._safe_float_conversion(value)
        results = []
        
        if position_size <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"仓位大小必须为正数: {position_size}",
                    current_value=position_size,
                    suggestion="请提供有效的正数仓位大小"
                )
            )
        elif position_size < min_value:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"仓位大小过小: {position_size} USDT，最小值为 {min_value} USDT",
                    current_value=position_size,
                    expected_range={"min": min_value, "max": max_value},
                    suggestion=f"建议增加仓位至 {min_value} USDT 或以上",
                    corrected_value=min_value
                )
            )
        elif position_size > max_value:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"仓位大小过大: {position_size} USDT，建议最大值 {max_value} USDT",
                    current_value=position_size,
                    expected_range={"min": min_value, "max": max_value},
                    suggestion=f"考虑降低仓位至 {max_value} USDT 以下"
                )
            )
        
        return results
    
    def _validate_position_ratio(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证仓位比例"""
        position_config = self.constraints.get("position_size", {})
        min_ratio = position_config.get("min_ratio", 0.001)
        max_ratio = position_config.get("max_ratio", 0.95)
        
        ratio = self._safe_float_conversion(value)
        results = []
        
        if ratio < 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"仓位比例不能为负数: {ratio}",
                    current_value=ratio,
                    suggestion="请提供有效的正数仓位比例"
                )
            )
        elif ratio < min_ratio:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"仓位比例过小: {ratio:.3%}，最小建议值 {min_ratio:.3%}",
                    current_value=ratio,
                    expected_range={"min": min_ratio, "max": max_ratio},
                    suggestion="过小的仓位可能无法产生有意义的收益"
                )
            )
        elif ratio > max_ratio:
            corrected_value = max_ratio
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"仓位比例过大: {ratio:.3%}，最大安全值 {max_ratio:.3%}",
                    current_value=ratio,
                    expected_range={"min": min_ratio, "max": max_ratio},
                    suggestion=f"建议降低仓位比例至 {corrected_value:.3%}",
                    corrected_value=corrected_value
                )
            )
        elif ratio > 0.5:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"仓位比例较高: {ratio:.3%}，存在集中度风险",
                    current_value=ratio,
                    suggestion="高仓位比例增加了组合风险，请确认风险承受能力"
                )
            )
        
        return results
    
    def _validate_risk_reward_ratio(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证风险收益比"""
        rrr_config = self.constraints.get("risk_reward_ratio", {})
        min_ratio = rrr_config.get("min", 0.5)
        good_ratio = rrr_config.get("good", 1.5)
        excellent_ratio = rrr_config.get("excellent", 3.0)
        
        rrr = self._safe_float_conversion(value)
        results = []
        
        if rrr <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"风险收益比必须为正数: {rrr}",
                    current_value=rrr,
                    suggestion="请重新计算风险收益比"
                )
            )
        elif rrr < min_ratio:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"风险收益比过低: {rrr:.2f}，最小建议值 {min_ratio:.2f}",
                    current_value=rrr,
                    expected_range={"min": min_ratio, "good": good_ratio, "excellent": excellent_ratio},
                    suggestion=f"建议调整止损止盈位置以获得更好的风险收益比",
                    corrected_value=good_ratio
                )
            )
        elif rrr < good_ratio:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"风险收益比一般: {rrr:.2f}，可以优化",
                    current_value=rrr,
                    suggestion=f"建议目标风险收益比 {good_ratio:.2f} 以上"
                )
            )
        elif rrr >= excellent_ratio:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    field_name=field_name,
                    message=f"优秀的风险收益比: {rrr:.2f}",
                    current_value=rrr,
                    suggestion="风险收益比很好，适合交易"
                )
            )
        
        return results
    
    def _validate_stop_loss(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证止损价格"""
        stop_config = self.constraints.get("stop_distance", {})
        min_distance = stop_config.get("min_stop_loss", 0.5)
        max_distance = stop_config.get("max_stop_loss", 15.0)
        
        stop_loss = self._safe_float_conversion(value)
        current_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.current_price"))
        direction = self._get_nested_value(data, "basic_params.direction")
        
        results = []
        
        if stop_loss <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"止损价格必须为正数: {stop_loss}",
                    current_value=stop_loss,
                    suggestion="请提供有效的止损价格"
                )
            )
            return results
        
        if current_price > 0:
            # 计算止损距离
            if direction == "long":
                distance = (current_price - stop_loss) / current_price * 100
            elif direction == "short":
                distance = (stop_loss - current_price) / current_price * 100
            else:
                # 如果没有方向信息，跳过距离检查
                return results
            
            if distance < min_distance:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name=field_name,
                        message=f"止损距离过近: {distance:.2f}%，可能频繁触发",
                        current_value=stop_loss,
                        expected_range={"min_distance": min_distance, "max_distance": max_distance},
                        suggestion=f"建议止损距离至少 {min_distance}%"
                    )
                )
            elif distance > max_distance:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name=field_name,
                        message=f"止损距离过远: {distance:.2f}%，风险过大",
                        current_value=stop_loss,
                        expected_range={"min_distance": min_distance, "max_distance": max_distance},
                        suggestion=f"建议止损距离不超过 {max_distance}%"
                    )
                )
        
        return results
    
    def _validate_take_profit(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证止盈价格"""
        stop_config = self.constraints.get("stop_distance", {})
        min_distance = stop_config.get("min_take_profit", 0.8)
        max_distance = stop_config.get("max_take_profit", 50.0)
        
        take_profit = self._safe_float_conversion(value)
        current_price = self._safe_float_conversion(self._get_nested_value(data, "basic_params.current_price"))
        direction = self._get_nested_value(data, "basic_params.direction")
        
        results = []
        
        if take_profit <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"止盈价格必须为正数: {take_profit}",
                    current_value=take_profit,
                    suggestion="请提供有效的止盈价格"
                )
            )
            return results
        
        if current_price > 0:
            # 计算止盈距离
            if direction == "long":
                distance = (take_profit - current_price) / current_price * 100
            elif direction == "short":
                distance = (current_price - take_profit) / current_price * 100
            else:
                return results
            
            if distance < min_distance:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name=field_name,
                        message=f"止盈距离过近: {distance:.2f}%，盈利空间有限",
                        current_value=take_profit,
                        expected_range={"min_distance": min_distance, "max_distance": max_distance},
                        suggestion=f"建议止盈距离至少 {min_distance}%"
                    )
                )
            elif distance > max_distance:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name=field_name,
                        message=f"止盈距离过远: {distance:.2f}%，可能难以达到",
                        current_value=take_profit,
                        expected_range={"min_distance": min_distance, "max_distance": max_distance},
                        suggestion=f"建议止盈距离不超过 {max_distance}%"
                    )
                )
        
        return results
    
    def _validate_risk_percentage(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证风险百分比"""
        risk_percentage = self._safe_float_conversion(value) * 100  # 转换为百分比
        results = []
        
        if risk_percentage <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"风险百分比必须为正数: {risk_percentage:.2f}%",
                    current_value=value,
                    suggestion="请提供有效的风险百分比"
                )
            )
        elif risk_percentage > 10:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"单笔交易风险过高: {risk_percentage:.2f}%，建议不超过 10%",
                    current_value=value,
                    suggestion="降低仓位大小或调整止损位置以减少风险",
                    corrected_value=0.05  # 建议5%风险
                )
            )
        elif risk_percentage > 5:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"单笔交易风险较高: {risk_percentage:.2f}%",
                    current_value=value,
                    suggestion="建议单笔交易风险控制在 2-3% 以内"
                )
            )
        
        return results
    
    def _validate_margin_utilization(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证保证金使用率"""
        margin_config = self.constraints.get("margin_utilization", {})
        warning_level = margin_config.get("warning_level", 0.7)
        critical_level = margin_config.get("critical_level", 0.85)
        emergency_level = margin_config.get("emergency_level", 0.95)
        
        utilization = self._safe_float_conversion(value)
        results = []
        
        if utilization < 0 or utilization > 1:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"保证金使用率超出有效范围: {utilization:.1%}",
                    current_value=utilization,
                    expected_range={"min": 0, "max": 1},
                    suggestion="保证金使用率应在 0-100% 之间"
                )
            )
        elif utilization >= emergency_level:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    field_name=field_name,
                    message=f"保证金使用率极高: {utilization:.1%}，面临强平风险",
                    current_value=utilization,
                    suggestion="立即降低仓位或增加保证金",
                    corrected_value=warning_level
                )
            )
        elif utilization >= critical_level:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"保证金使用率过高: {utilization:.1%}，存在强平风险",
                    current_value=utilization,
                    suggestion="建议降低仓位或增加保证金",
                    corrected_value=warning_level
                )
            )
        elif utilization >= warning_level:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"保证金使用率较高: {utilization:.1%}",
                    current_value=utilization,
                    suggestion="建议密切监控保证金水平"
                )
            )
        
        return results
    
    def _validate_expected_return(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证期望收益率"""
        return_config = self.constraints.get("return_expectations", {})
        min_return = return_config.get("min_annual_return", 0.1)
        
        expected_return = self._safe_float_conversion(value)
        results = []
        
        # 将期望收益转换为年化收益进行比较（假设这是交易的期望收益）
        holding_time = self._safe_float_conversion(self._get_nested_value(data, "risk_management.position_hold_time", 24))
        annual_return = expected_return * (8760 / holding_time)  # 年化
        
        if expected_return <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"期望收益率不能为负或零: {expected_return:.2%}",
                    current_value=expected_return,
                    suggestion="负收益率的交易不建议执行"
                )
            )
        elif annual_return < min_return:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name=field_name,
                    message=f"年化收益率偏低: {annual_return:.1%}，低于最小期望 {min_return:.1%}",
                    current_value=expected_return,
                    suggestion="考虑寻找更好的交易机会"
                )
            )
        
        return results
    
    def _validate_probability(self, field_name: str, value: Any, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证概率值"""
        probability = self._safe_float_conversion(value)
        results = []
        
        if probability < 0 or probability > 1:
            corrected_value = self._clamp_value(probability, 0, 1)
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name=field_name,
                    message=f"概率值超出有效范围: {probability:.1%}",
                    current_value=probability,
                    expected_range={"min": 0, "max": 1},
                    suggestion="概率值应在 0-100% 之间",
                    corrected_value=corrected_value
                )
            )
        
        return results