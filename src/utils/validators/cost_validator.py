"""
成本合理性验证器

验证交易成本是否合理，包括：
- 手续费成本分析
- 资金费率影响
- 滑点成本估算
- 持仓成本计算
- 收益率合理性检查
"""

from typing import Dict, Any, List, Optional
import math
from .base_validator import BaseValidator, ValidationResult, ValidationSeverity, ValidationMixin


class CostReasonabilityValidator(BaseValidator, ValidationMixin):
    """
    成本合理性验证器
    
    验证各种交易成本是否合理和可接受
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化成本合理性验证器
        
        Args:
            config: 验证器配置，包含成本约束规则
        """
        super().__init__("CostReasonabilityValidator", config)
        
        # 从配置中提取成本约束
        self.constraints = config.get("cost_constraints", {})
    
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """
        执行成本合理性验证
        
        Args:
            data: 待验证的交易数据
            context: 验证上下文信息
            
        Returns:
            验证结果列表
        """
        if not self.is_enabled():
            return []
        
        results = []
        
        # 验证交易手续费
        results.extend(self._validate_trading_fees(data, context))
        
        # 验证资金费率
        results.extend(self._validate_funding_rate(data, context))
        
        # 验证滑点成本
        results.extend(self._validate_slippage_cost(data, context))
        
        # 验证持仓成本
        results.extend(self._validate_holding_cost(data, context))
        
        # 验证收益率合理性
        results.extend(self._validate_return_expectations(data, context))
        
        # 验证成本效益分析
        results.extend(self._validate_cost_benefit_analysis(data, context))
        
        # 验证盈亏平衡分析
        results.extend(self._validate_break_even_analysis(data, context))
        
        # 记录验证结果
        for result in results:
            self.log_validation_result(result)
        
        return results
    
    def _validate_trading_fees(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证交易手续费"""
        results = []
        
        trading_fee = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.estimated_trading_fee")
        )
        position_size = self._safe_float_conversion(
            self._get_nested_value(data, "basic_params.position_size")
        )
        expected_return = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.expected_return")
        )
        target_profit = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.target_profit")
        )
        
        fee_config = self.constraints.get("trading_fees", {})
        max_fee_rate = fee_config.get("max_fee_rate", 0.001)
        high_fee_warning = fee_config.get("high_fee_warning", 0.0008)
        
        if trading_fee <= 0 or position_size <= 0:
            return results
        
        # 计算手续费率
        fee_rate = trading_fee / position_size
        
        # 检查手续费率是否超限
        if fee_rate > max_fee_rate:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="cost_benefit_analysis.estimated_trading_fee",
                    message=f"手续费率过高: {fee_rate:.3%}，超过最大限制 {max_fee_rate:.3%}",
                    current_value=trading_fee,
                    expected_range={"max_rate": max_fee_rate},
                    suggestion="检查交易所费率设置或考虑VIP等级优惠",
                    context={
                        "fee_rate": fee_rate,
                        "position_size": position_size
                    }
                )
            )
        elif fee_rate > high_fee_warning:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="cost_benefit_analysis.estimated_trading_fee",
                    message=f"手续费率偏高: {fee_rate:.3%}，高于警告线 {high_fee_warning:.3%}",
                    current_value=trading_fee,
                    suggestion="考虑优化交易频率或寻找更优惠的费率",
                    context={"fee_rate": fee_rate}
                )
            )
        
        # 检查手续费与预期收益的比例
        if expected_return > 0:
            fee_to_return_ratio = trading_fee / (position_size * expected_return)
            if fee_to_return_ratio > 0.1:  # 手续费超过预期收益的10%
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.estimated_trading_fee",
                        message=f"手续费占预期收益比例过高: {fee_to_return_ratio:.1%}",
                        current_value=trading_fee,
                        suggestion="手续费过高可能侵蚀大部分利润，考虑调整策略",
                        context={
                            "fee_to_return_ratio": fee_to_return_ratio,
                            "expected_return": expected_return
                        }
                    )
                )
        
        # 检查双向手续费成本（开仓+平仓）
        total_fee_impact = fee_rate * 2  # 开仓和平仓都需要手续费
        if target_profit > 0:
            profit_after_fees = target_profit - trading_fee * 2  # 假设平仓手续费相同
            if profit_after_fees <= 0:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="cost_benefit_analysis.target_profit",
                        message=f"手续费成本超过目标利润: 目标利润 {target_profit:.2f}，双向手续费 {trading_fee * 2:.2f}",
                        current_value=target_profit,
                        suggestion="增加目标利润或降低仓位大小以覆盖手续费成本",
                        context={
                            "total_fees": trading_fee * 2,
                            "profit_after_fees": profit_after_fees
                        }
                    )
                )
        
        return results
    
    def _validate_funding_rate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证资金费率影响"""
        results = []
        
        funding_rate = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.funding_rate")
        )
        funding_cost_8h = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.funding_cost_8h")
        )
        funding_cost_daily = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.funding_cost_daily")
        )
        position_hold_time = self._safe_float_conversion(
            self._get_nested_value(data, "risk_management.position_hold_time", 24)
        )
        position_size = self._safe_float_conversion(
            self._get_nested_value(data, "basic_params.position_size")
        )
        expected_return = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.expected_return")
        )
        
        funding_config = self.constraints.get("funding_rate", {})
        high_rate_threshold = funding_config.get("high_rate_threshold", 0.01)
        extreme_rate_threshold = funding_config.get("extreme_rate_threshold", 0.02)
        
        # 验证资金费率水平
        if abs(funding_rate) > extreme_rate_threshold:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="cost_benefit_analysis.funding_rate",
                    message=f"资金费率极端: {funding_rate:.3%}/8h，超过阈值 {extreme_rate_threshold:.3%}",
                    current_value=funding_rate,
                    expected_range={"extreme_threshold": extreme_rate_threshold},
                    suggestion="极端资金费率环境下建议避免长期持仓",
                    context={"rate_type": "extreme"}
                )
            )
        elif abs(funding_rate) > high_rate_threshold:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="cost_benefit_analysis.funding_rate",
                    message=f"资金费率偏高: {funding_rate:.3%}/8h，高于警告线 {high_rate_threshold:.3%}",
                    current_value=funding_rate,
                    suggestion="高资金费率下建议缩短持仓时间",
                    context={"rate_type": "high"}
                )
            )
        
        # 验证持仓期间的资金费用影响
        if position_hold_time > 0 and position_size > 0:
            # 计算预计总资金费用
            funding_periods = max(1, position_hold_time / 8)  # 8小时收取一次
            estimated_total_funding = funding_cost_8h * funding_periods
            
            # 资金费用占仓位比例
            funding_impact_ratio = abs(estimated_total_funding) / position_size
            
            if funding_impact_ratio > 0.005:  # 超过0.5%
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.funding_cost_8h",
                        message=f"预计资金费用影响较大: {funding_impact_ratio:.2%}，持仓时间 {position_hold_time}h",
                        current_value=funding_cost_8h,
                        suggestion="考虑缩短持仓时间或在资金费率较低时交易",
                        context={
                            "estimated_total_funding": estimated_total_funding,
                            "funding_periods": funding_periods,
                            "impact_ratio": funding_impact_ratio
                        }
                    )
                )
            
            # 资金费用与预期收益对比
            if expected_return > 0:
                expected_profit = position_size * expected_return
                funding_to_profit_ratio = abs(estimated_total_funding) / expected_profit
                
                if funding_to_profit_ratio > 0.2:  # 资金费用超过预期利润的20%
                    results.append(
                        self._create_result(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field_name="cost_benefit_analysis.funding_cost_daily",
                            message=f"资金费用侵蚀利润严重: 占预期利润 {funding_to_profit_ratio:.1%}",
                            current_value=funding_cost_daily,
                            suggestion="优化持仓时间或等待更有利的资金费率环境",
                            context={
                                "funding_to_profit_ratio": funding_to_profit_ratio,
                                "expected_profit": expected_profit
                            }
                        )
                    )
        
        return results
    
    def _validate_slippage_cost(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证滑点成本"""
        results = []
        
        # 从上下文获取滑点信息，如果没有则估算
        slippage_cost = 0
        if context and "estimated_slippage" in context:
            slippage_cost = self._safe_float_conversion(context["estimated_slippage"])
        
        position_size = self._safe_float_conversion(
            self._get_nested_value(data, "basic_params.position_size")
        )
        order_type = self._get_nested_value(data, "basic_params.order_type", "market")
        
        slippage_config = self.constraints.get("slippage", {})
        max_expected_slippage = slippage_config.get("max_expected_slippage", 0.001)
        high_slippage_warning = slippage_config.get("high_slippage_warning", 0.0005)
        
        # 根据订单类型估算滑点
        if slippage_cost == 0 and position_size > 0:
            if order_type == "market":
                # 市价单估算滑点（基于仓位大小）
                if position_size > 100000:  # 大额订单
                    estimated_slippage_rate = 0.0008
                elif position_size > 50000:
                    estimated_slippage_rate = 0.0005
                elif position_size > 10000:
                    estimated_slippage_rate = 0.0003
                else:
                    estimated_slippage_rate = 0.0001
                
                slippage_cost = position_size * estimated_slippage_rate
            else:  # 限价单滑点较小
                estimated_slippage_rate = 0.0001
                slippage_cost = position_size * estimated_slippage_rate
        
        if slippage_cost > 0 and position_size > 0:
            slippage_rate = slippage_cost / position_size
            
            # 检查滑点是否超限
            if slippage_rate > max_expected_slippage:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="execution.estimated_slippage",
                        message=f"预期滑点过高: {slippage_rate:.3%}，超过最大预期 {max_expected_slippage:.3%}",
                        current_value=slippage_cost,
                        expected_range={"max_slippage": max_expected_slippage},
                        suggestion="考虑分批执行或使用限价单减少滑点",
                        context={
                            "slippage_rate": slippage_rate,
                            "order_type": order_type
                        }
                    )
                )
            elif slippage_rate > high_slippage_warning:
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        field_name="execution.estimated_slippage",
                        message=f"滑点成本偏高: {slippage_rate:.3%}",
                        current_value=slippage_cost,
                        suggestion="考虑优化执行策略以降低滑点",
                        context={"slippage_rate": slippage_rate}
                    )
                )
            
            # 如果是大额订单且使用市价单，给出建议
            if position_size > 50000 and order_type == "market":
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        field_name="basic_params.order_type",
                        message=f"大额市价单可能产生较高滑点: 仓位 {position_size:.0f}",
                        current_value=order_type,
                        suggestion="大额订单建议使用限价单或分批执行策略",
                        context={"position_size": position_size}
                    )
                )
        
        return results
    
    def _validate_holding_cost(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证持仓成本"""
        results = []
        
        holding_cost_total = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.holding_cost_total")
        )
        position_size = self._safe_float_conversion(
            self._get_nested_value(data, "basic_params.position_size")
        )
        position_hold_time = self._safe_float_conversion(
            self._get_nested_value(data, "risk_management.position_hold_time", 24)
        )
        expected_return = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.expected_return")
        )
        
        holding_config = self.constraints.get("holding_cost", {})
        max_daily_cost_ratio = holding_config.get("max_daily_cost_ratio", 0.005)
        break_even_time_limit = holding_config.get("break_even_time_limit", 168)  # 7天
        
        if holding_cost_total > 0 and position_size > 0:
            # 计算日持仓成本比例
            daily_cost_ratio = (holding_cost_total / position_size) * (24 / position_hold_time)
            
            if daily_cost_ratio > max_daily_cost_ratio:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.holding_cost_total",
                        message=f"日持仓成本过高: {daily_cost_ratio:.2%}，超过限制 {max_daily_cost_ratio:.2%}",
                        current_value=holding_cost_total,
                        expected_range={"max_daily_ratio": max_daily_cost_ratio},
                        suggestion="高持仓成本下建议缩短持仓时间或选择成本更低的时间段",
                        context={
                            "daily_cost_ratio": daily_cost_ratio,
                            "hold_time": position_hold_time
                        }
                    )
                )
            
            # 检查持仓成本与收益的关系
            if expected_return > 0:
                expected_profit = position_size * expected_return
                cost_to_profit_ratio = holding_cost_total / expected_profit
                
                if cost_to_profit_ratio > 0.3:  # 持仓成本超过预期利润的30%
                    results.append(
                        self._create_result(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            field_name="cost_benefit_analysis.holding_cost_total",
                            message=f"持仓成本占预期利润比例过高: {cost_to_profit_ratio:.1%}",
                            current_value=holding_cost_total,
                            suggestion="优化持仓时间或等待更有利的成本环境",
                            context={
                                "cost_to_profit_ratio": cost_to_profit_ratio,
                                "expected_profit": expected_profit
                            }
                        )
                    )
        
        # 检查持仓时间是否过长
        if position_hold_time > break_even_time_limit:
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    field_name="risk_management.position_hold_time",
                    message=f"预期持仓时间过长: {position_hold_time}小时，超过建议限制 {break_even_time_limit}小时",
                    current_value=position_hold_time,
                    suggestion="长期持仓增加成本和风险，考虑缩短持仓周期",
                    context={"time_limit": break_even_time_limit}
                )
            )
        
        return results
    
    def _validate_return_expectations(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证收益率期望的合理性"""
        results = []
        
        expected_return = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.expected_return")
        )
        roi_annualized = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.roi_annualized")
        )
        profit_probability = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.profit_probability")
        )
        
        return_config = self.constraints.get("return_expectations", {})
        min_annual_return = return_config.get("min_annual_return", 0.1)
        risk_free_rate = return_config.get("risk_free_rate", 0.03)
        min_sharpe_ratio = return_config.get("min_sharpe_ratio", 1.0)
        
        # 验证年化收益率
        if roi_annualized > 0:
            if roi_annualized < min_annual_return:
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.roi_annualized",
                        message=f"年化收益率偏低: {roi_annualized:.1%}，低于最小期望 {min_annual_return:.1%}",
                        current_value=roi_annualized,
                        expected_range={"min_annual": min_annual_return},
                        suggestion="考虑寻找年化收益率更高的交易机会"
                    )
                )
            elif roi_annualized > 2.0:  # 年化收益率超过200%
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.roi_annualized",
                        message=f"年化收益率过高: {roi_annualized:.1%}，可能风险极大",
                        current_value=roi_annualized,
                        suggestion="超高收益率往往伴随极高风险，请谨慎评估",
                        context={"risk_warning": "high_return_high_risk"}
                    )
                )
        
        # 验证风险调整收益
        if roi_annualized > risk_free_rate:
            excess_return = roi_annualized - risk_free_rate
            # 如果有风险指标，计算夏普比率
            volatility = None
            if context and "volatility" in context:
                volatility = self._safe_float_conversion(context["volatility"])
            
            if volatility and volatility > 0:
                implied_sharpe = excess_return / volatility
                if implied_sharpe < min_sharpe_ratio:
                    results.append(
                        self._create_result(
                            is_valid=True,
                            severity=ValidationSeverity.WARNING,
                            field_name="cost_benefit_analysis.roi_annualized",
                            message=f"风险调整收益不佳: 隐含夏普比率 {implied_sharpe:.2f}，低于最小要求 {min_sharpe_ratio:.2f}",
                            current_value=roi_annualized,
                            suggestion="在当前风险水平下，收益率不够吸引",
                            context={
                                "implied_sharpe": implied_sharpe,
                                "volatility": volatility,
                                "excess_return": excess_return
                            }
                        )
                    )
        
        # 验证获利概率
        if profit_probability > 0:
            if profit_probability < 0.5:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.profit_probability",
                        message=f"获利概率偏低: {profit_probability:.1%}，低于50%",
                        current_value=profit_probability,
                        suggestion="低获利概率的交易需要更高的风险收益比来补偿",
                        context={"probability_threshold": 0.5}
                    )
                )
            elif profit_probability > 0.9:
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        field_name="cost_benefit_analysis.profit_probability",
                        message=f"获利概率很高: {profit_probability:.1%}",
                        current_value=profit_probability,
                        suggestion="高获利概率是好信号，但要注意预期收益是否合理"
                    )
                )
        
        return results
    
    def _validate_cost_benefit_analysis(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证整体成本效益分析"""
        results = []
        
        expected_value = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.expected_value")
        )
        target_profit = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.target_profit")
        )
        trading_fee = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.estimated_trading_fee")
        )
        holding_cost_total = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.holding_cost_total")
        )
        
        # 计算总成本
        total_cost = trading_fee + holding_cost_total
        
        # 验证期望值是否为正
        if expected_value <= 0:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    field_name="cost_benefit_analysis.expected_value",
                    message=f"期望值为负: {expected_value:.2f}，交易预期不盈利",
                    current_value=expected_value,
                    suggestion="负期望值的交易不建议执行，重新评估策略",
                    context={"recommendation": "avoid_trade"}
                )
            )
        elif expected_value < total_cost:
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="cost_benefit_analysis.expected_value",
                    message=f"期望值低于总成本: 期望值 {expected_value:.2f}，总成本 {total_cost:.2f}",
                    current_value=expected_value,
                    suggestion="期望收益难以覆盖交易成本，考虑优化策略",
                    context={"total_cost": total_cost}
                )
            )
        
        # 验证目标利润的合理性
        if target_profit > 0 and total_cost > 0:
            net_target_profit = target_profit - total_cost
            if net_target_profit <= 0:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name="cost_benefit_analysis.target_profit",
                        message=f"目标利润无法覆盖成本: 目标利润 {target_profit:.2f}，总成本 {total_cost:.2f}",
                        current_value=target_profit,
                        suggestion="增加目标利润或降低交易成本",
                        corrected_value=total_cost * 1.2,  # 建议增加20%缓冲
                        context={
                            "net_profit": net_target_profit,
                            "total_cost": total_cost
                        }
                    )
                )
            elif net_target_profit < total_cost * 0.5:  # 净利润小于成本的50%
                results.append(
                    self._create_result(
                        is_valid=True,
                        severity=ValidationSeverity.WARNING,
                        field_name="cost_benefit_analysis.target_profit",
                        message=f"目标净利润偏低: 净利润 {net_target_profit:.2f}，成本 {total_cost:.2f}",
                        current_value=target_profit,
                        suggestion="考虑设置更高的目标利润以获得更好的风险补偿"
                    )
                )
        
        return results
    
    def _validate_break_even_analysis(self, data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> List[ValidationResult]:
        """验证盈亏平衡分析"""
        results = []
        
        break_even_price = self._safe_float_conversion(
            self._get_nested_value(data, "cost_benefit_analysis.break_even_price")
        )
        current_price = self._safe_float_conversion(
            self._get_nested_value(data, "basic_params.current_price")
        )
        direction = self._get_nested_value(data, "basic_params.direction")
        take_profit = self._safe_float_conversion(
            self._get_nested_value(data, "risk_management.take_profit")
        )
        
        if not all([break_even_price > 0, current_price > 0, direction, take_profit > 0]):
            return results
        
        # 计算盈亏平衡距离
        if direction == "long":
            be_distance_pct = (break_even_price - current_price) / current_price * 100
            profit_range_pct = (take_profit - break_even_price) / break_even_price * 100
        elif direction == "short":
            be_distance_pct = (current_price - break_even_price) / current_price * 100
            profit_range_pct = (break_even_price - take_profit) / break_even_price * 100
        else:
            return results
        
        # 验证盈亏平衡距离
        if be_distance_pct > 5:  # 盈亏平衡距离超过5%
            results.append(
                self._create_result(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="cost_benefit_analysis.break_even_price",
                    message=f"盈亏平衡距离过远: {be_distance_pct:.1f}%，成本较高",
                    current_value=break_even_price,
                    suggestion="高成本导致盈亏平衡点较远，考虑优化成本结构",
                    context={
                        "be_distance_pct": be_distance_pct,
                        "current_price": current_price,
                        "direction": direction
                    }
                )
            )
        elif be_distance_pct > 2:  # 2-5%之间给出提醒
            results.append(
                self._create_result(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    field_name="cost_benefit_analysis.break_even_price",
                    message=f"盈亏平衡距离: {be_distance_pct:.1f}%",
                    current_value=break_even_price,
                    suggestion="关注成本控制以降低盈亏平衡点"
                )
            )
        
        # 验证盈利空间
        if profit_range_pct > 0:
            if profit_range_pct < be_distance_pct * 1.5:  # 盈利空间小于成本空间的1.5倍
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        field_name="risk_management.take_profit",
                        message=f"盈利空间不足: 盈利空间 {profit_range_pct:.1f}%，成本空间 {be_distance_pct:.1f}%",
                        current_value=take_profit,
                        suggestion="增加止盈目标或降低成本以获得更好的风险收益比",
                        context={
                            "profit_range_pct": profit_range_pct,
                            "be_distance_pct": be_distance_pct
                        }
                    )
                )
        
        return results