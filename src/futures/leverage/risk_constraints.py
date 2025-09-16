"""
风险约束配置模块

定义风险等级与杠杆倍数的强制约束关系，确保高风险环境下使用适当的杠杆。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级定义"""
    EMERGENCY = "emergency"  # 紧急风险 - 禁止杠杆
    CRITICAL = "critical"    # 危急风险 - 极低杠杆
    HIGH = "high"           # 高风险 - 低杠杆
    MEDIUM = "medium"       # 中等风险 - 中等杠杆
    LOW = "low"            # 低风险 - 正常杠杆


class MarketRiskLevel(Enum):
    """市场风险等级（来自市场分析器）"""
    EMERGENCY = "emergency"
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_string(cls, value: str) -> 'MarketRiskLevel':
        """从字符串转换为枚举"""
        value_lower = value.lower()
        for level in cls:
            if level.value == value_lower:
                return level
        # 默认返回中等风险
        logger.warning(f"未知的市场风险等级: {value}，使用默认值MEDIUM")
        return cls.MEDIUM


@dataclass
class RiskConstraint:
    """风险约束配置"""
    risk_level: RiskLevel
    max_leverage: float          # 最大杠杆
    recommended_leverage: float  # 推荐杠杆
    min_leverage: float = 1.0    # 最小杠杆
    warning_threshold: float = 0.8  # 警告阈值（最大杠杆的百分比）

    def validate_leverage(self, leverage: float) -> Tuple[float, str]:
        """
        验证并调整杠杆倍数

        Returns:
            (调整后的杠杆, 警告信息)
        """
        if leverage > self.max_leverage:
            return self.max_leverage, f"杠杆超过{self.risk_level.value}风险等级限制，已强制降至{self.max_leverage}x"

        if leverage > self.max_leverage * self.warning_threshold:
            return leverage, f"警告：杠杆接近{self.risk_level.value}风险等级上限"

        if leverage < self.min_leverage:
            return self.min_leverage, f"杠杆低于最小值，已调整至{self.min_leverage}x"

        return leverage, ""


class RiskConstraintManager:
    """风险约束管理器"""

    # 核心风险-杠杆映射表
    RISK_LEVERAGE_MAP = {
        RiskLevel.EMERGENCY: RiskConstraint(
            risk_level=RiskLevel.EMERGENCY,
            max_leverage=1.0,
            recommended_leverage=1.0,
            min_leverage=1.0,
            warning_threshold=1.0
        ),
        RiskLevel.CRITICAL: RiskConstraint(
            risk_level=RiskLevel.CRITICAL,
            max_leverage=1.5,
            recommended_leverage=1.2,
            min_leverage=1.0,
            warning_threshold=0.9
        ),
        RiskLevel.HIGH: RiskConstraint(
            risk_level=RiskLevel.HIGH,
            max_leverage=3.0,
            recommended_leverage=2.0,
            min_leverage=1.0,
            warning_threshold=0.85
        ),
        RiskLevel.MEDIUM: RiskConstraint(
            risk_level=RiskLevel.MEDIUM,
            max_leverage=8.0,
            recommended_leverage=5.0,
            min_leverage=2.0,
            warning_threshold=0.8
        ),
        RiskLevel.LOW: RiskConstraint(
            risk_level=RiskLevel.LOW,
            max_leverage=15.0,
            recommended_leverage=10.0,
            min_leverage=3.0,
            warning_threshold=0.75
        )
    }

    # 市场波动率到风险等级的映射
    VOLATILITY_RISK_MAP = {
        (0.0, 0.1): RiskLevel.LOW,      # 低波动
        (0.1, 0.2): RiskLevel.LOW,      # 正常波动
        (0.2, 0.3): RiskLevel.MEDIUM,   # 中等波动
        (0.3, 0.5): RiskLevel.HIGH,     # 高波动
        (0.5, 0.8): RiskLevel.CRITICAL, # 极高波动
        (0.8, float('inf')): RiskLevel.EMERGENCY  # 紧急波动
    }

    def __init__(self, custom_constraints: Optional[Dict[RiskLevel, RiskConstraint]] = None):
        """
        初始化风险约束管理器

        Args:
            custom_constraints: 自定义风险约束配置
        """
        self.constraints = self.RISK_LEVERAGE_MAP.copy()
        if custom_constraints:
            self.constraints.update(custom_constraints)

        self._validation_log = []

    def get_constraint(self, risk_level: RiskLevel) -> RiskConstraint:
        """获取风险约束配置"""
        return self.constraints.get(risk_level, self.constraints[RiskLevel.MEDIUM])

    def get_constraint_from_market_risk(self, market_risk: str) -> RiskConstraint:
        """根据市场风险字符串获取约束"""
        try:
            market_level = MarketRiskLevel.from_string(market_risk)
            risk_level = self._map_market_to_risk_level(market_level)
            return self.get_constraint(risk_level)
        except Exception as e:
            logger.error(f"解析市场风险等级失败: {e}")
            return self.get_constraint(RiskLevel.HIGH)  # 默认使用高风险约束

    def _map_market_to_risk_level(self, market_level: MarketRiskLevel) -> RiskLevel:
        """映射市场风险等级到内部风险等级"""
        mapping = {
            MarketRiskLevel.EMERGENCY: RiskLevel.EMERGENCY,
            MarketRiskLevel.CRITICAL: RiskLevel.CRITICAL,
            MarketRiskLevel.HIGH: RiskLevel.HIGH,
            MarketRiskLevel.MEDIUM: RiskLevel.MEDIUM,
            MarketRiskLevel.LOW: RiskLevel.LOW
        }
        return mapping.get(market_level, RiskLevel.MEDIUM)

    def determine_risk_level_from_volatility(self, volatility: float) -> RiskLevel:
        """根据波动率确定风险等级"""
        for (low, high), risk_level in self.VOLATILITY_RISK_MAP.items():
            if low <= volatility < high:
                return risk_level
        return RiskLevel.MEDIUM

    def apply_risk_constraint(
        self,
        leverage: float,
        risk_level: RiskLevel,
        source: str = "unknown"
    ) -> Tuple[float, RiskConstraint, str]:
        """
        应用风险约束到杠杆倍数

        Args:
            leverage: 原始杠杆倍数
            risk_level: 风险等级
            source: 调用来源

        Returns:
            (调整后的杠杆, 使用的约束, 调整说明)
        """
        constraint = self.get_constraint(risk_level)
        adjusted_leverage, warning = constraint.validate_leverage(leverage)

        # 记录验证日志
        self._validation_log.append({
            'source': source,
            'original_leverage': leverage,
            'adjusted_leverage': adjusted_leverage,
            'risk_level': risk_level.value,
            'constraint': constraint,
            'warning': warning
        })

        if warning:
            logger.warning(f"风险约束调整 [{source}]: {warning}")

        if adjusted_leverage != leverage:
            adjustment_msg = (f"风险约束生效: {risk_level.value}风险等级下，"
                            f"杠杆从{leverage:.1f}x调整至{adjusted_leverage:.1f}x")
            logger.info(adjustment_msg)
            return adjusted_leverage, constraint, adjustment_msg

        return adjusted_leverage, constraint, ""

    def get_safe_leverage_range(self, risk_level: RiskLevel) -> Tuple[float, float]:
        """
        获取安全的杠杆范围

        Returns:
            (最小杠杆, 最大杠杆)
        """
        constraint = self.get_constraint(risk_level)
        return constraint.min_leverage, constraint.max_leverage

    def calculate_risk_adjusted_leverage(
        self,
        base_leverage: float,
        risk_factors: Dict[str, float]
    ) -> Tuple[float, RiskLevel, str]:
        """
        基于多个风险因子计算风险调整后的杠杆

        Args:
            base_leverage: 基础杠杆
            risk_factors: 风险因子字典，包含:
                - volatility: 波动率
                - drawdown: 最大回撤
                - liquidity_score: 流动性评分
                - margin_ratio: 保证金使用率

        Returns:
            (调整后的杠杆, 综合风险等级, 说明)
        """
        # 评估各项风险
        risk_scores = []

        # 波动率风险
        if 'volatility' in risk_factors:
            vol_risk = self.determine_risk_level_from_volatility(risk_factors['volatility'])
            risk_scores.append(self._risk_level_to_score(vol_risk))

        # 回撤风险
        if 'drawdown' in risk_factors:
            dd = risk_factors['drawdown']
            if dd > 0.3:
                risk_scores.append(0.9)  # CRITICAL
            elif dd > 0.2:
                risk_scores.append(0.7)  # HIGH
            elif dd > 0.1:
                risk_scores.append(0.5)  # MEDIUM
            else:
                risk_scores.append(0.3)  # LOW

        # 流动性风险
        if 'liquidity_score' in risk_factors:
            liq = risk_factors['liquidity_score']
            if liq < 0.2:
                risk_scores.append(0.8)  # HIGH
            elif liq < 0.5:
                risk_scores.append(0.5)  # MEDIUM
            else:
                risk_scores.append(0.3)  # LOW

        # 保证金风险
        if 'margin_ratio' in risk_factors:
            margin = risk_factors['margin_ratio']
            if margin > 0.8:
                risk_scores.append(1.0)  # EMERGENCY
            elif margin > 0.6:
                risk_scores.append(0.8)  # CRITICAL
            elif margin > 0.4:
                risk_scores.append(0.6)  # HIGH
            else:
                risk_scores.append(0.4)  # MEDIUM

        # 计算综合风险评分（取最高风险）
        if risk_scores:
            max_risk_score = max(risk_scores)
            overall_risk_level = self._score_to_risk_level(max_risk_score)
        else:
            overall_risk_level = RiskLevel.MEDIUM

        # 应用风险约束
        adjusted_leverage, constraint, msg = self.apply_risk_constraint(
            base_leverage,
            overall_risk_level,
            source="risk_adjusted_calculation"
        )

        explanation = (f"综合风险评估: {overall_risk_level.value} | "
                      f"推荐杠杆: {constraint.recommended_leverage}x | "
                      f"最大允许: {constraint.max_leverage}x")

        if msg:
            explanation = f"{explanation} | {msg}"

        return adjusted_leverage, overall_risk_level, explanation

    def _risk_level_to_score(self, risk_level: RiskLevel) -> float:
        """风险等级转换为数值评分"""
        mapping = {
            RiskLevel.EMERGENCY: 1.0,
            RiskLevel.CRITICAL: 0.8,
            RiskLevel.HIGH: 0.6,
            RiskLevel.MEDIUM: 0.4,
            RiskLevel.LOW: 0.2
        }
        return mapping.get(risk_level, 0.5)

    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """数值评分转换为风险等级"""
        if score >= 0.9:
            return RiskLevel.EMERGENCY
        elif score >= 0.7:
            return RiskLevel.CRITICAL
        elif score >= 0.5:
            return RiskLevel.HIGH
        elif score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def get_validation_log(self) -> list:
        """获取验证日志"""
        return self._validation_log.copy()

    def clear_validation_log(self):
        """清空验证日志"""
        self._validation_log.clear()