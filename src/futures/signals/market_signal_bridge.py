"""
市场信号桥接器

实现市场分析器与期货策略系统之间的信号传递桥接机制，解决信号断层问题。
提供分层决策机制：
- Layer 1: 市场状态评估（来自市场分析器）
- Layer 2: 策略信号计算（来自MACD/RSI策略）
- Layer 3: 信号融合决策（综合两层信息）
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import numpy as np

# 导入相关模块
from ..models.data_models import TradingDirection, RiskLevel, OperationType
from ..market.market_analyzer import MarketConditionAnalysis, MarketCondition, LiquidityLevel

logger = logging.getLogger(__name__)


class SignalLayer(Enum):
    """信号层级"""
    MARKET_ASSESSMENT = "market_assessment"      # 市场状态评估层
    STRATEGY_SIGNAL = "strategy_signal"          # 策略信号层
    FUSION_DECISION = "fusion_decision"          # 融合决策层


class SignalPriority(Enum):
    """信号优先级"""
    CRITICAL = 5    # 关键信号（强制执行）
    HIGH = 4        # 高优先级
    MEDIUM = 3      # 中等优先级
    LOW = 2         # 低优先级
    IGNORE = 1      # 忽略


@dataclass
class MarketSignalWeight:
    """市场信号权重配置"""
    trend_weight: float = 0.3           # 趋势权重
    risk_weight: float = 0.4            # 风险权重
    liquidity_weight: float = 0.2       # 流动性权重
    sentiment_weight: float = 0.1       # 情绪权重

    def normalize(self) -> 'MarketSignalWeight':
        """归一化权重"""
        total = self.trend_weight + self.risk_weight + self.liquidity_weight + self.sentiment_weight
        if total > 0:
            return MarketSignalWeight(
                trend_weight=self.trend_weight / total,
                risk_weight=self.risk_weight / total,
                liquidity_weight=self.liquidity_weight / total,
                sentiment_weight=self.sentiment_weight / total
            )
        return self


@dataclass
class BridgedSignal:
    """桥接后的信号"""
    # 原始信号信息
    market_direction: TradingDirection
    market_risk_level: RiskLevel
    market_confidence: float

    # 策略信号信息
    strategy_direction: TradingDirection
    strategy_strength: float
    strategy_confidence: float

    # 融合结果
    final_direction: TradingDirection
    final_confidence: float
    final_strength: float
    final_risk_level: RiskLevel

    # 决策层级信息
    dominant_layer: SignalLayer
    signal_priority: SignalPriority

    # 元数据
    fusion_logic: str                    # 融合逻辑说明
    signal_weights: MarketSignalWeight   # 使用的权重
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "market_direction": self.market_direction.value,
            "market_risk_level": self.market_risk_level.value,
            "market_confidence": self.market_confidence,
            "strategy_direction": self.strategy_direction.value,
            "strategy_strength": self.strategy_strength,
            "strategy_confidence": self.strategy_confidence,
            "final_direction": self.final_direction.value,
            "final_confidence": self.final_confidence,
            "final_strength": self.final_strength,
            "final_risk_level": self.final_risk_level.value,
            "dominant_layer": self.dominant_layer.value,
            "signal_priority": self.signal_priority.value,
            "fusion_logic": self.fusion_logic,
            "timestamp": self.timestamp.isoformat()
        }


class MarketSignalBridge:
    """市场信号桥接器"""

    def __init__(self,
                 market_weight: MarketSignalWeight = None,
                 risk_preference: str = "moderate"):
        """
        初始化桥接器

        Args:
            market_weight: 市场信号权重配置
            risk_preference: 风险偏好 (conservative, moderate, aggressive)
        """
        self.market_weight = market_weight or MarketSignalWeight()
        self.market_weight = self.market_weight.normalize()
        self.risk_preference = risk_preference

        # 根据风险偏好调整权重
        self._adjust_weights_by_risk_preference()

        self.logger = logging.getLogger(__name__)

        # 信号历史记录（用于调试和分析）
        self.signal_history: List[BridgedSignal] = []

    def _adjust_weights_by_risk_preference(self):
        """根据风险偏好调整权重"""
        if self.risk_preference == "conservative":
            # 保守策略：增加风险权重，降低趋势权重
            self.market_weight.risk_weight *= 1.5
            self.market_weight.trend_weight *= 0.8
            self.market_weight.liquidity_weight *= 1.2
        elif self.risk_preference == "aggressive":
            # 激进策略：增加趋势权重，降低风险权重
            self.market_weight.trend_weight *= 1.3
            self.market_weight.risk_weight *= 0.7
            self.market_weight.sentiment_weight *= 1.2

        # 重新归一化
        self.market_weight = self.market_weight.normalize()

    def bridge_signals(self,
                      market_analysis: MarketConditionAnalysis,
                      strategy_direction: TradingDirection,
                      strategy_strength: float,
                      strategy_confidence: float) -> BridgedSignal:
        """
        桥接市场分析信号和策略信号

        Args:
            market_analysis: 市场条件分析结果
            strategy_direction: 策略信号方向
            strategy_strength: 策略信号强度
            strategy_confidence: 策略信号置信度

        Returns:
            BridgedSignal: 桥接后的综合信号
        """
        try:
            # Layer 1: 市场状态评估
            market_signal = self._assess_market_signals(market_analysis)

            # Layer 2: 策略信号处理
            processed_strategy = self._process_strategy_signals(
                strategy_direction, strategy_strength, strategy_confidence
            )

            # Layer 3: 信号融合决策
            fusion_result = self._fuse_signals(market_signal, processed_strategy, market_analysis)

            # 创建桥接信号
            bridged_signal = BridgedSignal(
                market_direction=market_signal['direction'],
                market_risk_level=market_signal['risk_level'],
                market_confidence=market_signal['confidence'],
                strategy_direction=strategy_direction,
                strategy_strength=strategy_strength,
                strategy_confidence=strategy_confidence,
                final_direction=fusion_result['direction'],
                final_confidence=fusion_result['confidence'],
                final_strength=fusion_result['strength'],
                final_risk_level=fusion_result['risk_level'],
                dominant_layer=fusion_result['dominant_layer'],
                signal_priority=fusion_result['priority'],
                fusion_logic=fusion_result['logic'],
                signal_weights=self.market_weight
            )

            # 记录信号历史
            self.signal_history.append(bridged_signal)
            if len(self.signal_history) > 100:  # 保持最近100个信号
                self.signal_history.pop(0)

            self.logger.info(f"信号桥接完成: 市场={market_signal['direction'].value} "
                           f"策略={strategy_direction.value} 最终={fusion_result['direction'].value} "
                           f"置信度={fusion_result['confidence']:.2f}")

            return bridged_signal

        except Exception as e:
            self.logger.error(f"信号桥接失败: {e}")
            # 返回保守的默认信号
            return self._create_default_signal(strategy_direction, strategy_strength, strategy_confidence)

    def _assess_market_signals(self, market_analysis: MarketConditionAnalysis) -> Dict[str, Any]:
        """Layer 1: 评估市场信号"""
        # 基于市场条件确定方向
        market_direction = market_analysis.trend_direction

        # 计算市场风险级别
        risk_factors = []

        # 市场压力风险
        if market_analysis.market_stress_level > 0.8:
            risk_factors.append("high_stress")
        elif market_analysis.market_stress_level > 0.6:
            risk_factors.append("medium_stress")

        # 流动性风险
        if market_analysis.liquidity_level == LiquidityLevel.VERY_LOW:
            risk_factors.append("liquidity_crisis")
        elif market_analysis.liquidity_level == LiquidityLevel.LOW:
            risk_factors.append("low_liquidity")

        # 相关性风险
        if market_analysis.correlation_breakdown:
            risk_factors.append("correlation_breakdown")

        # 流动性危机
        if market_analysis.liquidity_crisis:
            risk_factors.append("liquidity_crisis")

        # 确定风险级别
        risk_level = self._determine_market_risk_level(risk_factors, market_analysis)

        # 计算市场置信度
        market_confidence = self._calculate_market_confidence(market_analysis)

        return {
            'direction': market_direction,
            'risk_level': risk_level,
            'confidence': market_confidence,
            'risk_factors': risk_factors
        }

    def _process_strategy_signals(self, direction: TradingDirection,
                                strength: float, confidence: float) -> Dict[str, Any]:
        """Layer 2: 处理策略信号"""
        # 策略信号已经是处理过的，这里主要是归一化和验证
        normalized_strength = max(0.0, min(1.0, abs(strength)))
        normalized_confidence = max(0.0, min(1.0, confidence / 100.0))

        return {
            'direction': direction,
            'strength': normalized_strength,
            'confidence': normalized_confidence
        }

    def _fuse_signals(self, market_signal: Dict[str, Any],
                     strategy_signal: Dict[str, Any],
                     market_analysis: MarketConditionAnalysis) -> Dict[str, Any]:
        """Layer 3: 融合信号决策"""

        market_dir = market_signal['direction']
        strategy_dir = strategy_signal['direction']
        market_risk = market_signal['risk_level']

        # 决策逻辑
        fusion_logic = ""
        dominant_layer = SignalLayer.FUSION_DECISION
        priority = SignalPriority.MEDIUM

        # 1. 风险优先检查
        if market_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
            # 高风险时，市场信号占主导
            if market_dir == TradingDirection.NEUTRAL:
                final_direction = TradingDirection.NEUTRAL
                fusion_logic = f"高风险环境({market_risk.value})，暂停交易"
                priority = SignalPriority.CRITICAL
                dominant_layer = SignalLayer.MARKET_ASSESSMENT
            else:
                # 高风险但有明确市场方向时，跟随市场但降低信心
                final_direction = market_dir
                fusion_logic = f"高风险环境但市场方向明确，跟随市场信号"
                priority = SignalPriority.HIGH
                dominant_layer = SignalLayer.MARKET_ASSESSMENT

        # 2. 信号一致性检查
        elif market_dir == strategy_dir and market_dir != TradingDirection.NEUTRAL:
            # 信号一致且有明确方向
            final_direction = market_dir
            fusion_logic = "市场信号与策略信号一致，信心强化"
            priority = SignalPriority.HIGH

        elif market_dir == TradingDirection.NEUTRAL and strategy_dir != TradingDirection.NEUTRAL:
            # 市场中性，策略有方向
            if market_risk == RiskLevel.LOW:
                final_direction = strategy_dir
                fusion_logic = "市场中性且低风险，跟随策略信号"
                dominant_layer = SignalLayer.STRATEGY_SIGNAL
            else:
                final_direction = TradingDirection.NEUTRAL
                fusion_logic = "市场中性但风险偏高，保持观望"
                dominant_layer = SignalLayer.MARKET_ASSESSMENT

        elif market_dir != TradingDirection.NEUTRAL and strategy_dir == TradingDirection.NEUTRAL:
            # 市场有方向，策略中性
            final_direction = market_dir
            fusion_logic = "策略中性但市场有明确方向，跟随市场"
            dominant_layer = SignalLayer.MARKET_ASSESSMENT

        elif market_dir != strategy_dir and market_dir != TradingDirection.NEUTRAL and strategy_dir != TradingDirection.NEUTRAL:
            # 信号冲突
            if market_analysis.trend_strength > 0.7:
                # 强趋势时跟随市场
                final_direction = market_dir
                fusion_logic = f"信号冲突但市场趋势强劲({market_analysis.trend_strength:.2f})，跟随市场"
                dominant_layer = SignalLayer.MARKET_ASSESSMENT
            else:
                # 弱趋势时保持中性
                final_direction = TradingDirection.NEUTRAL
                fusion_logic = "信号冲突且趋势不明确，保持观望"
                priority = SignalPriority.LOW

        else:
            # 默认情况：都是中性
            final_direction = TradingDirection.NEUTRAL
            fusion_logic = "所有信号均为中性，保持观望"
            priority = SignalPriority.LOW

        # 计算融合置信度
        final_confidence = self._calculate_fusion_confidence(
            market_signal, strategy_signal, final_direction, market_analysis
        )

        # 计算融合强度
        final_strength = self._calculate_fusion_strength(
            market_signal, strategy_signal, final_direction, market_analysis
        )

        # 确定最终风险级别
        final_risk_level = self._determine_final_risk_level(market_risk, final_direction)

        return {
            'direction': final_direction,
            'confidence': final_confidence,
            'strength': final_strength,
            'risk_level': final_risk_level,
            'dominant_layer': dominant_layer,
            'priority': priority,
            'logic': fusion_logic
        }

    def _determine_market_risk_level(self, risk_factors: List[str],
                                   market_analysis: MarketConditionAnalysis) -> RiskLevel:
        """确定市场风险级别"""
        critical_factors = ["liquidity_crisis", "correlation_breakdown"]
        high_factors = ["high_stress", "low_liquidity"]

        if any(factor in critical_factors for factor in risk_factors):
            return RiskLevel.CRITICAL
        elif len([f for f in risk_factors if f in high_factors]) >= 2:
            return RiskLevel.HIGH
        elif any(factor in high_factors for factor in risk_factors):
            return RiskLevel.MEDIUM
        elif market_analysis.market_stress_level > 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _calculate_market_confidence(self, market_analysis: MarketConditionAnalysis) -> float:
        """计算市场信号置信度"""
        confidence_factors = []

        # 趋势强度贡献
        confidence_factors.append(market_analysis.trend_strength * self.market_weight.trend_weight)

        # 数据质量贡献
        confidence_factors.append(market_analysis.data_quality_score * 0.2)

        # 分析置信度贡献
        confidence_factors.append(market_analysis.analysis_confidence * 0.3)

        # 流动性贡献
        liquidity_score = self._get_liquidity_score(market_analysis.liquidity_level)
        confidence_factors.append(liquidity_score * self.market_weight.liquidity_weight)

        # 风险调整
        risk_penalty = market_analysis.market_stress_level * 0.3

        total_confidence = sum(confidence_factors) - risk_penalty
        return max(0.0, min(1.0, total_confidence))

    def _get_liquidity_score(self, liquidity_level: LiquidityLevel) -> float:
        """获取流动性评分"""
        scores = {
            LiquidityLevel.VERY_HIGH: 1.0,
            LiquidityLevel.HIGH: 0.8,
            LiquidityLevel.NORMAL: 0.6,
            LiquidityLevel.LOW: 0.3,
            LiquidityLevel.VERY_LOW: 0.1
        }
        return scores.get(liquidity_level, 0.5)

    def _calculate_fusion_confidence(self, market_signal: Dict[str, Any],
                                   strategy_signal: Dict[str, Any],
                                   final_direction: TradingDirection,
                                   market_analysis: MarketConditionAnalysis) -> float:
        """计算融合置信度"""
        market_conf = market_signal['confidence']
        strategy_conf = strategy_signal['confidence']

        # 基础置信度（加权平均）
        base_confidence = (market_conf * 0.6 + strategy_conf * 0.4)

        # 一致性奖励
        if market_signal['direction'] == strategy_signal['direction'] and final_direction != TradingDirection.NEUTRAL:
            base_confidence *= 1.2  # 一致性奖励
        elif market_signal['direction'] != strategy_signal['direction']:
            base_confidence *= 0.8  # 冲突惩罚

        # 趋势强度奖励
        if market_analysis.trend_strength > 0.6:
            base_confidence *= (1.0 + market_analysis.trend_strength * 0.2)

        return max(0.0, min(1.0, base_confidence))

    def _calculate_fusion_strength(self, market_signal: Dict[str, Any],
                                 strategy_signal: Dict[str, Any],
                                 final_direction: TradingDirection,
                                 market_analysis: MarketConditionAnalysis) -> float:
        """计算融合强度"""
        if final_direction == TradingDirection.NEUTRAL:
            return 0.0

        # 基于趋势强度和策略强度的加权组合
        trend_strength = market_analysis.trend_strength
        strategy_strength = strategy_signal['strength']

        # 加权融合
        fusion_strength = (trend_strength * 0.6 + strategy_strength * 0.4)

        # 信号一致性调整
        if market_signal['direction'] == strategy_signal['direction']:
            fusion_strength *= 1.1  # 一致性增强

        return max(0.0, min(1.0, fusion_strength))

    def _determine_final_risk_level(self, market_risk: RiskLevel,
                                  final_direction: TradingDirection) -> RiskLevel:
        """确定最终风险级别"""
        if final_direction == TradingDirection.NEUTRAL:
            return RiskLevel.LOW  # 观望状态风险较低

        # 交易时的风险不会低于市场风险
        risk_levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EMERGENCY]
        market_risk_index = risk_levels.index(market_risk)

        # 交易风险通常比市场风险高一级（因为有杠杆等因素）
        final_risk_index = min(market_risk_index + 1, len(risk_levels) - 1)
        return risk_levels[final_risk_index]

    def _create_default_signal(self, strategy_direction: TradingDirection,
                             strategy_strength: float, strategy_confidence: float) -> BridgedSignal:
        """创建默认保守信号"""
        return BridgedSignal(
            market_direction=TradingDirection.NEUTRAL,
            market_risk_level=RiskLevel.MEDIUM,
            market_confidence=0.5,
            strategy_direction=strategy_direction,
            strategy_strength=strategy_strength,
            strategy_confidence=strategy_confidence,
            final_direction=TradingDirection.NEUTRAL,
            final_confidence=0.3,
            final_strength=0.0,
            final_risk_level=RiskLevel.MEDIUM,
            dominant_layer=SignalLayer.MARKET_ASSESSMENT,
            signal_priority=SignalPriority.LOW,
            fusion_logic="桥接失败，采用保守默认信号",
            signal_weights=self.market_weight
        )

    def get_signal_statistics(self) -> Dict[str, Any]:
        """获取信号统计信息"""
        if not self.signal_history:
            return {"total_signals": 0}

        total = len(self.signal_history)
        directions = [s.final_direction for s in self.signal_history]
        layers = [s.dominant_layer for s in self.signal_history]

        return {
            "total_signals": total,
            "direction_distribution": {
                "long": sum(1 for d in directions if d == TradingDirection.LONG) / total,
                "short": sum(1 for d in directions if d == TradingDirection.SHORT) / total,
                "neutral": sum(1 for d in directions if d == TradingDirection.NEUTRAL) / total
            },
            "dominant_layer_distribution": {
                "market": sum(1 for l in layers if l == SignalLayer.MARKET_ASSESSMENT) / total,
                "strategy": sum(1 for l in layers if l == SignalLayer.STRATEGY_SIGNAL) / total,
                "fusion": sum(1 for l in layers if l == SignalLayer.FUSION_DECISION) / total
            },
            "average_confidence": sum(s.final_confidence for s in self.signal_history) / total,
            "average_strength": sum(s.final_strength for s in self.signal_history) / total
        }


__all__ = [
    'MarketSignalBridge', 'BridgedSignal', 'MarketSignalWeight',
    'SignalLayer', 'SignalPriority'
]