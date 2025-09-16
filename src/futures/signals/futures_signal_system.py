"""
期货信号系统

实现完整的期货交易信号生成系统，包括：
- 多时间框架信号聚合
- 信号强度计算和方向决策
- 杠杆和风险参数整合
- 操作类型决策（开仓/加仓/平仓/反向开仓）
- 止盈止损价格计算
- 预期收益和风险评分
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import asyncio
import logging
import pandas as pd
from enum import Enum
import numpy as np

# 导入项目模块
from .strategy_manager import FuturesStrategyManager, AggregatedSignal
from .base_strategy import FuturesBaseStrategy, SignalStrength
from .tp_sl_calculator import TpSlCalculator, TpSlCalculationParams, RiskRewardStrategy
from ..models.data_models import (
    FuturesSignal, TradingDirection, OperationType, RiskLevel,
    Position, MarginStatus, ValidationResult, ValidationSeverity
)
from ..leverage.leverage_controller import LeverageController, LeverageStrategy
from ..market.market_analyzer import MarketAnalyzer
from ..constants import RiskParameters
from ...utils.exceptions import (
    ContractTradingError,
    MarginInsufficientError,
    LeverageExceedsLimitError
)

logger = logging.getLogger(__name__)


class PositionOperation(Enum):
    """仓位操作类型"""
    OPEN_LONG = "open_long"           # 开多仓
    OPEN_SHORT = "open_short"         # 开空仓
    ADD_LONG = "add_long"             # 加多仓
    ADD_SHORT = "add_short"           # 加空仓
    REDUCE_LONG = "reduce_long"       # 减多仓
    REDUCE_SHORT = "reduce_short"     # 减空仓
    CLOSE_LONG = "close_long"         # 平多仓
    CLOSE_SHORT = "close_short"       # 平空仓
    REVERSE_TO_LONG = "reverse_to_long"   # 反向到多仓
    REVERSE_TO_SHORT = "reverse_to_short" # 反向到空仓
    HOLD = "hold"                     # 持仓不变


@dataclass
class TimeframeSignal:
    """单个时间框架的信号"""
    timeframe: str
    direction: TradingDirection
    strength: float
    confidence: float
    raw_signal: Optional[FuturesSignal] = None


@dataclass
class SignalConfidenceMetrics:
    """信号置信度指标"""
    signal_consistency: float = 0.0      # 信号一致性 (0-1)
    timeframe_agreement: float = 0.0     # 时间框架一致性 (0-1)
    volume_confirmation: float = 0.0     # 成交量确认度 (0-1)
    trend_alignment: float = 0.0         # 趋势对齐度 (0-1)
    momentum_strength: float = 0.0       # 动量强度 (0-1)
    support_resistance: float = 0.0      # 支撑阻力确认 (0-1)

    def calculate_overall_confidence(self) -> float:
        """计算总体置信度"""
        weights = {
            'signal_consistency': 0.25,
            'timeframe_agreement': 0.20,
            'volume_confirmation': 0.15,
            'trend_alignment': 0.15,
            'momentum_strength': 0.15,
            'support_resistance': 0.10
        }

        total_confidence = (
            self.signal_consistency * weights['signal_consistency'] +
            self.timeframe_agreement * weights['timeframe_agreement'] +
            self.volume_confirmation * weights['volume_confirmation'] +
            self.trend_alignment * weights['trend_alignment'] +
            self.momentum_strength * weights['momentum_strength'] +
            self.support_resistance * weights['support_resistance']
        )

        return min(max(total_confidence, 0.0), 1.0)


class FuturesSignalSystem:
    """
    期货信号系统

    核心功能：
    1. 多时间框架信号聚合
    2. 信号强度计算和方向决策
    3. 操作类型决策矩阵
    4. 杠杆和风险参数整合
    5. 止盈止损价格计算
    6. 预期收益和持仓时间预测
    """

    # 时间框架权重配置
    TIMEFRAME_WEIGHTS = {
        '5m': 0.10,
        '15m': 0.15,
        '30m': 0.20,
        '1h': 0.30,
        '4h': 0.25
    }

    # 信号强度阈值（已优化：进一步降低阈值提高敏感度）
    SIGNAL_THRESHOLDS = {
        'strong_long': 0.6,
        'long': 0.25,        # 从0.35进一步降低到0.25，更加敏感
        'neutral_upper': 0.15,
        'neutral_lower': -0.15,
        'short': -0.25,      # 从-0.35进一步降低到-0.25，更加敏感
        'strong_short': -0.6
    }

    def __init__(
        self,
        strategy_manager: Optional[FuturesStrategyManager] = None,
        leverage_controller: Optional[LeverageController] = None,
        market_analyzer: Optional[MarketAnalyzer] = None,
        tp_sl_calculator: Optional[TpSlCalculator] = None,
        risk_params: Optional[Dict[str, Any]] = None,
        enable_market_bridge: bool = True,
        market_bridge_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化期货信号系统

        Args:
            strategy_manager: 策略管理器
            leverage_controller: 杠杆控制器
            market_analyzer: 市场分析器
            tp_sl_calculator: 止盈止损计算器
            risk_params: 风险参数配置
            enable_market_bridge: 是否启用市场信号桥接
            market_bridge_config: 市场信号桥接配置
        """
        self.strategy_manager = strategy_manager or FuturesStrategyManager()
        self.leverage_controller = leverage_controller or LeverageController()
        self.market_analyzer = market_analyzer
        self.tp_sl_calculator = tp_sl_calculator or TpSlCalculator()
        self.risk_params = risk_params or {
            'max_position_ratio': RiskParameters.MAX_POSITION_RATIO,
            'min_position_size': 10.0,
            'max_position_size': 10000.0
        }

        # 市场信号桥接配置
        self.enable_market_bridge = enable_market_bridge
        self.market_signal_bridge = None
        
        if self.enable_market_bridge:
            from .market_signal_bridge import MarketSignalBridge, MarketSignalWeight
            
            # 从配置创建桥接器
            bridge_config = market_bridge_config or {}
            weight_config = bridge_config.get('signal_weights', {})
            
            market_weight = MarketSignalWeight(
                trend_weight=weight_config.get('trend_weight', 0.3),
                risk_weight=weight_config.get('risk_weight', 0.4),
                liquidity_weight=weight_config.get('liquidity_weight', 0.2),
                sentiment_weight=weight_config.get('sentiment_weight', 0.1)
            )
            
            risk_preference = bridge_config.get('risk_preference', 'moderate')
            self.market_signal_bridge = MarketSignalBridge(
                market_weight=market_weight,
                risk_preference=risk_preference
            )

        # 系统状态
        self.is_initialized = False
        self.last_signal_time: Optional[datetime] = None

        # 缓存和统计
        self._signal_cache: Dict[str, FuturesSignal] = {}
        self._performance_stats = {
            "total_signals": 0,
            "successful_signals": 0,
            "signal_accuracy": {},
            "avg_processing_time": 0.0,
            "direction_distribution": {},
            "bridge_statistics": {}  # 桥接统计信息
        }

        logger.info(f"期货信号系统已初始化（包含专业止盈止损计算器）"
                   f"{'，已启用市场信号桥接' if self.enable_market_bridge else ''}")

    def initialize(self) -> bool:
        """
        初始化信号系统

        Returns:
            是否初始化成功
        """
        try:
            # 初始化策略管理器
            if not self.strategy_manager.get_active_strategies():
                logger.warning("没有活跃的策略，请添加策略")

            # 验证组件可用性
            components_status = {
                "strategy_manager": len(self.strategy_manager.strategies) > 0,
                "leverage_controller": self.leverage_controller is not None,
                "market_analyzer": self.market_analyzer is not None
            }

            logger.info(f"期货信号系统组件状态: {components_status}")

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"信号系统初始化失败: {e}")
            return False

    async def generate_signal(
        self,
        ticker: str,
        market_data: Dict[str, pd.DataFrame],
        current_price: float,
        current_positions: Optional[List[Position]] = None,
        margin_status: Optional[MarginStatus] = None,
        leverage_strategy: LeverageStrategy = LeverageStrategy.MODERATE,
        **kwargs
    ) -> Optional[FuturesSignal]:
        """
        生成完整的期货交易信号

        Args:
            ticker: 交易对符号
            market_data: 多时间框架市场数据
            current_price: 当前价格
            current_positions: 当前持仓
            margin_status: 保证金状态
            leverage_strategy: 杠杆策略
            **kwargs: 其他参数

        Returns:
            完整的期货交易信号
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("信号系统未能初始化，无法生成信号")
                return None

        start_time = datetime.now()

        try:
            logger.info(f"开始生成期货信号: {ticker}")

            # 1. 收集多时间框架信号
            timeframe_signals = await self._collect_timeframe_signals(
                ticker, market_data, current_price, **kwargs
            )

            if not timeframe_signals:
                logger.warning(f"未能收集到有效的时间框架信号: {ticker}")
                return None

            # 2. 计算综合信号强度
            signal_strength = self._calculate_signal_strength(timeframe_signals)

            # 3. 确定交易方向
            direction = self._determine_direction(signal_strength)

            # 4. 计算置信度指标
            confidence_metrics = self._calculate_confidence_metrics(
                timeframe_signals, market_data, current_price
            )

            # === 新增：市场信号桥接层 ===
            # 5. 市场分析和信号桥接
            final_direction = direction
            final_confidence = confidence_metrics.calculate_overall_confidence()
            final_strength = abs(signal_strength)
            bridge_metadata = {}

            if self.enable_market_bridge and self.market_analyzer and self.market_signal_bridge:
                try:
                    # 获取市场条件分析
                    market_analysis = await self.market_analyzer.analyze_market_conditions(
                        ticker=ticker,
                        price_data=market_data.get('1h', list(market_data.values())[0]),
                        volume_data=None  # 可以从market_data中提取volume数据
                    )

                    # 桥接信号
                    bridged_signal = self.market_signal_bridge.bridge_signals(
                        market_analysis=market_analysis,
                        strategy_direction=direction,
                        strategy_strength=abs(signal_strength),
                        strategy_confidence=confidence_metrics.calculate_overall_confidence() * 100
                    )

                    # 使用桥接后的信号
                    final_direction = bridged_signal.final_direction
                    final_confidence = bridged_signal.final_confidence
                    final_strength = bridged_signal.final_strength

                    # 记录桥接信息
                    bridge_metadata = {
                        "bridge_enabled": True,
                        "market_analysis": market_analysis.to_dict(),
                        "bridged_signal": bridged_signal.to_dict(),
                        "original_direction": direction.value,
                        "final_direction": final_direction.value,
                        "signal_changed": direction != final_direction,
                        "dominant_layer": bridged_signal.dominant_layer.value,
                        "fusion_logic": bridged_signal.fusion_logic
                    }

                    logger.info(f"市场信号桥接完成: {ticker} "
                              f"原始={direction.value} → 最终={final_direction.value} "
                              f"置信度={final_confidence:.2f} 逻辑={bridged_signal.fusion_logic}")

                except Exception as e:
                    logger.error(f"市场信号桥接失败，使用原始信号: {e}")
                    bridge_metadata = {"bridge_enabled": False, "bridge_error": str(e)}
            else:
                bridge_metadata = {"bridge_enabled": False, "reason": "桥接未启用或组件缺失"}

            # 6. 确定操作类型（基于最终方向）
            operation = self._determine_operation(
                final_direction, current_positions, ticker, confidence_metrics
            )

            # 7. 计算杠杆建议（传递市场风险信息）
            # 构建增强的市场数据，包含风险等级
            enhanced_market_data = self._extract_market_data_for_leverage(market_data)

            # 如果有市场分析结果，添加风险等级
            if bridge_metadata.get('market_analysis'):
                market_analysis_dict = bridge_metadata['market_analysis']
                if 'risk_assessment' in market_analysis_dict:
                    risk_info = market_analysis_dict['risk_assessment']
                    # 将风险等级添加到市场数据中
                    enhanced_market_data['risk_level'] = risk_info.get('risk_level', 'medium')
                    enhanced_market_data['risk_score'] = risk_info.get('overall_risk_score', 50)
                    enhanced_market_data['volatility'] = risk_info.get('volatility_risk', 0.2)

                    logger.info(f"传递市场风险等级到杠杆控制器: {enhanced_market_data.get('risk_level', 'unknown')}")

            leverage_result = await self.leverage_controller.calculate_optimal_leverage(
                ticker=ticker,
                strategy=leverage_strategy,
                current_positions=current_positions,
                margin_status=margin_status,
                market_data=enhanced_market_data
            )

            # 8. 计算仓位大小（基于最终方向）
            position_size = self._calculate_position_size(
                ticker, final_direction, leverage_result, margin_status, current_price
            )

            # 9. 使用专业止盈止损计算器（基于最终方向和强度）
            tp_sl_result = await self._calculate_advanced_tp_sl(
                ticker, final_direction, current_price, leverage_result,
                final_strength, confidence_metrics, market_data
            )

            # 提取止盈止损价格
            tp_price = tp_sl_result.take_profit_price if tp_sl_result else None
            sl_price = tp_sl_result.stop_loss_price if tp_sl_result else None

            # 10. 预测收益和持仓时间（基于最终参数）
            expected_return, expected_duration = self._predict_return_and_duration(
                final_direction, final_strength, confidence_metrics, tp_price, sl_price, current_price
            )

            # 11. 计算风险评分
            risk_score = self._calculate_risk_score(
                leverage_result, confidence_metrics, margin_status
            )

            # 12. 创建完整信号（使用最终参数）
            signal = FuturesSignal(
                ticker=ticker,
                direction=final_direction,
                operation_type=operation,
                confidence=final_confidence * 100,
                strength=final_strength,
                suggested_leverage=leverage_result.applied_leverage,
                position_size=position_size,
                entry_price=current_price,
                current_price=current_price,
                take_profit_price=tp_price,
                stop_loss_price=sl_price,
                expiry_time=datetime.now() + timedelta(minutes=30),  # 信号30分钟有效
                risk_level=self._determine_risk_level(risk_score),
                strategy_source="futures_signal_system_with_bridge",
                signal_id=f"fs_{ticker}_{int(datetime.now().timestamp())}",
                metadata={
                    # 原始信号信息
                    "original_signal_strength": signal_strength,
                    "original_direction": direction.value,
                    "timeframe_signals": len(timeframe_signals),
                    "confidence_breakdown": confidence_metrics.__dict__,
                    
                    # 桥接信息
                    "bridge_metadata": bridge_metadata,
                    
                    # 最终结果
                    "final_direction": final_direction.value,
                    "final_confidence": final_confidence,
                    "final_strength": final_strength,
                    
                    # 其他元数据
                    "leverage_adjustment": leverage_result.get_adjustment_percentage(),
                    "expected_return": expected_return,
                    "expected_duration_hours": expected_duration,
                    "risk_score": risk_score,
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "tp_sl_result": tp_sl_result.__dict__ if tp_sl_result else None
                }
            )

            # 13. 验证信号
            validation_result = self._validate_signal(signal, margin_status, current_positions)
            if not validation_result:
                logger.warning(f"信号验证失败: {ticker}")
                return None

            # 14. 缓存信号
            self._cache_signal(ticker, signal)

            # 15. 更新统计（包括桥接统计）
            self._update_performance_stats(signal, datetime.now() - start_time)
            if self.enable_market_bridge and self.market_signal_bridge:
                self._performance_stats["bridge_statistics"] = self.market_signal_bridge.get_signal_statistics()

            logger.info(
                f"信号生成完成: {ticker} 方向={final_direction.value} "
                f"操作={operation.value} 杠杆={leverage_result.applied_leverage:.1f} "
                f"置信度={signal.confidence:.1f}% 强度={signal.strength:.3f}"
                f"{' [桥接生效]' if bridge_metadata.get('signal_changed', False) else ''}"
            )

            return signal

        except Exception as e:
            logger.error(f"生成期货信号失败 '{ticker}': {e}")
            return None

    async def _collect_timeframe_signals(
        self,
        ticker: str,
        market_data: Dict[str, pd.DataFrame],
        current_price: float,
        **kwargs
    ) -> List[TimeframeSignal]:
        """
        收集多时间框架信号

        Args:
            ticker: 交易对
            market_data: 多时间框架数据
            current_price: 当前价格

        Returns:
            时间框架信号列表
        """
        timeframe_signals = []

        try:
            for timeframe, weight in self.TIMEFRAME_WEIGHTS.items():
                if timeframe not in market_data:
                    logger.warning(f"缺少时间框架数据: {timeframe}")
                    continue

                # 传递完整的多时间框架数据给策略
                # 策略需要所有支持的时间框架数据来进行验证和分析
                aggregated_result = self.strategy_manager.analyze_ticker(
                    ticker=ticker,
                    data=market_data,  # 传递完整数据而不是单个时间框架
                    current_price=current_price,
                    primary_timeframe=timeframe,  # 指定当前分析的主要时间框架
                    **kwargs
                )

                if aggregated_result and aggregated_result.final_signal:
                    signal = aggregated_result.final_signal

                    # 创建时间框架信号
                    tf_signal = TimeframeSignal(
                        timeframe=timeframe,
                        direction=signal.direction,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        raw_signal=signal
                    )

                    timeframe_signals.append(tf_signal)

                    logger.debug(
                        f"时间框架信号 {timeframe}: 方向={signal.direction.value} "
                        f"强度={signal.strength:.3f} 置信度={signal.confidence:.1f}%"
                    )

            logger.info(f"收集到 {len(timeframe_signals)} 个时间框架信号")
            return timeframe_signals

        except Exception as e:
            logger.error(f"收集时间框架信号失败: {e}")
            return []

    def _calculate_signal_strength(self, timeframe_signals: List[TimeframeSignal]) -> float:
        """
        计算综合信号强度

        Args:
            timeframe_signals: 时间框架信号列表

        Returns:
            综合信号强度 (-1到1)
        """
        try:
            if not timeframe_signals:
                return 0.0

            total_weighted_strength = 0.0
            total_weight = 0.0

            for tf_signal in timeframe_signals:
                weight = self.TIMEFRAME_WEIGHTS.get(tf_signal.timeframe, 0.1)
                confidence_weight = tf_signal.confidence / 100.0  # 置信度权重

                # 方向性强度
                if tf_signal.direction == TradingDirection.LONG:
                    directional_strength = tf_signal.strength
                elif tf_signal.direction == TradingDirection.SHORT:
                    directional_strength = -tf_signal.strength
                else:  # NEUTRAL
                    directional_strength = 0.0

                # 加权累计
                effective_weight = weight * confidence_weight
                total_weighted_strength += directional_strength * effective_weight
                total_weight += effective_weight

            if total_weight == 0:
                return 0.0

            # 计算加权平均强度
            signal_strength = total_weighted_strength / total_weight

            # 限制在 [-1, 1] 范围内
            signal_strength = max(-1.0, min(1.0, signal_strength))

            # 详细的调试日志
            logger.debug(
                f"综合信号强度计算完成: {signal_strength:.4f} "
                f"(来源: {len(timeframe_signals)}个时间框架, 总权重: {total_weight:.3f})"
            )

            # 输出每个时间框架的贡献
            for tf_signal in timeframe_signals:
                weight = self.TIMEFRAME_WEIGHTS.get(tf_signal.timeframe, 0.1)
                confidence_weight = tf_signal.confidence / 100.0
                logger.debug(
                    f"  {tf_signal.timeframe}: 方向={tf_signal.direction.value} "
                    f"强度={tf_signal.strength:.3f} 置信度={tf_signal.confidence:.1f}% "
                    f"权重={weight*confidence_weight:.3f}"
                )

            return signal_strength

        except Exception as e:
            logger.error(f"计算信号强度失败: {e}")
            return 0.0

    def _determine_direction(self, signal_strength: float) -> TradingDirection:
        """
        根据信号强度确定交易方向

        Args:
            signal_strength: 信号强度 (-1到1)

        Returns:
            交易方向
        """
        try:
            direction = TradingDirection.NEUTRAL

            if signal_strength >= self.SIGNAL_THRESHOLDS['long']:
                direction = TradingDirection.LONG
            elif signal_strength <= self.SIGNAL_THRESHOLDS['short']:
                direction = TradingDirection.SHORT
            else:
                direction = TradingDirection.NEUTRAL

            # 添加详细调试日志
            logger.debug(
                f"信号方向判断: 强度={signal_strength:.4f} "
                f"阈值[多头>={self.SIGNAL_THRESHOLDS['long']}, 空头<={self.SIGNAL_THRESHOLDS['short']}] "
                f"→ 方向={direction.value}"
            )

            return direction

        except Exception as e:
            logger.error(f"确定交易方向失败: {e}")
            return TradingDirection.NEUTRAL

    def _calculate_confidence_metrics(
        self,
        timeframe_signals: List[TimeframeSignal],
        market_data: Dict[str, pd.DataFrame],
        current_price: float
    ) -> SignalConfidenceMetrics:
        """
        计算信号置信度指标

        Args:
            timeframe_signals: 时间框架信号
            market_data: 市场数据
            current_price: 当前价格

        Returns:
            置信度指标
        """
        metrics = SignalConfidenceMetrics()

        try:
            if not timeframe_signals:
                return metrics

            # 1. 信号一致性 - 同方向信号的占比
            direction_counts = {
                TradingDirection.LONG: 0,
                TradingDirection.SHORT: 0,
                TradingDirection.NEUTRAL: 0
            }

            total_confidence = 0.0
            for signal in timeframe_signals:
                direction_counts[signal.direction] += 1
                total_confidence += signal.confidence

            max_direction_count = max(direction_counts.values())
            metrics.signal_consistency = max_direction_count / len(timeframe_signals)

            # 2. 时间框架一致性 - 重要时间框架的权重一致性
            important_timeframes = ['1h', '4h']
            important_signals = [s for s in timeframe_signals if s.timeframe in important_timeframes]
            if important_signals:
                important_directions = [s.direction for s in important_signals]
                if len(set(important_directions)) == 1:  # 重要时间框架方向一致
                    metrics.timeframe_agreement = 1.0
                else:
                    metrics.timeframe_agreement = 0.3
            else:
                metrics.timeframe_agreement = 0.5

            # 3. 成交量确认度 - 基于成交量数据
            try:
                volume_confirmation = 0.0
                volume_data_count = 0

                for timeframe, data in market_data.items():
                    if 'volume' in data.columns and len(data) > 20:
                        recent_volume = data['volume'].iloc[-5:].mean()
                        avg_volume = data['volume'].iloc[-20:].mean()

                        if recent_volume > avg_volume * 1.2:  # 成交量放大
                            volume_confirmation += 0.8
                        elif recent_volume > avg_volume:
                            volume_confirmation += 0.6
                        else:
                            volume_confirmation += 0.3

                        volume_data_count += 1

                if volume_data_count > 0:
                    metrics.volume_confirmation = volume_confirmation / volume_data_count
                else:
                    metrics.volume_confirmation = 0.5

            except Exception as e:
                logger.debug(f"成交量确认度计算失败: {e}")
                metrics.volume_confirmation = 0.5

            # 4. 趋势对齐度 - 基于移动平均线
            try:
                trend_scores = []

                for timeframe, data in market_data.items():
                    if 'close' in data.columns and len(data) > 50:
                        # 计算多个周期的移动平均
                        ma_short = data['close'].rolling(10).mean().iloc[-1]
                        ma_medium = data['close'].rolling(20).mean().iloc[-1]
                        ma_long = data['close'].rolling(50).mean().iloc[-1]

                        # 趋势评分
                        if ma_short > ma_medium > ma_long:  # 明显上升趋势
                            trend_scores.append(0.9)
                        elif ma_short < ma_medium < ma_long:  # 明显下降趋势
                            trend_scores.append(0.9)
                        elif ma_short > ma_long:  # 弱上升
                            trend_scores.append(0.6)
                        elif ma_short < ma_long:  # 弱下降
                            trend_scores.append(0.6)
                        else:  # 横盘
                            trend_scores.append(0.3)

                if trend_scores:
                    metrics.trend_alignment = sum(trend_scores) / len(trend_scores)
                else:
                    metrics.trend_alignment = 0.5

            except Exception as e:
                logger.debug(f"趋势对齐度计算失败: {e}")
                metrics.trend_alignment = 0.5

            # 5. 动量强度 - 基于RSI和价格动量
            try:
                momentum_scores = []

                for timeframe, data in market_data.items():
                    if 'close' in data.columns and len(data) > 14:
                        # 计算RSI
                        delta = data['close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rs = gain / loss
                        rsi = 100 - (100 / (1 + rs))
                        current_rsi = rsi.iloc[-1]

                        # 价格变化率
                        price_change = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) * 100

                        # 动量评分
                        if abs(price_change) > 3 and (current_rsi > 70 or current_rsi < 30):
                            momentum_scores.append(0.8)
                        elif abs(price_change) > 1:
                            momentum_scores.append(0.6)
                        else:
                            momentum_scores.append(0.4)

                if momentum_scores:
                    metrics.momentum_strength = sum(momentum_scores) / len(momentum_scores)
                else:
                    metrics.momentum_strength = 0.5

            except Exception as e:
                logger.debug(f"动量强度计算失败: {e}")
                metrics.momentum_strength = 0.5

            # 6. 支撑阻力确认度 - 简化版本
            try:
                # 基于最近的高低点分析
                support_resistance_score = 0.5

                main_timeframe = '1h'
                if main_timeframe in market_data:
                    data = market_data[main_timeframe]
                    if 'high' in data.columns and 'low' in data.columns and len(data) > 20:
                        recent_highs = data['high'].iloc[-10:].max()
                        recent_lows = data['low'].iloc[-10:].min()
                        price_range = recent_highs - recent_lows

                        # 当前价格相对位置
                        if price_range > 0:
                            price_position = (current_price - recent_lows) / price_range

                            # 如果接近支撑或阻力位，给予更高确认度
                            if price_position < 0.1 or price_position > 0.9:
                                support_resistance_score = 0.8
                            elif price_position < 0.2 or price_position > 0.8:
                                support_resistance_score = 0.7
                            else:
                                support_resistance_score = 0.6

                metrics.support_resistance = support_resistance_score

            except Exception as e:
                logger.debug(f"支撑阻力确认度计算失败: {e}")
                metrics.support_resistance = 0.5

            logger.debug(f"置信度指标计算完成: {metrics.__dict__}")
            return metrics

        except Exception as e:
            logger.error(f"计算置信度指标失败: {e}")
            return SignalConfidenceMetrics()

    def _determine_operation(
        self,
        direction: TradingDirection,
        current_positions: Optional[List[Position]],
        ticker: str,
        confidence_metrics: SignalConfidenceMetrics
    ) -> OperationType:
        """
        确定操作类型（开仓、加仓、平仓、反向开仓等）

        Args:
            direction: 交易方向
            current_positions: 当前持仓
            ticker: 交易对
            confidence_metrics: 置信度指标

        Returns:
            操作类型
        """
        try:
            # 找到当前交易对的持仓
            current_position = None
            if current_positions:
                for pos in current_positions:
                    if pos.ticker == ticker and pos.size > 0:
                        current_position = pos
                        break

            # 如果没有持仓
            if not current_position:
                if direction == TradingDirection.NEUTRAL:
                    return OperationType.HOLD
                else:
                    return OperationType.OPEN

            # 如果有持仓，基于方向和置信度决策
            overall_confidence = confidence_metrics.calculate_overall_confidence()

            # 当前持仓方向
            if current_position.side.value == "LONG":
                current_direction = TradingDirection.LONG
            else:
                current_direction = TradingDirection.SHORT

            # 决策逻辑
            if direction == TradingDirection.NEUTRAL:
                # 信号中性，考虑平仓
                if overall_confidence > 0.7:  # 高置信度中性信号
                    return OperationType.CLOSE
                else:
                    return OperationType.HOLD

            elif direction == current_direction:
                # 方向相同，考虑加仓
                if overall_confidence > 0.8 and confidence_metrics.signal_consistency > 0.8:
                    return OperationType.ADD
                else:
                    return OperationType.HOLD

            else:
                # 方向相反，考虑平仓或反向
                if overall_confidence > 0.85:
                    # 高置信度反向信号，反向开仓
                    return OperationType.CLOSE  # 先平仓，后续可以开反向仓
                elif overall_confidence > 0.6:
                    # 中等置信度，减仓
                    return OperationType.REDUCE
                else:
                    # 低置信度，持仓观望
                    return OperationType.HOLD

        except Exception as e:
            logger.error(f"确定操作类型失败: {e}")
            return OperationType.HOLD

    def _calculate_position_size(
        self,
        ticker: str,
        direction: TradingDirection,
        leverage_result: Any,
        margin_status: Optional[MarginStatus],
        current_price: float
    ) -> float:
        """
        计算建议仓位大小

        Args:
            ticker: 交易对
            direction: 交易方向
            leverage_result: 杠杆计算结果
            margin_status: 保证金状态
            current_price: 当前价格

        Returns:
            建议仓位大小 (USDT)
        """
        try:
            if direction == TradingDirection.NEUTRAL:
                return 0.0

            if not margin_status or not margin_status.can_open_position:
                return 0.0

            # 基础仓位计算
            available_margin = margin_status.available_margin
            max_position_ratio = self.risk_params.get('max_position_ratio', 0.2)  # 最大仓位比例20%

            # 保守估算可用保证金
            usable_margin = available_margin * max_position_ratio * 0.8  # 80%安全边际

            # 考虑杠杆
            leverage = leverage_result.applied_leverage
            max_notional = usable_margin * leverage

            # 风险调整
            risk_adjustment = 1.0
            if leverage_result.risk_level in ['HIGH', 'CRITICAL', 'EMERGENCY']:
                risk_adjustment = 0.5
            elif leverage_result.risk_level == 'MEDIUM':
                risk_adjustment = 0.8

            # 最终仓位大小
            position_size = max_notional * risk_adjustment

            # 最小和最大限制
            min_position = self.risk_params.get('min_position_size', 10.0)  # 最小10 USDT
            max_position = self.risk_params.get('max_position_size', 10000.0)  # 最大10000 USDT

            position_size = max(min_position, min(position_size, max_position))

            logger.debug(f"仓位大小计算: {ticker} 杠杆={leverage:.1f} 大小={position_size:.2f} USDT")

            return position_size

        except Exception as e:
            logger.error(f"计算仓位大小失败: {e}")
            return 0.0

    async def _calculate_advanced_tp_sl(
        self,
        ticker: str,
        direction: TradingDirection,
        current_price: float,
        leverage_result: Any,
        signal_strength: float,
        confidence_metrics: SignalConfidenceMetrics,
        market_data: Dict[str, pd.DataFrame]
    ) -> Optional[Any]:
        """
        使用专业止盈止损计算器计算止盈止损

        Args:
            ticker: 交易对
            direction: 交易方向
            current_price: 当前价格
            leverage_result: 杠杆计算结果
            signal_strength: 信号强度
            confidence_metrics: 置信度指标
            market_data: 市场数据

        Returns:
            TpSlResult对象或None
        """
        try:
            if direction == TradingDirection.NEUTRAL:
                logger.debug("中性方向，跳过止盈止损计算")
                return None

            # 提取技术位信息（支撑阻力位）
            support_levels, resistance_levels = self._extract_support_resistance_levels(
                market_data, current_price
            )

            # 计算波动率
            volatility = self._calculate_market_volatility(market_data)

            # 计算趋势强度和动量
            trend_strength = self._calculate_trend_strength(market_data)
            momentum_score = confidence_metrics.momentum_strength

            # 根据置信度选择风险收益比策略
            confidence = confidence_metrics.calculate_overall_confidence()
            if confidence > 0.8 and abs(signal_strength) > 0.7:
                risk_strategy = RiskRewardStrategy.AGGRESSIVE
            elif confidence > 0.6:
                risk_strategy = RiskRewardStrategy.BALANCED
            elif confidence > 0.4:
                risk_strategy = RiskRewardStrategy.CONSERVATIVE
            else:
                risk_strategy = RiskRewardStrategy.CONSERVATIVE

            # 计算强平价格（如果有保证金信息）
            liquidation_price = self._estimate_liquidation_price(
                current_price, direction, leverage_result.applied_leverage
            )

            # 构建计算参数
            tp_sl_params = TpSlCalculationParams(
                ticker=ticker,
                direction=direction,
                current_price=current_price,
                leverage=leverage_result.applied_leverage,
                market_data=market_data,
                volatility=volatility,
                risk_reward_strategy=risk_strategy,
                liquidation_price=liquidation_price,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_strength=trend_strength,
                momentum_score=momentum_score,
                liquidity_score=getattr(leverage_result, 'liquidity_score', 1.0),
                max_holding_hours=24.0,  # 默认24小时最大持仓
                enable_trailing_stop=True,
                enable_tiered_tp=True
            )

            # 调用止盈止损计算器
            tp_sl_result = await self.tp_sl_calculator.calculate_tp_sl(tp_sl_params)

            if tp_sl_result and tp_sl_result.is_valid():
                logger.info(
                    f"专业止盈止损计算完成: {ticker} "
                    f"止盈={tp_sl_result.take_profit_price:.6f} "
                    f"止损={tp_sl_result.stop_loss_price:.6f} "
                    f"风险收益比=1:{tp_sl_result.risk_reward_ratio:.2f} "
                    f"置信度={tp_sl_result.confidence_score:.2f}"
                )
                return tp_sl_result
            else:
                logger.warning(f"专业止盈止损计算无效，使用简单方法: {ticker}")
                return self._fallback_simple_tp_sl_calculation(
                    ticker, direction, current_price, leverage_result.applied_leverage,
                    signal_strength, confidence_metrics
                )

        except Exception as e:
            logger.error(f"专业止盈止损计算失败: {e}")
            return self._fallback_simple_tp_sl_calculation(
                ticker, direction, current_price, leverage_result.applied_leverage,
                signal_strength, confidence_metrics
            )

    def _fallback_simple_tp_sl_calculation(
        self,
        ticker: str,
        direction: TradingDirection,
        current_price: float,
        leverage: float,
        signal_strength: float,
        confidence_metrics: SignalConfidenceMetrics
    ) -> Optional[Any]:
        """
        简单止盈止损计算（作为回退方案）

        Returns:
            简化的TpSlResult对象
        """
        try:
            from .tp_sl_calculator import TpSlResult

            # 基础风险收益比
            confidence = confidence_metrics.calculate_overall_confidence()
            if confidence > 0.8 and abs(signal_strength) > 0.7:
                risk_reward_ratio = 2.5
            elif confidence > 0.6:
                risk_reward_ratio = 2.0
            else:
                risk_reward_ratio = 1.5

            # 根据杠杆调整止损距离
            base_stop_loss_pct = 0.02  # 基础2%止损
            if leverage > 10:
                stop_loss_pct = base_stop_loss_pct * 0.5
            elif leverage > 5:
                stop_loss_pct = base_stop_loss_pct * 0.7
            else:
                stop_loss_pct = base_stop_loss_pct

            # 根据置信度调整
            volatility_adjustment = 1.0 + (1.0 - confidence) * 0.5
            stop_loss_pct *= volatility_adjustment

            # 计算价格
            if direction == TradingDirection.LONG:
                stop_loss_price = current_price * (1 - stop_loss_pct)
                take_profit_price = current_price * (1 + stop_loss_pct * risk_reward_ratio)
            else:  # SHORT
                stop_loss_price = current_price * (1 + stop_loss_pct)
                take_profit_price = current_price * (1 - stop_loss_pct * risk_reward_ratio)

            # 价格精度调整
            precision = self._get_price_precision(current_price)
            take_profit_price = round(take_profit_price, precision)
            stop_loss_price = round(stop_loss_price, precision)

            # 创建简化结果
            simple_result = TpSlResult(
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                risk_reward_ratio=risk_reward_ratio,
                stop_loss_pct=abs(stop_loss_pct * 100),
                take_profit_pct=abs(stop_loss_pct * risk_reward_ratio * 100),
                confidence_score=confidence,
                warnings=["使用简化计算方法"],
                recommendations=["建议使用完整的市场数据进行高级计算"]
            )

            logger.debug(f"简化止盈止损计算完成: {ticker}")
            return simple_result

        except Exception as e:
            logger.error(f"简化止盈止损计算失败: {e}")
            return None

    def _extract_support_resistance_levels(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_price: float
    ) -> Tuple[List[float], List[float]]:
        """提取支撑阻力位"""
        try:
            support_levels = []
            resistance_levels = []

            # 使用1小时数据分析支撑阻力
            timeframe = '1h'
            if timeframe in market_data:
                data = market_data[timeframe]
                if 'high' in data.columns and 'low' in data.columns and len(data) > 50:
                    # 简单的支撑阻力识别
                    recent_data = data.iloc[-50:]
                    highs = recent_data['high']
                    lows = recent_data['low']

                    # 寻找局部高点和低点
                    for i in range(2, len(recent_data) - 2):
                        # 局部高点（阻力位）
                        if (highs.iloc[i] > highs.iloc[i-1] and
                            highs.iloc[i] > highs.iloc[i+1] and
                            highs.iloc[i] > highs.iloc[i-2] and
                            highs.iloc[i] > highs.iloc[i+2]):
                            resistance_levels.append(highs.iloc[i])

                        # 局部低点（支撑位）
                        if (lows.iloc[i] < lows.iloc[i-1] and
                            lows.iloc[i] < lows.iloc[i+1] and
                            lows.iloc[i] < lows.iloc[i-2] and
                            lows.iloc[i] < lows.iloc[i+2]):
                            support_levels.append(lows.iloc[i])

                    # 去重并排序
                    support_levels = sorted(list(set([round(level, 6) for level in support_levels])))
                    resistance_levels = sorted(list(set([round(level, 6) for level in resistance_levels])))

            logger.debug(f"提取到支撑位: {len(support_levels)}个, 阻力位: {len(resistance_levels)}个")
            return support_levels, resistance_levels

        except Exception as e:
            logger.error(f"提取支撑阻力位失败: {e}")
            return [], []

    def _calculate_market_volatility(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """计算市场波动率"""
        try:
            # 使用1小时数据计算波动率
            timeframe = '1h'
            if timeframe in market_data:
                data = market_data[timeframe]
                if 'close' in data.columns and len(data) > 20:
                    returns = data['close'].pct_change().dropna()
                    volatility = returns.std() * (24 ** 0.5)  # 年化波动率
                    return min(max(volatility, 0.005), 0.20)  # 限制在0.5%-20%之间

            # 默认波动率
            return 0.02

        except Exception as e:
            logger.error(f"计算市场波动率失败: {e}")
            return 0.02

    def _calculate_trend_strength(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """计算趋势强度"""
        try:
            # 使用1小时数据计算趋势强度
            timeframe = '1h'
            if timeframe in market_data:
                data = market_data[timeframe]
                if 'close' in data.columns and len(data) > 50:
                    close = data['close']

                    # 简单的趋势强度：基于移动平均线的斜率
                    ma_short = close.rolling(10).mean()
                    ma_long = close.rolling(30).mean()

                    # 计算斜率
                    short_slope = (ma_short.iloc[-1] - ma_short.iloc[-10]) / ma_short.iloc[-10]
                    long_slope = (ma_long.iloc[-1] - ma_long.iloc[-30]) / ma_long.iloc[-30]

                    # 综合趋势强度
                    trend_strength = (short_slope + long_slope) / 2

                    # 限制在 -1 到 1 之间
                    return max(-1.0, min(1.0, trend_strength * 100))

            return 0.0

        except Exception as e:
            logger.error(f"计算趋势强度失败: {e}")
            return 0.0

    def _estimate_liquidation_price(
        self,
        current_price: float,
        direction: TradingDirection,
        leverage: float
    ) -> Optional[float]:
        """估算强平价格"""
        try:
            # 简化的强平价格估算
            maintenance_margin_rate = 0.004  # 假设0.4%维持保证金率

            if direction == TradingDirection.LONG:
                # 多头强平价格 = 入场价格 * (1 - 1/杠杆 + 维持保证金率)
                liquidation_price = current_price * (1 - (1 / leverage) + maintenance_margin_rate)
            else:  # SHORT
                # 空头强平价格 = 入场价格 * (1 + 1/杠杆 - 维持保证金率)
                liquidation_price = current_price * (1 + (1 / leverage) - maintenance_margin_rate)

            return liquidation_price

        except Exception as e:
            logger.error(f"估算强平价格失败: {e}")
            return None

    def _get_price_precision(self, price: float) -> int:
        """获取价格精度"""
        if price > 1000:
            return 0
        elif price > 100:
            return 1
        elif price > 10:
            return 2
        elif price > 1:
            return 3
        else:
            return 6

    def _predict_return_and_duration(
        self,
        direction: TradingDirection,
        signal_strength: float,
        confidence_metrics: SignalConfidenceMetrics,
        tp_price: Optional[float],
        sl_price: Optional[float],
        current_price: float
    ) -> Tuple[float, float]:
        """
        预测收益率和持仓时间

        Args:
            direction: 交易方向
            signal_strength: 信号强度
            confidence_metrics: 置信度指标
            tp_price: 止盈价格
            sl_price: 止损价格
            current_price: 当前价格

        Returns:
            (预期收益率%, 预期持仓时间小时)
        """
        try:
            if direction == TradingDirection.NEUTRAL or not tp_price or not sl_price:
                return 0.0, 0.0

            # 计算潜在收益和损失
            if direction == TradingDirection.LONG:
                potential_profit_pct = (tp_price - current_price) / current_price * 100
                potential_loss_pct = (current_price - sl_price) / current_price * 100
            else:  # SHORT
                potential_profit_pct = (current_price - tp_price) / current_price * 100
                potential_loss_pct = (sl_price - current_price) / current_price * 100

            # 基于置信度计算成功概率
            confidence = confidence_metrics.calculate_overall_confidence()
            signal_consistency = confidence_metrics.signal_consistency

            # 成功概率估算
            success_probability = confidence * 0.7 + signal_consistency * 0.3
            success_probability = min(max(success_probability, 0.3), 0.9)  # 限制在30%-90%

            # 期望收益计算
            expected_return = (
                success_probability * potential_profit_pct -
                (1 - success_probability) * potential_loss_pct
            )

            # 持仓时间预测（基于信号强度和市场条件）
            base_duration = 24.0  # 基础24小时

            # 信号强度调整
            strength_multiplier = 0.5 + abs(signal_strength) * 1.5

            # 置信度调整
            confidence_multiplier = 0.8 + confidence * 0.4

            expected_duration = base_duration * strength_multiplier * confidence_multiplier

            # 限制在合理范围
            expected_duration = max(4.0, min(expected_duration, 72.0))  # 4-72小时

            logger.debug(
                f"收益时间预测: 预期收益={expected_return:.2f}% "
                f"预期时间={expected_duration:.1f}h 成功概率={success_probability:.2f}"
            )

            return expected_return, expected_duration

        except Exception as e:
            logger.error(f"预测收益和持仓时间失败: {e}")
            return 0.0, 24.0

    def _calculate_risk_score(
        self,
        leverage_result: Any,
        confidence_metrics: SignalConfidenceMetrics,
        margin_status: Optional[MarginStatus]
    ) -> float:
        """
        计算综合风险评分

        Args:
            leverage_result: 杠杆计算结果
            confidence_metrics: 置信度指标
            margin_status: 保证金状态

        Returns:
            风险评分 (0-100, 越高越危险)
        """
        try:
            risk_components = []

            # 1. 杠杆风险
            leverage = leverage_result.applied_leverage
            if leverage <= 2:
                leverage_risk = 10
            elif leverage <= 5:
                leverage_risk = 25
            elif leverage <= 10:
                leverage_risk = 50
            elif leverage <= 20:
                leverage_risk = 75
            else:
                leverage_risk = 95

            risk_components.append(('leverage', leverage_risk, 0.3))

            # 2. 置信度风险（置信度越低风险越高）
            confidence = confidence_metrics.calculate_overall_confidence()
            confidence_risk = (1 - confidence) * 100
            risk_components.append(('confidence', confidence_risk, 0.25))

            # 3. 保证金风险
            margin_risk = 30  # 默认中等风险
            if margin_status:
                if margin_status.risk_level.value == 'EMERGENCY':
                    margin_risk = 95
                elif margin_status.risk_level.value == 'CRITICAL':
                    margin_risk = 85
                elif margin_status.risk_level.value == 'HIGH':
                    margin_risk = 70
                elif margin_status.risk_level.value == 'MEDIUM':
                    margin_risk = 40
                else:
                    margin_risk = 20

            risk_components.append(('margin', margin_risk, 0.2))

            # 4. 市场条件风险
            market_risk = 40  # 默认中等市场风险
            if hasattr(leverage_result, 'market_condition') and leverage_result.market_condition:
                volatility = leverage_result.market_condition.volatility
                market_risk = min(volatility * 100, 90)  # 波动率转换为风险评分

            risk_components.append(('market', market_risk, 0.15))

            # 5. 信号质量风险
            signal_risk = 50
            if confidence_metrics.signal_consistency > 0.8:
                signal_risk = 20
            elif confidence_metrics.signal_consistency > 0.6:
                signal_risk = 35
            elif confidence_metrics.signal_consistency < 0.4:
                signal_risk = 70

            risk_components.append(('signal_quality', signal_risk, 0.1))

            # 计算加权总风险
            total_risk = sum(risk * weight for _, risk, weight in risk_components)

            # 限制在0-100范围
            total_risk = max(0, min(total_risk, 100))

            logger.debug(f"风险评分计算完成: 总分={total_risk:.1f} 组件={risk_components}")

            return total_risk

        except Exception as e:
            logger.error(f"计算风险评分失败: {e}")
            return 50.0  # 返回中等风险评分

    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """
        根据风险评分确定风险等级

        Args:
            risk_score: 风险评分 (0-100)

        Returns:
            风险等级
        """
        if risk_score >= 80:
            return RiskLevel.EMERGENCY
        elif risk_score >= 65:
            return RiskLevel.CRITICAL
        elif risk_score >= 50:
            return RiskLevel.HIGH
        elif risk_score >= 30:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _extract_market_data_for_leverage(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        从市场数据中提取杠杆计算所需的信息

        Args:
            market_data: 多时间框架市场数据

        Returns:
            杠杆计算所需的市场数据
        """
        try:
            leverage_data = {}

            # 使用1小时数据作为主要参考
            main_timeframe = '1h'
            if main_timeframe in market_data:
                data = market_data[main_timeframe]

                if 'close' in data.columns and len(data) > 20:
                    # 计算波动率
                    returns = data['close'].pct_change().dropna()
                    volatility = returns.std() * (24 ** 0.5)  # 年化波动率
                    leverage_data['volatility'] = volatility

                    # 计算流动性评分（基于成交量）
                    if 'volume' in data.columns:
                        recent_volume = data['volume'].iloc[-5:].mean()
                        avg_volume = data['volume'].iloc[-20:].mean()
                        liquidity_score = min(recent_volume / avg_volume, 2.0) / 2.0
                        leverage_data['liquidity_score'] = liquidity_score

                    # 市场状态
                    price_change_5d = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) * 100
                    if abs(price_change_5d) > 10:
                        leverage_data['market_regime'] = 'volatile'
                    elif abs(price_change_5d) > 5:
                        leverage_data['market_regime'] = 'active'
                    else:
                        leverage_data['market_regime'] = 'calm'

            return leverage_data

        except Exception as e:
            logger.error(f"提取杠杆市场数据失败: {e}")
            return {}

    def _validate_signal(
        self,
        signal: FuturesSignal,
        margin_status: Optional[MarginStatus],
        current_positions: Optional[List[Position]]
    ) -> bool:
        """
        验证信号的有效性

        Args:
            signal: 期货信号
            margin_status: 保证金状态
            current_positions: 当前持仓

        Returns:
            是否通过验证
        """
        try:
            # 基础验证
            if not signal.is_valid():
                logger.warning(f"信号基础验证失败: {signal.ticker}")
                return False

            # 保证金验证
            if margin_status:
                if not margin_status.can_trade:
                    logger.warning(f"账户无法交易: {signal.ticker}")
                    return False

                if signal.operation_type == OperationType.OPEN and not margin_status.can_open_position:
                    logger.warning(f"账户无法开仓: {signal.ticker}")
                    return False

            # 杠杆验证
            if signal.suggested_leverage > 25:  # 硬性限制
                logger.warning(f"杠杆超过安全限制: {signal.suggested_leverage}")
                return False

            # 仓位大小验证
            if signal.position_size and signal.position_size < 10:  # 最小仓位
                logger.warning(f"仓位过小: {signal.position_size}")
                return False

            # 风险等级验证
            if signal.risk_level == RiskLevel.EMERGENCY:
                logger.warning(f"风险等级过高: {signal.risk_level.value}")
                return False

            logger.debug(f"信号验证通过: {signal.ticker}")
            return True

        except Exception as e:
            logger.error(f"信号验证失败: {e}")
            return False

    def _cache_signal(self, ticker: str, signal: FuturesSignal):
        """缓存信号"""
        try:
            cache_key = f"{ticker}_{signal.signal_id}"
            self._signal_cache[cache_key] = signal

            # 清理过期缓存
            current_time = datetime.now()
            expired_keys = [
                key for key, cached_signal in self._signal_cache.items()
                if cached_signal.expiry_time and current_time > cached_signal.expiry_time
            ]

            for key in expired_keys:
                del self._signal_cache[key]

        except Exception as e:
            logger.error(f"缓存信号失败: {e}")

    def _update_performance_stats(self, signal: FuturesSignal, processing_time: timedelta):
        """更新性能统计"""
        try:
            self._performance_stats["total_signals"] += 1
            self._performance_stats["successful_signals"] += 1

            # 更新方向分布
            direction = signal.direction.value
            if direction not in self._performance_stats["direction_distribution"]:
                self._performance_stats["direction_distribution"][direction] = 0
            self._performance_stats["direction_distribution"][direction] += 1

            # 更新平均处理时间
            current_avg = self._performance_stats["avg_processing_time"]
            total_signals = self._performance_stats["total_signals"]
            new_time = processing_time.total_seconds()

            self._performance_stats["avg_processing_time"] = (
                current_avg * (total_signals - 1) + new_time
            ) / total_signals

            self.last_signal_time = datetime.now()

        except Exception as e:
            logger.error(f"更新性能统计失败: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            系统状态信息
        """
        return {
            "is_initialized": self.is_initialized,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "performance_stats": self._performance_stats,
            "cached_signals": len(self._signal_cache),
            "active_strategies": len(self.strategy_manager.get_active_strategies()),
            "strategy_manager_status": self.strategy_manager.get_manager_status(),
            "leverage_controller_stats": self.leverage_controller.get_leverage_statistics() if hasattr(self.leverage_controller, 'get_leverage_statistics') else {},
            "tp_sl_calculator_stats": self.tp_sl_calculator.get_calculator_stats() if hasattr(self.tp_sl_calculator, 'get_calculator_stats') else {},
            "timeframe_weights": self.TIMEFRAME_WEIGHTS,
            "signal_thresholds": self.SIGNAL_THRESHOLDS
        }

    def reset_performance_stats(self):
        """重置性能统计"""
        self._performance_stats = {
            "total_signals": 0,
            "successful_signals": 0,
            "signal_accuracy": {},
            "avg_processing_time": 0.0,
            "direction_distribution": {}
        }
        logger.info("期货信号系统性能统计已重置")

    def clear_cache(self):
        """清理缓存"""
        self._signal_cache.clear()
        logger.info("期货信号系统缓存已清理")


# 导出主要类
__all__ = [
    'FuturesSignalSystem',
    'PositionOperation',
    'TimeframeSignal',
    'SignalConfidenceMetrics'
]