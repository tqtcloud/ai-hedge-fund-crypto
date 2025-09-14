"""
止盈止损计算模块

实现基于杠杆、波动率和市场条件的智能止盈止损价格计算，
支持多种计算策略、风险收益比验证和动态调整机制。

核心功能：
1. 杠杆适应性止损计算
2. 波动率驱动的止盈止损范围调整
3. 多种风险收益比策略
4. 技术位优化和流动性考量
5. 移动止损和分级止盈
6. 强平价格缓冲和时间止损
7. 完整的验证和异常处理机制
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import logging
import numpy as np
import pandas as pd
from math import sqrt, log, exp

# 导入项目模块
from ..models.data_models import (
    TradingDirection, RiskLevel, ValidationResult, ValidationSeverity
)
from ..constants import RiskParameters
from ...utils.exceptions import (
    ContractTradingError,
    LeverageExceedsLimitError
)

logger = logging.getLogger(__name__)


class TpSlCalculationMethod(Enum):
    """止盈止损计算方法"""
    ATR_BASED = "atr_based"                   # 基于ATR（平均真实波动范围）
    PERCENTAGE_BASED = "percentage_based"     # 基于百分比
    FIXED_POINTS = "fixed_points"            # 固定点数
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # 波动率调整
    SUPPORT_RESISTANCE = "support_resistance"    # 支撑阻力位
    ADAPTIVE_HYBRID = "adaptive_hybrid"           # 自适应混合方法


class RiskRewardStrategy(Enum):
    """风险收益比策略"""
    CONSERVATIVE = "conservative"     # 保守策略 (1:1.5)
    BALANCED = "balanced"            # 平衡策略 (1:2.0)
    AGGRESSIVE = "aggressive"        # 激进策略 (1:2.5)
    MAXIMUM = "maximum"              # 最大策略 (1:3.0)
    DYNAMIC = "dynamic"              # 动态策略（基于市场条件）


class TrailingStopType(Enum):
    """移动止损类型"""
    FIXED_DISTANCE = "fixed_distance"       # 固定距离
    PERCENTAGE_BASED = "percentage_based"   # 百分比跟踪
    ATR_TRAILING = "atr_trailing"          # ATR跟踪
    TIERED_TRAILING = "tiered_trailing"     # 分级跟踪


@dataclass
class TpSlResult:
    """止盈止损计算结果"""
    take_profit_price: float                        # 止盈价格
    stop_loss_price: float                          # 止损价格
    risk_reward_ratio: float                        # 实际风险收益比
    stop_loss_pct: float                           # 止损百分比
    take_profit_pct: float                         # 止盈百分比

    # 移动止损配置
    trailing_stop_config: Dict[str, Any] = field(default_factory=dict)

    # 分级止盈配置
    tiered_take_profit: List[Dict[str, Any]] = field(default_factory=list)

    # 时间和条件止损
    time_based_exit: Optional[datetime] = None      # 基于时间的退出
    max_holding_hours: float = 24.0                 # 最大持仓小时

    # 质量指标
    confidence_score: float = 0.0                   # 置信度评分
    liquidation_buffer: float = 0.0                 # 强平缓冲距离
    slippage_adjustment: float = 0.0                # 滑点调整

    # 警告和建议
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # 计算元数据
    calculation_method: TpSlCalculationMethod = TpSlCalculationMethod.ADAPTIVE_HYBRID
    risk_reward_strategy: RiskRewardStrategy = RiskRewardStrategy.BALANCED
    market_regime: str = "normal"                   # 市场状态
    volatility_regime: str = "normal"               # 波动率状态

    def __post_init__(self):
        """初始化后验证"""
        self._validate_prices()

    def _validate_prices(self):
        """验证价格的合理性"""
        if self.take_profit_price <= 0 or self.stop_loss_price <= 0:
            raise ValueError("止盈止损价格必须大于0")

        if self.risk_reward_ratio <= 0:
            raise ValueError("风险收益比必须大于0")

    def is_valid(self) -> bool:
        """检查结果是否有效"""
        try:
            return (
                self.take_profit_price > 0 and
                self.stop_loss_price > 0 and
                self.risk_reward_ratio > 0 and
                self.confidence_score >= 0.3  # 最低置信度阈值
            )
        except:
            return False

    def get_summary(self) -> Dict[str, Any]:
        """获取计算结果摘要"""
        return {
            "take_profit_price": self.take_profit_price,
            "stop_loss_price": self.stop_loss_price,
            "risk_reward_ratio": f"1:{self.risk_reward_ratio:.2f}",
            "stop_loss_pct": f"{self.stop_loss_pct:.2f}%",
            "take_profit_pct": f"{self.take_profit_pct:.2f}%",
            "confidence_score": f"{self.confidence_score:.2f}",
            "liquidation_buffer": f"{self.liquidation_buffer:.2f}%",
            "max_holding_hours": self.max_holding_hours,
            "calculation_method": self.calculation_method.value,
            "warnings_count": len(self.warnings),
            "has_trailing_stop": bool(self.trailing_stop_config),
            "has_tiered_tp": bool(self.tiered_take_profit)
        }


@dataclass
class TpSlCalculationParams:
    """止盈止损计算参数"""
    # 基础参数
    ticker: str
    direction: TradingDirection
    current_price: float
    leverage: float
    position_size: float = 0.0

    # 市场数据
    market_data: Optional[Dict[str, pd.DataFrame]] = None
    volatility: float = 0.02                        # 默认2%波动率
    atr_period: int = 14                            # ATR周期

    # 策略参数
    calculation_method: TpSlCalculationMethod = TpSlCalculationMethod.ADAPTIVE_HYBRID
    risk_reward_strategy: RiskRewardStrategy = RiskRewardStrategy.BALANCED

    # 风险参数
    max_stop_loss_pct: float = 0.05                # 最大止损5%
    min_stop_loss_pct: float = 0.005               # 最小止损0.5%
    max_risk_reward_ratio: float = 5.0             # 最大风险收益比
    min_risk_reward_ratio: float = 1.2             # 最小风险收益比

    # 期货特有参数
    liquidation_price: Optional[float] = None       # 强平价格
    min_liquidation_buffer: float = 0.15           # 最小强平缓冲15%
    funding_rate_impact: bool = True                # 考虑资金费率影响

    # 技术分析参数
    support_levels: List[float] = field(default_factory=list)    # 支撑位
    resistance_levels: List[float] = field(default_factory=list) # 阻力位
    trend_strength: float = 0.0                     # 趋势强度 (-1到1)
    momentum_score: float = 0.0                     # 动量评分 (0到1)

    # 移动止损参数
    enable_trailing_stop: bool = True               # 启用移动止损
    trailing_stop_type: TrailingStopType = TrailingStopType.PERCENTAGE_BASED
    trailing_activation_pct: float = 0.01          # 移动止损激活阈值1%

    # 分级止盈参数
    enable_tiered_tp: bool = True                   # 启用分级止盈
    tp_levels: List[Tuple[float, float]] = field(   # (价格比例, 平仓比例)
        default_factory=lambda: [(0.5, 0.3), (0.8, 0.4), (1.0, 0.3)]
    )

    # 时间参数
    signal_timeframe: str = "1h"                    # 信号时间框架
    max_holding_hours: float = 24.0                 # 最大持仓时间
    enable_time_exit: bool = True                   # 启用时间退出

    # 流动性参数
    liquidity_score: float = 1.0                    # 流动性评分 (0-1)
    expected_slippage: float = 0.001               # 预期滑点0.1%


class TpSlCalculator:
    """
    止盈止损计算器

    功能特性：
    1. 智能杠杆调整：杠杆越高止损越紧
    2. 波动率适应：根据市场波动率动态调整
    3. 多种计算方法：ATR、百分比、技术位等
    4. 风险收益比策略：从保守到激进的多种选择
    5. 移动止损：支持多种跟踪方式
    6. 分级止盈：分批获利了结
    7. 期货特有保护：强平缓冲、资金费率考虑
    8. 完整验证：多层次验证机制
    """

    # 默认风险收益比配置
    RISK_REWARD_RATIOS = {
        RiskRewardStrategy.CONSERVATIVE: 1.5,
        RiskRewardStrategy.BALANCED: 2.0,
        RiskRewardStrategy.AGGRESSIVE: 2.5,
        RiskRewardStrategy.MAXIMUM: 3.0
    }

    # 杠杆调整系数
    LEVERAGE_ADJUSTMENT_FACTORS = {
        (1, 2): 1.0,      # 1-2倍杠杆：正常止损
        (2, 5): 0.8,      # 2-5倍杠杆：收紧20%
        (5, 10): 0.6,     # 5-10倍杠杆：收紧40%
        (10, 20): 0.4,    # 10-20倍杠杆：收紧60%
        (20, 50): 0.2     # 20倍以上：收紧80%
    }

    # 波动率调整阈值
    VOLATILITY_THRESHOLDS = {
        'low': 0.01,      # 低波动率 < 1%
        'normal': 0.03,   # 正常波动率 1-3%
        'high': 0.05,     # 高波动率 3-5%
        'extreme': 0.10   # 极端波动率 > 5%
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化止盈止损计算器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 缓存和统计
        self._calculation_cache: Dict[str, TpSlResult] = {}
        self._calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "cache_hits": 0,
            "method_usage": {},
            "average_risk_reward_ratio": 0.0,
            "average_confidence_score": 0.0
        }

        logger.info("止盈止损计算器已初始化")

    async def calculate_tp_sl(
        self,
        params: TpSlCalculationParams
    ) -> Optional[TpSlResult]:
        """
        计算止盈止损价格

        Args:
            params: 计算参数

        Returns:
            计算结果，失败时返回None
        """
        try:
            logger.info(f"开始计算止盈止损: {params.ticker} {params.direction.value}")

            # 1. 参数验证
            if not self._validate_params(params):
                logger.error("参数验证失败")
                return None

            # 2. 检查缓存
            cache_key = self._generate_cache_key(params)
            if cache_key in self._calculation_cache:
                self._calculation_stats["cache_hits"] += 1
                logger.debug(f"使用缓存结果: {cache_key}")
                return self._calculation_cache[cache_key]

            # 3. 计算波动率指标
            volatility_metrics = self._calculate_volatility_metrics(params)

            # 4. 选择计算方法
            actual_method = self._select_calculation_method(params, volatility_metrics)

            # 5. 计算基础止损价格
            base_stop_loss = self._calculate_base_stop_loss(
                params, volatility_metrics, actual_method
            )

            if base_stop_loss is None:
                logger.error("基础止损计算失败")
                return None

            # 6. 应用杠杆调整
            adjusted_stop_loss = self._apply_leverage_adjustment(
                base_stop_loss, params.current_price, params.leverage, params.direction
            )

            # 7. 应用强平缓冲
            final_stop_loss = self._apply_liquidation_buffer(
                adjusted_stop_loss, params.current_price, params.liquidation_price,
                params.direction, params.min_liquidation_buffer
            )

            # 8. 计算风险收益比
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                params, volatility_metrics
            )

            # 9. 计算止盈价格
            take_profit_price = self._calculate_take_profit_price(
                final_stop_loss, params.current_price, risk_reward_ratio, params.direction
            )

            # 10. 计算百分比
            stop_loss_pct, take_profit_pct = self._calculate_percentages(
                params.current_price, final_stop_loss, take_profit_price, params.direction
            )

            # 11. 生成移动止损配置
            trailing_config = self._generate_trailing_stop_config(
                params, final_stop_loss, volatility_metrics
            )

            # 12. 生成分级止盈配置
            tiered_tp_config = self._generate_tiered_take_profit_config(
                params, take_profit_price
            )

            # 13. 计算时间退出
            time_exit = self._calculate_time_exit(params)

            # 14. 计算质量指标
            confidence_score, warnings, recommendations = self._calculate_quality_metrics(
                params, final_stop_loss, take_profit_price, risk_reward_ratio, volatility_metrics
            )

            # 15. 计算强平缓冲和滑点
            liquidation_buffer = self._calculate_liquidation_buffer_pct(
                params.current_price, params.liquidation_price, final_stop_loss, params.direction
            )

            # 16. 创建结果
            result = TpSlResult(
                take_profit_price=take_profit_price,
                stop_loss_price=final_stop_loss,
                risk_reward_ratio=risk_reward_ratio,
                stop_loss_pct=abs(stop_loss_pct),
                take_profit_pct=abs(take_profit_pct),
                trailing_stop_config=trailing_config,
                tiered_take_profit=tiered_tp_config,
                time_based_exit=time_exit,
                max_holding_hours=params.max_holding_hours,
                confidence_score=confidence_score,
                liquidation_buffer=liquidation_buffer,
                slippage_adjustment=params.expected_slippage * 100,
                warnings=warnings,
                recommendations=recommendations,
                calculation_method=actual_method,
                risk_reward_strategy=params.risk_reward_strategy,
                market_regime=volatility_metrics.get('market_regime', 'normal'),
                volatility_regime=volatility_metrics.get('volatility_regime', 'normal')
            )

            # 17. 最终验证
            if not self._validate_result(result, params):
                logger.error("结果验证失败")
                return None

            # 18. 缓存结果
            self._calculation_cache[cache_key] = result

            # 19. 更新统计
            self._update_stats(result, actual_method)

            logger.info(
                f"止盈止损计算完成: {params.ticker} "
                f"止盈={take_profit_price:.6f} 止损={final_stop_loss:.6f} "
                f"风险收益比=1:{risk_reward_ratio:.2f} 置信度={confidence_score:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"计算止盈止损失败: {e}")
            return None

    def _validate_params(self, params: TpSlCalculationParams) -> bool:
        """验证计算参数"""
        try:
            # 基础参数检查
            if not params.ticker or params.current_price <= 0:
                logger.error("无效的基础参数")
                return False

            if params.leverage <= 0 or params.leverage > 100:
                logger.error(f"无效的杠杆倍数: {params.leverage}")
                return False

            if params.volatility < 0 or params.volatility > 1:
                logger.error(f"无效的波动率: {params.volatility}")
                return False

            # 强平价格检查
            if params.liquidation_price:
                if params.direction == TradingDirection.LONG:
                    if params.liquidation_price >= params.current_price:
                        logger.error("多头强平价格不能高于当前价格")
                        return False
                elif params.direction == TradingDirection.SHORT:
                    if params.liquidation_price <= params.current_price:
                        logger.error("空头强平价格不能低于当前价格")
                        return False

            # 支撑阻力位检查
            for level in params.support_levels + params.resistance_levels:
                if level <= 0:
                    logger.error("支撑阻力位必须大于0")
                    return False

            logger.debug("参数验证通过")
            return True

        except Exception as e:
            logger.error(f"参数验证失败: {e}")
            return False

    def _generate_cache_key(self, params: TpSlCalculationParams) -> str:
        """生成缓存键"""
        try:
            key_components = [
                params.ticker,
                params.direction.value,
                f"{params.current_price:.6f}",
                f"{params.leverage:.1f}",
                f"{params.volatility:.4f}",
                params.calculation_method.value,
                params.risk_reward_strategy.value
            ]

            return "_".join(key_components)

        except Exception as e:
            logger.error(f"生成缓存键失败: {e}")
            return f"fallback_{int(datetime.now().timestamp())}"

    def _calculate_volatility_metrics(self, params: TpSlCalculationParams) -> Dict[str, Any]:
        """计算波动率指标"""
        try:
            metrics = {
                'current_volatility': params.volatility,
                'volatility_regime': 'normal',
                'market_regime': 'normal',
                'atr_value': 0.0,
                'price_volatility': params.volatility,
                'volume_weighted_volatility': params.volatility
            }

            # 分类波动率状态
            if params.volatility < self.VOLATILITY_THRESHOLDS['low']:
                metrics['volatility_regime'] = 'low'
            elif params.volatility < self.VOLATILITY_THRESHOLDS['normal']:
                metrics['volatility_regime'] = 'normal'
            elif params.volatility < self.VOLATILITY_THRESHOLDS['high']:
                metrics['volatility_regime'] = 'high'
            else:
                metrics['volatility_regime'] = 'extreme'

            # 计算ATR（如果有市场数据）
            if params.market_data:
                atr_value = self._calculate_atr(params.market_data, params.atr_period)
                if atr_value > 0:
                    metrics['atr_value'] = atr_value
                    # 基于ATR调整波动率
                    atr_based_volatility = atr_value / params.current_price
                    metrics['atr_based_volatility'] = atr_based_volatility

            # 计算趋势强度影响
            if abs(params.trend_strength) > 0.7:
                metrics['market_regime'] = 'trending'
            elif abs(params.trend_strength) < 0.3:
                metrics['market_regime'] = 'ranging'

            logger.debug(f"波动率指标计算完成: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"计算波动率指标失败: {e}")
            return {'current_volatility': params.volatility, 'volatility_regime': 'normal'}

    def _calculate_atr(self, market_data: Dict[str, pd.DataFrame], period: int = 14) -> float:
        """计算平均真实波动范围"""
        try:
            # 使用1小时数据计算ATR
            timeframe = '1h'
            if timeframe not in market_data:
                timeframe = list(market_data.keys())[0]  # 使用第一个可用时间框架

            data = market_data[timeframe]
            if len(data) < period + 1:
                return 0.0

            high = data['high']
            low = data['low']
            close = data['close']

            # 计算真实波动范围
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean().iloc[-1]

            return float(atr) if not pd.isna(atr) else 0.0

        except Exception as e:
            logger.error(f"ATR计算失败: {e}")
            return 0.0

    def _select_calculation_method(
        self,
        params: TpSlCalculationParams,
        volatility_metrics: Dict[str, Any]
    ) -> TpSlCalculationMethod:
        """选择最适合的计算方法"""
        try:
            # 如果指定了方法且不是自适应，直接使用
            if params.calculation_method != TpSlCalculationMethod.ADAPTIVE_HYBRID:
                return params.calculation_method

            # 自适应方法选择逻辑
            volatility_regime = volatility_metrics.get('volatility_regime', 'normal')
            has_market_data = params.market_data is not None
            has_technical_levels = bool(params.support_levels or params.resistance_levels)

            # 高波动率优先使用ATR
            if volatility_regime in ['high', 'extreme'] and has_market_data:
                return TpSlCalculationMethod.ATR_BASED

            # 有技术位时优先考虑技术位方法
            if has_technical_levels:
                return TpSlCalculationMethod.SUPPORT_RESISTANCE

            # 低波动率使用固定百分比
            if volatility_regime == 'low':
                return TpSlCalculationMethod.PERCENTAGE_BASED

            # 默认使用波动率调整方法
            return TpSlCalculationMethod.VOLATILITY_ADJUSTED

        except Exception as e:
            logger.error(f"选择计算方法失败: {e}")
            return TpSlCalculationMethod.PERCENTAGE_BASED

    def _calculate_base_stop_loss(
        self,
        params: TpSlCalculationParams,
        volatility_metrics: Dict[str, Any],
        method: TpSlCalculationMethod
    ) -> Optional[float]:
        """计算基础止损价格"""
        try:
            current_price = params.current_price
            direction = params.direction

            if method == TpSlCalculationMethod.ATR_BASED:
                return self._calculate_atr_stop_loss(params, volatility_metrics)

            elif method == TpSlCalculationMethod.PERCENTAGE_BASED:
                return self._calculate_percentage_stop_loss(params)

            elif method == TpSlCalculationMethod.VOLATILITY_ADJUSTED:
                return self._calculate_volatility_adjusted_stop_loss(params, volatility_metrics)

            elif method == TpSlCalculationMethod.SUPPORT_RESISTANCE:
                return self._calculate_technical_stop_loss(params)

            elif method == TpSlCalculationMethod.FIXED_POINTS:
                return self._calculate_fixed_points_stop_loss(params)

            else:
                # 默认使用百分比方法
                return self._calculate_percentage_stop_loss(params)

        except Exception as e:
            logger.error(f"计算基础止损失败: {e}")
            return None

    def _calculate_percentage_stop_loss(self, params: TpSlCalculationParams) -> float:
        """基于百分比计算止损"""
        try:
            # 基础止损百分比
            base_stop_pct = min(max(params.volatility * 2, params.min_stop_loss_pct), params.max_stop_loss_pct)

            # 根据方向计算止损价格
            if params.direction == TradingDirection.LONG:
                stop_loss_price = params.current_price * (1 - base_stop_pct)
            else:  # SHORT
                stop_loss_price = params.current_price * (1 + base_stop_pct)

            logger.debug(f"百分比止损: {stop_loss_price:.6f} ({base_stop_pct*100:.2f}%)")
            return stop_loss_price

        except Exception as e:
            logger.error(f"百分比止损计算失败: {e}")
            raise

    def _calculate_atr_stop_loss(
        self,
        params: TpSlCalculationParams,
        volatility_metrics: Dict[str, Any]
    ) -> float:
        """基于ATR计算止损"""
        try:
            atr_value = volatility_metrics.get('atr_value', 0.0)

            # 如果没有ATR值，回退到百分比方法
            if atr_value <= 0:
                logger.warning("ATR值无效，回退到百分比方法")
                return self._calculate_percentage_stop_loss(params)

            # ATR倍数设置（根据波动率状态调整）
            volatility_regime = volatility_metrics.get('volatility_regime', 'normal')
            atr_multipliers = {
                'low': 1.0,
                'normal': 1.5,
                'high': 2.0,
                'extreme': 2.5
            }

            multiplier = atr_multipliers.get(volatility_regime, 1.5)
            stop_distance = atr_value * multiplier

            # 确保在合理范围内
            max_distance = params.current_price * params.max_stop_loss_pct
            min_distance = params.current_price * params.min_stop_loss_pct
            stop_distance = min(max(stop_distance, min_distance), max_distance)

            # 计算止损价格
            if params.direction == TradingDirection.LONG:
                stop_loss_price = params.current_price - stop_distance
            else:  # SHORT
                stop_loss_price = params.current_price + stop_distance

            logger.debug(f"ATR止损: {stop_loss_price:.6f} (ATR={atr_value:.6f}, 倍数={multiplier})")
            return stop_loss_price

        except Exception as e:
            logger.error(f"ATR止损计算失败: {e}")
            return self._calculate_percentage_stop_loss(params)

    def _calculate_volatility_adjusted_stop_loss(
        self,
        params: TpSlCalculationParams,
        volatility_metrics: Dict[str, Any]
    ) -> float:
        """基于波动率调整计算止损"""
        try:
            current_volatility = volatility_metrics['current_volatility']
            volatility_regime = volatility_metrics.get('volatility_regime', 'normal')

            # 基础止损百分比
            base_stop_pct = params.volatility * 2

            # 根据波动率状态调整
            volatility_adjustments = {
                'low': 0.8,
                'normal': 1.0,
                'high': 1.3,
                'extreme': 1.6
            }

            adjustment = volatility_adjustments.get(volatility_regime, 1.0)
            adjusted_stop_pct = base_stop_pct * adjustment

            # 限制在合理范围内
            adjusted_stop_pct = min(max(adjusted_stop_pct, params.min_stop_loss_pct), params.max_stop_loss_pct)

            # 计算止损价格
            if params.direction == TradingDirection.LONG:
                stop_loss_price = params.current_price * (1 - adjusted_stop_pct)
            else:  # SHORT
                stop_loss_price = params.current_price * (1 + adjusted_stop_pct)

            logger.debug(f"波动率调整止损: {stop_loss_price:.6f} (调整后={adjusted_stop_pct*100:.2f}%)")
            return stop_loss_price

        except Exception as e:
            logger.error(f"波动率调整止损计算失败: {e}")
            return self._calculate_percentage_stop_loss(params)

    def _calculate_technical_stop_loss(self, params: TpSlCalculationParams) -> float:
        """基于技术位计算止损"""
        try:
            current_price = params.current_price
            direction = params.direction

            # 选择相关的技术位
            if direction == TradingDirection.LONG:
                # 多头使用支撑位作为止损参考
                relevant_levels = [level for level in params.support_levels if level < current_price]
            else:  # SHORT
                # 空头使用阻力位作为止损参考
                relevant_levels = [level for level in params.resistance_levels if level > current_price]

            if not relevant_levels:
                logger.warning("没有找到相关技术位，使用百分比方法")
                return self._calculate_percentage_stop_loss(params)

            # 选择最近的技术位
            if direction == TradingDirection.LONG:
                nearest_level = max(relevant_levels)  # 最近的支撑位
                # 在支撑位下方设置一定缓冲
                buffer_pct = 0.005  # 0.5%缓冲
                stop_loss_price = nearest_level * (1 - buffer_pct)
            else:  # SHORT
                nearest_level = min(relevant_levels)  # 最近的阻力位
                # 在阻力位上方设置一定缓冲
                buffer_pct = 0.005  # 0.5%缓冲
                stop_loss_price = nearest_level * (1 + buffer_pct)

            # 确保止损距离合理
            stop_distance_pct = abs(stop_loss_price - current_price) / current_price
            if stop_distance_pct < params.min_stop_loss_pct:
                logger.warning(f"技术位止损过近({stop_distance_pct*100:.2f}%)，调整到最小距离")
                if direction == TradingDirection.LONG:
                    stop_loss_price = current_price * (1 - params.min_stop_loss_pct)
                else:
                    stop_loss_price = current_price * (1 + params.min_stop_loss_pct)
            elif stop_distance_pct > params.max_stop_loss_pct:
                logger.warning(f"技术位止损过远({stop_distance_pct*100:.2f}%)，调整到最大距离")
                if direction == TradingDirection.LONG:
                    stop_loss_price = current_price * (1 - params.max_stop_loss_pct)
                else:
                    stop_loss_price = current_price * (1 + params.max_stop_loss_pct)

            logger.debug(f"技术位止损: {stop_loss_price:.6f} (技术位={nearest_level:.6f})")
            return stop_loss_price

        except Exception as e:
            logger.error(f"技术位止损计算失败: {e}")
            return self._calculate_percentage_stop_loss(params)

    def _calculate_fixed_points_stop_loss(self, params: TpSlCalculationParams) -> float:
        """基于固定点数计算止损"""
        try:
            # 根据价格确定固定点数
            if params.current_price >= 1000:
                fixed_points = 10.0
            elif params.current_price >= 100:
                fixed_points = 1.0
            elif params.current_price >= 10:
                fixed_points = 0.1
            elif params.current_price >= 1:
                fixed_points = 0.01
            else:
                fixed_points = 0.001

            # 根据波动率调整点数
            volatility_multiplier = max(0.5, min(2.0, params.volatility / 0.02))
            adjusted_points = fixed_points * volatility_multiplier

            # 计算止损价格
            if params.direction == TradingDirection.LONG:
                stop_loss_price = params.current_price - adjusted_points
            else:  # SHORT
                stop_loss_price = params.current_price + adjusted_points

            logger.debug(f"固定点数止损: {stop_loss_price:.6f} (点数={adjusted_points})")
            return stop_loss_price

        except Exception as e:
            logger.error(f"固定点数止损计算失败: {e}")
            return self._calculate_percentage_stop_loss(params)

    def _apply_leverage_adjustment(
        self,
        base_stop_loss: float,
        current_price: float,
        leverage: float,
        direction: TradingDirection
    ) -> float:
        """应用杠杆调整"""
        try:
            # 获取杠杆调整因子
            adjustment_factor = 1.0
            for (min_lev, max_lev), factor in self.LEVERAGE_ADJUSTMENT_FACTORS.items():
                if min_lev < leverage <= max_lev:
                    adjustment_factor = factor
                    break

            if adjustment_factor == 1.0:
                logger.debug("无需杠杆调整")
                return base_stop_loss

            # 计算调整后的止损距离
            if direction == TradingDirection.LONG:
                original_distance = current_price - base_stop_loss
                adjusted_distance = original_distance * adjustment_factor
                adjusted_stop_loss = current_price - adjusted_distance
            else:  # SHORT
                original_distance = base_stop_loss - current_price
                adjusted_distance = original_distance * adjustment_factor
                adjusted_stop_loss = current_price + adjusted_distance

            logger.debug(
                f"杠杆调整: {base_stop_loss:.6f} -> {adjusted_stop_loss:.6f} "
                f"(杠杆={leverage:.1f}, 系数={adjustment_factor})"
            )

            return adjusted_stop_loss

        except Exception as e:
            logger.error(f"杠杆调整失败: {e}")
            return base_stop_loss

    def _apply_liquidation_buffer(
        self,
        stop_loss_price: float,
        current_price: float,
        liquidation_price: Optional[float],
        direction: TradingDirection,
        min_buffer_pct: float
    ) -> float:
        """应用强平价格缓冲"""
        try:
            if not liquidation_price:
                logger.debug("无强平价格，跳过缓冲调整")
                return stop_loss_price

            # 计算强平缓冲区域
            if direction == TradingDirection.LONG:
                # 多头：止损价格应高于强平价格
                min_stop_price = liquidation_price * (1 + min_buffer_pct)
                if stop_loss_price < min_stop_price:
                    logger.warning(
                        f"止损过近强平价格，调整: {stop_loss_price:.6f} -> {min_stop_price:.6f}"
                    )
                    return min_stop_price
            else:  # SHORT
                # 空头：止损价格应低于强平价格
                max_stop_price = liquidation_price * (1 - min_buffer_pct)
                if stop_loss_price > max_stop_price:
                    logger.warning(
                        f"止损过近强平价格，调整: {stop_loss_price:.6f} -> {max_stop_price:.6f}"
                    )
                    return max_stop_price

            logger.debug("强平缓冲检查通过")
            return stop_loss_price

        except Exception as e:
            logger.error(f"强平缓冲调整失败: {e}")
            return stop_loss_price

    def _calculate_risk_reward_ratio(
        self,
        params: TpSlCalculationParams,
        volatility_metrics: Dict[str, Any]
    ) -> float:
        """计算风险收益比"""
        try:
            if params.risk_reward_strategy == RiskRewardStrategy.DYNAMIC:
                # 动态风险收益比基于市场条件
                return self._calculate_dynamic_risk_reward_ratio(params, volatility_metrics)
            else:
                # 固定风险收益比
                base_ratio = self.RISK_REWARD_RATIOS[params.risk_reward_strategy]

                # 根据置信度微调
                confidence_adjustment = 1.0
                if params.momentum_score > 0.8:  # 高动量
                    confidence_adjustment = 1.2
                elif params.momentum_score < 0.3:  # 低动量
                    confidence_adjustment = 0.8

                adjusted_ratio = base_ratio * confidence_adjustment

                # 限制在合理范围内
                final_ratio = min(max(adjusted_ratio, params.min_risk_reward_ratio), params.max_risk_reward_ratio)

                logger.debug(f"风险收益比: {final_ratio:.2f} (基础={base_ratio:.2f})")
                return final_ratio

        except Exception as e:
            logger.error(f"计算风险收益比失败: {e}")
            return 2.0  # 默认值

    def _calculate_dynamic_risk_reward_ratio(
        self,
        params: TpSlCalculationParams,
        volatility_metrics: Dict[str, Any]
    ) -> float:
        """计算动态风险收益比"""
        try:
            # 基础比例
            base_ratio = 2.0

            # 趋势强度调整
            trend_adjustment = 1.0 + abs(params.trend_strength) * 0.5

            # 波动率调整
            volatility_regime = volatility_metrics.get('volatility_regime', 'normal')
            volatility_adjustments = {
                'low': 0.8,      # 低波动率降低目标
                'normal': 1.0,   # 正常波动率
                'high': 1.2,     # 高波动率提高目标
                'extreme': 0.9   # 极端波动率保守一些
            }
            volatility_adjustment = volatility_adjustments.get(volatility_regime, 1.0)

            # 动量调整
            momentum_adjustment = 0.8 + params.momentum_score * 0.4

            # 流动性调整
            liquidity_adjustment = 0.9 + params.liquidity_score * 0.2

            # 综合计算
            dynamic_ratio = (
                base_ratio *
                trend_adjustment *
                volatility_adjustment *
                momentum_adjustment *
                liquidity_adjustment
            )

            # 限制范围
            final_ratio = min(max(dynamic_ratio, params.min_risk_reward_ratio), params.max_risk_reward_ratio)

            logger.debug(
                f"动态风险收益比: {final_ratio:.2f} "
                f"(趋势={trend_adjustment:.2f}, 波动率={volatility_adjustment:.2f}, "
                f"动量={momentum_adjustment:.2f}, 流动性={liquidity_adjustment:.2f})"
            )

            return final_ratio

        except Exception as e:
            logger.error(f"计算动态风险收益比失败: {e}")
            return 2.0

    def _calculate_take_profit_price(
        self,
        stop_loss_price: float,
        current_price: float,
        risk_reward_ratio: float,
        direction: TradingDirection
    ) -> float:
        """计算止盈价格"""
        try:
            # 计算止损距离
            if direction == TradingDirection.LONG:
                stop_distance = current_price - stop_loss_price
                take_profit_price = current_price + (stop_distance * risk_reward_ratio)
            else:  # SHORT
                stop_distance = stop_loss_price - current_price
                take_profit_price = current_price - (stop_distance * risk_reward_ratio)

            logger.debug(
                f"止盈价格计算: {take_profit_price:.6f} "
                f"(风险收益比=1:{risk_reward_ratio:.2f})"
            )

            return take_profit_price

        except Exception as e:
            logger.error(f"计算止盈价格失败: {e}")
            raise

    def _calculate_percentages(
        self,
        current_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        direction: TradingDirection
    ) -> Tuple[float, float]:
        """计算百分比"""
        try:
            if direction == TradingDirection.LONG:
                stop_loss_pct = (current_price - stop_loss_price) / current_price * 100
                take_profit_pct = (take_profit_price - current_price) / current_price * 100
            else:  # SHORT
                stop_loss_pct = (stop_loss_price - current_price) / current_price * 100
                take_profit_pct = (current_price - take_profit_price) / current_price * 100

            return stop_loss_pct, take_profit_pct

        except Exception as e:
            logger.error(f"计算百分比失败: {e}")
            return 2.0, 4.0  # 默认值

    def _generate_trailing_stop_config(
        self,
        params: TpSlCalculationParams,
        stop_loss_price: float,
        volatility_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成移动止损配置"""
        try:
            if not params.enable_trailing_stop:
                return {}

            config = {
                "enabled": True,
                "type": params.trailing_stop_type.value,
                "activation_threshold_pct": params.trailing_activation_pct * 100,
                "initial_stop_price": stop_loss_price
            }

            # 根据移动止损类型配置参数
            if params.trailing_stop_type == TrailingStopType.PERCENTAGE_BASED:
                config.update({
                    "trail_percentage": params.trailing_activation_pct * 100,
                    "min_profit_threshold_pct": 1.0  # 最少1%利润后激活
                })

            elif params.trailing_stop_type == TrailingStopType.ATR_TRAILING:
                atr_value = volatility_metrics.get('atr_value', 0.0)
                if atr_value > 0:
                    config.update({
                        "atr_multiplier": 1.5,
                        "atr_value": atr_value,
                        "update_frequency": "1m"  # 每分钟更新
                    })
                else:
                    # 回退到百分比跟踪
                    config["type"] = TrailingStopType.PERCENTAGE_BASED.value
                    config["trail_percentage"] = params.trailing_activation_pct * 100

            elif params.trailing_stop_type == TrailingStopType.TIERED_TRAILING:
                config.update({
                    "tiers": [
                        {"profit_threshold_pct": 2.0, "trail_distance_pct": 1.0},
                        {"profit_threshold_pct": 5.0, "trail_distance_pct": 2.0},
                        {"profit_threshold_pct": 10.0, "trail_distance_pct": 3.0}
                    ]
                })

            logger.debug(f"移动止损配置生成完成: {config}")
            return config

        except Exception as e:
            logger.error(f"生成移动止损配置失败: {e}")
            return {}

    def _generate_tiered_take_profit_config(
        self,
        params: TpSlCalculationParams,
        final_take_profit_price: float
    ) -> List[Dict[str, Any]]:
        """生成分级止盈配置"""
        try:
            if not params.enable_tiered_tp:
                return []

            tiered_config = []
            current_price = params.current_price

            for i, (price_ratio, quantity_ratio) in enumerate(params.tp_levels):
                # 计算分级止盈价格
                if params.direction == TradingDirection.LONG:
                    tp_distance = final_take_profit_price - current_price
                    tier_price = current_price + (tp_distance * price_ratio)
                else:  # SHORT
                    tp_distance = current_price - final_take_profit_price
                    tier_price = current_price - (tp_distance * price_ratio)

                tier_config = {
                    "tier": i + 1,
                    "price": tier_price,
                    "quantity_ratio": quantity_ratio,
                    "price_ratio": price_ratio,
                    "triggered": False
                }

                tiered_config.append(tier_config)

            logger.debug(f"分级止盈配置生成完成: {len(tiered_config)}层")
            return tiered_config

        except Exception as e:
            logger.error(f"生成分级止盈配置失败: {e}")
            return []

    def _calculate_time_exit(self, params: TpSlCalculationParams) -> Optional[datetime]:
        """计算时间退出"""
        try:
            if not params.enable_time_exit:
                return None

            return datetime.now() + timedelta(hours=params.max_holding_hours)

        except Exception as e:
            logger.error(f"计算时间退出失败: {e}")
            return None

    def _calculate_quality_metrics(
        self,
        params: TpSlCalculationParams,
        stop_loss_price: float,
        take_profit_price: float,
        risk_reward_ratio: float,
        volatility_metrics: Dict[str, Any]
    ) -> Tuple[float, List[str], List[str]]:
        """计算质量指标"""
        try:
            warnings = []
            recommendations = []
            quality_scores = []

            # 1. 风险收益比评分
            if risk_reward_ratio >= 2.5:
                quality_scores.append(0.9)
            elif risk_reward_ratio >= 2.0:
                quality_scores.append(0.8)
            elif risk_reward_ratio >= 1.5:
                quality_scores.append(0.6)
            else:
                quality_scores.append(0.3)
                warnings.append(f"风险收益比偏低: 1:{risk_reward_ratio:.2f}")

            # 2. 止损距离合理性评分
            stop_distance_pct = abs(stop_loss_price - params.current_price) / params.current_price
            if 0.01 <= stop_distance_pct <= 0.03:
                quality_scores.append(0.9)
            elif 0.005 <= stop_distance_pct <= 0.05:
                quality_scores.append(0.7)
            else:
                quality_scores.append(0.4)
                if stop_distance_pct < 0.005:
                    warnings.append("止损距离过近，可能频繁触发")
                else:
                    warnings.append("止损距离过远，风险较大")

            # 3. 强平缓冲评分
            if params.liquidation_price:
                liquidation_buffer_pct = self._calculate_liquidation_buffer_pct(
                    params.current_price, params.liquidation_price, stop_loss_price, params.direction
                )
                if liquidation_buffer_pct >= 0.15:
                    quality_scores.append(0.9)
                elif liquidation_buffer_pct >= 0.10:
                    quality_scores.append(0.7)
                elif liquidation_buffer_pct >= 0.05:
                    quality_scores.append(0.5)
                    warnings.append("强平缓冲距离较小")
                else:
                    quality_scores.append(0.2)
                    warnings.append("强平缓冲距离过小，风险极高")
            else:
                quality_scores.append(0.6)  # 没有强平价格信息时的中性评分

            # 4. 杠杆合理性评分
            if params.leverage <= 5:
                quality_scores.append(0.9)
            elif params.leverage <= 10:
                quality_scores.append(0.7)
            elif params.leverage <= 20:
                quality_scores.append(0.5)
                warnings.append(f"杠杆较高: {params.leverage:.1f}倍")
            else:
                quality_scores.append(0.2)
                warnings.append(f"杠杆过高: {params.leverage:.1f}倍，风险极大")

            # 5. 市场条件评分
            volatility_regime = volatility_metrics.get('volatility_regime', 'normal')
            if volatility_regime == 'normal':
                quality_scores.append(0.8)
            elif volatility_regime in ['low', 'high']:
                quality_scores.append(0.6)
            else:  # extreme
                quality_scores.append(0.3)
                warnings.append("市场波动率极端，建议降低仓位")

            # 计算综合置信度
            confidence_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5

            # 生成建议
            if confidence_score < 0.5:
                recommendations.append("建议重新评估交易参数")
            if params.leverage > 10:
                recommendations.append("建议降低杠杆以减少风险")
            if risk_reward_ratio < 2.0:
                recommendations.append("建议提高止盈目标或收紧止损")
            if volatility_regime == 'extreme':
                recommendations.append("极端波动市场，建议等待稳定后交易")

            logger.debug(f"质量评分: {confidence_score:.2f}, 警告: {len(warnings)}, 建议: {len(recommendations)}")

            return confidence_score, warnings, recommendations

        except Exception as e:
            logger.error(f"计算质量指标失败: {e}")
            return 0.5, ["质量评估失败"], []

    def _calculate_liquidation_buffer_pct(
        self,
        current_price: float,
        liquidation_price: Optional[float],
        stop_loss_price: float,
        direction: TradingDirection
    ) -> float:
        """计算强平缓冲百分比"""
        try:
            if not liquidation_price:
                return 0.0

            if direction == TradingDirection.LONG:
                buffer_pct = (stop_loss_price - liquidation_price) / current_price
            else:  # SHORT
                buffer_pct = (liquidation_price - stop_loss_price) / current_price

            return max(0.0, buffer_pct)

        except Exception as e:
            logger.error(f"计算强平缓冲失败: {e}")
            return 0.0

    def _validate_result(self, result: TpSlResult, params: TpSlCalculationParams) -> bool:
        """验证计算结果"""
        try:
            # 基础验证
            if not result.is_valid():
                logger.error("结果基础验证失败")
                return False

            # 价格合理性检查
            current_price = params.current_price
            if params.direction == TradingDirection.LONG:
                if result.stop_loss_price >= current_price:
                    logger.error("多头止损价格不能高于当前价格")
                    return False
                if result.take_profit_price <= current_price:
                    logger.error("多头止盈价格不能低于当前价格")
                    return False
            else:  # SHORT
                if result.stop_loss_price <= current_price:
                    logger.error("空头止损价格不能低于当前价格")
                    return False
                if result.take_profit_price >= current_price:
                    logger.error("空头止盈价格不能高于当前价格")
                    return False

            # 风险收益比检查
            if result.risk_reward_ratio < params.min_risk_reward_ratio:
                logger.error(f"风险收益比过低: {result.risk_reward_ratio}")
                return False

            # 止损距离检查
            if result.stop_loss_pct > params.max_stop_loss_pct * 100:
                logger.error(f"止损距离过大: {result.stop_loss_pct}%")
                return False

            logger.debug("结果验证通过")
            return True

        except Exception as e:
            logger.error(f"结果验证失败: {e}")
            return False

    def _update_stats(self, result: TpSlResult, method: TpSlCalculationMethod):
        """更新统计信息"""
        try:
            self._calculation_stats["total_calculations"] += 1
            self._calculation_stats["successful_calculations"] += 1

            # 更新方法使用统计
            method_name = method.value
            if method_name not in self._calculation_stats["method_usage"]:
                self._calculation_stats["method_usage"][method_name] = 0
            self._calculation_stats["method_usage"][method_name] += 1

            # 更新平均指标
            total = self._calculation_stats["successful_calculations"]
            current_avg_rr = self._calculation_stats["average_risk_reward_ratio"]
            current_avg_conf = self._calculation_stats["average_confidence_score"]

            self._calculation_stats["average_risk_reward_ratio"] = (
                current_avg_rr * (total - 1) + result.risk_reward_ratio
            ) / total

            self._calculation_stats["average_confidence_score"] = (
                current_avg_conf * (total - 1) + result.confidence_score
            ) / total

        except Exception as e:
            logger.error(f"更新统计信息失败: {e}")

    def get_calculator_stats(self) -> Dict[str, Any]:
        """获取计算器统计信息"""
        return {
            "calculation_stats": self._calculation_stats,
            "cache_size": len(self._calculation_cache),
            "supported_methods": [method.value for method in TpSlCalculationMethod],
            "supported_strategies": [strategy.value for strategy in RiskRewardStrategy],
            "leverage_adjustment_factors": self.LEVERAGE_ADJUSTMENT_FACTORS,
            "volatility_thresholds": self.VOLATILITY_THRESHOLDS
        }

    def clear_cache(self):
        """清理缓存"""
        self._calculation_cache.clear()
        logger.info("止盈止损计算器缓存已清理")

    def reset_stats(self):
        """重置统计信息"""
        self._calculation_stats = {
            "total_calculations": 0,
            "successful_calculations": 0,
            "cache_hits": 0,
            "method_usage": {},
            "average_risk_reward_ratio": 0.0,
            "average_confidence_score": 0.0
        }
        logger.info("止盈止损计算器统计信息已重置")


# 导出主要类
__all__ = [
    'TpSlCalculator',
    'TpSlResult',
    'TpSlCalculationParams',
    'TpSlCalculationMethod',
    'RiskRewardStrategy',
    'TrailingStopType'
]