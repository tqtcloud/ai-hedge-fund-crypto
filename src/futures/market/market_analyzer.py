"""
期货市场数据分析器

提供全面的市场分析功能，包括波动率分析、技术指标集成、
市场条件分析、相关性分析和风险度量等核心功能。
支持多时间框架分析和实时数据处理。
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from collections import defaultdict
from enum import Enum
import numpy as np
import pandas as pd

# 尝试导入，如果失败则使用本地定义
try:
    from ..models.data_models import (
        TradingDirection, RiskLevel, ValidationSeverity,
        FuturesSignal, Position, MarginStatus, ValidationResult
    )
    from ...utils.exceptions import (
        ContractTradingError, MarginInsufficientError,
        LeverageExceedsLimitError, LiquidationRiskError,
        PositionSizeError
    )
except ImportError:
    # 如果导入失败，定义基本枚举类
    from enum import Enum
    
    class TradingDirection(Enum):
        LONG = "long"
        SHORT = "short"
        NEUTRAL = "neutral"
    
    class RiskLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
        EMERGENCY = "emergency"
    
    class ValidationSeverity(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
# 尝试导入技术指标，如果失败则使用内置实现
try:
    from ...indicators.general_indicators import (
        calculate_rsi, calculate_bollinger_bands, calculate_ema,
        calculate_adx, calculate_atr, calculate_hurst_exponent
    )
except ImportError:
    # 内置简化版技术指标实现
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        # 增强版RSI计算 - 支持多周期和更高精度
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        # 使用Wilder平滑法（更准确的RSI计算）
        alpha = 1.0 / period
        avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

        # 避免除零错误
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))

        # 处理边界情况
        rsi = np.where(avg_loss == 0, 100, rsi)
        rsi = np.where((avg_gain == 0) & (avg_loss == 0), 50, rsi)

        return pd.Series(rsi, index=df.index)
    
    def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20) -> tuple:
        sma = df["close"].rolling(window).mean()
        std_dev = df["close"].rolling(window).std()
        upper_band = sma + (std_dev * 2)
        lower_band = sma - (std_dev * 2)
        return upper_band, lower_band
    
    def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
        return df["close"].ewm(span=window, adjust=False).mean()
    
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        # 增强版ADX计算 - 提高准确性
        df_copy = df.copy()

        # 计算真实范围 (True Range)
        df_copy["high_low"] = df_copy["high"] - df_copy["low"]
        df_copy["high_close"] = abs(df_copy["high"] - df_copy["close"].shift())
        df_copy["low_close"] = abs(df_copy["low"] - df_copy["close"].shift())
        df_copy["tr"] = df_copy[["high_low", "high_close", "low_close"]].max(axis=1)

        # 计算方向性移动 (Directional Movement)
        df_copy["up_move"] = df_copy["high"] - df_copy["high"].shift()
        df_copy["down_move"] = df_copy["low"].shift() - df_copy["low"]

        # 修正方向性移动计算
        df_copy["plus_dm"] = np.where(
            (df_copy["up_move"] > df_copy["down_move"]) & (df_copy["up_move"] > 0),
            df_copy["up_move"], 0
        )
        df_copy["minus_dm"] = np.where(
            (df_copy["down_move"] > df_copy["up_move"]) & (df_copy["down_move"] > 0),
            df_copy["down_move"], 0
        )

        # 使用Wilder平滑法计算平滑的TR和DM
        alpha = 1.0 / period
        df_copy["tr_smooth"] = df_copy["tr"].ewm(alpha=alpha, adjust=False).mean()
        df_copy["plus_dm_smooth"] = df_copy["plus_dm"].ewm(alpha=alpha, adjust=False).mean()
        df_copy["minus_dm_smooth"] = df_copy["minus_dm"].ewm(alpha=alpha, adjust=False).mean()

        # 计算方向性指标
        df_copy["+di"] = 100 * (df_copy["plus_dm_smooth"] / df_copy["tr_smooth"])
        df_copy["-di"] = 100 * (df_copy["minus_dm_smooth"] / df_copy["tr_smooth"])

        # 计算DX和ADX
        di_sum = df_copy["+di"] + df_copy["-di"]
        di_diff = abs(df_copy["+di"] - df_copy["-di"])
        df_copy["dx"] = np.where(di_sum != 0, 100 * (di_diff / di_sum), 0)
        df_copy["adx"] = df_copy["dx"].ewm(alpha=alpha, adjust=False).mean()

        # 清理NaN值
        result_df = df_copy[["adx", "+di", "-di"]].fillna(0)

        return result_df
    
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()
    
    def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 20) -> float:
        try:
            lags = range(2, max_lag)
            tau = []
            for lag in lags:
                diff_series = np.subtract(price_series[lag:], price_series[:-lag])
                # 检查差分序列有效性
                if len(diff_series) > 0 and not np.all(np.isnan(diff_series)):
                    valid_diffs = diff_series[~np.isnan(diff_series)]
                    if len(valid_diffs) > 0:
                        std_val = np.std(valid_diffs)
                        tau.append(max(1e-8, np.sqrt(std_val)))
                    else:
                        tau.append(1e-8)
                else:
                    tau.append(1e-8)
            
            if len(tau) < 2:
                return 0.5
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        except (ValueError, RuntimeWarning):
            return 0.5

logger = logging.getLogger(__name__)


class MarketCondition(Enum):
    """市场状态枚举"""
    BULLISH = "bullish"           # 看涨市场
    BEARISH = "bearish"           # 看跌市场  
    SIDEWAYS = "sideways"         # 横盘市场
    VOLATILE = "volatile"         # 高波动市场
    TRENDING = "trending"         # 趋势市场
    UNCERTAIN = "uncertain"       # 不确定市场


class VolatilityRegime(Enum):
    """波动率状态枚举"""
    LOW = "low"                   # 低波动率
    NORMAL = "normal"             # 正常波动率
    HIGH = "high"                 # 高波动率
    EXTREME = "extreme"           # 极端波动率


class LiquidityLevel(Enum):
    """流动性水平枚举"""
    VERY_LOW = "very_low"         # 极低流动性
    LOW = "low"                   # 低流动性
    NORMAL = "normal"             # 正常流动性
    HIGH = "high"                 # 高流动性
    VERY_HIGH = "very_high"       # 极高流动性


@dataclass
class VolatilityMetrics:
    """波动率指标类"""
    historical_volatility: float           # 历史波动率（年化）
    realized_volatility: float             # 实现波动率（年化）
    parkinson_volatility: float            # Parkinson波动率估计
    garman_klass_volatility: float         # Garman-Klass波动率估计
    yang_zhang_volatility: float           # Yang-Zhang波动率估计
    
    # 波动率状态
    volatility_regime: VolatilityRegime     # 波动率状态
    volatility_percentile: float           # 波动率百分位
    volatility_z_score: float              # 波动率Z分数
    
    # 时间信息
    calculation_period: int = 21            # 计算周期（天数）
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "historical_volatility": self.historical_volatility,
            "realized_volatility": self.realized_volatility,
            "parkinson_volatility": self.parkinson_volatility,
            "garman_klass_volatility": self.garman_klass_volatility,
            "yang_zhang_volatility": self.yang_zhang_volatility,
            "volatility_regime": self.volatility_regime.value,
            "volatility_percentile": self.volatility_percentile,
            "volatility_z_score": self.volatility_z_score,
            "calculation_period": self.calculation_period,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class VolumeAnalysis:
    """成交量分析结果类"""
    volume_trend: str                       # 成交量趋势（increasing/decreasing/stable）
    volume_strength: float                  # 成交量强度（0-1）
    volume_distribution: Dict[str, float]   # 成交量分布统计
    volume_profile: Dict[str, Any]          # 成交量价格分布
    abnormal_volume_detected: bool          # 是否检测到异常成交量
    volume_breakout_signal: bool            # 成交量突破信号
    volume_climax: bool                     # 成交量高潮
    volume_divergence: Optional[str] = None # 量价背离（bullish/bearish/none）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "volume_trend": self.volume_trend,
            "volume_strength": self.volume_strength,
            "volume_distribution": self.volume_distribution,
            "volume_profile": self.volume_profile,
            "abnormal_volume_detected": self.abnormal_volume_detected,
            "volume_breakout_signal": self.volume_breakout_signal,
            "volume_climax": self.volume_climax,
            "volume_divergence": self.volume_divergence
        }


@dataclass
class TechnicalIndicators:
    """技术指标集合类"""
    # 趋势指标
    ema_9: float                            # 9周期EMA
    ema_21: float                           # 21周期EMA
    ema_50: float                           # 50周期EMA
    ema_200: float                          # 200周期EMA
    adx: float                              # ADX趋势强度
    adx_di_plus: float                      # +DI
    adx_di_minus: float                     # -DI

    # 震荡指标
    rsi_14: float                           # 14周期RSI
    rsi_28: float                           # 28周期RSI
    stoch_k: float                          # 随机指标%K
    stoch_d: float                          # 随机指标%D

    # 波动指标
    bb_upper: float                         # 布林带上轨
    bb_middle: float                        # 布林带中轨
    bb_lower: float                         # 布林带下轨
    bb_percent: float                       # 布林带百分比
    bb_width: float                         # 布林带宽度
    atr: float                              # ATR

    # 增强成交量指标
    volume_sma: float                       # 成交量移动平均
    volume_ratio: float                     # 成交量比率

    # 可选字段（必须在最后）
    rsi_7: Optional[float] = None           # 7周期RSI（短期）
    rsi_21: Optional[float] = None          # 21周期RSI（中期）
    vwap: Optional[float] = None            # 成交量加权平均价
    volume_analysis: Optional[VolumeAnalysis] = None  # 详细成交量分析
    support_level: Optional[float] = None   # 支撑位
    resistance_level: Optional[float] = None # 阻力位

    # 默认值字段
    price_position: float = 0.0             # 价格在区间中的位置（0-1）
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_trend_signal(self) -> Tuple[TradingDirection, float]:
        """基于技术指标获取趋势信号"""
        signals = []
        confidence_scores = []
        
        # EMA趋势信号
        if self.ema_9 > self.ema_21 > self.ema_50:
            signals.append(1)
            confidence_scores.append(0.8)
        elif self.ema_9 < self.ema_21 < self.ema_50:
            signals.append(-1)
            confidence_scores.append(0.8)
        else:
            signals.append(0)
            confidence_scores.append(0.3)
        
        # ADX趋势强度
        if self.adx > 25:
            if self.adx_di_plus > self.adx_di_minus:
                signals.append(1)
                confidence_scores.append(min(self.adx / 50.0, 1.0))
            else:
                signals.append(-1)
                confidence_scores.append(min(self.adx / 50.0, 1.0))
        else:
            signals.append(0)
            confidence_scores.append(0.2)
        
        # RSI信号
        if self.rsi_14 > 70:
            signals.append(-1)
            confidence_scores.append(0.6)
        elif self.rsi_14 < 30:
            signals.append(1)
            confidence_scores.append(0.6)
        else:
            signals.append(0)
            confidence_scores.append(0.3)
        
        # 布林带信号
        if self.bb_percent > 0.8:
            signals.append(-1)
            confidence_scores.append(0.5)
        elif self.bb_percent < 0.2:
            signals.append(1)
            confidence_scores.append(0.5)
        else:
            signals.append(0)
            confidence_scores.append(0.3)
        
        # 计算加权平均信号
        weighted_signal = sum(s * c for s, c in zip(signals, confidence_scores)) / sum(confidence_scores)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # 转换为方向
        if weighted_signal > 0.2:
            direction = TradingDirection.LONG
        elif weighted_signal < -0.2:
            direction = TradingDirection.SHORT
        else:
            direction = TradingDirection.NEUTRAL
        
        return direction, avg_confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "ema_9": self.ema_9,
            "ema_21": self.ema_21,
            "ema_50": self.ema_50,
            "ema_200": self.ema_200,
            "adx": self.adx,
            "adx_di_plus": self.adx_di_plus,
            "adx_di_minus": self.adx_di_minus,
            "rsi_14": self.rsi_14,
            "rsi_28": self.rsi_28,
            "rsi_7": self.rsi_7,
            "rsi_21": self.rsi_21,
            "stoch_k": self.stoch_k,
            "stoch_d": self.stoch_d,
            "bb_upper": self.bb_upper,
            "bb_middle": self.bb_middle,
            "bb_lower": self.bb_lower,
            "bb_percent": self.bb_percent,
            "bb_width": self.bb_width,
            "atr": self.atr,
            "volume_sma": self.volume_sma,
            "volume_ratio": self.volume_ratio,
            "vwap": self.vwap,
            "price_position": self.price_position,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "timestamp": self.timestamp.isoformat()
        }

        # 添加成交量分析（如果存在）
        if self.volume_analysis:
            result["volume_analysis"] = self.volume_analysis.to_dict()

        return result


@dataclass
class CorrelationAnalysis:
    """相关性分析结果类"""
    asset_correlations: Dict[str, float]    # 与主要资产的相关性
    correlation_strength: str               # 相关性强度（strong/medium/weak）
    correlation_stability: float            # 相关性稳定性（0-1）
    market_coupling: float                  # 市场耦合度（0-1）
    correlation_breakdown_risk: float       # 相关性失效风险（0-1）
    diversification_benefit: float          # 分散化收益（0-1）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "asset_correlations": self.asset_correlations,
            "correlation_strength": self.correlation_strength,
            "correlation_stability": self.correlation_stability,
            "market_coupling": self.market_coupling,
            "correlation_breakdown_risk": self.correlation_breakdown_risk,
            "diversification_benefit": self.diversification_benefit
        }


@dataclass
class MarketSentimentAnalysis:
    """市场情绪分析结果类"""
    fear_greed_index: float                 # 恐贪指数（0-100）
    sentiment_label: str                    # 情绪标签（extreme_fear/fear/neutral/greed/extreme_greed）
    market_stress_indicators: Dict[str, float] # 市场压力指标集合
    volatility_sentiment: float             # 波动率情绪（0-1）
    momentum_sentiment: float               # 动量情绪（0-1）
    contrarian_signals: List[str]           # 逆向指标信号
    sentiment_divergence: Optional[str] = None  # 情绪背离

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "fear_greed_index": self.fear_greed_index,
            "sentiment_label": self.sentiment_label,
            "market_stress_indicators": self.market_stress_indicators,
            "volatility_sentiment": self.volatility_sentiment,
            "momentum_sentiment": self.momentum_sentiment,
            "contrarian_signals": self.contrarian_signals,
            "sentiment_divergence": self.sentiment_divergence
        }


@dataclass
class MarketConditionAnalysis:
    """市场条件分析结果类"""
    primary_condition: MarketCondition      # 主要市场状态
    secondary_conditions: List[MarketCondition] = field(default_factory=list)  # 次要状态

    # 市场环境指标
    trend_strength: float = 0.0             # 趋势强度（0-1）
    trend_direction: TradingDirection = TradingDirection.NEUTRAL  # 趋势方向
    liquidity_level: LiquidityLevel = LiquidityLevel.NORMAL      # 流动性水平
    market_sentiment: float = 0.5           # 市场情绪（0-1，0=极度恐慌，1=极度贪婪）

    # 增强分析模块
    correlation_analysis: Optional[CorrelationAnalysis] = None    # 相关性分析
    sentiment_analysis: Optional[MarketSentimentAnalysis] = None  # 情绪分析

    # 风险指标
    market_stress_level: float = 0.0        # 市场压力水平（0-1）
    correlation_breakdown: bool = False     # 相关性是否失效
    liquidity_crisis: bool = False          # 是否存在流动性危机

    # 置信度和可靠性
    analysis_confidence: float = 0.8        # 分析置信度
    data_quality_score: float = 1.0         # 数据质量评分

    # 时间信息
    analysis_time: datetime = field(default_factory=datetime.now)
    valid_until: Optional[datetime] = None   # 分析有效期
    
    def is_favorable_for_trading(self, risk_tolerance: str = "medium") -> bool:
        """判断当前市场条件是否适合交易"""
        if self.market_stress_level > 0.8:
            return False
        
        if self.liquidity_crisis or self.correlation_breakdown:
            return False
        
        if risk_tolerance == "low":
            return (self.trend_strength > 0.6 and 
                   self.liquidity_level in [LiquidityLevel.HIGH, LiquidityLevel.VERY_HIGH] and
                   self.market_stress_level < 0.3)
        elif risk_tolerance == "medium":
            return (self.trend_strength > 0.4 and 
                   self.liquidity_level != LiquidityLevel.VERY_LOW and
                   self.market_stress_level < 0.6)
        else:  # high risk tolerance
            return self.liquidity_level != LiquidityLevel.VERY_LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "primary_condition": self.primary_condition.value,
            "secondary_conditions": [c.value for c in self.secondary_conditions],
            "trend_strength": self.trend_strength,
            "trend_direction": self.trend_direction.value,
            "liquidity_level": self.liquidity_level.value,
            "market_sentiment": self.market_sentiment,
            "market_stress_level": self.market_stress_level,
            "correlation_breakdown": self.correlation_breakdown,
            "liquidity_crisis": self.liquidity_crisis,
            "analysis_confidence": self.analysis_confidence,
            "data_quality_score": self.data_quality_score,
            "analysis_time": self.analysis_time.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None
        }

        # 添加增强分析模块（如果存在）
        if self.correlation_analysis:
            result["correlation_analysis"] = self.correlation_analysis.to_dict()

        if self.sentiment_analysis:
            result["sentiment_analysis"] = self.sentiment_analysis.to_dict()

        return result


@dataclass
class RiskMetrics:
    """风险度量指标类"""
    # VaR相关指标
    var_1d_95: float                        # 1日95%VaR
    var_1d_99: float                        # 1日99%VaR
    var_5d_95: float                        # 5日95%VaR
    cvar_1d_95: float                       # 1日95%条件VaR
    
    # 回撤指标
    max_drawdown: float                     # 最大回撤
    current_drawdown: float                 # 当前回撤
    drawdown_duration: int                  # 回撤持续天数
    recovery_time: Optional[int] = None     # 预期恢复时间
    
    # 波动性风险
    volatility_risk: float = 0.0            # 波动性风险评分
    tail_risk: float = 0.0                  # 尾部风险
    skewness: float = 0.0                   # 收益分布偏度
    kurtosis: float = 0.0                   # 收益分布峰度
    
    # 流动性风险
    liquidity_risk: float = 0.0             # 流动性风险评分
    bid_ask_spread: Optional[float] = None  # 买卖价差
    market_impact: Optional[float] = None   # 市场冲击成本
    
    # 综合风险评级
    overall_risk_score: float = 0.0         # 综合风险评分（0-100）
    risk_level: RiskLevel = RiskLevel.MEDIUM # 风险等级
    
    # 计算参数
    calculation_period: int = 252           # 计算周期（交易日）
    confidence_level: float = 0.95          # 置信水平
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_risk_adjustment_factor(self) -> float:
        """获取风险调整系数（用于仓位大小调整）"""
        if self.risk_level == RiskLevel.LOW:
            return 1.2  # 可以适度增加仓位
        elif self.risk_level == RiskLevel.MEDIUM:
            return 1.0  # 标准仓位
        elif self.risk_level == RiskLevel.HIGH:
            return 0.7  # 减少仓位
        elif self.risk_level == RiskLevel.CRITICAL:
            return 0.4  # 大幅减少仓位
        else:  # EMERGENCY
            return 0.1  # 极小仓位或暂停交易
    
    def should_reduce_exposure(self) -> bool:
        """判断是否应该减少风险敞口"""
        return (self.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY] or
                self.max_drawdown > 0.15 or
                self.overall_risk_score > 80)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "var_1d_95": self.var_1d_95,
            "var_1d_99": self.var_1d_99,
            "var_5d_95": self.var_5d_95,
            "cvar_1d_95": self.cvar_1d_95,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "recovery_time": self.recovery_time,
            "volatility_risk": self.volatility_risk,
            "tail_risk": self.tail_risk,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "liquidity_risk": self.liquidity_risk,
            "bid_ask_spread": self.bid_ask_spread,
            "market_impact": self.market_impact,
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level.value,
            "calculation_period": self.calculation_period,
            "confidence_level": self.confidence_level,
            "timestamp": self.timestamp.isoformat()
        }


class MarketAnalyzer:
    """
    期货市场数据分析器
    
    提供全面的市场分析功能，包括：
    1. 波动率计算和分析
    2. 技术指标集成和信号生成
    3. 市场条件识别和分析
    4. 相关性分析和市场情绪计算
    5. 风险度量和风险管理
    6. 多时间框架分析支持
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化市场分析器
        
        Args:
            config: 分析器配置参数
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 缓存数据
        self._price_data_cache: Dict[str, pd.DataFrame] = {}
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_timeout = self.config.get("cache_timeout", 300)  # 5分钟缓存
        
        # 分析参数
        self.volatility_periods = self.config.get("volatility_periods", [21, 63, 252])
        self.technical_periods = self.config.get("technical_periods", {
            "rsi": [14, 28],
            "ema": [9, 21, 50, 200],
            "bb": 20,
            "atr": 14,
            "adx": 14
        })
        
        # 风险参数
        self.var_confidence_levels = self.config.get("var_confidence_levels", [0.95, 0.99])
        self.risk_calculation_periods = self.config.get("risk_periods", [1, 5, 21])
        
        # 市场条件阈值
        self.market_condition_thresholds = self.config.get("market_thresholds", {
            "trend_strength_min": 0.6,
            "volatility_regime_boundaries": [0.8, 1.2],
            "liquidity_thresholds": [0.2, 0.5, 1.5, 2.5],
            "sentiment_extremes": [0.2, 0.8]
        })
        
        self.logger.info("市场分析器初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "cache_timeout": 300,
            "enable_async": True,
            "volatility_periods": [21, 63, 252],
            "technical_periods": {
                "rsi": [14, 28],
                "ema": [9, 21, 50, 200],
                "bb": 20,
                "atr": 14,
                "adx": 14
            },
            "var_confidence_levels": [0.95, 0.99],
            "risk_periods": [1, 5, 21],
            "market_thresholds": {
                "trend_strength_min": 0.6,
                "volatility_regime_boundaries": [0.8, 1.2],
                "liquidity_thresholds": [0.2, 0.5, 1.5, 2.5],
                "sentiment_extremes": [0.2, 0.8]
            }
        }
    
    async def analyze_market_comprehensive(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行全面的市场分析
        
        Args:
            ticker: 交易对符号
            price_data: 价格数据（OHLC）
            volume_data: 成交量数据（可选）
            timeframes: 分析时间框架列表（可选）
            
        Returns:
            Dict[str, Any]: 完整的市场分析结果
        """
        try:
            self.logger.info(f"开始对 {ticker} 进行全面市场分析")
            
            # 验证数据
            if not self._validate_price_data(price_data):
                raise ValueError(f"价格数据验证失败: {ticker}")
            
            # 执行各个分析模块
            analysis_tasks = [
                self.calculate_volatility_metrics(ticker, price_data),
                self.calculate_technical_indicators(ticker, price_data, volume_data),
                self.analyze_market_conditions(ticker, price_data, volume_data),
                self.calculate_risk_metrics(ticker, price_data),
            ]
            
            # 并行执行分析任务
            if self.config.get("enable_async", True):
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            else:
                results = []
                for task in analysis_tasks:
                    try:
                        if asyncio.iscoroutine(task):
                            result = await task
                        else:
                            result = task
                        results.append(result)
                    except Exception as e:
                        results.append(e)
            
            # 处理结果
            volatility_metrics, technical_indicators, market_conditions, risk_metrics = results
            
            # 检查是否有异常
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"分析任务 {i} 失败: {result}")
                    raise result
            
            # 生成综合分析结果
            comprehensive_analysis = {
                "ticker": ticker,
                "analysis_timestamp": datetime.now().isoformat(),
                "volatility_metrics": volatility_metrics.to_dict() if volatility_metrics else None,
                "technical_indicators": technical_indicators.to_dict() if technical_indicators else None,
                "market_conditions": market_conditions.to_dict() if market_conditions else None,
                "risk_metrics": risk_metrics.to_dict() if risk_metrics else None,
                "data_quality": self._assess_data_quality(price_data),
                "analysis_confidence": self._calculate_overall_confidence(results),
            }
            
            # 生成交易建议
            trading_recommendation = self._generate_trading_recommendation(
                volatility_metrics, technical_indicators, market_conditions, risk_metrics
            )
            comprehensive_analysis["trading_recommendation"] = trading_recommendation
            
            # 多时间框架分析
            if timeframes:
                multi_timeframe_analysis = await self._analyze_multiple_timeframes(
                    ticker, price_data, timeframes
                )
                comprehensive_analysis["multi_timeframe_analysis"] = multi_timeframe_analysis
            
            # 缓存结果
            self._cache_analysis_result(ticker, comprehensive_analysis)
            
            self.logger.info(f"{ticker} 市场分析完成，置信度: {comprehensive_analysis['analysis_confidence']:.2f}")
            return comprehensive_analysis
            
        except Exception as e:
            self.logger.error(f"市场分析失败 {ticker}: {e}")
            raise
    
    async def calculate_volatility_metrics(
        self, 
        ticker: str, 
        price_data: pd.DataFrame,
        period: int = 21
    ) -> VolatilityMetrics:
        """
        计算全面的波动率指标
        
        Args:
            ticker: 交易对符号
            price_data: 价格数据
            period: 计算周期
            
        Returns:
            VolatilityMetrics: 波动率指标对象
        """
        try:
            # 计算收益率
            returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
            
            if len(returns) < period:
                raise ValueError(f"数据不足，需要至少 {period} 个数据点")
            
            # 1. 历史波动率（基于收盘价）
            historical_vol = returns.rolling(window=period).std() * math.sqrt(252)
            current_historical_vol = historical_vol.iloc[-1] if len(historical_vol) > 0 else 0.0
            
            # 2. 实现波动率（基于日内价格变动）
            if all(col in price_data.columns for col in ['high', 'low']):
                daily_range = np.log(price_data['high'] / price_data['low'])
                realized_vol = daily_range.rolling(window=period).std() * math.sqrt(252)
                current_realized_vol = realized_vol.iloc[-1] if len(realized_vol) > 0 else current_historical_vol
            else:
                current_realized_vol = current_historical_vol
            
            # 3. Parkinson波动率估计
            if all(col in price_data.columns for col in ['high', 'low']):
                parkinson_vol = math.sqrt(
                    (np.log(price_data['high'] / price_data['low']) ** 2).rolling(window=period).mean().iloc[-1] * 252
                ) if len(price_data) >= period else current_historical_vol
            else:
                parkinson_vol = current_historical_vol
            
            # 4. Garman-Klass波动率估计
            if all(col in price_data.columns for col in ['high', 'low', 'open', 'close']):
                hl_vol = 0.5 * (np.log(price_data['high'] / price_data['low']) ** 2)
                oc_vol = (2 * math.log(2) - 1) * (np.log(price_data['close'] / price_data['open']) ** 2)
                gk_vol = math.sqrt((hl_vol - oc_vol).rolling(window=period).mean().iloc[-1] * 252) if len(price_data) >= period else parkinson_vol
            else:
                gk_vol = parkinson_vol
            
            # 5. Yang-Zhang波动率估计
            if all(col in price_data.columns for col in ['high', 'low', 'open', 'close']):
                # 简化版Yang-Zhang估计
                overnight = np.log(price_data['open'] / price_data['close'].shift(1))
                rs = np.log(price_data['high'] / price_data['close']) * np.log(price_data['high'] / price_data['open']) + \
                     np.log(price_data['low'] / price_data['close']) * np.log(price_data['low'] / price_data['open'])
                yz_vol = math.sqrt((overnight.rolling(window=period).var() + rs.rolling(window=period).mean()).iloc[-1] * 252) \
                    if len(price_data) >= period else gk_vol
            else:
                yz_vol = gk_vol
            
            # 计算波动率状态
            vol_history = historical_vol.dropna()
            if len(vol_history) >= 63:  # 至少3个月数据
                vol_percentile = (vol_history <= current_historical_vol).sum() / len(vol_history)
                vol_mean = vol_history.rolling(window=63).mean().iloc[-1]
                vol_std = vol_history.rolling(window=63).std().iloc[-1]
                vol_z_score = (current_historical_vol - vol_mean) / vol_std if vol_std > 0 else 0.0
                
                # 确定波动率状态
                if vol_percentile <= 0.2:
                    volatility_regime = VolatilityRegime.LOW
                elif vol_percentile >= 0.8:
                    volatility_regime = VolatilityRegime.HIGH
                elif abs(vol_z_score) > 2:
                    volatility_regime = VolatilityRegime.EXTREME
                else:
                    volatility_regime = VolatilityRegime.NORMAL
            else:
                vol_percentile = 0.5
                vol_z_score = 0.0
                volatility_regime = VolatilityRegime.NORMAL
            
            return VolatilityMetrics(
                historical_volatility=float(current_historical_vol),
                realized_volatility=float(current_realized_vol),
                parkinson_volatility=float(parkinson_vol),
                garman_klass_volatility=float(gk_vol),
                yang_zhang_volatility=float(yz_vol),
                volatility_regime=volatility_regime,
                volatility_percentile=float(vol_percentile),
                volatility_z_score=float(vol_z_score),
                calculation_period=period
            )
            
        except Exception as e:
            self.logger.error(f"波动率计算失败 {ticker}: {e}")
            # 返回默认值
            return VolatilityMetrics(
                historical_volatility=0.2,
                realized_volatility=0.2,
                parkinson_volatility=0.2,
                garman_klass_volatility=0.2,
                yang_zhang_volatility=0.2,
                volatility_regime=VolatilityRegime.NORMAL,
                volatility_percentile=0.5,
                volatility_z_score=0.0,
                calculation_period=period
            )
    
    async def analyze_volume_comprehensive(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> VolumeAnalysis:
        """
        全面的成交量分析

        Args:
            ticker: 交易对符号
            price_data: 价格数据
            volume_data: 成交量数据（可选）

        Returns:
            VolumeAnalysis: 成交量分析结果
        """
        try:
            # 获取成交量数据
            if volume_data is not None and 'volume' in volume_data.columns:
                volumes = volume_data['volume']
            elif 'volume' in price_data.columns:
                volumes = price_data['volume']
            else:
                # 如果没有成交量数据，返回默认分析
                return self._create_default_volume_analysis()

            prices = price_data['close']

            # 1. 成交量趋势分析
            volume_trend = self._analyze_volume_trend(volumes)

            # 2. 成交量强度计算
            volume_strength = self._calculate_volume_strength(volumes, prices)

            # 3. 成交量分布统计
            volume_distribution = self._calculate_volume_distribution(volumes)

            # 4. 成交量价格分布（简化版Volume Profile）
            volume_profile = self._calculate_volume_profile(prices, volumes)

            # 5. 异常成交量检测
            abnormal_volume_detected = self._detect_abnormal_volume(volumes)

            # 6. 成交量突破信号
            volume_breakout_signal = self._detect_volume_breakout(volumes, prices)

            # 7. 成交量高潮检测
            volume_climax = self._detect_volume_climax(volumes, prices)

            # 8. 量价背离分析
            volume_divergence = self._analyze_price_volume_divergence(prices, volumes)

            return VolumeAnalysis(
                volume_trend=volume_trend,
                volume_strength=volume_strength,
                volume_distribution=volume_distribution,
                volume_profile=volume_profile,
                abnormal_volume_detected=abnormal_volume_detected,
                volume_breakout_signal=volume_breakout_signal,
                volume_climax=volume_climax,
                volume_divergence=volume_divergence
            )

        except Exception as e:
            self.logger.error(f"成交量分析失败 {ticker}: {e}")
            return self._create_default_volume_analysis()

    def _analyze_volume_trend(self, volumes: pd.Series, period: int = 20) -> str:
        """分析成交量趋势"""
        try:
            if len(volumes) < period:
                return "stable"

            recent_avg = volumes.tail(period // 2).mean()
            previous_avg = volumes.tail(period).head(period // 2).mean()

            change_ratio = (recent_avg / previous_avg - 1) if previous_avg > 0 else 0

            if change_ratio > 0.2:
                return "increasing"
            elif change_ratio < -0.2:
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "stable"

    def _calculate_volume_strength(self, volumes: pd.Series, prices: pd.Series) -> float:
        """计算成交量强度"""
        try:
            if len(volumes) < 20:
                return 0.5

            # 计算成交量的相对强度
            volume_sma_20 = volumes.rolling(20).mean()
            volume_sma_5 = volumes.rolling(5).mean()
            current_volume = volumes.iloc[-1]

            # 当前成交量与短期平均的比较
            short_term_strength = current_volume / volume_sma_5.iloc[-1] if volume_sma_5.iloc[-1] > 0 else 1

            # 短期平均与长期平均的比较
            long_term_strength = volume_sma_5.iloc[-1] / volume_sma_20.iloc[-1] if volume_sma_20.iloc[-1] > 0 else 1

            # 综合强度评分
            combined_strength = (short_term_strength * 0.6 + long_term_strength * 0.4) / 2

            # 标准化到0-1范围
            return min(1.0, max(0.0, combined_strength))

        except Exception:
            return 0.5

    def _calculate_volume_distribution(self, volumes: pd.Series) -> Dict[str, float]:
        """计算成交量分布统计"""
        try:
            if len(volumes) == 0:
                return {}

            return {
                "mean": float(volumes.mean()),
                "median": float(volumes.median()),
                "std": float(volumes.std()),
                "percentile_25": float(volumes.quantile(0.25)),
                "percentile_75": float(volumes.quantile(0.75)),
                "percentile_90": float(volumes.quantile(0.90)),
                "max": float(volumes.max()),
                "min": float(volumes.min()),
                "skewness": float(volumes.skew()) if len(volumes) > 1 else 0.0,
                "kurtosis": float(volumes.kurtosis()) if len(volumes) > 1 else 0.0
            }
        except Exception:
            return {}

    def _calculate_volume_profile(self, prices: pd.Series, volumes: pd.Series) -> Dict[str, Any]:
        """计算简化版成交量价格分布"""
        try:
            if len(prices) != len(volumes) or len(prices) < 20:
                return {}

            # 将价格分成10个区间
            price_bins = 10
            min_price = prices.min()
            max_price = prices.max()

            if max_price <= min_price:
                return {}

            price_range = max_price - min_price
            bin_size = price_range / price_bins

            volume_profile = {}
            total_volume = volumes.sum()

            for i in range(price_bins):
                lower_bound = min_price + i * bin_size
                upper_bound = min_price + (i + 1) * bin_size

                # 找到在此价格区间的数据点
                mask = (prices >= lower_bound) & (prices < upper_bound)
                if i == price_bins - 1:  # 最后一个区间包含上边界
                    mask = (prices >= lower_bound) & (prices <= upper_bound)

                bin_volume = volumes[mask].sum()
                volume_percentage = (bin_volume / total_volume * 100) if total_volume > 0 else 0

                volume_profile[f"price_{lower_bound:.2f}_{upper_bound:.2f}"] = {
                    "volume": float(bin_volume),
                    "volume_percentage": float(volume_percentage),
                    "price_range": [float(lower_bound), float(upper_bound)]
                }

            # 找到成交量最大的价格区间（POC - Point of Control）
            max_volume_bin = max(volume_profile.items(), key=lambda x: x[1]["volume"])

            return {
                "profile": volume_profile,
                "point_of_control": {
                    "price_range": max_volume_bin[1]["price_range"],
                    "volume_percentage": max_volume_bin[1]["volume_percentage"]
                },
                "total_volume": float(total_volume)
            }

        except Exception as e:
            self.logger.warning(f"计算成交量分布失败: {e}")
            return {}

    def _detect_abnormal_volume(self, volumes: pd.Series, threshold: float = 2.5) -> bool:
        """检测异常成交量"""
        try:
            if len(volumes) < 20:
                return False

            # 计算最近20期的平均成交量和标准差
            recent_volumes = volumes.tail(20)
            mean_volume = recent_volumes.mean()
            std_volume = recent_volumes.std()

            if std_volume == 0:
                return False

            # 检查最新成交量是否异常
            current_volume = volumes.iloc[-1]
            z_score = abs(current_volume - mean_volume) / std_volume

            return z_score > threshold

        except Exception:
            return False

    def _detect_volume_breakout(self, volumes: pd.Series, prices: pd.Series) -> bool:
        """检测成交量突破信号"""
        try:
            if len(volumes) < 10 or len(prices) < 10:
                return False

            # 成交量突破：当前成交量显著高于平均水平且价格有明显变动
            current_volume = volumes.iloc[-1]
            avg_volume = volumes.tail(10).mean()
            volume_breakout = current_volume > avg_volume * 1.5

            # 价格变动
            price_change = abs(prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) >= 2 else 0
            significant_price_move = price_change > 0.02  # 2%以上的价格变动

            return volume_breakout and significant_price_move

        except Exception:
            return False

    def _detect_volume_climax(self, volumes: pd.Series, prices: pd.Series) -> bool:
        """检测成交量高潮"""
        try:
            if len(volumes) < 5 or len(prices) < 5:
                return False

            # 成交量高潮：极高成交量伴随价格反转迹象
            current_volume = volumes.iloc[-1]
            max_recent_volume = volumes.tail(20).max() if len(volumes) >= 20 else volumes.max()

            # 当前成交量是否达到近期峰值
            volume_peak = current_volume >= max_recent_volume * 0.9

            # 价格是否出现反转迹象（简化判断）
            if len(prices) >= 3:
                recent_prices = prices.tail(3)
                price_reversal = (
                    (recent_prices.iloc[0] < recent_prices.iloc[1] > recent_prices.iloc[2]) or
                    (recent_prices.iloc[0] > recent_prices.iloc[1] < recent_prices.iloc[2])
                )
            else:
                price_reversal = False

            return volume_peak and price_reversal

        except Exception:
            return False

    def _analyze_price_volume_divergence(self, prices: pd.Series, volumes: pd.Series, period: int = 14) -> Optional[str]:
        """分析量价背离"""
        try:
            if len(prices) < period or len(volumes) < period:
                return None

            # 计算价格和成交量的趋势
            recent_prices = prices.tail(period)
            recent_volumes = volumes.tail(period)

            # 价格趋势（线性回归斜率）
            price_x = np.arange(len(recent_prices))
            price_trend = np.polyfit(price_x, recent_prices, 1)[0]

            # 成交量趋势
            volume_x = np.arange(len(recent_volumes))
            volume_trend = np.polyfit(volume_x, recent_volumes, 1)[0]

            # 判断背离
            if price_trend > 0 and volume_trend < 0:
                return "bearish"  # 价格上涨但成交量下降，看跌背离
            elif price_trend < 0 and volume_trend > 0:
                return "bullish"  # 价格下跌但成交量上升，看涨背离
            else:
                return None  # 无明显背离

        except Exception:
            return None

    def _create_default_volume_analysis(self) -> VolumeAnalysis:
        """创建默认的成交量分析结果"""
        return VolumeAnalysis(
            volume_trend="stable",
            volume_strength=0.5,
            volume_distribution={},
            volume_profile={},
            abnormal_volume_detected=False,
            volume_breakout_signal=False,
            volume_climax=False,
            volume_divergence=None
        )

    async def calculate_technical_indicators(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> TechnicalIndicators:
        """
        计算全面的技术指标
        
        Args:
            ticker: 交易对符号
            price_data: 价格数据
            volume_data: 成交量数据（可选）
            
        Returns:
            TechnicalIndicators: 技术指标对象
        """
        try:
            # 检查数据长度
            if len(price_data) < 200:  # 确保有足够数据计算200日EMA
                self.logger.warning(f"数据长度不足 {ticker}: {len(price_data)} < 200")
            
            current_price = float(price_data['close'].iloc[-1])
            
            # 1. 趋势指标
            ema_9 = float(calculate_ema(price_data, 9).iloc[-1])
            ema_21 = float(calculate_ema(price_data, 21).iloc[-1])
            ema_50 = float(calculate_ema(price_data, 50).iloc[-1]) if len(price_data) >= 50 else ema_21
            ema_200 = float(calculate_ema(price_data, 200).iloc[-1]) if len(price_data) >= 200 else ema_50
            
            # ADX指标
            adx_data = calculate_adx(price_data.copy(), 14)
            adx = float(adx_data['adx'].iloc[-1])
            adx_di_plus = float(adx_data['+di'].iloc[-1])
            adx_di_minus = float(adx_data['-di'].iloc[-1])
            
            # 2. 震荡指标（增强多周期RSI）
            rsi_14 = float(calculate_rsi(price_data, 14).iloc[-1])
            rsi_28 = float(calculate_rsi(price_data, 28).iloc[-1])
            rsi_7 = float(calculate_rsi(price_data, 7).iloc[-1]) if len(price_data) >= 7 else rsi_14
            rsi_21 = float(calculate_rsi(price_data, 21).iloc[-1]) if len(price_data) >= 21 else rsi_28
            
            # 随机指标
            stoch_k, stoch_d = self._calculate_stochastic(price_data)
            
            # 3. 波动指标
            bb_upper, bb_lower = calculate_bollinger_bands(price_data, 20)
            bb_middle = price_data['close'].rolling(20).mean()
            
            bb_upper_val = float(bb_upper.iloc[-1])
            bb_lower_val = float(bb_lower.iloc[-1])
            bb_middle_val = float(bb_middle.iloc[-1])
            
            # 布林带百分比和宽度
            bb_percent = (current_price - bb_lower_val) / (bb_upper_val - bb_lower_val) if bb_upper_val > bb_lower_val else 0.5
            bb_width = (bb_upper_val - bb_lower_val) / bb_middle_val if bb_middle_val > 0 else 0.0
            
            # ATR
            atr = float(calculate_atr(price_data, 14).iloc[-1])
            
            # 4. 增强成交量指标
            volume_analysis = None
            if volume_data is not None and 'volume' in volume_data.columns:
                volume_sma = float(volume_data['volume'].rolling(21).mean().iloc[-1])
                current_volume = float(volume_data['volume'].iloc[-1])
                volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0

                # VWAP计算
                if all(col in price_data.columns for col in ['high', 'low', 'close']):
                    typical_price = (price_data['high'] + price_data['low'] + price_data['close']) / 3
                    vwap = (typical_price * volume_data['volume']).rolling(21).sum() / volume_data['volume'].rolling(21).sum()
                    vwap_val = float(vwap.iloc[-1]) if not vwap.empty else None
                else:
                    vwap_val = None

                # 执行详细成交量分析
                volume_analysis = await self.analyze_volume_comprehensive(ticker, price_data, volume_data)

            elif 'volume' in price_data.columns:
                volume_sma = float(price_data['volume'].rolling(21).mean().iloc[-1])
                current_volume = float(price_data['volume'].iloc[-1])
                volume_ratio = current_volume / volume_sma if volume_sma > 0 else 1.0
                vwap_val = None

                # 执行详细成交量分析
                volume_analysis = await self.analyze_volume_comprehensive(ticker, price_data)
            else:
                volume_sma = 0.0
                volume_ratio = 1.0
                vwap_val = None
            
            # 5. 价格位置指标
            high_52w = price_data['high'].rolling(252).max().iloc[-1] if len(price_data) >= 252 else price_data['high'].max()
            low_52w = price_data['low'].rolling(252).min().iloc[-1] if len(price_data) >= 252 else price_data['low'].min()
            price_position = (current_price - low_52w) / (high_52w - low_52w) if high_52w > low_52w else 0.5
            
            # 支撑和阻力位（简化版）
            recent_highs = price_data['high'].rolling(21).max()
            recent_lows = price_data['low'].rolling(21).min()
            resistance_level = float(recent_highs.iloc[-1])
            support_level = float(recent_lows.iloc[-1])
            
            return TechnicalIndicators(
                ema_9=ema_9,
                ema_21=ema_21,
                ema_50=ema_50,
                ema_200=ema_200,
                adx=adx,
                adx_di_plus=adx_di_plus,
                adx_di_minus=adx_di_minus,
                rsi_14=rsi_14,
                rsi_28=rsi_28,
                rsi_7=rsi_7,
                rsi_21=rsi_21,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                bb_upper=bb_upper_val,
                bb_middle=bb_middle_val,
                bb_lower=bb_lower_val,
                bb_percent=float(bb_percent),
                bb_width=float(bb_width),
                atr=atr,
                volume_sma=volume_sma,
                volume_ratio=volume_ratio,
                vwap=vwap_val,
                volume_analysis=volume_analysis,
                price_position=float(price_position),
                support_level=support_level,
                resistance_level=resistance_level
            )
            
        except Exception as e:
            self.logger.error(f"技术指标计算失败 {ticker}: {e}")
            raise
    
    def _calculate_stochastic(self, price_data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """计算随机指标"""
        try:
            high_max = price_data['high'].rolling(window=k_period).max()
            low_min = price_data['low'].rolling(window=k_period).min()
            
            stoch_k = 100 * (price_data['close'] - low_min) / (high_max - low_min)
            stoch_d = stoch_k.rolling(window=d_period).mean()
            
            return float(stoch_k.iloc[-1]), float(stoch_d.iloc[-1])
        except Exception:
            return 50.0, 50.0  # 默认中性值
    
    async def analyze_correlation_comprehensive(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        reference_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> CorrelationAnalysis:
        """
        全面的相关性分析

        Args:
            ticker: 交易对符号
            price_data: 价格数据
            reference_data: 参考资产数据字典（可选）

        Returns:
            CorrelationAnalysis: 相关性分析结果
        """
        try:
            # 如果没有参考数据，使用模拟的主要资产相关性
            if reference_data is None:
                reference_data = self._generate_reference_correlations(ticker)

            returns = price_data['close'].pct_change(fill_method=None).dropna()

            # 1. 计算与主要资产的相关性
            asset_correlations = self._calculate_asset_correlations(returns, reference_data)

            # 2. 评估相关性强度
            correlation_strength = self._evaluate_correlation_strength(asset_correlations)

            # 3. 计算相关性稳定性
            correlation_stability = self._calculate_correlation_stability(returns, reference_data)

            # 4. 市场耦合度
            market_coupling = self._calculate_market_coupling(asset_correlations)

            # 5. 相关性失效风险
            correlation_breakdown_risk = self._assess_correlation_breakdown_risk(returns, reference_data)

            # 6. 分散化收益
            diversification_benefit = self._calculate_diversification_benefit(asset_correlations)

            return CorrelationAnalysis(
                asset_correlations=asset_correlations,
                correlation_strength=correlation_strength,
                correlation_stability=correlation_stability,
                market_coupling=market_coupling,
                correlation_breakdown_risk=correlation_breakdown_risk,
                diversification_benefit=diversification_benefit
            )

        except Exception as e:
            self.logger.error(f"相关性分析失败 {ticker}: {e}")
            return self._create_default_correlation_analysis()

    def _generate_reference_correlations(self, ticker: str) -> Dict[str, float]:
        """生成参考资产相关性（模拟数据）"""
        try:
            # 基于ticker类型生成合理的相关性值
            ticker_upper = ticker.upper()

            correlations = {}

            if 'BTC' in ticker_upper:
                correlations = {
                    'ETH': 0.75,
                    'SP500': 0.45,
                    'GOLD': -0.15,
                    'DXY': -0.35,
                    'VIX': -0.25
                }
            elif 'ETH' in ticker_upper:
                correlations = {
                    'BTC': 0.75,
                    'SP500': 0.40,
                    'GOLD': -0.10,
                    'DXY': -0.30,
                    'VIX': -0.20
                }
            else:
                # 其他加密货币
                correlations = {
                    'BTC': 0.60,
                    'ETH': 0.55,
                    'SP500': 0.35,
                    'GOLD': -0.05,
                    'DXY': -0.25,
                    'VIX': -0.15
                }

            return correlations

        except Exception:
            return {}

    def _calculate_asset_correlations(self, returns: pd.Series, reference_data: Dict[str, Any]) -> Dict[str, float]:
        """计算与主要资产的相关性"""
        try:
            if isinstance(reference_data, dict) and 'BTC' in reference_data:
                # 如果reference_data包含实际价格数据
                correlations = {}
                for asset, data in reference_data.items():
                    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
                        asset_returns = data['close'].pct_change(fill_method=None).dropna()
                        # 确保时间序列对齐
                        aligned_returns = returns.align(asset_returns, join='inner')[0]
                        aligned_asset_returns = returns.align(asset_returns, join='inner')[1]

                        if len(aligned_returns) > 10:  # 需要足够的数据点
                            correlation = aligned_returns.corr(aligned_asset_returns)
                            correlations[asset] = float(correlation) if not pd.isna(correlation) else 0.0
                        else:
                            correlations[asset] = 0.0
                return correlations
            else:
                # 使用模拟相关性数据
                return reference_data if isinstance(reference_data, dict) else {}

        except Exception:
            return {}

    def _evaluate_correlation_strength(self, correlations: Dict[str, float]) -> str:
        """评估相关性强度"""
        try:
            if not correlations:
                return "weak"

            avg_abs_correlation = sum(abs(corr) for corr in correlations.values()) / len(correlations)

            if avg_abs_correlation > 0.7:
                return "strong"
            elif avg_abs_correlation > 0.4:
                return "medium"
            else:
                return "weak"

        except Exception:
            return "weak"

    def _calculate_correlation_stability(self, returns: pd.Series, reference_data: Dict[str, Any], window: int = 30) -> float:
        """计算相关性稳定性"""
        try:
            if len(returns) < window * 2:
                return 0.5

            # 计算滚动相关性的标准差作为稳定性指标
            if isinstance(reference_data, dict) and any(isinstance(v, pd.DataFrame) for v in reference_data.values()):
                # 使用实际数据计算稳定性
                stabilities = []
                for asset, data in reference_data.items():
                    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
                        asset_returns = data['close'].pct_change(fill_method=None).dropna()
                        aligned_returns = returns.align(asset_returns, join='inner')[0]
                        aligned_asset_returns = returns.align(asset_returns, join='inner')[1]

                        if len(aligned_returns) > window:
                            rolling_corr = aligned_returns.rolling(window).corr(aligned_asset_returns)
                            corr_stability = 1.0 - rolling_corr.std() if rolling_corr.std() > 0 else 1.0
                            stabilities.append(max(0.0, min(1.0, corr_stability)))

                return sum(stabilities) / len(stabilities) if stabilities else 0.5
            else:
                # 使用价格波动性作为稳定性代理
                volatility = returns.rolling(window).std().std()
                # 低波动性通常意味着更稳定的相关性
                stability = max(0.0, min(1.0, 1.0 - volatility * 10))
                return stability

        except Exception:
            return 0.5

    def _calculate_market_coupling(self, correlations: Dict[str, float]) -> float:
        """计算市场耦合度"""
        try:
            if not correlations:
                return 0.5

            # 与传统金融市场（如SP500）的相关性作为耦合度指标
            traditional_assets = ['SP500', 'GOLD', 'DXY']
            traditional_correlations = [abs(correlations.get(asset, 0)) for asset in traditional_assets]

            if traditional_correlations:
                coupling = sum(traditional_correlations) / len(traditional_correlations)
                return min(1.0, max(0.0, coupling))
            else:
                # 如果没有传统资产相关性，使用平均相关性
                avg_correlation = sum(abs(corr) for corr in correlations.values()) / len(correlations)
                return min(1.0, max(0.0, avg_correlation))

        except Exception:
            return 0.5

    def _assess_correlation_breakdown_risk(self, returns: pd.Series, reference_data: Dict[str, Any]) -> float:
        """评估相关性失效风险"""
        try:
            # 使用价格波动性和相关性变化作为失效风险指标
            if len(returns) < 60:  # 需要至少2个月数据
                return 0.5

            # 计算最近的波动性变化
            recent_vol = returns.tail(30).std()
            historical_vol = returns.std()
            vol_change = abs(recent_vol / historical_vol - 1) if historical_vol > 0 else 0

            # 计算相关性变化（简化版本）
            correlation_instability = 1.0 - self._calculate_correlation_stability(returns, reference_data)

            # 综合风险评分
            risk_score = (vol_change * 0.6 + correlation_instability * 0.4)
            return min(1.0, max(0.0, risk_score))

        except Exception:
            return 0.5

    def _calculate_diversification_benefit(self, correlations: Dict[str, float]) -> float:
        """计算分散化收益"""
        try:
            if not correlations:
                return 0.5

            # 分散化收益与相关性成反比
            avg_correlation = sum(abs(corr) for corr in correlations.values()) / len(correlations)
            diversification_benefit = 1.0 - avg_correlation

            return min(1.0, max(0.0, diversification_benefit))

        except Exception:
            return 0.5

    def _create_default_correlation_analysis(self) -> CorrelationAnalysis:
        """创建默认的相关性分析结果"""
        return CorrelationAnalysis(
            asset_correlations={},
            correlation_strength="weak",
            correlation_stability=0.5,
            market_coupling=0.5,
            correlation_breakdown_risk=0.5,
            diversification_benefit=0.5
        )

    async def analyze_market_sentiment_comprehensive(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> MarketSentimentAnalysis:
        """
        全面的市场情绪分析

        Args:
            ticker: 交易对符号
            price_data: 价格数据
            volume_data: 成交量数据（可选）

        Returns:
            MarketSentimentAnalysis: 市场情绪分析结果
        """
        try:
            returns = price_data['close'].pct_change(fill_method=None).dropna()

            # 1. 计算恐贪指数
            fear_greed_index = self._calculate_fear_greed_index(price_data, returns, volume_data)

            # 2. 情绪标签
            sentiment_label = self._determine_sentiment_label(fear_greed_index)

            # 3. 市场压力指标
            market_stress_indicators = self._calculate_market_stress_indicators(returns, price_data)

            # 4. 波动率情绪
            volatility_sentiment = self._calculate_volatility_sentiment(returns)

            # 5. 动量情绪
            momentum_sentiment = self._calculate_momentum_sentiment(returns)

            # 6. 逆向指标信号
            contrarian_signals = self._identify_contrarian_signals(fear_greed_index, returns)

            # 7. 情绪背离
            sentiment_divergence = self._detect_sentiment_divergence(returns, fear_greed_index)

            return MarketSentimentAnalysis(
                fear_greed_index=fear_greed_index,
                sentiment_label=sentiment_label,
                market_stress_indicators=market_stress_indicators,
                volatility_sentiment=volatility_sentiment,
                momentum_sentiment=momentum_sentiment,
                contrarian_signals=contrarian_signals,
                sentiment_divergence=sentiment_divergence
            )

        except Exception as e:
            self.logger.error(f"市场情绪分析失败 {ticker}: {e}")
            return self._create_default_sentiment_analysis()

    def _calculate_fear_greed_index(self, price_data: pd.DataFrame, returns: pd.Series, volume_data: Optional[pd.DataFrame]) -> float:
        """计算恐贪指数（0-100）"""
        try:
            components = {}

            # 1. 价格动量（30%权重）
            if len(returns) >= 30:
                momentum_30d = (price_data['close'].iloc[-1] / price_data['close'].iloc[-30] - 1) * 100
                momentum_score = min(100, max(0, 50 + momentum_30d * 2))  # 标准化到0-100
                components['momentum'] = momentum_score * 0.3

            # 2. 波动率（25%权重）
            volatility = returns.std() * math.sqrt(252) * 100  # 年化波动率百分比
            volatility_score = max(0, min(100, 100 - volatility))  # 高波动率 = 恐慌
            components['volatility'] = volatility_score * 0.25

            # 3. RSI指标（20%权重）
            if len(price_data) >= 14:
                rsi = calculate_rsi(price_data, 14).iloc[-1]
                rsi_score = min(100, max(0, rsi))
                components['rsi'] = rsi_score * 0.2

            # 4. 成交量指标（15%权重）
            if volume_data is not None or 'volume' in price_data.columns:
                volumes = volume_data['volume'] if volume_data is not None else price_data['volume']
                if len(volumes) >= 20:
                    current_volume = volumes.iloc[-1]
                    avg_volume = volumes.rolling(20).mean().iloc[-1]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    volume_score = min(100, max(0, 50 + (volume_ratio - 1) * 50))
                    components['volume'] = volume_score * 0.15
                else:
                    components['volume'] = 50 * 0.15
            else:
                components['volume'] = 50 * 0.15

            # 5. 市场支配地位（10%权重）- 简化版
            if len(price_data) >= 52:
                price_position = (price_data['close'].iloc[-1] - price_data['close'].rolling(52).min().iloc[-1]) / (price_data['close'].rolling(52).max().iloc[-1] - price_data['close'].rolling(52).min().iloc[-1])
                dominance_score = price_position * 100
                components['dominance'] = dominance_score * 0.1
            else:
                components['dominance'] = 50 * 0.1

            # 综合恐贪指数
            fear_greed_index = sum(components.values())
            return min(100, max(0, fear_greed_index))

        except Exception:
            return 50.0  # 中性值

    def _determine_sentiment_label(self, fear_greed_index: float) -> str:
        """根据恐贪指数确定情绪标签"""
        if fear_greed_index <= 10:
            return "extreme_fear"
        elif fear_greed_index <= 25:
            return "fear"
        elif fear_greed_index <= 45:
            return "neutral"
        elif fear_greed_index <= 75:
            return "greed"
        else:
            return "extreme_greed"

    def _calculate_market_stress_indicators(self, returns: pd.Series, price_data: pd.DataFrame) -> Dict[str, float]:
        """计算市场压力指标集合"""
        try:
            indicators = {}

            # VIX指数代理（基于21日滚动波动率）
            if len(returns) >= 21:
                rolling_vol = returns.rolling(21).std() * math.sqrt(252) * 100
                vix_proxy = rolling_vol.iloc[-1]
                indicators['vix_proxy'] = min(100, max(0, vix_proxy))

            # 最大回撤压力
            if len(price_data) >= 30:
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.rolling(30).max()
                drawdown = ((cumulative - rolling_max) / rolling_max).abs()
                max_drawdown_30d = drawdown.max() * 100
                indicators['drawdown_stress'] = min(100, max(0, max_drawdown_30d))

            # 价格跳跃频率
            if len(returns) >= 20:
                large_moves = (abs(returns) > 2 * returns.std()).sum()
                jump_frequency = (large_moves / len(returns)) * 100
                indicators['jump_frequency'] = min(100, max(0, jump_frequency))

            return indicators

        except Exception:
            return {}

    def _calculate_volatility_sentiment(self, returns: pd.Series) -> float:
        """计算波动率情绪"""
        try:
            if len(returns) < 30:
                return 0.5

            current_vol = returns.tail(7).std() * math.sqrt(252)  # 最近一周年化波动率
            historical_vol = returns.std() * math.sqrt(252)  # 历史波动率

            if historical_vol == 0:
                return 0.5

            vol_ratio = current_vol / historical_vol

            # 转换为情绪分数（高波动率 = 恐慌情绪 = 低分）
            sentiment = max(0, min(1, 1 / (1 + vol_ratio)))

            return sentiment

        except Exception:
            return 0.5

    def _calculate_momentum_sentiment(self, returns: pd.Series) -> float:
        """计算动量情绪"""
        try:
            if len(returns) < 10:
                return 0.5

            # 使用短期动量作为情绪指标
            momentum_5d = returns.tail(5).mean()
            momentum_10d = returns.tail(10).mean()

            # 组合动量
            combined_momentum = momentum_5d * 0.6 + momentum_10d * 0.4

            # 转换为0-1范围
            sentiment = max(0, min(1, 0.5 + combined_momentum * 10))

            return sentiment

        except Exception:
            return 0.5

    def _identify_contrarian_signals(self, fear_greed_index: float, returns: pd.Series) -> List[str]:
        """识别逆向指标信号"""
        signals = []

        try:
            # 极端恐慌信号
            if fear_greed_index <= 10:
                signals.append("extreme_fear_buy_signal")

            # 极端贪婪信号
            if fear_greed_index >= 90:
                signals.append("extreme_greed_sell_signal")

            # 波动率异常低
            if len(returns) >= 20:
                recent_vol = returns.tail(10).std()
                historical_vol = returns.std()
                if recent_vol < historical_vol * 0.5:
                    signals.append("low_volatility_warning")

            # 连续上涨警告
            if len(returns) >= 7:
                consecutive_up = all(returns.tail(7) > 0)
                if consecutive_up:
                    signals.append("consecutive_gains_warning")

            return signals

        except Exception:
            return []

    def _detect_sentiment_divergence(self, returns: pd.Series, fear_greed_index: float) -> Optional[str]:
        """检测情绪背离"""
        try:
            if len(returns) < 20:
                return None

            # 价格动量
            price_momentum = returns.tail(10).mean()

            # 情绪vs价格背离
            if fear_greed_index > 70 and price_momentum < -0.005:  # 高贪婪但价格下跌
                return "bearish_divergence"
            elif fear_greed_index < 30 and price_momentum > 0.005:  # 高恐慌但价格上涨
                return "bullish_divergence"

            return None

        except Exception:
            return None

    def _create_default_sentiment_analysis(self) -> MarketSentimentAnalysis:
        """创建默认的市场情绪分析结果"""
        return MarketSentimentAnalysis(
            fear_greed_index=50.0,
            sentiment_label="neutral",
            market_stress_indicators={},
            volatility_sentiment=0.5,
            momentum_sentiment=0.5,
            contrarian_signals=[],
            sentiment_divergence=None
        )

    async def analyze_market_conditions(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.DataFrame] = None
    ) -> MarketConditionAnalysis:
        """
        分析市场条件和环境
        
        Args:
            ticker: 交易对符号
            price_data: 价格数据
            volume_data: 成交量数据（可选）
            
        Returns:
            MarketConditionAnalysis: 市场条件分析结果
        """
        try:
            # 计算基础指标
            returns = price_data['close'].pct_change(fill_method=None).dropna()

            # 1. 趋势分析
            trend_strength, trend_direction = self._analyze_trend(price_data)

            # 2. 流动性分析
            liquidity_level = self._analyze_liquidity(price_data, volume_data)

            # 3. 基础市场情绪分析
            market_sentiment = self._analyze_market_sentiment(returns)

            # 4. 市场压力分析
            market_stress_level = self._calculate_market_stress(returns, price_data)

            # 5. 相关性分析
            correlation_breakdown = self._detect_correlation_breakdown(returns)

            # 6. 流动性危机检测
            liquidity_crisis = self._detect_liquidity_crisis(price_data, volume_data)

            # 7. 增强分析模块
            # 执行全面的相关性分析
            correlation_analysis = await self.analyze_correlation_comprehensive(ticker, price_data)

            # 执行全面的市场情绪分析
            sentiment_analysis = await self.analyze_market_sentiment_comprehensive(ticker, price_data, volume_data)

            # 使用增强的市场情绪
            market_sentiment = sentiment_analysis.fear_greed_index / 100.0  # 转换为0-1范围

            # 8. 确定主要市场条件
            primary_condition = self._determine_primary_market_condition(
                trend_strength, trend_direction, market_stress_level, liquidity_level
            )

            # 9. 次要条件
            secondary_conditions = self._identify_secondary_conditions(
                market_sentiment, liquidity_level, market_stress_level
            )

            # 10. 计算分析置信度
            analysis_confidence = self._calculate_analysis_confidence(price_data, volume_data)

            # 11. 数据质量评估
            data_quality_score = self._assess_data_quality(price_data)

            return MarketConditionAnalysis(
                primary_condition=primary_condition,
                secondary_conditions=secondary_conditions,
                trend_strength=trend_strength,
                trend_direction=trend_direction,
                liquidity_level=liquidity_level,
                market_sentiment=market_sentiment,
                correlation_analysis=correlation_analysis,
                sentiment_analysis=sentiment_analysis,
                market_stress_level=market_stress_level,
                correlation_breakdown=correlation_breakdown,
                liquidity_crisis=liquidity_crisis,
                analysis_confidence=analysis_confidence,
                data_quality_score=data_quality_score,
                valid_until=datetime.now() + timedelta(hours=1)  # 1小时有效期
            )
            
        except Exception as e:
            self.logger.error(f"市场条件分析失败 {ticker}: {e}")
            raise
    
    def _analyze_trend(self, price_data: pd.DataFrame) -> Tuple[float, TradingDirection]:
        """分析趋势强度和方向"""
        try:
            # 使用多个EMA判断趋势
            ema_9 = calculate_ema(price_data, 9)
            ema_21 = calculate_ema(price_data, 21)
            ema_50 = calculate_ema(price_data, 50)
            
            current_price = price_data['close'].iloc[-1]
            
            # 趋势方向评分
            trend_scores = []
            
            # EMA排列
            if len(ema_50) > 0:
                if ema_9.iloc[-1] > ema_21.iloc[-1] > ema_50.iloc[-1]:
                    trend_scores.append(1.0)  # 强烈看涨
                elif ema_9.iloc[-1] < ema_21.iloc[-1] < ema_50.iloc[-1]:
                    trend_scores.append(-1.0)  # 强烈看跌
                else:
                    trend_scores.append(0.0)  # 趋势不明确
            
            # 价格相对EMA位置
            if len(ema_21) > 0:
                price_vs_ema = (current_price / ema_21.iloc[-1] - 1) * 10  # 放大差异
                trend_scores.append(max(-1, min(1, price_vs_ema)))
            
            # ADX趋势强度
            try:
                adx_data = calculate_adx(price_data.copy(), 14)
                adx_strength = adx_data['adx'].iloc[-1] / 50.0  # 标准化到0-1
                adx_direction = 1 if adx_data['+di'].iloc[-1] > adx_data['-di'].iloc[-1] else -1
                trend_scores.append(adx_direction * min(1.0, adx_strength))
            except Exception:
                pass
            
            # 计算综合趋势评分
            if trend_scores:
                avg_trend_score = sum(trend_scores) / len(trend_scores)
                trend_strength = abs(avg_trend_score)
                
                if avg_trend_score > 0.3:
                    trend_direction = TradingDirection.LONG
                elif avg_trend_score < -0.3:
                    trend_direction = TradingDirection.SHORT
                else:
                    trend_direction = TradingDirection.NEUTRAL
            else:
                trend_strength = 0.0
                trend_direction = TradingDirection.NEUTRAL
            
            return min(1.0, trend_strength), trend_direction
            
        except Exception as e:
            self.logger.warning(f"趋势分析失败: {e}")
            return 0.0, TradingDirection.NEUTRAL
    
    def _analyze_liquidity(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> LiquidityLevel:
        """分析流动性水平"""
        try:
            # 基于价格波动性分析流动性
            returns = price_data['close'].pct_change(fill_method=None).dropna()
            volatility = returns.std() * math.sqrt(252)  # 年化波动率
            
            # 基于ATR分析流动性
            atr = calculate_atr(price_data, 14)
            atr_ratio = atr.iloc[-1] / price_data['close'].iloc[-1] if len(atr) > 0 else 0.02
            
            # 如果有成交量数据
            volume_score = 0.5  # 默认值
            if volume_data is not None and 'volume' in volume_data.columns:
                volume_avg = volume_data['volume'].rolling(21).mean().iloc[-1]
                volume_current = volume_data['volume'].iloc[-1]
                volume_score = min(2.0, volume_current / volume_avg) if volume_avg > 0 else 0.5
            elif 'volume' in price_data.columns:
                volume_avg = price_data['volume'].rolling(21).mean().iloc[-1]
                volume_current = price_data['volume'].iloc[-1]
                volume_score = min(2.0, volume_current / volume_avg) if volume_avg > 0 else 0.5
            
            # 综合流动性评分
            liquidity_score = (2.0 - atr_ratio * 50) * volume_score  # ATR越小，成交量越大，流动性越好
            
            thresholds = self.market_condition_thresholds["liquidity_thresholds"]
            if liquidity_score <= thresholds[0]:
                return LiquidityLevel.VERY_LOW
            elif liquidity_score <= thresholds[1]:
                return LiquidityLevel.LOW
            elif liquidity_score <= thresholds[2]:
                return LiquidityLevel.NORMAL
            elif liquidity_score <= thresholds[3]:
                return LiquidityLevel.HIGH
            else:
                return LiquidityLevel.VERY_HIGH
                
        except Exception as e:
            self.logger.warning(f"流动性分析失败: {e}")
            return LiquidityLevel.NORMAL
    
    def _analyze_market_sentiment(self, returns: pd.Series) -> float:
        """分析市场情绪"""
        try:
            if len(returns) < 21:
                return 0.5  # 中性情绪
            
            # 1. 基于收益分布的情绪分析
            recent_returns = returns.tail(21)  # 最近21天
            positive_days = (recent_returns > 0).sum()
            sentiment_ratio = positive_days / len(recent_returns)
            
            # 2. 基于收益大小的情绪分析
            return_magnitude = recent_returns.abs().mean()
            return_bias = recent_returns.mean()
            
            # 情绪评分结合比例和偏向
            sentiment_score = sentiment_ratio * 0.6 + (0.5 + return_bias / (2 * return_magnitude)) * 0.4 \
                if return_magnitude > 0 else sentiment_ratio
            
            # 确保在0-1范围内
            return max(0.0, min(1.0, sentiment_score))
            
        except Exception as e:
            self.logger.warning(f"市场情绪分析失败: {e}")
            return 0.5
    
    def _calculate_market_stress(self, returns: pd.Series, price_data: pd.DataFrame) -> float:
        """计算市场压力水平"""
        try:
            if len(returns) < 21:
                return 0.0
            
            # 1. 基于波动率的压力指标
            volatility = returns.rolling(21).std().iloc[-1] * math.sqrt(252)
            historical_vol = returns.std() * math.sqrt(252)
            vol_stress = min(1.0, volatility / (historical_vol * 1.5)) if historical_vol > 0 else 0.0
            
            # 2. 基于价格跳跃的压力指标
            price_jumps = abs(returns) > 2 * returns.std()
            jump_stress = price_jumps.rolling(21).sum().iloc[-1] / 21 if len(price_jumps) > 21 else 0.0
            
            # 3. 基于连续下跌的压力指标
            negative_streak = 0
            recent_returns = returns.tail(10)
            for i in range(len(recent_returns) - 1, -1, -1):
                ret = recent_returns.iloc[i]
                if ret < -0.01:  # 连续1%以上下跌
                    negative_streak += 1
                else:
                    break
            streak_stress = min(1.0, negative_streak / 5.0)
            
            # 综合压力评分
            stress_score = (vol_stress * 0.5 + jump_stress * 0.3 + streak_stress * 0.2)
            return max(0.0, min(1.0, stress_score))
            
        except Exception as e:
            self.logger.warning(f"市场压力计算失败: {e}")
            return 0.0
    
    def _detect_correlation_breakdown(self, returns: pd.Series) -> bool:
        """检测相关性失效"""
        # 简化版本，实际应该与其他资产对比
        try:
            if len(returns) < 63:
                return False
            
            # 检查收益率的自相关性是否异常
            correlation = returns.autocorr(lag=1)
            return abs(correlation) > 0.3  # 如果自相关性过强，可能表示市场异常
            
        except Exception:
            return False
    
    def _detect_liquidity_crisis(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> bool:
        """检测流动性危机"""
        try:
            # 基于ATR和价格跳跃检测流动性危机
            atr = calculate_atr(price_data, 14)
            if len(atr) < 14:
                return False
            
            current_atr = atr.iloc[-1]
            avg_atr = atr.rolling(63).mean().iloc[-1] if len(atr) >= 63 else current_atr
            
            # ATR异常增大可能表示流动性问题
            atr_spike = current_atr > avg_atr * 2 if avg_atr > 0 else False
            
            # 如果有成交量数据，检查成交量是否异常萎缩
            volume_crisis = False
            if volume_data is not None and 'volume' in volume_data.columns:
                current_volume = volume_data['volume'].iloc[-1]
                avg_volume = volume_data['volume'].rolling(21).mean().iloc[-1]
                volume_crisis = current_volume < avg_volume * 0.3 if avg_volume > 0 else False
            
            return atr_spike or volume_crisis
            
        except Exception:
            return False
    
    def _determine_primary_market_condition(
        self,
        trend_strength: float,
        trend_direction: TradingDirection,
        market_stress: float,
        liquidity_level: LiquidityLevel
    ) -> MarketCondition:
        """确定主要市场条件"""
        
        # 高压力市场
        if market_stress > 0.7:
            return MarketCondition.VOLATILE
        
        # 流动性极差的市场
        if liquidity_level == LiquidityLevel.VERY_LOW:
            return MarketCondition.UNCERTAIN
        
        # 强趋势市场
        if trend_strength > self.market_condition_thresholds["trend_strength_min"]:
            if trend_direction == TradingDirection.LONG:
                return MarketCondition.BULLISH
            elif trend_direction == TradingDirection.SHORT:
                return MarketCondition.BEARISH
            else:
                return MarketCondition.TRENDING
        
        # 弱趋势或中性市场
        if trend_strength < 0.3:
            return MarketCondition.SIDEWAYS
        
        # 默认为不确定
        return MarketCondition.UNCERTAIN
    
    def _identify_secondary_conditions(
        self,
        market_sentiment: float,
        liquidity_level: LiquidityLevel,
        market_stress: float
    ) -> List[MarketCondition]:
        """识别次要市场条件"""
        secondary = []
        
        # 基于市场情绪
        extremes = self.market_condition_thresholds["sentiment_extremes"]
        if market_sentiment <= extremes[0]:
            secondary.append(MarketCondition.BEARISH)
        elif market_sentiment >= extremes[1]:
            secondary.append(MarketCondition.BULLISH)
        
        # 基于流动性
        if liquidity_level in [LiquidityLevel.VERY_LOW, LiquidityLevel.LOW]:
            secondary.append(MarketCondition.UNCERTAIN)
        
        # 基于市场压力
        if market_stress > 0.5:
            secondary.append(MarketCondition.VOLATILE)
        
        return secondary
    
    def _calculate_analysis_confidence(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame]) -> float:
        """计算分析置信度"""
        confidence_factors = []
        
        # 数据长度因子
        data_length_factor = min(1.0, len(price_data) / 252)  # 一年数据为满分
        confidence_factors.append(data_length_factor)
        
        # 数据完整性因子
        completeness_factor = 1.0 - price_data['close'].isna().sum() / len(price_data)
        confidence_factors.append(completeness_factor)
        
        # 成交量数据可用性
        if volume_data is not None or 'volume' in price_data.columns:
            confidence_factors.append(1.0)
        else:
            confidence_factors.append(0.8)  # 没有成交量数据会降低置信度
        
        # 价格数据质量（检查异常值）
        returns = price_data['close'].pct_change().dropna()
        if len(returns) > 0:
            # 检查极端收益率
            extreme_returns = (abs(returns) > returns.std() * 5).sum()
            quality_factor = max(0.5, 1.0 - extreme_returns / len(returns))
            confidence_factors.append(quality_factor)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    async def calculate_risk_metrics(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> RiskMetrics:
        """
        计算全面的风险指标
        
        Args:
            ticker: 交易对符号
            price_data: 价格数据
            confidence_level: 置信水平
            
        Returns:
            RiskMetrics: 风险指标对象
        """
        try:
            returns = np.log(price_data['close'] / price_data['close'].shift(1)).dropna()
            
            if len(returns) < 21:
                raise ValueError("数据不足，无法计算风险指标")
            
            # 1. VaR计算
            var_1d_95 = float(np.percentile(returns, (1 - 0.95) * 100))
            var_1d_99 = float(np.percentile(returns, (1 - 0.99) * 100))
            var_5d_95 = float(var_1d_95 * math.sqrt(5))  # 简化的5日VaR
            
            # 条件VaR (CVaR/ES)
            cvar_1d_95 = float(returns[returns <= var_1d_95].mean()) if any(returns <= var_1d_95) else var_1d_95
            
            # 2. 回撤计算
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            
            max_drawdown = float(abs(drawdowns.min()))
            current_drawdown = float(abs(drawdowns.iloc[-1]))
            
            # 回撤持续时间
            drawdown_duration = self._calculate_drawdown_duration(drawdowns)
            
            # 3. 波动性风险
            volatility = returns.std() * math.sqrt(252)
            volatility_risk = min(100.0, volatility * 100)  # 转换为百分比
            
            # 4. 尾部风险
            tail_risk = self._calculate_tail_risk(returns)
            
            # 5. 分布特征
            skewness = float(returns.skew())
            kurtosis = float(returns.kurtosis())
            
            # 6. 流动性风险（简化版）
            liquidity_risk = self._estimate_liquidity_risk(price_data)
            
            # 7. 综合风险评分
            overall_risk_score = self._calculate_overall_risk_score(
                max_drawdown, volatility, tail_risk, liquidity_risk
            )
            
            # 8. 风险等级
            risk_level = self._determine_risk_level(overall_risk_score)
            
            return RiskMetrics(
                var_1d_95=var_1d_95,
                var_1d_99=var_1d_99,
                var_5d_95=var_5d_95,
                cvar_1d_95=cvar_1d_95,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                drawdown_duration=drawdown_duration,
                volatility_risk=volatility_risk,
                tail_risk=tail_risk,
                skewness=skewness,
                kurtosis=kurtosis,
                liquidity_risk=liquidity_risk,
                overall_risk_score=overall_risk_score,
                risk_level=risk_level,
                confidence_level=confidence_level
            )
            
        except Exception as e:
            self.logger.error(f"风险指标计算失败 {ticker}: {e}")
            raise
    
    def _calculate_drawdown_duration(self, drawdowns: pd.Series) -> int:
        """计算当前回撤持续天数"""
        try:
            if drawdowns.iloc[-1] >= 0:
                return 0  # 当前不在回撤中
            
            # 从最后一个数据点向前找，找到最后一次不在回撤中的点
            for i in range(len(drawdowns) - 1, -1, -1):
                if drawdowns.iloc[i] >= -0.001:  # 允许小的数值误差
                    return len(drawdowns) - 1 - i
            
            return len(drawdowns)  # 整个期间都在回撤中
        except Exception:
            return 0
    
    def _calculate_tail_risk(self, returns: pd.Series) -> float:
        """计算尾部风险"""
        try:
            # 使用超额峰度作为尾部风险的代理
            kurtosis = returns.kurtosis()
            # 标准化尾部风险评分 (0-100)
            tail_risk = max(0, min(100, kurtosis * 10))
            return float(tail_risk)
        except Exception:
            return 0.0
    
    def _estimate_liquidity_risk(self, price_data: pd.DataFrame) -> float:
        """估计流动性风险"""
        try:
            # 使用价格跳跃和ATR作为流动性风险的代理
            atr = calculate_atr(price_data, 14)
            if len(atr) == 0:
                return 50.0
            
            current_atr = atr.iloc[-1]
            price = price_data['close'].iloc[-1]
            atr_ratio = current_atr / price if price > 0 else 0.02
            
            # 将ATR比率转换为0-100的风险评分
            liquidity_risk = min(100.0, atr_ratio * 1000)
            return float(liquidity_risk)
        except Exception:
            return 50.0  # 默认中等流动性风险
    
    def _calculate_overall_risk_score(
        self, 
        max_drawdown: float, 
        volatility: float, 
        tail_risk: float, 
        liquidity_risk: float
    ) -> float:
        """计算综合风险评分"""
        try:
            # 各项风险的权重
            weights = {
                'drawdown': 0.3,
                'volatility': 0.3,
                'tail': 0.2,
                'liquidity': 0.2
            }
            
            # 标准化风险指标到0-100范围
            drawdown_score = min(100.0, max_drawdown * 200)  # 50%回撤对应100分
            volatility_score = min(100.0, volatility * 100)  # 100%年化波动率对应100分
            
            # 加权平均
            overall_score = (
                drawdown_score * weights['drawdown'] +
                volatility_score * weights['volatility'] +
                tail_risk * weights['tail'] +
                liquidity_risk * weights['liquidity']
            )
            
            return float(overall_score)
        except Exception:
            return 50.0  # 默认中等风险
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """根据风险评分确定风险等级"""
        if risk_score <= 20:
            return RiskLevel.LOW
        elif risk_score <= 40:
            return RiskLevel.MEDIUM
        elif risk_score <= 60:
            return RiskLevel.HIGH
        elif risk_score <= 80:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.EMERGENCY
    
    def _validate_price_data(self, price_data: pd.DataFrame) -> bool:
        """验证价格数据的有效性"""
        try:
            required_columns = ['close']
            optional_columns = ['high', 'low', 'open', 'volume']
            
            # 检查必需列
            for col in required_columns:
                if col not in price_data.columns:
                    self.logger.error(f"缺少必需的列: {col}")
                    return False
            
            # 检查数据长度
            if len(price_data) < 10:
                self.logger.error("数据长度不足")
                return False
            
            # 检查空值
            if price_data['close'].isna().sum() > len(price_data) * 0.1:
                self.logger.error("空值过多")
                return False
            
            # 检查价格为正
            if (price_data['close'] <= 0).any():
                self.logger.error("存在非正价格")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"数据验证失败: {e}")
            return False
    
    def _assess_data_quality(self, price_data: pd.DataFrame) -> float:
        """评估数据质量"""
        try:
            quality_score = 1.0
            
            # 检查缺失值
            missing_ratio = price_data['close'].isna().sum() / len(price_data)
            quality_score *= (1 - missing_ratio)
            
            # 检查数据长度
            if len(price_data) < 252:  # 少于一年数据
                quality_score *= len(price_data) / 252
            
            # 检查价格连续性（避免过大跳跃）
            returns = price_data['close'].pct_change(fill_method=None).dropna()
            if len(returns) > 0:
                extreme_returns = (abs(returns) > 0.2).sum()  # 20%以上变动
                if extreme_returns > 0:
                    quality_score *= max(0.5, 1 - extreme_returns / len(returns))
            
            return max(0.0, min(1.0, quality_score))
        except Exception:
            return 0.8  # 默认质量评分
    
    def _calculate_overall_confidence(self, analysis_results: List[Any]) -> float:
        """计算总体分析置信度"""
        try:
            # 检查各个分析模块是否成功
            success_count = sum(1 for result in analysis_results if not isinstance(result, Exception))
            success_rate = success_count / len(analysis_results)
            
            # 基础置信度基于成功率
            base_confidence = success_rate * 0.8
            
            # 如果所有模块都成功，给予额外奖励
            if success_rate == 1.0:
                base_confidence += 0.2
            
            return max(0.0, min(1.0, base_confidence))
        except Exception:
            return 0.6  # 默认置信度
    
    def _generate_trading_recommendation(
        self,
        volatility_metrics: Optional[VolatilityMetrics],
        technical_indicators: Optional[TechnicalIndicators],
        market_conditions: Optional[MarketConditionAnalysis],
        risk_metrics: Optional[RiskMetrics]
    ) -> Dict[str, Any]:
        """生成交易建议"""
        try:
            recommendation = {
                "action": "hold",
                "confidence": 0.5,
                "reasons": [],
                "risk_adjustment": 1.0,
                "max_position_ratio": 0.1,
                "stop_loss_suggestion": None,
                "take_profit_suggestion": None
            }
            
            # 基于技术指标的建议
            if technical_indicators:
                tech_signal, tech_confidence = technical_indicators.get_trend_signal()
                if tech_signal == TradingDirection.LONG:
                    recommendation["action"] = "buy"
                    recommendation["reasons"].append("技术指标显示看涨信号")
                elif tech_signal == TradingDirection.SHORT:
                    recommendation["action"] = "sell"
                    recommendation["reasons"].append("技术指标显示看跌信号")
                recommendation["confidence"] = max(recommendation["confidence"], tech_confidence)
            
            # 基于市场条件的调整
            if market_conditions:
                if not market_conditions.is_favorable_for_trading():
                    recommendation["action"] = "hold"
                    recommendation["reasons"].append("市场条件不利于交易")
                    recommendation["confidence"] *= 0.5
                
                if market_conditions.market_stress_level > 0.7:
                    recommendation["max_position_ratio"] *= 0.5
                    recommendation["reasons"].append("高市场压力，减少仓位")
            
            # 基于风险指标的调整
            if risk_metrics:
                risk_adjustment = risk_metrics.get_risk_adjustment_factor()
                recommendation["risk_adjustment"] = risk_adjustment
                recommendation["max_position_ratio"] *= risk_adjustment
                
                if risk_metrics.should_reduce_exposure():
                    if recommendation["action"] != "hold":
                        recommendation["action"] = "reduce"
                    recommendation["reasons"].append("风险指标建议减少敞口")
            
            # 基于波动率的调整
            if volatility_metrics:
                if volatility_metrics.volatility_regime == VolatilityRegime.EXTREME:
                    recommendation["max_position_ratio"] *= 0.3
                    recommendation["reasons"].append("极端波动率，大幅减少仓位")
                elif volatility_metrics.volatility_regime == VolatilityRegime.HIGH:
                    recommendation["max_position_ratio"] *= 0.7
                    recommendation["reasons"].append("高波动率，适度减少仓位")
            
            # 确保最大仓位比例在合理范围内
            recommendation["max_position_ratio"] = max(0.01, min(0.5, recommendation["max_position_ratio"]))
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"生成交易建议失败: {e}")
            return {
                "action": "hold",
                "confidence": 0.3,
                "reasons": ["分析过程中出现错误，建议观望"],
                "risk_adjustment": 0.5,
                "max_position_ratio": 0.05
            }
    
    async def _analyze_multiple_timeframes(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        timeframes: List[str]
    ) -> Dict[str, Any]:
        """多时间框架分析"""
        try:
            multi_tf_analysis = {}
            
            for tf in timeframes:
                try:
                    # 这里简化处理，实际应该根据时间框架重采样数据
                    tf_analysis = {
                        "timeframe": tf,
                        "trend_direction": "neutral",
                        "trend_strength": 0.5,
                        "key_levels": {
                            "support": None,
                            "resistance": None
                        }
                    }
                    multi_tf_analysis[tf] = tf_analysis
                except Exception as e:
                    self.logger.warning(f"时间框架 {tf} 分析失败: {e}")
            
            return multi_tf_analysis
        except Exception as e:
            self.logger.error(f"多时间框架分析失败: {e}")
            return {}
    
    def _cache_analysis_result(self, ticker: str, analysis: Dict[str, Any]) -> None:
        """缓存分析结果"""
        try:
            cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            self._analysis_cache[cache_key] = {
                "data": analysis,
                "timestamp": datetime.now()
            }
            
            # 清理过期缓存
            self._cleanup_cache()
        except Exception as e:
            self.logger.warning(f"缓存分析结果失败: {e}")
    
    def _cleanup_cache(self) -> None:
        """清理过期缓存"""
        try:
            current_time = datetime.now()
            expired_keys = [
                key for key, value in self._analysis_cache.items()
                if (current_time - value["timestamp"]).total_seconds() > self._cache_timeout
            ]
            
            for key in expired_keys:
                del self._analysis_cache[key]
                
        except Exception as e:
            self.logger.warning(f"清理缓存失败: {e}")
    
    async def get_leverage_support_data(
        self,
        ticker: str,
        price_data: Optional[pd.DataFrame] = None,
        volume_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        为杠杆控制器提供增强数据支持

        Args:
            ticker: 交易对符号
            price_data: 价格数据（可选）
            volume_data: 成交量数据（可选）

        Returns:
            Dict[str, Any]: 杠杆控制器所需的增强数据
        """
        try:
            # 如果没有提供数据，尝试获取
            if price_data is None:
                # 如果有增强版分析器，尝试获取实时数据
                if hasattr(self, 'get_real_time_klines'):
                    price_data = await self.get_real_time_klines(ticker, "1h", 200)

                if price_data is None or len(price_data) < 50:
                    self.logger.warning(f"无法获取足够的价格数据用于杠杆分析: {ticker}")
                    return self._generate_default_leverage_data()

            # 执行各项分析
            tasks = [
                self.calculate_volatility_metrics(ticker, price_data),
                self.analyze_volume_comprehensive(ticker, price_data, volume_data),
                self.analyze_correlation_comprehensive(ticker, price_data),
                self.analyze_market_sentiment_comprehensive(ticker, price_data, volume_data),
                self.calculate_risk_metrics(ticker, price_data)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            volatility_metrics, volume_analysis, correlation_analysis, sentiment_analysis, risk_metrics = results

            # 检查结果有效性
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"杠杆支持数据分析任务 {i} 失败: {result}")

            # 构造杠杆控制器专用数据
            leverage_data = {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,

                # 多周期波动率计算
                "enhanced_volatility": {
                    "current": float(volatility_metrics.historical_volatility) if volatility_metrics else 0.2,
                    "regime": volatility_metrics.volatility_regime.value if volatility_metrics else "normal",
                    "percentile": float(volatility_metrics.volatility_percentile) if volatility_metrics else 0.5,
                    "z_score": float(volatility_metrics.volatility_z_score) if volatility_metrics else 0.0,
                    "multiple_estimators": {
                        "parkinson": float(volatility_metrics.parkinson_volatility) if volatility_metrics else 0.2,
                        "garman_klass": float(volatility_metrics.garman_klass_volatility) if volatility_metrics else 0.2,
                        "yang_zhang": float(volatility_metrics.yang_zhang_volatility) if volatility_metrics else 0.2
                    }
                },

                # 成交量分布特征
                "volume_characteristics": {
                    "strength": float(volume_analysis.volume_strength) if volume_analysis else 0.5,
                    "trend": volume_analysis.volume_trend if volume_analysis else "stable",
                    "abnormal_detected": volume_analysis.abnormal_volume_detected if volume_analysis else False,
                    "breakout_signal": volume_analysis.volume_breakout_signal if volume_analysis else False,
                    "distribution": volume_analysis.volume_distribution if volume_analysis else {},
                    "volume_profile": volume_analysis.volume_profile if volume_analysis else {}
                },

                # 市场关联度计算
                "market_correlation": {
                    "strength": correlation_analysis.correlation_strength if correlation_analysis else "weak",
                    "stability": float(correlation_analysis.correlation_stability) if correlation_analysis else 0.5,
                    "market_coupling": float(correlation_analysis.market_coupling) if correlation_analysis else 0.5,
                    "breakdown_risk": float(correlation_analysis.correlation_breakdown_risk) if correlation_analysis else 0.5,
                    "diversification_benefit": float(correlation_analysis.diversification_benefit) if correlation_analysis else 0.5,
                    "asset_correlations": correlation_analysis.asset_correlations if correlation_analysis else {}
                },

                # 恐贪指数和市场状态
                "market_sentiment": {
                    "fear_greed_index": float(sentiment_analysis.fear_greed_index) if sentiment_analysis else 50.0,
                    "sentiment_label": sentiment_analysis.sentiment_label if sentiment_analysis else "neutral",
                    "volatility_sentiment": float(sentiment_analysis.volatility_sentiment) if sentiment_analysis else 0.5,
                    "momentum_sentiment": float(sentiment_analysis.momentum_sentiment) if sentiment_analysis else 0.5,
                    "stress_indicators": sentiment_analysis.market_stress_indicators if sentiment_analysis else {},
                    "contrarian_signals": sentiment_analysis.contrarian_signals if sentiment_analysis else []
                },

                # 风险调整因子
                "risk_adjustments": {
                    "overall_risk_score": float(risk_metrics.overall_risk_score) if risk_metrics else 50.0,
                    "risk_level": risk_metrics.risk_level.value if risk_metrics else "medium",
                    "adjustment_factor": risk_metrics.get_risk_adjustment_factor() if risk_metrics else 1.0,
                    "max_drawdown": float(risk_metrics.max_drawdown) if risk_metrics else 0.1,
                    "volatility_risk": float(risk_metrics.volatility_risk) if risk_metrics else 20.0,
                    "tail_risk": float(risk_metrics.tail_risk) if risk_metrics else 10.0
                },

                # 性能优化标记
                "calculation_metrics": {
                    "data_points": len(price_data),
                    "analysis_modules_completed": sum(1 for r in results if not isinstance(r, Exception)),
                    "computation_time_estimate": "< 5ms"  # 目标性能
                }
            }

            return leverage_data

        except Exception as e:
            self.logger.error(f"获取杠杆支持数据失败 {ticker}: {e}")
            return self._generate_default_leverage_data()

    def _generate_default_leverage_data(self) -> Dict[str, Any]:
        """生成默认的杠杆支持数据"""
        return {
            "timestamp": datetime.now().isoformat(),
            "ticker": "UNKNOWN",
            "enhanced_volatility": {
                "current": 0.2,
                "regime": "normal",
                "percentile": 0.5,
                "z_score": 0.0,
                "multiple_estimators": {
                    "parkinson": 0.2,
                    "garman_klass": 0.2,
                    "yang_zhang": 0.2
                }
            },
            "volume_characteristics": {
                "strength": 0.5,
                "trend": "stable",
                "abnormal_detected": False,
                "breakout_signal": False,
                "distribution": {},
                "volume_profile": {}
            },
            "market_correlation": {
                "strength": "weak",
                "stability": 0.5,
                "market_coupling": 0.5,
                "breakdown_risk": 0.5,
                "diversification_benefit": 0.5,
                "asset_correlations": {}
            },
            "market_sentiment": {
                "fear_greed_index": 50.0,
                "sentiment_label": "neutral",
                "volatility_sentiment": 0.5,
                "momentum_sentiment": 0.5,
                "stress_indicators": {},
                "contrarian_signals": []
            },
            "risk_adjustments": {
                "overall_risk_score": 50.0,
                "risk_level": "medium",
                "adjustment_factor": 1.0,
                "max_drawdown": 0.1,
                "volatility_risk": 20.0,
                "tail_risk": 10.0
            },
            "calculation_metrics": {
                "data_points": 0,
                "analysis_modules_completed": 0,
                "computation_time_estimate": "< 5ms"
            }
        }

    def get_analysis_summary(self, ticker: str) -> Optional[Dict[str, Any]]:
        """获取分析摘要"""
        try:
            # 从缓存中查找最近的分析结果
            ticker_keys = [key for key in self._analysis_cache.keys() if key.startswith(ticker)]
            if not ticker_keys:
                return None

            latest_key = max(ticker_keys, key=lambda k: self._analysis_cache[k]["timestamp"])
            return self._analysis_cache[latest_key]["data"]
        except Exception as e:
            self.logger.error(f"获取分析摘要失败: {e}")
            return None


# 导出主要类和函数
__all__ = [
    'MarketAnalyzer',
    'VolatilityMetrics',
    'TechnicalIndicators',
    'MarketConditionAnalysis',
    'RiskMetrics',
    'VolumeAnalysis',
    'CorrelationAnalysis',
    'MarketSentimentAnalysis',
    'MarketCondition',
    'VolatilityRegime',
    'LiquidityLevel'
]