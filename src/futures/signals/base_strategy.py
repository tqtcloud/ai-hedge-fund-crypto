"""
期货策略基类接口

定义期货策略的统一接口，支持long/short/neutral输出语义，
并提供期货特有的参数处理（leverage, volatility等）。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import logging

try:
    from ..models.data_models import FuturesSignal, TradingDirection, OperationType, RiskLevel
    from ..constants import RiskParameters, TradingTimeConstants
except ImportError:
    # 处理相对导入问题
    from src.futures.models.data_models import FuturesSignal, TradingDirection, OperationType, RiskLevel
    from src.futures.constants import RiskParameters, TradingTimeConstants

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """策略类型枚举"""
    TREND_FOLLOWING = "trend_following"      # 趋势跟踪
    MEAN_REVERSION = "mean_reversion"        # 均值回归
    MOMENTUM = "momentum"                    # 动量策略
    SCALPING = "scalping"                    # 剥头皮策略
    SWING = "swing"                          # 摆动交易
    GRID = "grid"                           # 网格策略
    ARBITRAGE = "arbitrage"                  # 套利策略
    MULTI_TIMEFRAME = "multi_timeframe"      # 多时间框架
    HYBRID = "hybrid"                        # 混合策略


class SignalStrength(Enum):
    """信号强度枚举"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class StrategyConfig:
    """策略配置类"""
    name: str                                # 策略名称
    strategy_type: StrategyType              # 策略类型
    enabled: bool = True                     # 是否启用

    # 基础参数
    timeframes: List[str] = field(default_factory=lambda: ["1h"])  # 支持的时间框架
    symbols: List[str] = field(default_factory=list)               # 支持的交易对

    # 风险参数
    max_leverage: float = 10.0               # 最大杠杆
    default_leverage: float = 5.0            # 默认杠杆
    max_position_size: float = 1000.0        # 最大仓位大小(USDT)
    stop_loss_ratio: float = 0.02            # 默认止损比例
    take_profit_ratio: float = 0.06          # 默认止盈比例

    # 信号参数
    min_confidence: float = 50.0             # 最小置信度阈值
    signal_expiry_seconds: int = 1800        # 信号有效期(秒)

    # 特定参数
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """验证配置有效性"""
        try:
            if self.max_leverage <= 0 or self.max_leverage > 125:
                return False
            if self.default_leverage <= 0 or self.default_leverage > self.max_leverage:
                return False
            if not 0 <= self.min_confidence <= 100:
                return False
            if self.stop_loss_ratio <= 0 or self.take_profit_ratio <= 0:
                return False
            return True
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False


@dataclass
class StrategyOutput:
    """策略输出结果"""
    signal: FuturesSignal                    # 交易信号
    metadata: Dict[str, Any]                 # 元数据
    diagnostics: Dict[str, Any] = field(default_factory=dict)  # 诊断信息
    performance_metrics: Dict[str, Any] = field(default_factory=dict)  # 性能指标

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "signal": self.signal.to_dict() if self.signal else None,
            "metadata": self.metadata,
            "diagnostics": self.diagnostics,
            "performance_metrics": self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }


class FuturesBaseStrategy(ABC):
    """
    期货策略基类接口

    定义所有期货策略必须实现的统一接口，确保输出格式一致，
    支持期货特有的long/short/neutral语义。
    """

    def __init__(self, config: StrategyConfig):
        """
        初始化策略

        Args:
            config: 策略配置
        """
        self.config = config
        self.last_signal_time: Optional[datetime] = None
        self.signal_count = 0
        self.performance_stats = {
            "total_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "avg_confidence": 0.0
        }

        # 验证配置
        if not config.validate():
            raise ValueError(f"策略配置无效: {config.name}")

        logger.info(f"初始化期货策略: {config.name} ({config.strategy_type.value})")

    @property
    def name(self) -> str:
        """策略名称"""
        return self.config.name

    @property
    def strategy_type(self) -> StrategyType:
        """策略类型"""
        return self.config.strategy_type

    @property
    def is_enabled(self) -> bool:
        """是否启用"""
        return self.config.enabled

    @abstractmethod
    def analyze(self,
                ticker: str,
                data: Dict[str, pd.DataFrame],
                current_price: float,
                **kwargs) -> Optional[StrategyOutput]:
        """
        执行策略分析

        Args:
            ticker: 交易对符号
            data: 多时间框架数据，格式: {timeframe: DataFrame}
            current_price: 当前价格
            **kwargs: 其他参数

        Returns:
            策略输出结果，包含FuturesSignal或None
        """
        pass

    @abstractmethod
    def get_signal_strength(self, data: Dict[str, Any]) -> SignalStrength:
        """
        计算信号强度

        Args:
            data: 分析数据

        Returns:
            信号强度枚举
        """
        pass

    @abstractmethod
    def calculate_confidence(self, data: Dict[str, Any]) -> float:
        """
        计算信号置信度

        Args:
            data: 分析数据

        Returns:
            置信度(0-100)
        """
        pass

    @abstractmethod
    def get_supported_timeframes(self) -> List[str]:
        """
        获取策略支持的时间框架

        Returns:
            时间框架列表
        """
        pass

    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """
        获取策略需要的技术指标

        Returns:
            指标名称列表
        """
        pass

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        验证输入数据完整性

        Args:
            data: 多时间框架数据

        Returns:
            数据是否有效
        """
        try:
            required_timeframes = self.get_supported_timeframes()

            # 检查必需的时间框架
            for timeframe in required_timeframes:
                if timeframe not in data:
                    logger.warning(f"缺少必需时间框架数据: {timeframe}")
                    return False

                df = data[timeframe]
                if df.empty:
                    logger.warning(f"时间框架{timeframe}数据为空")
                    return False

                # 检查必需的列
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col.lower() not in df.columns.str.lower():
                        logger.warning(f"缺少必需列: {col}")
                        return False

            return True

        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return False

    def calculate_position_size(self,
                              ticker: str,
                              confidence: float,
                              current_price: float,
                              available_balance: float,
                              leverage: float = None) -> float:
        """
        计算仓位大小

        Args:
            ticker: 交易对
            confidence: 信号置信度
            current_price: 当前价格
            available_balance: 可用余额
            leverage: 杠杆倍数

        Returns:
            建议仓位大小(USDT)
        """
        try:
            # 使用配置的杠杆或传入的杠杆
            leverage = leverage or self.config.default_leverage

            # 基于置信度调整仓位大小
            confidence_factor = min(confidence / 100.0, 1.0)

            # 计算基础仓位大小
            max_position = min(
                self.config.max_position_size,
                available_balance * RiskParameters.MAX_POSITION_RATIO
            )

            position_size = max_position * confidence_factor

            # 确保仓位大小合理
            position_size = max(5.0, min(position_size, max_position))

            return position_size

        except Exception as e:
            logger.error(f"计算仓位大小失败: {e}")
            return 100.0  # 默认仓位大小

    def calculate_stop_loss_take_profit(self,
                                      entry_price: float,
                                      direction: TradingDirection,
                                      volatility: float = None) -> Dict[str, float]:
        """
        计算止损止盈价格

        Args:
            entry_price: 入场价格
            direction: 交易方向
            volatility: 市场波动率

        Returns:
            包含止损止盈价格的字典
        """
        try:
            # 根据波动率调整止损止盈比例
            if volatility is not None:
                # 高波动率环境下，适当放宽止损，收紧止盈
                vol_adjustment = min(volatility / 0.02, 2.0)  # 基准2%波动率
                stop_loss_ratio = self.config.stop_loss_ratio * (1 + vol_adjustment * 0.5)
                take_profit_ratio = self.config.take_profit_ratio * (1 + vol_adjustment * 0.3)
            else:
                stop_loss_ratio = self.config.stop_loss_ratio
                take_profit_ratio = self.config.take_profit_ratio

            if direction == TradingDirection.LONG:
                stop_loss_price = entry_price * (1 - stop_loss_ratio)
                take_profit_price = entry_price * (1 + take_profit_ratio)
            elif direction == TradingDirection.SHORT:
                stop_loss_price = entry_price * (1 + stop_loss_ratio)
                take_profit_price = entry_price * (1 - take_profit_ratio)
            else:  # NEUTRAL
                stop_loss_price = None
                take_profit_price = None

            return {
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "stop_loss_ratio": stop_loss_ratio,
                "take_profit_ratio": take_profit_ratio
            }

        except Exception as e:
            logger.error(f"计算止损止盈失败: {e}")
            return {
                "stop_loss_price": None,
                "take_profit_price": None,
                "stop_loss_ratio": self.config.stop_loss_ratio,
                "take_profit_ratio": self.config.take_profit_ratio
            }

    def create_futures_signal(self,
                            ticker: str,
                            direction: TradingDirection,
                            operation_type: OperationType,
                            confidence: float,
                            strength: SignalStrength,
                            current_price: float,
                            leverage: float = None,
                            metadata: Dict[str, Any] = None) -> FuturesSignal:
        """
        创建标准化的期货交易信号

        Args:
            ticker: 交易对
            direction: 交易方向
            operation_type: 操作类型
            confidence: 置信度
            strength: 信号强度
            current_price: 当前价格
            leverage: 杠杆倍数
            metadata: 元数据

        Returns:
            标准化的期货信号
        """
        try:
            leverage = leverage or self.config.default_leverage

            # 计算仓位大小（这里使用默认可用余额）
            position_size = self.calculate_position_size(
                ticker, confidence, current_price, 10000.0, leverage
            )

            # 计算止损止盈
            sl_tp = self.calculate_stop_loss_take_profit(
                current_price, direction,
                metadata.get('volatility') if metadata else None
            )

            # 计算信号过期时间
            expiry_time = datetime.now() + timedelta(seconds=self.config.signal_expiry_seconds)

            # 确定风险等级
            risk_level = self._calculate_risk_level(leverage, confidence, strength)

            signal = FuturesSignal(
                ticker=ticker,
                direction=direction,
                operation_type=operation_type,
                confidence=confidence,
                strength=self._strength_to_numeric(strength),
                suggested_leverage=leverage,
                position_size=position_size,
                entry_price=current_price,
                current_price=current_price,
                take_profit_price=sl_tp.get("take_profit_price"),
                stop_loss_price=sl_tp.get("stop_loss_price"),
                take_profit_ratio=sl_tp.get("take_profit_ratio"),
                stop_loss_ratio=sl_tp.get("stop_loss_ratio"),
                expiry_time=expiry_time,
                risk_level=risk_level,
                strategy_source=self.name,
                signal_id=f"{self.name}_{ticker}_{int(datetime.now().timestamp())}",
                metadata=metadata or {}
            )

            # 更新统计
            self.last_signal_time = datetime.now()
            self.signal_count += 1
            self.performance_stats["total_signals"] += 1

            return signal

        except Exception as e:
            logger.error(f"创建期货信号失败: {e}")
            raise

    def _strength_to_numeric(self, strength: SignalStrength) -> float:
        """转换信号强度为数值"""
        strength_mapping = {
            SignalStrength.VERY_WEAK: 0.2,
            SignalStrength.WEAK: 0.4,
            SignalStrength.MODERATE: 0.6,
            SignalStrength.STRONG: 0.8,
            SignalStrength.VERY_STRONG: 1.0
        }
        return strength_mapping.get(strength, 0.5)

    def _calculate_risk_level(self,
                            leverage: float,
                            confidence: float,
                            strength: SignalStrength) -> RiskLevel:
        """计算风险等级"""
        try:
            # 基于杠杆、置信度和信号强度计算风险
            leverage_risk = min(leverage / 20.0, 1.0)  # 杠杆风险
            confidence_risk = 1.0 - (confidence / 100.0)  # 置信度风险
            strength_risk = 1.0 - self._strength_to_numeric(strength)  # 强度风险

            # 综合风险评分
            overall_risk = (leverage_risk * 0.4 + confidence_risk * 0.3 + strength_risk * 0.3)

            if overall_risk >= 0.8:
                return RiskLevel.CRITICAL
            elif overall_risk >= 0.6:
                return RiskLevel.HIGH
            elif overall_risk >= 0.4:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except Exception:
            return RiskLevel.MEDIUM

    def update_performance(self, signal_result: str, pnl: float = None):
        """
        更新策略性能统计

        Args:
            signal_result: 信号结果 ("success" 或 "failed")
            pnl: 盈亏金额（可选）
        """
        try:
            if signal_result == "success":
                self.performance_stats["successful_signals"] += 1
            elif signal_result == "failed":
                self.performance_stats["failed_signals"] += 1

            # 计算平均置信度（需要重构以包含历史数据）
            if self.performance_stats["total_signals"] > 0:
                self.performance_stats["success_rate"] = (
                    self.performance_stats["successful_signals"] /
                    self.performance_stats["total_signals"]
                )
        except Exception as e:
            logger.error(f"更新性能统计失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        获取策略状态信息

        Returns:
            策略状态字典
        """
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "enabled": self.is_enabled,
            "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None,
            "signal_count": self.signal_count,
            "performance_stats": self.performance_stats,
            "config": {
                "max_leverage": self.config.max_leverage,
                "default_leverage": self.config.default_leverage,
                "min_confidence": self.config.min_confidence,
                "supported_timeframes": self.get_supported_timeframes(),
                "required_indicators": self.get_required_indicators()
            }
        }

    def reset_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            "total_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "avg_confidence": 0.0
        }
        self.signal_count = 0
        logger.info(f"重置策略性能统计: {self.name}")

    def enable(self):
        """启用策略"""
        self.config.enabled = True
        logger.info(f"启用策略: {self.name}")

    def disable(self):
        """禁用策略"""
        self.config.enabled = False
        logger.info(f"禁用策略: {self.name}")

    def __str__(self) -> str:
        """字符串表示"""
        return f"FuturesStrategy(name='{self.name}', type='{self.strategy_type.value}', enabled={self.is_enabled})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"FuturesStrategy(name='{self.name}', type='{self.strategy_type.value}', "
                f"enabled={self.is_enabled}, signals={self.signal_count}, "
                f"success_rate={self.performance_stats.get('success_rate', 0):.2%})")