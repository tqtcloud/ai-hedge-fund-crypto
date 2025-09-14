"""
期货策略工厂

提供期货策略的统一创建和管理机制，支持策略类型注册、
动态创建策略实例和策略配置管理。
"""

from typing import Dict, Type, Optional, List, Any, Union
import importlib
import logging
from datetime import datetime

try:
    from .base_strategy import FuturesBaseStrategy, StrategyConfig, StrategyType
    from ..constants import DefaultConfig
except ImportError:
    # 处理相对导入问题
    from src.futures.signals.base_strategy import FuturesBaseStrategy, StrategyConfig, StrategyType
    from src.futures.constants import DefaultConfig

logger = logging.getLogger(__name__)


class StrategyRegistrationError(Exception):
    """策略注册错误"""
    pass


class StrategyCreationError(Exception):
    """策略创建错误"""
    pass


class FuturesStrategyFactory:
    """
    期货策略工厂类

    负责管理所有可用的期货策略类型，提供统一的策略创建接口，
    支持动态注册和策略配置管理。
    """

    def __init__(self):
        """初始化策略工厂"""
        self._strategies: Dict[str, Type[FuturesBaseStrategy]] = {}
        self._default_configs: Dict[str, StrategyConfig] = {}
        self._creation_count = 0

        logger.info("初始化期货策略工厂")

        # 自动注册内置策略
        self._register_builtin_strategies()

    def register_strategy(self,
                         name: str,
                         strategy_class: Type[FuturesBaseStrategy],
                         default_config: Optional[StrategyConfig] = None,
                         override: bool = False) -> None:
        """
        注册策略类型

        Args:
            name: 策略名称
            strategy_class: 策略类
            default_config: 默认配置
            override: 是否覆盖已存在的策略

        Raises:
            StrategyRegistrationError: 注册失败时抛出
        """
        try:
            # 检查策略是否已存在
            if name in self._strategies and not override:
                raise StrategyRegistrationError(f"策略 '{name}' 已存在，设置override=True以覆盖")

            # 验证策略类
            if not issubclass(strategy_class, FuturesBaseStrategy):
                raise StrategyRegistrationError(f"策略类必须继承自FuturesBaseStrategy: {strategy_class}")

            # 注册策略
            self._strategies[name] = strategy_class

            # 设置默认配置
            if default_config:
                self._default_configs[name] = default_config
            else:
                self._default_configs[name] = self._create_default_config(name, strategy_class)

            logger.info(f"成功注册期货策略: {name} -> {strategy_class.__name__}")

        except Exception as e:
            error_msg = f"注册策略失败 '{name}': {e}"
            logger.error(error_msg)
            raise StrategyRegistrationError(error_msg)

    def unregister_strategy(self, name: str) -> bool:
        """
        注销策略

        Args:
            name: 策略名称

        Returns:
            是否成功注销
        """
        try:
            if name in self._strategies:
                del self._strategies[name]
                if name in self._default_configs:
                    del self._default_configs[name]
                logger.info(f"成功注销策略: {name}")
                return True
            else:
                logger.warning(f"策略 '{name}' 不存在，无法注销")
                return False
        except Exception as e:
            logger.error(f"注销策略失败 '{name}': {e}")
            return False

    def create_strategy(self,
                       name: str,
                       config: Optional[Union[StrategyConfig, Dict[str, Any]]] = None,
                       **kwargs) -> FuturesBaseStrategy:
        """
        创建策略实例

        Args:
            name: 策略名称
            config: 策略配置（StrategyConfig对象或字典）
            **kwargs: 额外的配置参数

        Returns:
            策略实例

        Raises:
            StrategyCreationError: 创建失败时抛出
        """
        try:
            # 检查策略是否已注册
            if name not in self._strategies:
                raise StrategyCreationError(f"未注册的策略: {name}. 可用策略: {list(self._strategies.keys())}")

            strategy_class = self._strategies[name]

            # 准备配置
            if config is None:
                # 使用默认配置
                strategy_config = self._default_configs[name]
            elif isinstance(config, dict):
                # 从字典创建配置
                strategy_config = self._create_config_from_dict(name, config, **kwargs)
            elif isinstance(config, StrategyConfig):
                # 直接使用提供的配置
                strategy_config = config
            else:
                raise StrategyCreationError(f"无效的配置类型: {type(config)}")

            # 创建策略实例
            strategy_instance = strategy_class(strategy_config)

            # 更新创建计数
            self._creation_count += 1

            logger.info(f"成功创建策略实例: {name} (#{self._creation_count})")

            return strategy_instance

        except Exception as e:
            error_msg = f"创建策略实例失败 '{name}': {e}"
            logger.error(error_msg)
            raise StrategyCreationError(error_msg)

    def create_multiple_strategies(self,
                                 strategy_configs: Dict[str, Union[StrategyConfig, Dict[str, Any]]]) -> Dict[str, FuturesBaseStrategy]:
        """
        批量创建策略实例

        Args:
            strategy_configs: 策略配置字典 {策略名: 配置}

        Returns:
            策略实例字典 {策略名: 实例}
        """
        strategies = {}
        failed_strategies = []

        for strategy_name, config in strategy_configs.items():
            try:
                strategy = self.create_strategy(strategy_name, config)
                strategies[strategy_name] = strategy
            except Exception as e:
                logger.error(f"批量创建策略失败 '{strategy_name}': {e}")
                failed_strategies.append(strategy_name)

        if failed_strategies:
            logger.warning(f"以下策略创建失败: {failed_strategies}")

        logger.info(f"批量创建策略完成: 成功{len(strategies)}个，失败{len(failed_strategies)}个")

        return strategies

    def get_available_strategies(self) -> List[str]:
        """
        获取可用策略列表

        Returns:
            策略名称列表
        """
        return list(self._strategies.keys())

    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取策略信息

        Args:
            name: 策略名称

        Returns:
            策略信息字典或None
        """
        if name not in self._strategies:
            return None

        strategy_class = self._strategies[name]
        default_config = self._default_configs.get(name)

        return {
            "name": name,
            "class": strategy_class.__name__,
            "module": strategy_class.__module__,
            "strategy_type": default_config.strategy_type.value if default_config else None,
            "default_config": default_config.to_dict() if hasattr(default_config, 'to_dict') else None,
            "docstring": strategy_class.__doc__
        }

    def list_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有策略信息

        Returns:
            所有策略信息字典
        """
        strategies_info = {}
        for name in self._strategies:
            strategies_info[name] = self.get_strategy_info(name)
        return strategies_info

    def get_strategies_by_type(self, strategy_type: StrategyType) -> List[str]:
        """
        根据策略类型获取策略列表

        Args:
            strategy_type: 策略类型

        Returns:
            匹配的策略名称列表
        """
        matching_strategies = []
        for name, config in self._default_configs.items():
            if config.strategy_type == strategy_type:
                matching_strategies.append(name)
        return matching_strategies

    def validate_strategy_config(self, name: str, config: Union[StrategyConfig, Dict[str, Any]]) -> bool:
        """
        验证策略配置

        Args:
            name: 策略名称
            config: 配置对象或字典

        Returns:
            配置是否有效
        """
        try:
            if name not in self._strategies:
                logger.error(f"策略不存在: {name}")
                return False

            if isinstance(config, dict):
                temp_config = self._create_config_from_dict(name, config)
            elif isinstance(config, StrategyConfig):
                temp_config = config
            else:
                logger.error(f"无效的配置类型: {type(config)}")
                return False

            return temp_config.validate()

        except Exception as e:
            logger.error(f"验证策略配置失败 '{name}': {e}")
            return False

    def _register_builtin_strategies(self) -> None:
        """注册内置策略"""
        try:
            # 注册MACD策略
            self._register_macd_strategy()

            # 注册RSI策略
            self._register_rsi_strategy()

            logger.info("内置策略注册完成")

        except Exception as e:
            logger.error(f"注册内置策略失败: {e}")

    def _register_macd_strategy(self) -> None:
        """注册MACD策略（使用适配器）"""
        try:
            # 使用适配器替代原始策略
            try:
                from .strategy_adapters import FuturesMacdStrategyAdapter
            except ImportError:
                from src.futures.signals.strategy_adapters import FuturesMacdStrategyAdapter

            # 创建MACD策略配置
            macd_config = StrategyConfig(
                name="futures_macd_strategy",
                strategy_type=StrategyType.TREND_FOLLOWING,
                timeframes=["1h", "4h", "1d"],
                default_leverage=10.0,
                max_leverage=20.0,
                min_confidence=60.0,
                signal_expiry_seconds=1800,
                custom_params={
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                }
            )

            self.register_strategy("futures_macd_strategy", FuturesMacdStrategyAdapter, macd_config)

        except ImportError as e:
            logger.warning(f"MACD策略适配器导入失败，跳过注册: {e}")
        except Exception as e:
            logger.error(f"注册MACD策略失败: {e}")

    def _register_rsi_strategy(self) -> None:
        """注册RSI策略（使用适配器）"""
        try:
            # 使用适配器替代原始策略
            try:
                from .strategy_adapters import FuturesRSIStrategyAdapter
            except ImportError:
                from src.futures.signals.strategy_adapters import FuturesRSIStrategyAdapter

            # 创建RSI策略配置
            rsi_config = StrategyConfig(
                name="futures_rsi_strategy",
                strategy_type=StrategyType.MEAN_REVERSION,
                timeframes=["1h", "4h"],
                default_leverage=8.0,
                max_leverage=15.0,
                min_confidence=55.0,
                signal_expiry_seconds=1200,
                custom_params={
                    "rsi_period": 14,
                    "overbought_threshold": 70,
                    "oversold_threshold": 30
                }
            )

            self.register_strategy("futures_rsi_strategy", FuturesRSIStrategyAdapter, rsi_config)

        except ImportError as e:
            logger.warning(f"RSI策略适配器导入失败，跳过注册: {e}")
        except Exception as e:
            logger.error(f"注册RSI策略失败: {e}")

    def _create_default_config(self, name: str, strategy_class: Type[FuturesBaseStrategy]) -> StrategyConfig:
        """
        创建默认策略配置

        Args:
            name: 策略名称
            strategy_class: 策略类

        Returns:
            默认配置
        """
        # 尝试从策略类推断类型
        strategy_type = StrategyType.HYBRID  # 默认类型

        if "macd" in name.lower() or "trend" in name.lower():
            strategy_type = StrategyType.TREND_FOLLOWING
        elif "rsi" in name.lower() or "mean" in name.lower():
            strategy_type = StrategyType.MEAN_REVERSION
        elif "momentum" in name.lower():
            strategy_type = StrategyType.MOMENTUM

        return StrategyConfig(
            name=name,
            strategy_type=strategy_type,
            timeframes=["1h"],
            max_leverage=DefaultConfig.DEFAULT_TRADING_CONFIG["leverage"],
            default_leverage=DefaultConfig.DEFAULT_TRADING_CONFIG["leverage"],
            stop_loss_ratio=DefaultConfig.DEFAULT_TRADING_CONFIG["stop_loss_ratio"],
            take_profit_ratio=DefaultConfig.DEFAULT_TRADING_CONFIG["take_profit_ratio"]
        )

    def _create_config_from_dict(self, name: str, config_dict: Dict[str, Any], **kwargs) -> StrategyConfig:
        """
        从字典创建策略配置

        Args:
            name: 策略名称
            config_dict: 配置字典
            **kwargs: 额外参数

        Returns:
            策略配置对象
        """
        # 合并配置字典和额外参数
        merged_config = {**config_dict, **kwargs}

        # 获取默认配置作为基础
        if name in self._default_configs:
            base_config = self._default_configs[name]
            config_args = {
                "name": merged_config.get("name", base_config.name),
                "strategy_type": StrategyType(merged_config.get("strategy_type", base_config.strategy_type.value)),
                "enabled": merged_config.get("enabled", base_config.enabled),
                "timeframes": merged_config.get("timeframes", base_config.timeframes),
                "symbols": merged_config.get("symbols", base_config.symbols),
                "max_leverage": merged_config.get("max_leverage", base_config.max_leverage),
                "default_leverage": merged_config.get("default_leverage", base_config.default_leverage),
                "max_position_size": merged_config.get("max_position_size", base_config.max_position_size),
                "stop_loss_ratio": merged_config.get("stop_loss_ratio", base_config.stop_loss_ratio),
                "take_profit_ratio": merged_config.get("take_profit_ratio", base_config.take_profit_ratio),
                "min_confidence": merged_config.get("min_confidence", base_config.min_confidence),
                "signal_expiry_seconds": merged_config.get("signal_expiry_seconds", base_config.signal_expiry_seconds),
                "custom_params": merged_config.get("custom_params", base_config.custom_params)
            }
        else:
            # 创建新配置
            config_args = {
                "name": merged_config.get("name", name),
                "strategy_type": StrategyType(merged_config.get("strategy_type", "hybrid")),
                "enabled": merged_config.get("enabled", True),
                "timeframes": merged_config.get("timeframes", ["1h"]),
                "symbols": merged_config.get("symbols", []),
                "max_leverage": merged_config.get("max_leverage", 10.0),
                "default_leverage": merged_config.get("default_leverage", 5.0),
                "max_position_size": merged_config.get("max_position_size", 1000.0),
                "stop_loss_ratio": merged_config.get("stop_loss_ratio", 0.02),
                "take_profit_ratio": merged_config.get("take_profit_ratio", 0.06),
                "min_confidence": merged_config.get("min_confidence", 50.0),
                "signal_expiry_seconds": merged_config.get("signal_expiry_seconds", 1800),
                "custom_params": merged_config.get("custom_params", {})
            }

        return StrategyConfig(**config_args)

    def get_factory_stats(self) -> Dict[str, Any]:
        """
        获取工厂统计信息

        Returns:
            统计信息字典
        """
        return {
            "registered_strategies": len(self._strategies),
            "available_strategies": self.get_available_strategies(),
            "creation_count": self._creation_count,
            "strategy_types": {
                strategy_type.value: len(self.get_strategies_by_type(strategy_type))
                for strategy_type in StrategyType
            },
            "created_at": datetime.now().isoformat()
        }

    def reset_factory(self) -> None:
        """重置工厂状态"""
        self._strategies.clear()
        self._default_configs.clear()
        self._creation_count = 0

        # 重新注册内置策略
        self._register_builtin_strategies()

        logger.info("期货策略工厂已重置")

    def __str__(self) -> str:
        """字符串表示"""
        return f"FuturesStrategyFactory(strategies={len(self._strategies)}, created={self._creation_count})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"FuturesStrategyFactory(strategies={list(self._strategies.keys())}, created={self._creation_count})"


# 创建全局策略工厂实例
strategy_factory = FuturesStrategyFactory()