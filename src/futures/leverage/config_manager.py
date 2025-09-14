"""
杠杆控制器配置管理器

提供杠杆控制系统的配置加载、验证和管理功能。
"""
import logging
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from .leverage_models import (
    LeverageConfig,
    LeverageLimits,
    LeverageStrategy,
    MarketRegime,
    LiquidityLevel
)
from ..models.data_models import RiskLevel
from ...utils.exceptions import ContractTradingError

logger = logging.getLogger(__name__)


class LeverageConfigManager:
    """
    杠杆配置管理器

    负责加载、验证、保存和管理杠杆控制系统的配置。
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径（可选）
        """
        self.config_path = config_path
        self._config_cache: Optional[LeverageConfig] = None
        self._config_timestamp: Optional[float] = None

    def load_config(self, config_path: Optional[Path] = None) -> LeverageConfig:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径（可选，使用初始化时的路径）

        Returns:
            杠杆配置对象

        Raises:
            FileNotFoundError: 配置文件不存在
            ValueError: 配置文件格式错误
        """
        path = config_path or self.config_path

        try:
            if path and path.exists():
                # 检查文件是否被修改
                current_timestamp = path.stat().st_mtime
                if (self._config_cache and
                    self._config_timestamp and
                    current_timestamp == self._config_timestamp):
                    logger.debug("使用缓存的配置")
                    return self._config_cache

                # 加载配置文件
                with open(path, 'r', encoding='utf-8') as f:
                    if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)

                # 解析配置
                config = self._parse_config_data(config_data)

                # 缓存配置
                self._config_cache = config
                self._config_timestamp = current_timestamp

                logger.info(f"已加载杠杆配置: {path}")
                return config

            else:
                # 创建默认配置
                logger.info("使用默认杠杆配置")
                config = self._create_default_config()
                self._config_cache = config
                return config

        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            # 返回默认配置
            return self._create_default_config()

    def save_config(
        self,
        config: LeverageConfig,
        config_path: Optional[Path] = None,
        format_type: str = 'yaml'
    ) -> bool:
        """
        保存配置到文件

        Args:
            config: 杠杆配置对象
            config_path: 保存路径（可选）
            format_type: 文件格式 ('yaml' 或 'json')

        Returns:
            是否保存成功
        """
        path = config_path or self.config_path

        try:
            if not path:
                logger.error("未指定配置文件保存路径")
                return False

            # 确保目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            # 转换为字典格式
            config_dict = config.to_dict()

            # 保存文件
            with open(path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # 更新缓存
            self._config_cache = config
            self._config_timestamp = path.stat().st_mtime

            logger.info(f"配置已保存到: {path}")
            return True

        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            return False

    def validate_config(self, config: LeverageConfig) -> List[str]:
        """
        验证配置有效性

        Args:
            config: 杠杆配置对象

        Returns:
            验证错误列表（空列表表示无错误）
        """
        errors = []

        try:
            # 验证基础杠杆设置
            if config.default_leverage <= 0:
                errors.append("默认杠杆必须大于0")

            if config.default_leverage > 200:
                errors.append("默认杠杆不应超过200")

            # 验证策略杠杆设置
            for strategy, leverage in config.base_leverage_by_strategy.items():
                if leverage <= 0:
                    errors.append(f"策略{strategy.value}的杠杆必须大于0")
                if leverage > 200:
                    errors.append(f"策略{strategy.value}的杠杆不应超过200")

            # 验证调整参数
            if not 0 <= config.max_adjustment_percentage <= 100:
                errors.append("最大调整百分比必须在0-100范围内")

            if not 0 <= config.min_adjustment_percentage <= 100:
                errors.append("最小调整百分比必须在0-100范围内")

            if config.min_adjustment_percentage > config.max_adjustment_percentage:
                errors.append("最小调整百分比不能大于最大调整百分比")

            # 验证阈值设置
            if not 0 <= config.high_volatility_threshold <= 1:
                errors.append("高波动阈值必须在0-1范围内")

            if not 0 <= config.low_liquidity_threshold <= 1:
                errors.append("低流动性阈值必须在0-1范围内")

            # 验证时间窗口设置
            if config.volatility_window_minutes <= 0:
                errors.append("波动率窗口必须大于0分钟")

            if config.liquidity_window_minutes <= 0:
                errors.append("流动性窗口必须大于0分钟")

            if config.position_check_interval_seconds <= 0:
                errors.append("仓位检查间隔必须大于0秒")

            # 验证交易对特定限制
            for ticker, limits in config.symbol_specific_limits.items():
                symbol_errors = self._validate_symbol_limits(ticker, limits)
                errors.extend(symbol_errors)

        except Exception as e:
            errors.append(f"配置验证时发生错误: {str(e)}")

        return errors

    def _validate_symbol_limits(self, ticker: str, limits: LeverageLimits) -> List[str]:
        """验证交易对限制"""
        errors = []

        try:
            if limits.max_leverage <= 0:
                errors.append(f"{ticker}: 最大杠杆必须大于0")

            if limits.min_leverage <= 0:
                errors.append(f"{ticker}: 最小杠杆必须大于0")

            if limits.min_leverage > limits.max_leverage:
                errors.append(f"{ticker}: 最小杠杆不能大于最大杠杆")

            # 验证风险等级限制
            for risk_level, leverage in limits.risk_based_limits.items():
                if leverage <= 0:
                    errors.append(f"{ticker}: 风险等级{risk_level.value}的杠杆必须大于0")
                if leverage > limits.max_leverage:
                    errors.append(f"{ticker}: 风险等级{risk_level.value}的杠杆不能超过最大杠杆")

            # 验证策略限制
            for strategy, leverage in limits.strategy_based_limits.items():
                if leverage <= 0:
                    errors.append(f"{ticker}: 策略{strategy.value}的杠杆必须大于0")
                if leverage > limits.max_leverage:
                    errors.append(f"{ticker}: 策略{strategy.value}的杠杆不能超过最大杠杆")

        except Exception as e:
            errors.append(f"{ticker}: 验证限制时发生错误: {str(e)}")

        return errors

    def _parse_config_data(self, config_data: Dict[str, Any]) -> LeverageConfig:
        """
        解析配置数据

        Args:
            config_data: 配置字典

        Returns:
            杠杆配置对象
        """
        try:
            # 解析基础设置
            config = LeverageConfig(
                default_leverage=config_data.get('default_leverage', 5.0),
                max_adjustment_percentage=config_data.get('max_adjustment_percentage', 50.0),
                min_adjustment_percentage=config_data.get('min_adjustment_percentage', 10.0),
                emergency_leverage_cap=config_data.get('emergency_leverage_cap', 2.0),
                high_volatility_threshold=config_data.get('high_volatility_threshold', 0.4),
                low_liquidity_threshold=config_data.get('low_liquidity_threshold', 0.3),
                volatility_window_minutes=config_data.get('volatility_window_minutes', 60),
                liquidity_window_minutes=config_data.get('liquidity_window_minutes', 30),
                position_check_interval_seconds=config_data.get('position_check_interval_seconds', 30),
                enable_emergency_reduction=config_data.get('enable_emergency_reduction', True),
                enable_position_size_limits=config_data.get('enable_position_size_limits', True),
                enable_correlation_checks=config_data.get('enable_correlation_checks', True)
            )

            # 解析策略杠杆设置
            strategy_leverages = config_data.get('base_leverage_by_strategy', {})
            for strategy_name, leverage in strategy_leverages.items():
                try:
                    strategy = LeverageStrategy(strategy_name)
                    config.base_leverage_by_strategy[strategy] = leverage
                except ValueError:
                    logger.warning(f"未知杠杆策略: {strategy_name}")

            # 解析交易对特定限制
            symbol_limits_data = config_data.get('symbol_specific_limits', {})
            for ticker, limits_data in symbol_limits_data.items():
                limits = self._parse_symbol_limits(ticker, limits_data)
                config.symbol_specific_limits[ticker] = limits

            # 验证配置
            errors = self.validate_config(config)
            if errors:
                logger.warning(f"配置验证发现问题: {'; '.join(errors)}")

            return config

        except Exception as e:
            logger.error(f"解析配置数据失败: {e}")
            return self._create_default_config()

    def _parse_symbol_limits(self, ticker: str, limits_data: Dict[str, Any]) -> LeverageLimits:
        """解析交易对限制数据"""
        try:
            limits = LeverageLimits(
                ticker=ticker,
                max_leverage=limits_data.get('max_leverage', 50.0),
                min_leverage=limits_data.get('min_leverage', 1.0),
                volatility_threshold=limits_data.get('volatility_threshold', 0.3),
                liquidity_threshold=limits_data.get('liquidity_threshold', 0.5),
                emergency_reduction_factor=limits_data.get('emergency_reduction_factor', 0.5)
            )

            # 解析风险等级限制
            risk_limits = limits_data.get('risk_based_limits', {})
            for risk_name, leverage in risk_limits.items():
                try:
                    risk_level = RiskLevel(risk_name)
                    limits.risk_based_limits[risk_level] = leverage
                except ValueError:
                    logger.warning(f"未知风险等级: {risk_name}")

            # 解析策略限制
            strategy_limits = limits_data.get('strategy_based_limits', {})
            for strategy_name, leverage in strategy_limits.items():
                try:
                    strategy = LeverageStrategy(strategy_name)
                    limits.strategy_based_limits[strategy] = leverage
                except ValueError:
                    logger.warning(f"未知杠杆策略: {strategy_name}")

            # 解析市场状态限制
            market_limits = limits_data.get('market_regime_limits', {})
            for regime_name, leverage in market_limits.items():
                try:
                    regime = MarketRegime(regime_name)
                    limits.market_regime_limits[regime] = leverage
                except ValueError:
                    logger.warning(f"未知市场状态: {regime_name}")

            return limits

        except Exception as e:
            logger.error(f"解析{ticker}限制失败: {e}")
            return LeverageLimits(ticker=ticker, max_leverage=50.0)

    def _create_default_config(self) -> LeverageConfig:
        """创建默认配置"""
        config = LeverageConfig()

        # 添加主要交易对的特定限制
        major_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        for ticker in major_pairs:
            limits = LeverageLimits(ticker=ticker, max_leverage=125.0)
            config.symbol_specific_limits[ticker] = limits

        # 添加二线币种限制
        alt_pairs = ['ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'SOLUSDT']
        for ticker in alt_pairs:
            limits = LeverageLimits(ticker=ticker, max_leverage=75.0)
            config.symbol_specific_limits[ticker] = limits

        return config

    def get_config_template(self) -> Dict[str, Any]:
        """
        获取配置文件模板

        Returns:
            配置文件模板字典
        """
        template = {
            "default_leverage": 5.0,
            "base_leverage_by_strategy": {
                "conservative": 3.0,
                "moderate": 5.0,
                "aggressive": 8.0,
                "expert": 12.0
            },
            "max_adjustment_percentage": 50.0,
            "min_adjustment_percentage": 10.0,
            "emergency_leverage_cap": 2.0,
            "high_volatility_threshold": 0.4,
            "low_liquidity_threshold": 0.3,
            "volatility_window_minutes": 60,
            "liquidity_window_minutes": 30,
            "position_check_interval_seconds": 30,
            "enable_emergency_reduction": True,
            "enable_position_size_limits": True,
            "enable_correlation_checks": True,
            "symbol_specific_limits": {
                "BTCUSDT": {
                    "max_leverage": 125.0,
                    "min_leverage": 1.0,
                    "volatility_threshold": 0.3,
                    "liquidity_threshold": 0.5,
                    "emergency_reduction_factor": 0.5,
                    "risk_based_limits": {
                        "low": 125.0,
                        "medium": 100.0,
                        "high": 75.0,
                        "critical": 50.0,
                        "emergency": 25.0
                    },
                    "strategy_based_limits": {
                        "conservative": 25.0,
                        "moderate": 50.0,
                        "aggressive": 75.0,
                        "expert": 125.0
                    },
                    "market_regime_limits": {
                        "calm": 125.0,
                        "volatile": 100.0,
                        "trending": 112.0,
                        "ranging": 87.0,
                        "high_volatility": 62.0
                    }
                },
                "ETHUSDT": {
                    "max_leverage": 100.0,
                    "min_leverage": 1.0,
                    "volatility_threshold": 0.3,
                    "liquidity_threshold": 0.5,
                    "emergency_reduction_factor": 0.5,
                    "risk_based_limits": {
                        "low": 100.0,
                        "medium": 80.0,
                        "high": 60.0,
                        "critical": 40.0,
                        "emergency": 20.0
                    },
                    "strategy_based_limits": {
                        "conservative": 20.0,
                        "moderate": 40.0,
                        "aggressive": 60.0,
                        "expert": 100.0
                    },
                    "market_regime_limits": {
                        "calm": 100.0,
                        "volatile": 80.0,
                        "trending": 90.0,
                        "ranging": 70.0,
                        "high_volatility": 50.0
                    }
                }
            }
        }

        return template

    def create_template_file(self, file_path: Path, format_type: str = 'yaml') -> bool:
        """
        创建配置模板文件

        Args:
            file_path: 文件路径
            format_type: 文件格式 ('yaml' 或 'json')

        Returns:
            是否创建成功
        """
        try:
            template = self.get_config_template()

            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存模板文件
            with open(file_path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'yaml':
                    yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(template, f, indent=2, ensure_ascii=False)

            logger.info(f"配置模板已创建: {file_path}")
            return True

        except Exception as e:
            logger.error(f"创建配置模板失败: {e}")
            return False


# 导出主要类
__all__ = ['LeverageConfigManager']