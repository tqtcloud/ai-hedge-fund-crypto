"""
验证器工厂

负责创建和配置验证器实例，提供便捷的验证器获取接口。
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from .composite_validator import CompositeValidator

logger = logging.getLogger(__name__)


class ValidatorFactory:
    """
    验证器工厂类
    
    负责创建和管理验证器实例
    """
    
    _instance = None
    _validators_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._config = None
        self._config_path = None
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        加载验证器配置
        
        Args:
            config_path: 配置文件路径，为None时使用默认路径
            
        Returns:
            验证器配置字典
        """
        if config_path is None:
            # 使用默认配置路径
            default_path = Path(__file__).parent.parent.parent.parent / "config" / "validation_constraints.yaml"
            config_path = str(default_path)
        
        # 如果配置已加载且路径相同，直接返回缓存的配置
        if self._config is not None and self._config_path == config_path:
            return self._config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            self._config_path = config_path
            logger.info(f"加载验证器配置成功: {config_path}")
            return self._config
        except FileNotFoundError:
            logger.error(f"验证器配置文件未找到: {config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"验证器配置文件格式错误: {e}")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"加载验证器配置失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "validation_level": "moderate",
            "numerical_constraints": {
                "leverage": {"min": 1, "max": 125, "warning_threshold": 50},
                "confidence": {"min": 0, "max": 100, "reliable_threshold": 65},
                "price": {"min": 0.000001, "max": 1000000},
                "position_size": {"min_value": 10.0, "max_value": 1000000.0}
            },
            "logical_constraints": {
                "direction_stop_logic": {},
                "leverage_volatility": {"high_volatility_max_leverage": 10}
            },
            "risk_constraints": {
                "margin_utilization": {"warning_level": 0.7, "critical_level": 0.85},
                "liquidation_distance": {"min_safe_distance": 15.0}
            },
            "cost_constraints": {
                "trading_fees": {"max_fee_rate": 0.001},
                "funding_rate": {"high_rate_threshold": 0.01}
            },
            "validation_levels": {
                "moderate": {
                    "enabled_validators": ["numerical", "logical", "risk", "cost"],
                    "warning_only": []
                }
            },
            "auto_correction": {
                "enabled": True,
                "max_adjustments": 3,
                "preserve_intent": True
            }
        }
    
    def create_validator(
        self, 
        validation_level: str = "moderate",
        config_path: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> CompositeValidator:
        """
        创建验证器实例
        
        Args:
            validation_level: 验证级别 (strict, moderate, lenient)
            config_path: 配置文件路径
            custom_config: 自定义配置，会覆盖文件配置
            
        Returns:
            复合验证器实例
        """
        # 生成缓存键
        cache_key = f"{validation_level}_{config_path}_{id(custom_config) if custom_config else 'none'}"
        
        # 检查缓存
        if cache_key in self._validators_cache:
            logger.debug(f"使用缓存的验证器: {cache_key}")
            return self._validators_cache[cache_key]
        
        # 加载配置
        config = self.load_config(config_path)
        
        # 应用自定义配置
        if custom_config:
            config = self._merge_configs(config, custom_config)
        
        # 设置验证级别
        config["validation_level"] = validation_level
        
        # 创建验证器实例
        validator = CompositeValidator(config)
        
        # 缓存验证器实例
        self._validators_cache[cache_key] = validator
        
        logger.info(f"创建验证器实例: 级别={validation_level}, "
                   f"启用验证器={list(validator.validators.keys())}")
        
        return validator
    
    def _merge_configs(self, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置字典
        
        Args:
            base_config: 基础配置
            custom_config: 自定义配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        
        def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> None:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(merged, custom_config)
        return merged
    
    def get_available_levels(self) -> List[str]:
        """
        获取可用的验证级别列表
        
        Returns:
            验证级别列表
        """
        config = self.load_config()
        return list(config.get("validation_levels", {}).keys())
    
    def get_validator_for_level(self, level: str) -> CompositeValidator:
        """
        为指定级别创建验证器
        
        Args:
            level: 验证级别
            
        Returns:
            验证器实例
        """
        return self.create_validator(validation_level=level)
    
    def clear_cache(self) -> None:
        """清除验证器缓存"""
        self._validators_cache.clear()
        logger.info("验证器缓存已清除")
    
    def reload_config(self, config_path: Optional[str] = None) -> None:
        """
        重新加载配置
        
        Args:
            config_path: 配置文件路径
        """
        self._config = None
        self._config_path = None
        self.clear_cache()
        self.load_config(config_path)
        logger.info("验证器配置已重新加载")


# 全局工厂实例
validator_factory = ValidatorFactory()


def create_validator(
    validation_level: str = "moderate",
    config_path: Optional[str] = None,
    custom_config: Optional[Dict[str, Any]] = None
) -> CompositeValidator:
    """
    便捷函数：创建验证器实例
    
    Args:
        validation_level: 验证级别
        config_path: 配置文件路径
        custom_config: 自定义配置
        
    Returns:
        验证器实例
    """
    return validator_factory.create_validator(validation_level, config_path, custom_config)


def get_validator_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    便捷函数：获取验证器配置
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    return validator_factory.load_config(config_path)