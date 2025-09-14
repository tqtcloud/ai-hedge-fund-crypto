"""
Binance官方SDK配置工厂

为官方binance-connector-python SDK创建配置对象的工厂类。
支持从项目配置文件自动生成SDK所需的配置类实例。
"""

import logging
import ssl
from typing import Optional, Dict, Any, Union
from enum import Enum

from .settings import BinanceSDKSettings

logger = logging.getLogger(__name__)


class SDKConnectionMode(Enum):
    """SDK连接模式枚举"""
    SINGLE = "single"
    POOL = "pool"


class BinanceSDKConfigurationFactory:
    """Binance官方SDK配置工厂类"""
    
    def __init__(self, sdk_settings: BinanceSDKSettings):
        """
        初始化配置工厂
        
        Args:
            sdk_settings: SDK配置对象
        """
        self.sdk_settings = sdk_settings
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def create_rest_api_configuration(self) -> Dict[str, Any]:
        """
        创建REST API配置对象
        
        Returns:
            Dict[str, Any]: 适用于 ConfigurationRestAPI 的配置字典
        """
        try:
            config = self.sdk_settings.create_rest_api_config_dict()
            
            # 添加环境特定的端点URL
            environment = self.sdk_settings.get_effective_environment()
            if not self.sdk_settings.endpoints.rest_api_base_url:
                config["base_path"] = self._get_default_rest_endpoint(environment)
            
            self.logger.info(f"创建REST API配置成功 (环境: {environment})")
            return config
            
        except Exception as e:
            self.logger.error(f"创建REST API配置失败: {e}")
            raise
    
    def create_websocket_api_configuration(self) -> Dict[str, Any]:
        """
        创建WebSocket API配置对象
        
        Returns:
            Dict[str, Any]: 适用于 ConfigurationWebSocketAPI 的配置字典
        """
        try:
            config = self.sdk_settings.create_websocket_api_config_dict()
            
            # 添加环境特定的端点URL
            environment = self.sdk_settings.get_effective_environment()
            if not self.sdk_settings.endpoints.websocket_api_url:
                config["stream_url"] = self._get_default_websocket_api_endpoint(environment)
            
            # 转换连接模式
            if config.get("mode") == "pool":
                try:
                    # 导入WebsocketMode枚举
                    from binance_common.constants import WebsocketMode
                    config["mode"] = WebsocketMode.POOL
                except ImportError:
                    self.logger.warning("无法导入WebsocketMode，使用字符串模式")
                    config["mode"] = "pool"
            
            self.logger.info(f"创建WebSocket API配置成功 (环境: {environment})")
            return config
            
        except Exception as e:
            self.logger.error(f"创建WebSocket API配置失败: {e}")
            raise
    
    def create_websocket_streams_configuration(self) -> Dict[str, Any]:
        """
        创建WebSocket流配置对象
        
        Returns:
            Dict[str, Any]: 适用于 ConfigurationWebSocketStreams 的配置字典
        """
        try:
            config = self.sdk_settings.create_websocket_streams_config_dict()
            
            # 添加环境特定的端点URL
            environment = self.sdk_settings.get_effective_environment()
            if not self.sdk_settings.endpoints.websocket_streams_url:
                config["stream_url"] = self._get_default_websocket_streams_endpoint(environment)
            
            # 转换连接模式
            if config.get("mode") == "pool":
                try:
                    # 导入WebsocketMode枚举
                    from binance_common.constants import WebsocketMode
                    config["mode"] = WebsocketMode.POOL
                except ImportError:
                    self.logger.warning("无法导入WebsocketMode，使用字符串模式")
                    config["mode"] = "pool"
            
            self.logger.info(f"创建WebSocket流配置成功 (环境: {environment})")
            return config
            
        except Exception as e:
            self.logger.error(f"创建WebSocket流配置失败: {e}")
            raise
    
    def create_futures_usds_client_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        创建USDS期货客户端配置
        
        Returns:
            Dict[str, Dict[str, Any]]: 包含所有配置类型的字典
        """
        return {
            "rest_api": self.create_rest_api_configuration(),
            "websocket_api": self.create_websocket_api_configuration(),
            "websocket_streams": self.create_websocket_streams_configuration()
        }
    
    def create_futures_coin_client_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        创建COIN期货客户端配置
        
        Returns:
            Dict[str, Dict[str, Any]]: 包含所有配置类型的字典
        """
        # COIN期货使用相同的配置，但端点不同
        configs = self.create_futures_usds_client_configs()
        
        environment = self.sdk_settings.get_effective_environment()
        
        # 更新为COIN期货特定的端点
        if not self.sdk_settings.endpoints.rest_api_base_url:
            configs["rest_api"]["base_path"] = self._get_default_coin_futures_rest_endpoint(environment)
            
        if not self.sdk_settings.endpoints.websocket_api_url:
            configs["websocket_api"]["stream_url"] = self._get_default_coin_futures_websocket_api_endpoint(environment)
            
        if not self.sdk_settings.endpoints.websocket_streams_url:
            configs["websocket_streams"]["stream_url"] = self._get_default_coin_futures_websocket_streams_endpoint(environment)
        
        return configs
    
    def _get_default_rest_endpoint(self, environment: str) -> str:
        """获取默认REST端点"""
        if environment == "mainnet":
            return "https://fapi.binance.com"
        else:
            return "https://testnet.binancefuture.com"
    
    def _get_default_websocket_api_endpoint(self, environment: str) -> str:
        """获取默认WebSocket API端点"""
        try:
            if environment == "mainnet":
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_API_PROD_URL
                return DERIVATIVES_TRADING_USDS_FUTURES_WS_API_PROD_URL
            else:
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_API_TESTNET_URL
                return DERIVATIVES_TRADING_USDS_FUTURES_WS_API_TESTNET_URL
        except ImportError:
            self.logger.warning("无法导入SDK常量，使用备用端点")
            if environment == "mainnet":
                return "wss://fstream.binance.com/ws"
            else:
                return "wss://stream.binancefuture.com/ws"
    
    def _get_default_websocket_streams_endpoint(self, environment: str) -> str:
        """获取默认WebSocket流端点"""
        try:
            if environment == "mainnet":
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL
                return DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL
            else:
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
                return DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
        except ImportError:
            self.logger.warning("无法导入SDK常量，使用备用端点")
            if environment == "mainnet":
                return "wss://fstream.binance.com/stream"
            else:
                return "wss://stream.binancefuture.com/stream"
    
    def _get_default_coin_futures_rest_endpoint(self, environment: str) -> str:
        """获取默认COIN期货REST端点"""
        try:
            if environment == "mainnet":
                from binance_common.constants import DERIVATIVES_TRADING_COIN_FUTURES_REST_API_PROD_URL
                return DERIVATIVES_TRADING_COIN_FUTURES_REST_API_PROD_URL
            else:
                from binance_common.constants import DERIVATIVES_TRADING_COIN_FUTURES_REST_API_TESTNET_URL
                return DERIVATIVES_TRADING_COIN_FUTURES_REST_API_TESTNET_URL
        except ImportError:
            self.logger.warning("无法导入SDK常量，使用备用端点")
            if environment == "mainnet":
                return "https://dapi.binance.com"
            else:
                return "https://testnet.binancefuture.com"
    
    def _get_default_coin_futures_websocket_api_endpoint(self, environment: str) -> str:
        """获取默认COIN期货WebSocket API端点"""
        try:
            if environment == "mainnet":
                from binance_common.constants import DERIVATIVES_TRADING_COIN_FUTURES_WS_API_PROD_URL
                return DERIVATIVES_TRADING_COIN_FUTURES_WS_API_PROD_URL
            else:
                from binance_common.constants import DERIVATIVES_TRADING_COIN_FUTURES_WS_API_TESTNET_URL
                return DERIVATIVES_TRADING_COIN_FUTURES_WS_API_TESTNET_URL
        except ImportError:
            self.logger.warning("无法导入SDK常量，使用备用端点")
            if environment == "mainnet":
                return "wss://dstream.binance.com/ws"
            else:
                return "wss://stream.binancefuture.com/ws"
    
    def _get_default_coin_futures_websocket_streams_endpoint(self, environment: str) -> str:
        """获取默认COIN期货WebSocket流端点"""
        try:
            if environment == "mainnet":
                from binance_common.constants import DERIVATIVES_TRADING_COIN_FUTURES_WS_STREAMS_PROD_URL
                return DERIVATIVES_TRADING_COIN_FUTURES_WS_STREAMS_PROD_URL
            else:
                from binance_common.constants import DERIVATIVES_TRADING_COIN_FUTURES_WS_STREAMS_TESTNET_URL
                return DERIVATIVES_TRADING_COIN_FUTURES_WS_STREAMS_TESTNET_URL
        except ImportError:
            self.logger.warning("无法导入SDK常量，使用备用端点")
            if environment == "mainnet":
                return "wss://dstream.binance.com/stream"
            else:
                return "wss://stream.binancefuture.com/stream"


def create_binance_sdk_factory(sdk_settings: BinanceSDKSettings) -> BinanceSDKConfigurationFactory:
    """
    创建Binance SDK配置工厂实例
    
    Args:
        sdk_settings: SDK配置对象
        
    Returns:
        BinanceSDKConfigurationFactory: 配置工厂实例
    """
    return BinanceSDKConfigurationFactory(sdk_settings)


def create_usds_futures_client_from_settings(sdk_settings: BinanceSDKSettings):
    """
    从配置创建USDS期货客户端
    
    Args:
        sdk_settings: SDK配置对象
        
    Returns:
        DerivativesTradingUsdsFutures: 期货交易客户端 (如果SDK已安装)
    """
    try:
        from binance_common.configuration import (
            ConfigurationRestAPI, 
            ConfigurationWebSocketAPI, 
            ConfigurationWebSocketStreams
        )
        from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
            DerivativesTradingUsdsFutures
        )
        
        factory = create_binance_sdk_factory(sdk_settings)
        configs = factory.create_futures_usds_client_configs()
        
        # 创建配置对象
        rest_config = ConfigurationRestAPI(**configs["rest_api"])
        ws_api_config = ConfigurationWebSocketAPI(**configs["websocket_api"])
        ws_streams_config = ConfigurationWebSocketStreams(**configs["websocket_streams"])
        
        # 创建客户端
        client = DerivativesTradingUsdsFutures(
            config_rest_api=rest_config,
            config_ws_api=ws_api_config,
            config_ws_streams=ws_streams_config
        )
        
        logger.info("USDS期货客户端创建成功")
        return client
        
    except ImportError as e:
        logger.error(f"Binance SDK未安装或版本不兼容: {e}")
        raise ImportError("请安装binance-connector-python SDK")
    except Exception as e:
        logger.error(f"创建USDS期货客户端失败: {e}")
        raise


def create_coin_futures_client_from_settings(sdk_settings: BinanceSDKSettings):
    """
    从配置创建COIN期货客户端
    
    Args:
        sdk_settings: SDK配置对象
        
    Returns:
        DerivativesTradingCoinFutures: COIN期货交易客户端 (如果SDK已安装)
    """
    try:
        from binance_common.configuration import (
            ConfigurationRestAPI, 
            ConfigurationWebSocketAPI, 
            ConfigurationWebSocketStreams
        )
        from binance_sdk_derivatives_trading_coin_futures.derivatives_trading_coin_futures import (
            DerivativesTradingCoinFutures
        )
        
        factory = create_binance_sdk_factory(sdk_settings)
        configs = factory.create_futures_coin_client_configs()
        
        # 创建配置对象
        rest_config = ConfigurationRestAPI(**configs["rest_api"])
        ws_api_config = ConfigurationWebSocketAPI(**configs["websocket_api"])
        ws_streams_config = ConfigurationWebSocketStreams(**configs["websocket_streams"])
        
        # 创建客户端
        client = DerivativesTradingCoinFutures(
            config_rest_api=rest_config,
            config_ws_api=ws_api_config,
            config_ws_streams=ws_streams_config
        )
        
        logger.info("COIN期货客户端创建成功")
        return client
        
    except ImportError as e:
        logger.error(f"Binance SDK未安装或版本不兼容: {e}")
        raise ImportError("请安装binance-connector-python SDK")
    except Exception as e:
        logger.error(f"创建COIN期货客户端失败: {e}")
        raise


def validate_sdk_availability() -> Dict[str, bool]:
    """
    验证SDK可用性
    
    Returns:
        Dict[str, bool]: SDK模块可用性状态
    """
    availability = {
        "binance_common": False,
        "usds_futures": False,
        "coin_futures": False,
        "portfolio_margin": False
    }
    
    try:
        import binance_common
        availability["binance_common"] = True
    except ImportError:
        pass
    
    try:
        import binance_sdk_derivatives_trading_usds_futures
        availability["usds_futures"] = True
    except ImportError:
        pass
    
    try:
        import binance_sdk_derivatives_trading_coin_futures
        availability["coin_futures"] = True
    except ImportError:
        pass
    
    try:
        import binance_sdk_derivatives_trading_portfolio_margin
        availability["portfolio_margin"] = True
    except ImportError:
        pass
    
    return availability


def get_sdk_installation_guide() -> Dict[str, str]:
    """
    获取SDK安装指南
    
    Returns:
        Dict[str, str]: 安装命令和说明
    """
    return {
        "pip_commands": [
            "pip install binance-connector-python[futures]",
            "# 或者单独安装具体模块:",
            "pip install binance-sdk-derivatives-trading-usds-futures",
            "pip install binance-sdk-derivatives-trading-coin-futures"
        ],
        "note": "请参考官方文档选择合适的SDK模块: https://github.com/binance/binance-connector-python",
        "requirements_file": "# 添加到 requirements.txt:\nbinance-connector-python[futures]>=1.0.0"
    }