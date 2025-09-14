"""
Binance SDK配置管理器

为项目提供统一的Binance SDK配置管理和客户端创建服务。
集成官方SDK的配置模式，支持testnet/mainnet环境切换。
"""
import os
import logging
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

from ..constants import (
    get_sdk_environment_config,
    create_sdk_rest_config,
    create_sdk_websocket_config,
    get_current_environment,
    is_testnet_environment,
    BinanceSDKConfig,
    SDKErrorCodes
)

if TYPE_CHECKING:
    try:
        from binance_common.configuration import (
            ConfigurationRestAPI,
            ConfigurationWebSocketStreams,
            ConfigurationWebSocketAPI
        )
        from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
            DerivativesTradingUsdsFutures
        )
    except ImportError:
        ConfigurationRestAPI = Any
        ConfigurationWebSocketStreams = Any
        ConfigurationWebSocketAPI = Any
        DerivativesTradingUsdsFutures = Any
else:
    try:
        from binance_common.configuration import (
            ConfigurationRestAPI,
            ConfigurationWebSocketStreams,
            ConfigurationWebSocketAPI
        )
        from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
            DerivativesTradingUsdsFutures
        )
        SDK_AVAILABLE = True
    except ImportError:
        ConfigurationRestAPI = None
        ConfigurationWebSocketStreams = None
        ConfigurationWebSocketAPI = None
        DerivativesTradingUsdsFutures = None
        SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SDKCredentials:
    """SDK认证凭据配置"""
    api_key: str
    api_secret: Optional[str] = None
    private_key: Optional[str] = None
    private_key_passphrase: Optional[str] = None
    
    def __post_init__(self):
        """验证认证凭据"""
        if not self.api_key:
            raise ValueError("API密钥不能为空")
        
        # 至少需要api_secret或private_key之一
        if not self.api_secret and not self.private_key:
            raise ValueError("必须提供api_secret或private_key")


class BinanceSDKManager:
    """
    Binance SDK配置管理器
    
    负责创建和管理Binance SDK客户端实例，处理不同环境的配置，
    并提供统一的SDK客户端访问接口。
    """
    
    def __init__(
        self,
        credentials: Optional[SDKCredentials] = None,
        environment: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化SDK管理器
        
        Args:
            credentials: SDK认证凭据
            environment: 环境名称（testnet/mainnet），默认从环境变量获取
            custom_config: 自定义配置
        """
        self.credentials = credentials
        self.environment = environment or get_current_environment()
        self.custom_config = custom_config or {}
        
        # 验证SDK可用性
        if not SDK_AVAILABLE:
            logger.warning("Binance SDK不可用，某些功能可能受限")
        
        # 获取环境配置
        try:
            self._env_config = get_sdk_environment_config(self.environment)
        except ValueError as e:
            logger.error(f"获取环境配置失败: {e}")
            raise
        
        # 客户端实例缓存
        self._rest_client: Optional[DerivativesTradingUsdsFutures] = None
        self._ws_streams_client: Optional[DerivativesTradingUsdsFutures] = None
        self._ws_api_client: Optional[DerivativesTradingUsdsFutures] = None
        
        logger.info(f"SDK管理器已初始化 - 环境: {self.environment}, SDK可用: {SDK_AVAILABLE}")
    
    @classmethod
    def from_env_vars(cls, environment: Optional[str] = None) -> 'BinanceSDKManager':
        """
        从环境变量创建SDK管理器
        
        环境变量:
        - BINANCE_API_KEY: API密钥
        - BINANCE_API_SECRET: API密钥秘密
        - BINANCE_PRIVATE_KEY: 私钥（可选，与API_SECRET二选一）
        - BINANCE_PRIVATE_KEY_PASSPHRASE: 私钥密码（可选）
        - BINANCE_ENVIRONMENT: 环境名称（testnet/mainnet）
        
        Args:
            environment: 覆盖环境变量的环境设置
            
        Returns:
            BinanceSDKManager实例
        """
        api_key = os.getenv('BINANCE_API_KEY')
        if not api_key:
            raise ValueError("未找到BINANCE_API_KEY环境变量")
        
        api_secret = os.getenv('BINANCE_API_SECRET')
        private_key = os.getenv('BINANCE_PRIVATE_KEY')
        private_key_passphrase = os.getenv('BINANCE_PRIVATE_KEY_PASSPHRASE')
        
        credentials = SDKCredentials(
            api_key=api_key,
            api_secret=api_secret,
            private_key=private_key,
            private_key_passphrase=private_key_passphrase
        )
        
        return cls(
            credentials=credentials,
            environment=environment or os.getenv('BINANCE_ENVIRONMENT', 'testnet')
        )
    
    def is_sdk_available(self) -> bool:
        """检查SDK是否可用"""
        return SDK_AVAILABLE
    
    def get_environment_info(self) -> Dict[str, Any]:
        """
        获取当前环境信息
        
        Returns:
            环境配置信息
        """
        return {
            "environment": self.environment,
            "is_testnet": is_testnet_environment(),
            "sdk_available": SDK_AVAILABLE,
            "rest_api_url": self._env_config["rest_api_url"],
            "ws_streams_url": self._env_config["ws_streams_url"],
            "ws_api_url": self._env_config["ws_api_url"],
            "max_leverage": self._env_config["max_leverage"],
            "config": self._env_config["sdk_config"]
        }
    
    def create_rest_client(self, **kwargs) -> 'DerivativesTradingUsdsFutures':
        """
        创建REST API客户端
        
        Args:
            **kwargs: 额外配置参数
            
        Returns:
            DerivativesTradingUsdsFutures客户端实例
        """
        if not SDK_AVAILABLE:
            raise RuntimeError("Binance SDK不可用，无法创建REST客户端")
        
        if not self.credentials:
            raise ValueError("未设置SDK认证凭据")
        
        # 创建配置
        config_dict = create_sdk_rest_config(
            api_key=self.credentials.api_key,
            api_secret=self.credentials.api_secret,
            private_key=self.credentials.private_key,
            private_key_passphrase=self.credentials.private_key_passphrase,
            environment=self.environment,
            **self.custom_config,
            **kwargs
        )
        
        # 创建官方SDK配置对象
        config = ConfigurationRestAPI(**config_dict)
        
        # 创建客户端
        client = DerivativesTradingUsdsFutures(config_rest_api=config)
        
        logger.info(f"REST客户端已创建 - 环境: {self.environment}")
        return client
    
    def create_websocket_streams_client(self, **kwargs) -> 'DerivativesTradingUsdsFutures':
        """
        创建WebSocket Streams客户端（用于接收数据流）
        
        Args:
            **kwargs: 额外配置参数
            
        Returns:
            DerivativesTradingUsdsFutures客户端实例
        """
        if not SDK_AVAILABLE:
            raise RuntimeError("Binance SDK不可用，无法创建WebSocket Streams客户端")
        
        # WebSocket Streams不需要认证
        config_dict = create_sdk_websocket_config(
            environment=self.environment,
            connection_type="streams",
            **self.custom_config,
            **kwargs
        )
        
        # 创建官方SDK配置对象
        config = ConfigurationWebSocketStreams(**config_dict)
        
        # 创建客户端
        client = DerivativesTradingUsdsFutures(config_ws_streams=config)
        
        logger.info(f"WebSocket Streams客户端已创建 - 环境: {self.environment}")
        return client
    
    def create_websocket_api_client(self, **kwargs) -> 'DerivativesTradingUsdsFutures':
        """
        创建WebSocket API客户端（用于API调用）
        
        Args:
            **kwargs: 额外配置参数
            
        Returns:
            DerivativesTradingUsdsFutures客户端实例
        """
        if not SDK_AVAILABLE:
            raise RuntimeError("Binance SDK不可用，无法创建WebSocket API客户端")
        
        if not self.credentials:
            raise ValueError("未设置SDK认证凭据")
        
        # 创建配置
        config_dict = create_sdk_websocket_config(
            api_key=self.credentials.api_key,
            api_secret=self.credentials.api_secret,
            private_key=self.credentials.private_key,
            private_key_passphrase=self.credentials.private_key_passphrase,
            environment=self.environment,
            connection_type="api",
            **self.custom_config,
            **kwargs
        )
        
        # 创建官方SDK配置对象
        config = ConfigurationWebSocketAPI(**config_dict)
        
        # 创建客户端
        client = DerivativesTradingUsdsFutures(config_ws_api=config)
        
        logger.info(f"WebSocket API客户端已创建 - 环境: {self.environment}")
        return client
    
    def get_rest_client(self) -> 'DerivativesTradingUsdsFutures':
        """获取缓存的REST客户端（懒加载）"""
        if self._rest_client is None:
            self._rest_client = self.create_rest_client()
        return self._rest_client
    
    def get_websocket_streams_client(self) -> 'DerivativesTradingUsdsFutures':
        """获取缓存的WebSocket Streams客户端（懒加载）"""
        if self._ws_streams_client is None:
            self._ws_streams_client = self.create_websocket_streams_client()
        return self._ws_streams_client
    
    def get_websocket_api_client(self) -> 'DerivativesTradingUsdsFutures':
        """获取缓存的WebSocket API客户端（懒加载）"""
        if self._ws_api_client is None:
            self._ws_api_client = self.create_websocket_api_client()
        return self._ws_api_client
    
    def switch_environment(self, new_environment: str):
        """
        切换环境
        
        Args:
            new_environment: 新环境名称
        """
        if new_environment == self.environment:
            return
        
        logger.info(f"切换环境: {self.environment} -> {new_environment}")
        
        # 更新环境配置
        self.environment = new_environment
        self._env_config = get_sdk_environment_config(new_environment)
        
        # 清理缓存的客户端
        self._rest_client = None
        self._ws_streams_client = None
        self._ws_api_client = None
    
    def update_credentials(self, credentials: SDKCredentials):
        """
        更新认证凭据
        
        Args:
            credentials: 新的认证凭据
        """
        logger.info("更新SDK认证凭据")
        self.credentials = credentials
        
        # 清理需要认证的客户端缓存
        self._rest_client = None
        self._ws_api_client = None
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试连接
        
        Returns:
            连接测试结果
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "environment": self.environment,
            "sdk_available": SDK_AVAILABLE,
            "rest_api": {"connected": False, "error": None},
            "websocket_streams": {"connected": False, "error": None}
        }
        
        if not SDK_AVAILABLE:
            return result
        
        # 测试REST API
        try:
            client = self.get_rest_client()
            # 这里可以调用一个简单的API来测试连接
            # response = client.rest_api.exchange_information()
            result["rest_api"]["connected"] = True
            logger.info("REST API连接测试通过")
        except Exception as e:
            result["rest_api"]["error"] = str(e)
            logger.error(f"REST API连接测试失败: {e}")
        
        # 测试WebSocket Streams（这里只是创建客户端，不实际连接）
        try:
            client = self.get_websocket_streams_client()
            result["websocket_streams"]["connected"] = True
            logger.info("WebSocket Streams客户端创建成功")
        except Exception as e:
            result["websocket_streams"]["error"] = str(e)
            logger.error(f"WebSocket Streams客户端创建失败: {e}")
        
        return result


# 全局SDK管理器实例（单例）
_global_sdk_manager: Optional[BinanceSDKManager] = None


def get_global_sdk_manager() -> BinanceSDKManager:
    """
    获取全局SDK管理器实例
    
    Returns:
        全局BinanceSDKManager实例
    """
    global _global_sdk_manager
    
    if _global_sdk_manager is None:
        try:
            _global_sdk_manager = BinanceSDKManager.from_env_vars()
        except ValueError as e:
            logger.warning(f"无法从环境变量创建SDK管理器: {e}")
            # 创建一个无认证的管理器用于基本功能
            _global_sdk_manager = BinanceSDKManager()
    
    return _global_sdk_manager


def set_global_sdk_manager(manager: BinanceSDKManager):
    """
    设置全局SDK管理器实例
    
    Args:
        manager: BinanceSDKManager实例
    """
    global _global_sdk_manager
    _global_sdk_manager = manager


def reset_global_sdk_manager():
    """重置全局SDK管理器实例"""
    global _global_sdk_manager
    _global_sdk_manager = None


# 便捷函数
def create_sdk_manager_from_config(
    api_key: str,
    api_secret: Optional[str] = None,
    private_key: Optional[str] = None,
    environment: str = "testnet",
    **kwargs
) -> BinanceSDKManager:
    """
    从配置参数创建SDK管理器
    
    Args:
        api_key: API密钥
        api_secret: API密钥秘密
        private_key: 私钥
        environment: 环境名称
        **kwargs: 其他配置参数
        
    Returns:
        BinanceSDKManager实例
    """
    credentials = SDKCredentials(
        api_key=api_key,
        api_secret=api_secret,
        private_key=private_key
    )
    
    return BinanceSDKManager(
        credentials=credentials,
        environment=environment,
        custom_config=kwargs
    )