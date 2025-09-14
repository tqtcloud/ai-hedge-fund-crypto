from pydantic_settings import BaseSettings
from pydantic import model_validator, BaseModel, Field
from datetime import datetime
import yaml
import os
import logging
import ssl
from typing import List, Optional, Dict, Any, Union
from dotenv import load_dotenv
from .constants import Interval

load_dotenv()
logger = logging.getLogger(__name__)


class SignalSettings(BaseModel):
    intervals: List[Interval]
    tickers: List[str]
    strategies: List[str]


class ModelSettings(BaseModel):
    name: str
    provider: str
    base_url: Optional[str] = None


class FuturesRiskSettings(BaseModel):
    """期货风险配置"""
    max_leverage: int = 20
    max_position_size: float = 1000.0
    stop_loss_percentage: float = 0.02
    take_profit_percentage: float = 0.05
    daily_loss_limit: float = 500.0
    max_open_positions: int = 5
    margin_requirement: float = 0.1


class FuturesAPISettings(BaseModel):
    """期货API端点配置"""
    base_url: str
    api_url: str
    websocket_url: str
    websocket_stream_url: str


class FuturesAPIConfig(BaseModel):
    """期货API配置"""
    testnet: FuturesAPISettings
    mainnet: FuturesAPISettings
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0


class FuturesWebSocketSettings(BaseModel):
    """期货WebSocket配置"""
    enable_user_data_stream: bool = True
    enable_market_data_stream: bool = True
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30
    buffer_size: int = 1024
    connection_timeout: int = 10
    ping_interval: int = 20
    pong_timeout: int = 10


class FuturesSecuritySettings(BaseModel):
    """期货安全配置"""
    api_key_env: str = "BINANCE_FUTURES_API_KEY"
    secret_key_env: str = "BINANCE_FUTURES_SECRET_KEY"
    enable_signature: bool = True
    recv_window: int = 5000
    verify_ssl: bool = True
    enable_ip_whitelist: bool = False


class FuturesMonitoringSettings(BaseModel):
    """期货监控配置"""
    enable_performance_monitoring: bool = True
    enable_detailed_logging: bool = False
    log_level: str = "INFO"
    metrics_collection_interval: int = 60
    health_check_interval: int = 30


class FuturesSettings(BaseModel):
    """期货配置"""
    environment: str = "testnet"
    api: FuturesAPIConfig
    margin_type: str = "CROSSED"
    symbols: List[str] = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    default_leverage: Dict[str, int] = {}
    allowed_order_types: List[str] = ["MARKET", "LIMIT", "STOP", "TAKE_PROFIT"]
    min_order_size: Dict[str, float] = {}
    price_precision: Dict[str, int] = {}
    quantity_precision: Dict[str, int] = {}
    risk: FuturesRiskSettings = FuturesRiskSettings()
    websocket: FuturesWebSocketSettings = FuturesWebSocketSettings()
    security: FuturesSecuritySettings = FuturesSecuritySettings()
    monitoring: FuturesMonitoringSettings = FuturesMonitoringSettings()
    
    def get_api_endpoints(self) -> FuturesAPISettings:
        """根据当前环境获取API端点配置"""
        if self.environment == "mainnet":
            return self.api.mainnet
        return self.api.testnet
    
    def get_api_credentials(self) -> Dict[str, Optional[str]]:
        """获取API凭证"""
        return {
            "api_key": os.getenv(self.security.api_key_env),
            "secret_key": os.getenv(self.security.secret_key_env)
        }
    
    def validate_api_credentials(self) -> bool:
        """验证API凭证是否配置"""
        credentials = self.get_api_credentials()
        return bool(credentials["api_key"]) and bool(credentials["secret_key"])


class EnvironmentProfile(BaseModel):
    """环境配置文件"""
    futures_environment: str = "testnet"
    log_level: str = "INFO"
    enable_detailed_logging: bool = False


class EnvironmentSettings(BaseModel):
    """环境配置"""
    active: str = "development"
    profiles: Dict[str, EnvironmentProfile] = {}
    
    def get_active_profile(self) -> EnvironmentProfile:
        """获取当前激活的环境配置"""
        return self.profiles.get(self.active, EnvironmentProfile())


# ============ Binance SDK 配置类 ============

class BinanceSDKAPISettings(BaseModel):
    """Binance SDK API认证配置"""
    key: str = Field(default="${BINANCE_API_KEY}", description="API密钥 (支持环境变量)")
    secret: str = Field(default="${BINANCE_API_SECRET}", description="API密钥 (支持环境变量)")
    private_key_path: Optional[str] = Field(default=None, description="RSA私钥文件路径")
    private_key_passphrase: Optional[str] = Field(default=None, description="私钥密码短语")
    
    def get_resolved_credentials(self) -> Dict[str, Optional[str]]:
        """获取解析后的API凭证"""
        resolved_key = None
        resolved_secret = None
        
        # 解析API Key
        if self.key.startswith("${") and self.key.endswith("}"):
            env_var = self.key[2:-1]
            resolved_key = os.getenv(env_var)
        else:
            resolved_key = self.key
            
        # 解析API Secret
        if self.secret.startswith("${") and self.secret.endswith("}"):
            env_var = self.secret[2:-1]
            resolved_secret = os.getenv(env_var)
        else:
            resolved_secret = self.secret
            
        return {
            "api_key": resolved_key,
            "api_secret": resolved_secret
        }
    
    def get_private_key_content(self) -> Optional[str]:
        """读取私钥文件内容"""
        if not self.private_key_path:
            return None
        try:
            with open(self.private_key_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"读取私钥文件失败: {e}")
            return None


class BinanceSDKEnvironmentSettings(BaseModel):
    """Binance SDK环境配置"""
    type: str = Field(default="testnet", description="环境类型: testnet/mainnet")
    auto_detect: bool = Field(default=True, description="是否自动检测环境")
    
    @model_validator(mode='after')
    def validate_environment_type(self):
        if self.type not in ['testnet', 'mainnet']:
            raise ValueError(f"不支持的环境类型: {self.type}")
        return self


class BinanceSDKProxySettings(BaseModel):
    """代理配置"""
    enabled: bool = False
    host: Optional[str] = None
    port: Optional[int] = None
    protocol: str = "http"  # http/https
    auth: Optional[Dict[str, Optional[str]]] = None
    
    def to_proxy_dict(self) -> Optional[Dict[str, Any]]:
        """转换为SDK代理配置字典"""
        if not self.enabled or not self.host or not self.port:
            return None
            
        proxy_config = {
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol
        }
        
        if self.auth and self.auth.get("username") and self.auth.get("password"):
            proxy_config["auth"] = {
                "username": self.auth["username"],
                "password": self.auth["password"]
            }
            
        return proxy_config


class BinanceSDKRestAPISettings(BaseModel):
    """Binance SDK REST API配置"""
    timeout: int = Field(default=10000, description="请求超时时间(毫秒)")
    retries: int = Field(default=3, description="重试次数")
    backoff: int = Field(default=1000, description="重试延迟(毫秒)")
    keep_alive: bool = Field(default=True, description="HTTP keep-alive")
    compression: bool = Field(default=True, description="响应压缩")
    proxy: BinanceSDKProxySettings = Field(default_factory=BinanceSDKProxySettings)
    
    @model_validator(mode='after')
    def validate_settings(self):
        if self.timeout <= 0:
            raise ValueError("timeout必须大于0")
        if self.retries < 0:
            raise ValueError("retries不能小于0")
        if self.backoff < 0:
            raise ValueError("backoff不能小于0")
        return self


class BinanceSDKWebSocketSettings(BaseModel):
    """Binance SDK WebSocket配置基类"""
    timeout: int = Field(default=10000, description="超时时间(毫秒)")
    reconnect_delay: int = Field(default=5000, description="重连延迟(毫秒)")
    compression: int = Field(default=0, description="压缩级别")
    pool_size: int = Field(default=3, description="连接池大小")
    mode: str = Field(default="single", description="连接模式: single/pool")
    proxy: BinanceSDKProxySettings = Field(default_factory=BinanceSDKProxySettings)
    
    @model_validator(mode='after')
    def validate_websocket_settings(self):
        if self.timeout <= 0:
            raise ValueError("timeout必须大于0")
        if self.reconnect_delay < 0:
            raise ValueError("reconnect_delay不能小于0")
        if self.compression < 0:
            raise ValueError("compression不能小于0")
        if self.pool_size <= 0:
            raise ValueError("pool_size必须大于0")
        if self.mode not in ['single', 'pool']:
            raise ValueError(f"不支持的连接模式: {self.mode}")
        return self


class BinanceSDKWebSocketAPISettings(BinanceSDKWebSocketSettings):
    """Binance SDK WebSocket API配置"""
    pass


class BinanceSDKWebSocketStreamsSettings(BinanceSDKWebSocketSettings):
    """Binance SDK WebSocket流配置"""
    user_agent: str = Field(default="ai-hedge-fund-crypto/1.0", description="用户代理")


class BinanceSDKNetworkSettings(BaseModel):
    """Binance SDK网络配置"""
    https_agent: Dict[str, Any] = Field(
        default_factory=lambda: {
            "verify_ssl": True,
            "custom_ca_path": None
        }
    )
    
    def get_https_context(self) -> Optional[ssl.SSLContext]:
        """获取HTTPS SSL上下文"""
        try:
            context = ssl.create_default_context()
            
            if not self.https_agent.get("verify_ssl", True):
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                
            custom_ca_path = self.https_agent.get("custom_ca_path")
            if custom_ca_path and os.path.exists(custom_ca_path):
                context.load_verify_locations(custom_ca_path)
                
            return context
        except Exception as e:
            logger.error(f"创建HTTPS上下文失败: {e}")
            return None


class BinanceSDKEndpointsSettings(BaseModel):
    """Binance SDK端点配置"""
    rest_api_base_url: Optional[str] = Field(default=None, description="REST API基础URL")
    websocket_api_url: Optional[str] = Field(default=None, description="WebSocket API URL")
    websocket_streams_url: Optional[str] = Field(default=None, description="WebSocket流URL")


class BinanceSDKSettings(BaseModel):
    """Binance官方SDK配置"""
    api: BinanceSDKAPISettings = Field(default_factory=BinanceSDKAPISettings)
    environment: BinanceSDKEnvironmentSettings = Field(default_factory=BinanceSDKEnvironmentSettings)
    rest_api: BinanceSDKRestAPISettings = Field(default_factory=BinanceSDKRestAPISettings)
    websocket_api: BinanceSDKWebSocketAPISettings = Field(default_factory=BinanceSDKWebSocketAPISettings)
    websocket_streams: BinanceSDKWebSocketStreamsSettings = Field(default_factory=BinanceSDKWebSocketStreamsSettings)
    network: BinanceSDKNetworkSettings = Field(default_factory=BinanceSDKNetworkSettings)
    endpoints: BinanceSDKEndpointsSettings = Field(default_factory=BinanceSDKEndpointsSettings)
    
    def validate_credentials(self) -> bool:
        """验证API凭证是否配置"""
        credentials = self.api.get_resolved_credentials()
        return bool(credentials.get("api_key")) and bool(credentials.get("api_secret"))
    
    def get_effective_environment(self) -> str:
        """获取实际使用的环境"""
        if self.environment.auto_detect:
            # 可以通过API凭证或其他逻辑自动检测环境
            # 这里简化处理，直接返回配置的环境
            pass
        return self.environment.type
    
    def create_rest_api_config_dict(self) -> Dict[str, Any]:
        """创建REST API配置字典"""
        credentials = self.api.get_resolved_credentials()
        config = {
            "api_key": credentials.get("api_key"),
            "api_secret": credentials.get("api_secret"),
            "timeout": self.rest_api.timeout,
            "retries": self.rest_api.retries,
            "backoff": self.rest_api.backoff,
            "keep_alive": self.rest_api.keep_alive,
            "compression": self.rest_api.compression,
        }
        
        # 添加代理配置
        proxy_config = self.rest_api.proxy.to_proxy_dict()
        if proxy_config:
            config["proxy"] = proxy_config
            
        # 添加HTTPS配置
        https_context = self.network.get_https_context()
        if https_context:
            config["https_agent"] = https_context
            
        # 添加私钥配置
        if self.api.private_key_path:
            private_key_content = self.api.get_private_key_content()
            if private_key_content:
                config["private_key"] = private_key_content
                if self.api.private_key_passphrase:
                    config["private_key_passphrase"] = self.api.private_key_passphrase
                    
        # 添加自定义端点
        if self.endpoints.rest_api_base_url:
            config["base_path"] = self.endpoints.rest_api_base_url
            
        return config
    
    def create_websocket_api_config_dict(self) -> Dict[str, Any]:
        """创建WebSocket API配置字典"""
        credentials = self.api.get_resolved_credentials()
        config = {
            "api_key": credentials.get("api_key"),
            "api_secret": credentials.get("api_secret"),
            "timeout": self.websocket_api.timeout,
            "reconnect_delay": self.websocket_api.reconnect_delay,
            "compression": self.websocket_api.compression,
            "mode": self.websocket_api.mode,
            "pool_size": self.websocket_api.pool_size,
        }
        
        # 添加代理配置
        proxy_config = self.websocket_api.proxy.to_proxy_dict()
        if proxy_config:
            config["proxy"] = proxy_config
            
        # 添加HTTPS配置
        https_context = self.network.get_https_context()
        if https_context:
            config["https_agent"] = https_context
            
        # 添加私钥配置
        if self.api.private_key_path:
            private_key_content = self.api.get_private_key_content()
            if private_key_content:
                config["private_key"] = private_key_content
                if self.api.private_key_passphrase:
                    config["private_key_passphrase"] = self.api.private_key_passphrase
                    
        # 添加自定义端点
        if self.endpoints.websocket_api_url:
            config["stream_url"] = self.endpoints.websocket_api_url
            
        return config
    
    def create_websocket_streams_config_dict(self) -> Dict[str, Any]:
        """创建WebSocket流配置字典"""
        config = {
            "reconnect_delay": self.websocket_streams.reconnect_delay,
            "compression": self.websocket_streams.compression,
            "mode": self.websocket_streams.mode,
            "pool_size": self.websocket_streams.pool_size,
        }
        
        # 添加代理配置
        proxy_config = self.websocket_streams.proxy.to_proxy_dict()
        if proxy_config:
            config["proxy"] = proxy_config
            
        # 添加HTTPS配置
        https_context = self.network.get_https_context()
        if https_context:
            config["https_agent"] = https_context
            
        # 添加自定义端点
        if self.endpoints.websocket_streams_url:
            config["stream_url"] = self.endpoints.websocket_streams_url
            
        return config


class Settings(BaseSettings):
    mode: str
    start_date: datetime
    end_date: datetime
    primary_interval: Interval
    initial_cash: int
    margin_requirement: float
    show_reasoning: bool
    show_agent_graph: bool = True
    signals: SignalSettings
    model: ModelSettings
    # 期货配置（可选）
    futures: Optional[FuturesSettings] = None
    # 环境配置（可选）
    environment: Optional[EnvironmentSettings] = None
    # Binance官方SDK配置（可选）
    binance_sdk: Optional[BinanceSDKSettings] = None
    
    def get_effective_futures_environment(self) -> str:
        """获取实际的期货交易环境"""
        if self.environment:
            profile = self.environment.get_active_profile()
            return profile.futures_environment
        elif self.futures:
            return self.futures.environment
        return "testnet"  # 默认安全环境
    
    def get_effective_log_level(self) -> str:
        """获取实际的日志级别"""
        if self.environment:
            profile = self.environment.get_active_profile()
            return profile.log_level
        elif self.futures and self.futures.monitoring:
            return self.futures.monitoring.log_level
        return "INFO"  # 默认日志级别
    
    def switch_environment(self, env_name: str) -> None:
        """切换环境"""
        if self.environment and env_name in self.environment.profiles:
            self.environment.active = env_name
            logger.info(f"已切换到环境: {env_name}")
        else:
            logger.warning(f"环境 {env_name} 不存在")
    
    def get_binance_sdk_config(self) -> Optional[BinanceSDKSettings]:
        """获取Binance SDK配置"""
        return self.binance_sdk
    
    def validate_binance_sdk_credentials(self) -> bool:
        """验证Binance SDK凭证"""
        if not self.binance_sdk:
            return False
        return self.binance_sdk.validate_credentials()
    
    def get_sdk_environment_compatibility(self) -> Dict[str, Any]:
        """检查SDK配置与期货配置的兼容性"""
        compatibility_info = {
            "sdk_configured": self.binance_sdk is not None,
            "futures_configured": self.futures is not None,
            "environment_match": False,
            "credentials_valid": False
        }
        
        if self.binance_sdk:
            compatibility_info["credentials_valid"] = self.binance_sdk.validate_credentials()
            sdk_env = self.binance_sdk.get_effective_environment()
            
            if self.futures:
                futures_env = self.get_effective_futures_environment()
                compatibility_info["environment_match"] = sdk_env == futures_env
                compatibility_info["sdk_environment"] = sdk_env
                compatibility_info["futures_environment"] = futures_env
            else:
                compatibility_info["sdk_environment"] = sdk_env
                
        return compatibility_info

    @model_validator(mode='after')
    def check_primary_interval_in_intervals(self):
        if self.primary_interval not in self.signals.intervals:
            raise ValueError(
                f"primary_interval '{self.primary_interval}' must be in signals.intervals {self.signals.intervals}")
        return self

    @model_validator(mode='after') 
    def validate_futures_config(self):
        """验证期货配置"""
        if self.futures:
            # 获取实际的期货环境
            effective_env = self.get_effective_futures_environment()
            
            # 验证环境配置
            if effective_env not in ['testnet', 'mainnet']:
                raise ValueError(f"不支持的期货环境: {effective_env}")
            
            # 验证保证金类型
            if self.futures.margin_type not in ['CROSSED', 'ISOLATED']:
                raise ValueError(f"不支持的保证金类型: {self.futures.margin_type}")
            
            # 在生产模式下使用mainnet时发出警告
            if self.mode == 'live' and effective_env == 'mainnet':
                logger.warning(
                    "在生产模式下使用mainnet环境，请确保已充分测试！"
                )
            
            # 验证API凭证
            if self.mode == 'live' and not self.futures.validate_api_credentials():
                raise ValueError(
                    f"实盘模式需要配置API凭证: {self.futures.security.api_key_env}, "
                    f"{self.futures.security.secret_key_env}"
                )
        
        return self
    
    @model_validator(mode='after')
    def validate_environment_config(self):
        """验证环境配置"""
        if self.environment:
            # 验证激活的环境是否存在
            if self.environment.active not in self.environment.profiles:
                raise ValueError(f"激活的环境 '{self.environment.active}' 不存在于配置中")
            
            # 同步期货环境配置
            if self.futures:
                profile = self.environment.get_active_profile()
                if profile.futures_environment != self.futures.environment:
                    logger.warning(
                        f"环境配置中的期货环境 ({profile.futures_environment}) 与直接配置的期货环境 "
                        f"({self.futures.environment}) 不匹配，将使用环境配置中的设置"
                    )
                    self.futures.environment = profile.futures_environment
        
        return self
    
    @model_validator(mode='after')
    def validate_binance_sdk_config(self):
        """验证Binance SDK配置"""
        if self.binance_sdk:
            # 验证环境配置兼容性
            compatibility = self.get_sdk_environment_compatibility()
            
            if compatibility["futures_configured"] and not compatibility["environment_match"]:
                logger.warning(
                    f"Binance SDK环境 ({compatibility['sdk_environment']}) 与期货配置环境 "
                    f"({compatibility['futures_environment']}) 不匹配"
                )
            
            # 在实盘模式下验证凭证
            if self.mode == 'live':
                if not compatibility["credentials_valid"]:
                    raise ValueError("实盘模式需要有效的Binance SDK API凭证")
                    
                # 在mainnet环境下发出警告
                if self.binance_sdk.get_effective_environment() == 'mainnet':
                    logger.warning("在实盘模式下使用mainnet环境，请确保已充分测试！")
        
        return self


def load_settings(yaml_path: str = "config.yaml") -> Settings:
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    return Settings(**yaml_data)


def load_settings_with_futures_validation(yaml_path: str = "config.yaml") -> Settings:
    """
    加载设置并进行期货配置验证
    
    Args:
        yaml_path: 配置文件路径
        
    Returns:
        Settings: 验证后的设置对象
    """
    try:
        # 加载基础配置
        settings = load_settings(yaml_path)
        
        # 如果包含期货配置，进行额外验证
        if settings.futures:
            from src.futures.config.config_validator import FuturesConfigValidator
            
            validator = FuturesConfigValidator()
            
            # 重新加载原始配置数据进行期货验证
            with open(yaml_path, "r", encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            # 执行期货配置验证
            futures_config = validator.validate_config(yaml_data)
            
            logger.info(
                f"期货配置验证成功: {futures_config.environment}, "
                f"交易对: {len(futures_config.symbols)}"
            )
            
            # 验证API端点连接性（可选）
            if settings.mode == 'live':
                _validate_api_connectivity(settings.futures)
        
        return settings
        
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        raise


def _validate_api_connectivity(futures_config: FuturesSettings) -> None:
    """
    验证API端点连接性
    
    Args:
        futures_config: 期货配置
    """
    try:
        import requests

        api_endpoints = futures_config.get_api_endpoints()

        # 测试API连接 - 直接使用完整的ping端点URL
        test_url = f"{api_endpoints.api_url}/ping"
        response = requests.get(
            test_url,
            timeout=futures_config.api.timeout,
            verify=futures_config.security.verify_ssl
        )
        
        if response.status_code == 200:
            logger.info(f"API连接测试成功: {api_endpoints.base_url}")
        else:
            logger.warning(f"API连接测试失败: {response.status_code}")
            
    except Exception as e:
        logger.warning(f"API连接性验证失败: {e}")


def reload_settings(yaml_path: str = "config.yaml") -> Settings:
    """
    重新加载配置
    
    Args:
        yaml_path: 配置文件路径
        
    Returns:
        Settings: 新的设置对象
    """
    logger.info(f"重新加载配置文件: {yaml_path}")
    return load_settings_with_futures_validation(yaml_path)


def get_environment_config(settings: Settings) -> Dict[str, Any]:
    """
    获取环境相关配置摘要
    
    Args:
        settings: 设置对象
        
    Returns:
        Dict[str, Any]: 环境配置摘要
    """
    env_info = {
        "mode": settings.mode,
        "futures_environment": settings.get_effective_futures_environment(),
        "log_level": settings.get_effective_log_level(),
    }
    
    if settings.environment:
        env_info["active_profile"] = settings.environment.active
        env_info["available_profiles"] = list(settings.environment.profiles.keys())
    
    if settings.futures:
        env_info["api_credentials_configured"] = settings.futures.validate_api_credentials()
        api_endpoints = settings.futures.get_api_endpoints()
        env_info["api_base_url"] = api_endpoints.base_url
    
    return env_info


# Load and use
settings = load_settings_with_futures_validation()

# print(settings.model.name)
# print(settings.model.provider)
# print(settings.mode)
# print(settings.primary_interval)
# print(settings.start_date)
# print(settings.end_date)
# print(settings.signals)
