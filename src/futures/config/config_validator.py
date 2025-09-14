"""
期货配置验证器

提供期货交易相关配置的完整性和正确性验证功能。
支持testnet/mainnet环境验证，验证保证金、杠杆等期货特有参数。
"""

import logging
import re
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from decimal import Decimal
from enum import Enum
import yaml
from urllib.parse import urlparse

from ...utils.exceptions import ValidationError
from ...utils.settings import BinanceSDKSettings


logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """交易模式枚举"""
    BACKTEST = "backtest"
    LIVE = "live"
    PAPER = "paper"  # 模拟交易


class MarginType(Enum):
    """保证金类型枚举"""
    ISOLATED = "ISOLATED"  # 逐仓
    CROSSED = "CROSSED"    # 全仓


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "MARKET"
    LIMIT = "LIMIT" 
    STOP = "STOP"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


@dataclass
class FuturesRiskConfig:
    """期货风险控制配置"""
    max_leverage: int  # 最大杠杆倍数
    max_position_size: float  # 最大仓位大小（基准货币）
    stop_loss_percentage: float  # 止损百分比
    take_profit_percentage: float  # 止盈百分比
    daily_loss_limit: float  # 日损失限额
    max_open_positions: int  # 最大开仓数量
    margin_requirement: float  # 保证金要求比例
    
    def __post_init__(self):
        """初始化后验证"""
        if self.max_leverage < 1 or self.max_leverage > 125:
            raise ValidationError("max_leverage必须在1-125之间")
        if self.stop_loss_percentage <= 0 or self.stop_loss_percentage >= 1:
            raise ValidationError("stop_loss_percentage必须在0-1之间")
        if self.take_profit_percentage <= 0:
            raise ValidationError("take_profit_percentage必须大于0")


@dataclass
class FuturesAPIEndpoint:
    """期货API端点配置"""
    base_url: str
    api_url: str
    websocket_url: str
    websocket_stream_url: str
    
    def __post_init__(self):
        """API端点验证"""
        urls = [self.base_url, self.api_url, self.websocket_url, self.websocket_stream_url]
        for url in urls:
            if not self._validate_url(url):
                raise ValidationError(f"无效的URL格式: {url}")
    
    @staticmethod
    def _validate_url(url: str) -> bool:
        """验证URL格式"""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme) and bool(parsed.netloc)
        except Exception:
            return False


@dataclass
class FuturesSecurityConfig:
    """期货安全配置"""
    api_key_env: str
    secret_key_env: str
    enable_signature: bool = True
    recv_window: int = 5000
    verify_ssl: bool = True
    enable_ip_whitelist: bool = False
    
    def __post_init__(self):
        """安全配置验证"""
        if self.recv_window <= 0 or self.recv_window > 60000:
            raise ValidationError("recv_window必须在0-60000毫秒之间")


@dataclass  
class FuturesWebSocketConfig:
    """期货WebSocket配置"""
    enable_user_data_stream: bool = True
    enable_market_data_stream: bool = True
    reconnect_interval: int = 5  # 重连间隔（秒）
    max_reconnect_attempts: int = 10
    heartbeat_interval: int = 30  # 心跳间隔（秒）
    buffer_size: int = 1024  # 缓冲区大小
    connection_timeout: int = 10
    ping_interval: int = 20
    pong_timeout: int = 10


@dataclass
class FuturesConfig:
    """期货交易配置"""
    # 基础配置
    trading_mode: TradingMode
    environment: str  # testnet | mainnet
    margin_type: MarginType
    
    # API端点配置
    api_endpoints: Dict[str, FuturesAPIEndpoint]  # testnet/mainnet endpoints
    
    # 安全配置
    security: FuturesSecurityConfig
    
    # 风险控制
    risk: FuturesRiskConfig
    
    # WebSocket配置
    websocket: FuturesWebSocketConfig
    
    # 交易对配置
    symbols: List[str]  # 交易对列表
    default_leverage: Dict[str, int]  # 默认杠杆配置
    
    # 订单配置
    allowed_order_types: List[OrderType]
    min_order_size: Dict[str, float]  # 最小订单大小
    
    # 其他配置
    price_precision: Dict[str, int]  # 价格精度
    quantity_precision: Dict[str, int]  # 数量精度
    
    # API配置（有默认值）
    api_timeout: int = 10
    api_max_retries: int = 3
    api_retry_delay: float = 1.0
    
    def get_current_api_endpoint(self) -> FuturesAPIEndpoint:
        """获取当前API端点"""
        return self.api_endpoints.get(self.environment)
    
    def validate_api_credentials(self) -> bool:
        """验证API凭证是否存在"""
        api_key = os.getenv(self.security.api_key_env)
        secret_key = os.getenv(self.security.secret_key_env)
        return bool(api_key) and bool(secret_key)


class FuturesConfigValidator:
    """期货配置验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 支持的交易对列表 (可以从API获取)
        self.supported_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT",
            "XRPUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT", "UNIUSDT",
            "SOLUSDT", "AVAXUSDT", "MATICUSDT", "ATOMUSDT", "FILUSDT"
        ]
        
        # 杠杆限制配置
        self.leverage_limits = {
            "BTCUSDT": 125,
            "ETHUSDT": 100, 
            "BNBUSDT": 50,
            "ADAUSDT": 50,
            "DOTUSDT": 50,
            "XRPUSDT": 75,
            "default": 20
        }
        
        # 最小订单大小
        self.min_order_sizes = {
            "BTCUSDT": 0.001,
            "ETHUSDT": 0.01,
            "BNBUSDT": 0.1,
            "ADAUSDT": 1.0,
            "XRPUSDT": 10.0,
            "default": 0.001
        }
        
        # 安全配置验证模式
        self.security_patterns = {
            "api_key": re.compile(r'^[A-Za-z0-9_-]+$'),
            "secret_key": re.compile(r'^[A-Za-z0-9+/=]+$')
        }

    def validate_config(self, config_data: Dict[str, Any]) -> FuturesConfig:
        """
        验证期货配置数据
        
        Args:
            config_data: 原始配置数据
            
        Returns:
            FuturesConfig: 验证后的配置对象
            
        Raises:
            ValidationError: 配置验证失败
        """
        try:
            self.logger.info("开始验证期货配置")
            
            # 1. 基础配置验证
            self._validate_basic_config(config_data)
            
            # 2. 环境配置验证
            self._validate_environment_config(config_data)
            
            # 3. 风险配置验证
            risk_config = self._validate_risk_config(config_data.get('futures', {}).get('risk', {}))
            
            # 4. WebSocket配置验证
            ws_config = self._validate_websocket_config(config_data.get('futures', {}).get('websocket', {}))
            
            # 5. 交易对配置验证
            self._validate_symbols_config(config_data)
            
            # 6. 杠杆配置验证
            self._validate_leverage_config(config_data)
            
            # 7. 订单配置验证
            self._validate_order_config(config_data)
            
            # 8. Binance SDK配置验证（如果存在）
            self._validate_binance_sdk_config(config_data)
            
            # 构建配置对象
            futures_data = config_data.get('futures', {})
            
            # 构建API端点配置
            api_endpoints = {}
            api_data = futures_data.get('api', {})
            
            testnet_config = api_data.get('testnet', {})
            api_endpoints['testnet'] = FuturesAPIEndpoint(
                base_url=testnet_config.get('base_url', 'https://testnet.binancefuture.com'),
                api_url=testnet_config.get('api_url', 'https://testnet.binancefuture.com/fapi/v1'),
                websocket_url=testnet_config.get('websocket_url', 'wss://stream.binancefuture.com/ws'),
                websocket_stream_url=testnet_config.get('websocket_stream_url', 'wss://stream.binancefuture.com/stream')
            )
            
            mainnet_config = api_data.get('mainnet', {})
            api_endpoints['mainnet'] = FuturesAPIEndpoint(
                base_url=mainnet_config.get('base_url', 'https://fapi.binance.com'),
                api_url=mainnet_config.get('api_url', 'https://fapi.binance.com/fapi/v1'),
                websocket_url=mainnet_config.get('websocket_url', 'wss://fstream.binance.com/ws'),
                websocket_stream_url=mainnet_config.get('websocket_stream_url', 'wss://fstream.binance.com/stream')
            )
            
            # 构建安全配置
            security_data = futures_data.get('security', {})
            security_config = FuturesSecurityConfig(
                api_key_env=security_data.get('api_key_env', 'BINANCE_FUTURES_API_KEY'),
                secret_key_env=security_data.get('secret_key_env', 'BINANCE_FUTURES_SECRET_KEY'),
                enable_signature=security_data.get('enable_signature', True),
                recv_window=security_data.get('recv_window', 5000),
                verify_ssl=security_data.get('verify_ssl', True),
                enable_ip_whitelist=security_data.get('enable_ip_whitelist', False)
            )
            
            futures_config = FuturesConfig(
                trading_mode=TradingMode(config_data.get('mode', 'backtest')),
                environment=futures_data.get('environment', 'testnet'),
                margin_type=MarginType(futures_data.get('margin_type', 'CROSSED')),
                api_endpoints=api_endpoints,
                security=security_config,
                risk=risk_config,
                websocket=ws_config,
                symbols=futures_data.get('symbols', ['BTCUSDT']),
                default_leverage=futures_data.get('default_leverage', {}),
                allowed_order_types=[OrderType(ot) for ot in futures_data.get('allowed_order_types', ['MARKET', 'LIMIT'])],
                min_order_size=futures_data.get('min_order_size', {}),
                price_precision=futures_data.get('price_precision', {}),
                quantity_precision=futures_data.get('quantity_precision', {})
            )
            
            self.logger.info("期货配置验证成功")
            return futures_config
            
        except Exception as e:
            self.logger.error(f"期货配置验证失败: {e}")
            raise ValidationError(f"期货配置验证失败: {e}")

    def _validate_basic_config(self, config_data: Dict[str, Any]) -> None:
        """验证基础配置"""
        mode = config_data.get('mode')
        if mode not in ['backtest', 'live', 'paper']:
            raise ValidationError(f"不支持的交易模式: {mode}")
        
        if 'futures' not in config_data:
            raise ValidationError("缺少期货配置部分")

    def _validate_environment_config(self, config_data: Dict[str, Any]) -> None:
        """验证环境配置"""
        futures_config = config_data.get('futures', {})
        environment = futures_config.get('environment', 'testnet')
        
        if environment not in ['testnet', 'mainnet']:
            raise ValidationError(f"不支持的环境: {environment}")
        
        # 在生产模式下，建议使用testnet进行测试
        if environment == 'mainnet' and config_data.get('mode') == 'live':
            self.logger.warning("使用mainnet环境进行实盘交易，请确保已充分测试")

    def _validate_risk_config(self, risk_data: Dict[str, Any]) -> FuturesRiskConfig:
        """验证风险控制配置"""
        try:
            risk_config = FuturesRiskConfig(
                max_leverage=risk_data.get('max_leverage', 10),
                max_position_size=risk_data.get('max_position_size', 1000.0),
                stop_loss_percentage=risk_data.get('stop_loss_percentage', 0.02),
                take_profit_percentage=risk_data.get('take_profit_percentage', 0.05),
                daily_loss_limit=risk_data.get('daily_loss_limit', 500.0),
                max_open_positions=risk_data.get('max_open_positions', 5),
                margin_requirement=risk_data.get('margin_requirement', 0.1)
            )
            return risk_config
        except Exception as e:
            raise ValidationError(f"风险配置验证失败: {e}")

    def _validate_websocket_config(self, ws_data: Dict[str, Any]) -> FuturesWebSocketConfig:
        """验证WebSocket配置"""
        return FuturesWebSocketConfig(
            enable_user_data_stream=ws_data.get('enable_user_data_stream', True),
            enable_market_data_stream=ws_data.get('enable_market_data_stream', True),
            reconnect_interval=ws_data.get('reconnect_interval', 5),
            max_reconnect_attempts=ws_data.get('max_reconnect_attempts', 10),
            heartbeat_interval=ws_data.get('heartbeat_interval', 30),
            buffer_size=ws_data.get('buffer_size', 1024)
        )

    def _validate_symbols_config(self, config_data: Dict[str, Any]) -> None:
        """验证交易对配置"""
        futures_config = config_data.get('futures', {})
        symbols = futures_config.get('symbols', [])
        
        if not symbols:
            raise ValidationError("必须配置至少一个交易对")
        
        # 检查交易对是否受支持
        for symbol in symbols:
            if symbol not in self.supported_symbols:
                self.logger.warning(f"交易对 {symbol} 可能不受支持，请确认")

    def _validate_leverage_config(self, config_data: Dict[str, Any]) -> None:
        """验证杠杆配置"""
        futures_config = config_data.get('futures', {})
        default_leverage = futures_config.get('default_leverage', {})
        symbols = futures_config.get('symbols', [])
        
        for symbol in symbols:
            leverage = default_leverage.get(symbol, 1)
            max_leverage = self.leverage_limits.get(symbol, self.leverage_limits['default'])
            
            if leverage > max_leverage:
                raise ValidationError(f"交易对 {symbol} 的杠杆 {leverage} 超过最大限制 {max_leverage}")

    def _validate_order_config(self, config_data: Dict[str, Any]) -> None:
        """验证订单配置"""
        futures_config = config_data.get('futures', {})
        allowed_order_types = futures_config.get('allowed_order_types', [])
        
        if not allowed_order_types:
            raise ValidationError("必须配置至少一种订单类型")
        
        # 检查订单类型是否有效
        valid_order_types = [ot.value for ot in OrderType]
        for order_type in allowed_order_types:
            if order_type not in valid_order_types:
                raise ValidationError(f"无效的订单类型: {order_type}")

    def _validate_binance_sdk_config(self, config_data: Dict[str, Any]) -> None:
        """验证Binance SDK配置"""
        binance_sdk_data = config_data.get('binance_sdk')
        if not binance_sdk_data:
            self.logger.info("未配置Binance SDK，跳过SDK配置验证")
            return
        
        try:
            # 验证SDK配置的完整性
            sdk_config = BinanceSDKSettings(**binance_sdk_data)
            
            # 验证环境配置兼容性
            sdk_env = sdk_config.get_effective_environment()
            futures_env = config_data.get('futures', {}).get('environment', 'testnet')
            
            if sdk_env != futures_env:
                self.logger.warning(
                    f"Binance SDK环境 ({sdk_env}) 与期货配置环境 ({futures_env}) 不匹配"
                )
            
            # 验证API凭证格式
            self._validate_sdk_api_credentials(sdk_config)
            
            # 验证代理配置（如果启用）
            self._validate_sdk_proxy_configs(sdk_config)
            
            # 验证超时和重连配置
            self._validate_sdk_timeout_configs(sdk_config)
            
            self.logger.info("Binance SDK配置验证成功")
            
        except Exception as e:
            raise ValidationError(f"Binance SDK配置验证失败: {e}")

    def _validate_sdk_api_credentials(self, sdk_config: BinanceSDKSettings) -> None:
        """验证SDK API凭证配置"""
        credentials = sdk_config.api.get_resolved_credentials()
        
        # 检查凭证格式
        api_key = credentials.get("api_key")
        api_secret = credentials.get("api_secret")
        
        if api_key and len(api_key) < 32:
            self.logger.warning("API Key长度可能不正确，请检查配置")
        
        if api_secret and len(api_secret) < 32:
            self.logger.warning("API Secret长度可能不正确，请检查配置")
        
        # 检查私钥配置
        if sdk_config.api.private_key_path:
            if not os.path.exists(sdk_config.api.private_key_path):
                raise ValidationError(f"私钥文件不存在: {sdk_config.api.private_key_path}")
            
            # 尝试读取私钥内容验证格式
            try:
                private_key_content = sdk_config.api.get_private_key_content()
                if not private_key_content:
                    raise ValidationError("无法读取私钥文件内容")
                if "-----BEGIN" not in private_key_content:
                    raise ValidationError("私钥文件格式不正确")
            except Exception as e:
                raise ValidationError(f"私钥文件验证失败: {e}")

    def _validate_sdk_proxy_configs(self, sdk_config: BinanceSDKSettings) -> None:
        """验证SDK代理配置"""
        proxy_configs = [
            sdk_config.rest_api.proxy,
            sdk_config.websocket_api.proxy,
            sdk_config.websocket_streams.proxy
        ]
        
        for i, proxy_config in enumerate(proxy_configs):
            if not proxy_config.enabled:
                continue
                
            config_names = ["REST API", "WebSocket API", "WebSocket Streams"]
            config_name = config_names[i]
            
            if not proxy_config.host:
                raise ValidationError(f"{config_name} 代理配置缺少主机地址")
            
            if not proxy_config.port:
                raise ValidationError(f"{config_name} 代理配置缺少端口")
            
            if proxy_config.port <= 0 or proxy_config.port > 65535:
                raise ValidationError(f"{config_name} 代理端口 {proxy_config.port} 无效")
            
            if proxy_config.protocol not in ['http', 'https']:
                raise ValidationError(f"{config_name} 代理协议 {proxy_config.protocol} 不支持")
            
            # 验证认证配置
            if proxy_config.auth:
                if not proxy_config.auth.get("username") or not proxy_config.auth.get("password"):
                    self.logger.warning(f"{config_name} 代理认证配置不完整")

    def _validate_sdk_timeout_configs(self, sdk_config: BinanceSDKSettings) -> None:
        """验证SDK超时配置"""
        # 验证REST API超时配置
        rest_config = sdk_config.rest_api
        if rest_config.timeout <= 0:
            raise ValidationError(f"REST API超时时间必须大于0: {rest_config.timeout}")
        if rest_config.timeout > 300000:  # 5分钟
            self.logger.warning(f"REST API超时时间过长: {rest_config.timeout}ms")
        
        # 验证WebSocket超时配置
        ws_configs = [
            ("WebSocket API", sdk_config.websocket_api),
            ("WebSocket Streams", sdk_config.websocket_streams)
        ]
        
        for config_name, ws_config in ws_configs:
            if ws_config.timeout <= 0:
                raise ValidationError(f"{config_name}超时时间必须大于0: {ws_config.timeout}")
            if ws_config.timeout > 300000:  # 5分钟
                self.logger.warning(f"{config_name}超时时间过长: {ws_config.timeout}ms")
            
            if ws_config.reconnect_delay < 0:
                raise ValidationError(f"{config_name}重连延迟不能小于0: {ws_config.reconnect_delay}")
            if ws_config.reconnect_delay > 60000:  # 1分钟
                self.logger.warning(f"{config_name}重连延迟过长: {ws_config.reconnect_delay}ms")
            
            if ws_config.pool_size <= 0:
                raise ValidationError(f"{config_name}连接池大小必须大于0: {ws_config.pool_size}")
            if ws_config.pool_size > 10:
                self.logger.warning(f"{config_name}连接池大小过大: {ws_config.pool_size}")

    def validate_from_file(self, config_path: str) -> FuturesConfig:
        """
        从文件验证期货配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            FuturesConfig: 验证后的配置对象
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            return self.validate_config(config_data)
            
        except FileNotFoundError:
            raise ValidationError(f"配置文件不存在: {config_path}")
        except yaml.YAMLError as e:
            raise ValidationError(f"配置文件格式错误: {e}")

    def generate_default_config(self) -> Dict[str, Any]:
        """
        生成默认期货配置
        
        Returns:
            Dict[str, Any]: 默认配置数据
        """
        return {
            'mode': 'backtest',
            'futures': {
                'environment': 'testnet',  # 默认使用测试网络确保安全
                'margin_type': 'CROSSED',
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'default_leverage': {
                    'BTCUSDT': 10,
                    'ETHUSDT': 10
                },
                'allowed_order_types': ['MARKET', 'LIMIT'],
                'min_order_size': {
                    'BTCUSDT': 0.001,
                    'ETHUSDT': 0.01
                },
                'price_precision': {
                    'BTCUSDT': 2,
                    'ETHUSDT': 2
                },
                'quantity_precision': {
                    'BTCUSDT': 3,
                    'ETHUSDT': 2
                },
                'risk': {
                    'max_leverage': 20,
                    'max_position_size': 1000.0,
                    'stop_loss_percentage': 0.02,  # 2%
                    'take_profit_percentage': 0.05,  # 5%
                    'daily_loss_limit': 500.0,
                    'max_open_positions': 5,
                    'margin_requirement': 0.1  # 10%
                },
                'websocket': {
                    'enable_user_data_stream': True,
                    'enable_market_data_stream': True,
                    'reconnect_interval': 5,
                    'max_reconnect_attempts': 10,
                    'heartbeat_interval': 30,
                    'buffer_size': 1024
                }
            }
        }