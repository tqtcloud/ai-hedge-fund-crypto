"""
期货API安全管理器

提供安全管理Binance期货API密钥的功能。
支持testnet/mainnet不同的API配置，API权限和有效性检查，环境变量安全加载。
"""

import os
import logging
import asyncio
import hashlib
import hmac
import time
from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
from enum import Enum
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from cryptography.fernet import Fernet

from ...utils.exceptions import ValidationError


logger = logging.getLogger(__name__)


class APIEnvironment(Enum):
    """API环境枚举"""
    TESTNET = "testnet"
    MAINNET = "mainnet"


class APIPermission(Enum):
    """API权限枚举"""
    SPOT = "SPOT"
    FUTURES = "FUTURES"
    MARGIN = "MARGIN"
    OPTIONS = "OPTIONS"


@dataclass
class APICredentials:
    """API凭证数据类"""
    api_key: str
    api_secret: str
    environment: APIEnvironment
    permissions: list[APIPermission]
    is_encrypted: bool = False
    
    def __post_init__(self):
        """初始化后验证"""
        if not self.api_key or not self.api_secret:
            raise ValidationError("API密钥和秘钥不能为空")
        
        # 验证API密钥格式
        if len(self.api_key) < 20:
            raise ValidationError("API密钥格式不正确")
        
        if len(self.api_secret) < 20:
            raise ValidationError("API秘钥格式不正确")


@dataclass
class APIEndpoints:
    """API端点配置"""
    base_url: str
    futures_url: str
    websocket_url: str
    websocket_futures_url: str


class FuturesAPIManager:
    """期货API安全管理器"""
    
    # API端点配置
    ENDPOINTS = {
        APIEnvironment.TESTNET: APIEndpoints(
            base_url="https://testnet.binance.vision/api",
            futures_url="https://testnet.binancefuture.com/fapi",
            websocket_url="wss://testnet.binance.vision/ws",
            websocket_futures_url="wss://testnet.binancefuture.com/ws-fapi/v1"
        ),
        APIEnvironment.MAINNET: APIEndpoints(
            base_url="https://api.binance.com/api",
            futures_url="https://fapi.binance.com/fapi",
            websocket_url="wss://stream.binance.com:9443/ws",
            websocket_futures_url="wss://fstream.binance.com/ws"
        )
    }
    
    def __init__(self, environment: APIEnvironment = APIEnvironment.TESTNET):
        """
        初始化API管理器
        
        Args:
            environment: API环境，默认为testnet确保安全
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.environment = environment
        self.endpoints = self.ENDPOINTS[environment]
        self._credentials: Optional[APICredentials] = None
        self._encryption_key: Optional[bytes] = None
        
        # 加载环境变量
        load_dotenv()
        
        self.logger.info(f"初始化期货API管理器，环境: {environment.value}")

    def load_credentials_from_env(self, prefix: str = "BINANCE") -> APICredentials:
        """
        从环境变量加载API凭证
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            APICredentials: API凭证对象
            
        Raises:
            ValidationError: 凭证加载或验证失败
        """
        try:
            # 根据环境选择对应的环境变量
            if self.environment == APIEnvironment.TESTNET:
                api_key_env = f"{prefix}_FUTURES_TESTNET_API_KEY"
                api_secret_env = f"{prefix}_FUTURES_TESTNET_API_SECRET"
            else:
                api_key_env = f"{prefix}_FUTURES_API_KEY"
                api_secret_env = f"{prefix}_FUTURES_API_SECRET"
            
            api_key = os.getenv(api_key_env)
            api_secret = os.getenv(api_secret_env)
            
            # 如果期货专用变量不存在，回退到通用变量
            if not api_key:
                api_key = os.getenv(f"{prefix}_API_KEY")
            if not api_secret:
                api_secret = os.getenv(f"{prefix}_API_SECRET")
            
            if not api_key or not api_secret:
                raise ValidationError(
                    f"未找到API凭证环境变量: {api_key_env}, {api_secret_env}"
                )
            
            credentials = APICredentials(
                api_key=api_key,
                api_secret=api_secret,
                environment=self.environment,
                permissions=[APIPermission.FUTURES]  # 默认期货权限
            )
            
            self._credentials = credentials
            self.logger.info("从环境变量成功加载API凭证")
            return credentials
            
        except Exception as e:
            self.logger.error(f"从环境变量加载API凭证失败: {e}")
            raise ValidationError(f"API凭证加载失败: {e}")

    def load_credentials_from_file(self, file_path: str, encrypted: bool = False) -> APICredentials:
        """
        从文件加载API凭证
        
        Args:
            file_path: 凭证文件路径
            encrypted: 文件是否加密
            
        Returns:
            APICredentials: API凭证对象
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise ValidationError(f"凭证文件不存在: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if encrypted:
                if not self._encryption_key:
                    raise ValidationError("需要设置加密密钥才能读取加密文件")
                content = self._decrypt_content(content)
            
            # 解析凭证内容 (假设格式为 key=value)
            lines = content.strip().split('\n')
            cred_data = {}
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    cred_data[key.strip()] = value.strip()
            
            credentials = APICredentials(
                api_key=cred_data.get('api_key', ''),
                api_secret=cred_data.get('api_secret', ''),
                environment=self.environment,
                permissions=[APIPermission.FUTURES],
                is_encrypted=encrypted
            )
            
            self._credentials = credentials
            self.logger.info("从文件成功加载API凭证")
            return credentials
            
        except Exception as e:
            self.logger.error(f"从文件加载API凭证失败: {e}")
            raise ValidationError(f"API凭证文件加载失败: {e}")

    def set_encryption_key(self, key: Optional[str] = None) -> None:
        """
        设置加密密钥
        
        Args:
            key: 加密密钥，如果为None则自动生成
        """
        if key:
            self._encryption_key = key.encode()
        else:
            self._encryption_key = Fernet.generate_key()
        
        self.logger.info("加密密钥已设置")

    def _decrypt_content(self, encrypted_content: str) -> str:
        """解密内容"""
        if not self._encryption_key:
            raise ValidationError("未设置加密密钥")
        
        fernet = Fernet(self._encryption_key)
        decrypted = fernet.decrypt(encrypted_content.encode())
        return decrypted.decode()

    def _encrypt_content(self, content: str) -> str:
        """加密内容"""
        if not self._encryption_key:
            raise ValidationError("未设置加密密钥")
        
        fernet = Fernet(self._encryption_key)
        encrypted = fernet.encrypt(content.encode())
        return encrypted.decode()

    async def validate_api_credentials(self) -> Dict[str, Any]:
        """
        验证API凭证有效性
        
        Returns:
            Dict[str, Any]: 验证结果和账户信息
            
        Raises:
            ValidationError: 凭证验证失败
        """
        if not self._credentials:
            raise ValidationError("未加载API凭证")
        
        try:
            # 构建请求参数
            timestamp = int(time.time() * 1000)
            params = {
                'timestamp': timestamp,
                'recvWindow': 10000
            }
            
            # 生成签名
            query_string = self._build_query_string(params)
            signature = self._generate_signature(query_string)
            params['signature'] = signature
            
            # 构建请求头
            headers = {
                'X-MBX-APIKEY': self._credentials.api_key,
                'Content-Type': 'application/json'
            }
            
            # 发送验证请求
            url = f"{self.endpoints.futures_url}/v2/account"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        account_info = await response.json()
                        self.logger.info("API凭证验证成功")
                        
                        return {
                            'valid': True,
                            'environment': self.environment.value,
                            'account_info': account_info,
                            'permissions': [p.value for p in self._credentials.permissions]
                        }
                    else:
                        error_data = await response.json()
                        raise ValidationError(f"API验证失败: {error_data}")
                        
        except aiohttp.ClientError as e:
            self.logger.error(f"API请求失败: {e}")
            raise ValidationError(f"API连接失败: {e}")
        except Exception as e:
            self.logger.error(f"API凭证验证失败: {e}")
            raise ValidationError(f"API凭证验证失败: {e}")

    def validate_api_credentials_sync(self) -> Dict[str, Any]:
        """
        同步验证API凭证有效性
        
        Returns:
            Dict[str, Any]: 验证结果和账户信息
        """
        return asyncio.run(self.validate_api_credentials())

    def _build_query_string(self, params: Dict[str, Any]) -> str:
        """构建查询字符串"""
        return '&'.join([f"{k}={v}" for k, v in sorted(params.items()) if v is not None])

    def _generate_signature(self, query_string: str) -> str:
        """生成HMAC签名"""
        if not self._credentials:
            raise ValidationError("未加载API凭证")
        
        return hmac.new(
            self._credentials.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def check_api_permissions(self) -> Dict[str, Any]:
        """
        检查API权限
        
        Returns:
            Dict[str, Any]: 权限检查结果
        """
        if not self._credentials:
            raise ValidationError("未加载API凭证")
        
        try:
            # 检查期货交易权限
            futures_permissions = await self._check_futures_permissions()
            
            # 检查其他权限
            other_permissions = await self._check_other_permissions()
            
            return {
                'futures': futures_permissions,
                'other': other_permissions,
                'environment': self.environment.value
            }
            
        except Exception as e:
            self.logger.error(f"权限检查失败: {e}")
            raise ValidationError(f"API权限检查失败: {e}")

    async def _check_futures_permissions(self) -> Dict[str, bool]:
        """检查期货权限"""
        permissions = {
            'spot_and_margin_trading': False,
            'futures_trading': False,
            'wallet_management': False
        }
        
        try:
            # 尝试获取期货账户信息
            account_info = await self.validate_api_credentials()
            if account_info['valid']:
                permissions['futures_trading'] = True
                permissions['wallet_management'] = True
                
        except Exception:
            pass
        
        return permissions

    async def _check_other_permissions(self) -> Dict[str, bool]:
        """检查其他权限"""
        # 此处可以添加对其他权限的检查
        return {}

    def get_api_endpoints(self) -> APIEndpoints:
        """
        获取当前环境的API端点
        
        Returns:
            APIEndpoints: API端点配置
        """
        return self.endpoints

    def switch_environment(self, new_environment: APIEnvironment) -> None:
        """
        切换API环境
        
        Args:
            new_environment: 新环境
        """
        self.environment = new_environment
        self.endpoints = self.ENDPOINTS[new_environment]
        self._credentials = None  # 清空凭证，需要重新加载
        
        self.logger.info(f"API环境已切换至: {new_environment.value}")

    def get_request_headers(self) -> Dict[str, str]:
        """
        获取API请求头
        
        Returns:
            Dict[str, str]: 请求头字典
        """
        if not self._credentials:
            raise ValidationError("未加载API凭证")
        
        return {
            'X-MBX-APIKEY': self._credentials.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'ai-hedge-fund-crypto/1.0.0'
        }

    def create_signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建带签名的请求参数
        
        Args:
            params: 原始参数
            
        Returns:
            Dict[str, Any]: 带签名的参数
        """
        if not self._credentials:
            raise ValidationError("未加载API凭证")
        
        # 添加时间戳和接收窗口
        timestamp = int(time.time() * 1000)
        params['timestamp'] = timestamp
        params['recvWindow'] = 10000
        
        # 生成签名
        query_string = self._build_query_string(params)
        signature = self._generate_signature(query_string)
        params['signature'] = signature
        
        return params

    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        掩码敏感数据用于日志记录
        
        Args:
            data: 原始数据
            
        Returns:
            Dict[str, Any]: 掩码后的数据
        """
        masked_data = data.copy()
        
        sensitive_keys = ['api_key', 'api_secret', 'signature', 'password']
        
        for key in sensitive_keys:
            if key in masked_data:
                value = str(masked_data[key])
                if len(value) > 8:
                    masked_data[key] = value[:4] + '*' * (len(value) - 8) + value[-4:]
                else:
                    masked_data[key] = '*' * len(value)
        
        return masked_data

    def get_credentials_status(self) -> Dict[str, Any]:
        """
        获取凭证状态信息
        
        Returns:
            Dict[str, Any]: 凭证状态
        """
        if not self._credentials:
            return {
                'loaded': False,
                'environment': self.environment.value,
                'encrypted': False,
                'permissions': []
            }
        
        return {
            'loaded': True,
            'environment': self._credentials.environment.value,
            'encrypted': self._credentials.is_encrypted,
            'permissions': [p.value for p in self._credentials.permissions],
            'api_key_prefix': self._credentials.api_key[:8] + '...' if len(self._credentials.api_key) > 8 else 'short_key'
        }