"""
环境配置检查器

提供自动检测和切换testnet/mainnet环境的功能。
验证不同环境的配置完整性，提供环境状态报告。
"""

import os
import logging
import asyncio
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import yaml

from .api_manager import FuturesAPIManager, APIEnvironment
from .config_validator import FuturesConfigValidator
from ...utils.exceptions import ValidationError


logger = logging.getLogger(__name__)


class EnvironmentStatus(Enum):
    """环境状态枚举"""
    HEALTHY = "healthy"  # 健康
    DEGRADED = "degraded"  # 降级
    UNAVAILABLE = "unavailable"  # 不可用


class ConnectivityLevel(Enum):
    """连通性级别枚举"""
    FULL = "full"  # 完全连通
    LIMITED = "limited"  # 受限连通
    NONE = "none"  # 无连通


@dataclass
class EnvironmentHealth:
    """环境健康状态"""
    status: EnvironmentStatus
    connectivity: ConnectivityLevel
    api_responsive: bool
    websocket_available: bool
    latency_ms: float
    last_check: datetime
    error_message: Optional[str] = None


@dataclass
class EnvironmentReport:
    """环境报告"""
    environment: APIEnvironment
    health: EnvironmentHealth
    api_permissions: Dict[str, bool]
    configuration_valid: bool
    recommendations: List[str]
    warnings: List[str]


class EnvironmentChecker:
    """环境配置检查器"""
    
    # 环境检查超时时间（秒）
    CHECK_TIMEOUT = 10
    
    # 网络连通性检查地址
    CONNECTIVITY_HOSTS = {
        APIEnvironment.TESTNET: [
            ('testnet.binance.vision', 443),
            ('testnet.binancefuture.com', 443)
        ],
        APIEnvironment.MAINNET: [
            ('api.binance.com', 443),
            ('fapi.binance.com', 443)
        ]
    }
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_validator = FuturesConfigValidator()
        self.api_managers: Dict[APIEnvironment, FuturesAPIManager] = {}
        
        # 初始化API管理器
        for env in APIEnvironment:
            self.api_managers[env] = FuturesAPIManager(env)

    async def check_environment_health(self, environment: APIEnvironment) -> EnvironmentHealth:
        """
        检查环境健康状态
        
        Args:
            environment: 要检查的环境
            
        Returns:
            EnvironmentHealth: 环境健康状态
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"开始检查环境健康状态: {environment.value}")
            
            # 1. 网络连通性检查
            connectivity = await self._check_network_connectivity(environment)
            
            # 2. API响应性检查
            api_responsive, latency = await self._check_api_responsiveness(environment)
            
            # 3. WebSocket可用性检查
            websocket_available = await self._check_websocket_availability(environment)
            
            # 确定整体状态
            status = self._determine_environment_status(
                connectivity, api_responsive, websocket_available
            )
            
            health = EnvironmentHealth(
                status=status,
                connectivity=connectivity,
                api_responsive=api_responsive,
                websocket_available=websocket_available,
                latency_ms=latency,
                last_check=datetime.now()
            )
            
            self.logger.info(f"环境健康检查完成: {environment.value}, 状态: {status.value}")
            return health
            
        except Exception as e:
            self.logger.error(f"环境健康检查失败 {environment.value}: {e}")
            return EnvironmentHealth(
                status=EnvironmentStatus.UNAVAILABLE,
                connectivity=ConnectivityLevel.NONE,
                api_responsive=False,
                websocket_available=False,
                latency_ms=float('inf'),
                last_check=datetime.now(),
                error_message=str(e)
            )

    async def _check_network_connectivity(self, environment: APIEnvironment) -> ConnectivityLevel:
        """检查网络连通性"""
        hosts = self.CONNECTIVITY_HOSTS[environment]
        successful_connections = 0
        
        for host, port in hosts:
            try:
                # 尝试TCP连接
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=self.CHECK_TIMEOUT
                )
                writer.close()
                await writer.wait_closed()
                successful_connections += 1
                
            except Exception as e:
                self.logger.debug(f"连接失败 {host}:{port} - {e}")
        
        # 根据成功连接的比例确定连通性级别
        success_rate = successful_connections / len(hosts)
        if success_rate >= 0.8:
            return ConnectivityLevel.FULL
        elif success_rate >= 0.3:
            return ConnectivityLevel.LIMITED
        else:
            return ConnectivityLevel.NONE

    async def _check_api_responsiveness(self, environment: APIEnvironment) -> Tuple[bool, float]:
        """检查API响应性"""
        api_manager = self.api_managers[environment]
        endpoints = api_manager.get_api_endpoints()
        
        # 检查系统状态端点
        url = f"{endpoints.futures_url}/v1/ping"
        
        try:
            start_time = datetime.now()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.CHECK_TIMEOUT) as response:
                    end_time = datetime.now()
                    latency = (end_time - start_time).total_seconds() * 1000
                    
                    if response.status == 200:
                        return True, latency
                    else:
                        return False, latency
                        
        except Exception as e:
            self.logger.debug(f"API响应性检查失败: {e}")
            return False, float('inf')

    async def _check_websocket_availability(self, environment: APIEnvironment) -> bool:
        """检查WebSocket可用性"""
        api_manager = self.api_managers[environment]
        endpoints = api_manager.get_api_endpoints()
        
        # 简化的WebSocket连通性检查
        try:
            import websockets
            
            uri = endpoints.websocket_futures_url
            
            # 尝试建立WebSocket连接
            async with websockets.connect(uri, timeout=self.CHECK_TIMEOUT) as websocket:
                # 发送ping消息
                await websocket.ping()
                return True
                
        except Exception as e:
            self.logger.debug(f"WebSocket可用性检查失败: {e}")
            return False

    def _determine_environment_status(
        self, 
        connectivity: ConnectivityLevel, 
        api_responsive: bool, 
        websocket_available: bool
    ) -> EnvironmentStatus:
        """确定环境状态"""
        if connectivity == ConnectivityLevel.FULL and api_responsive and websocket_available:
            return EnvironmentStatus.HEALTHY
        elif connectivity != ConnectivityLevel.NONE and api_responsive:
            return EnvironmentStatus.DEGRADED
        else:
            return EnvironmentStatus.UNAVAILABLE

    async def generate_environment_report(
        self, 
        environment: APIEnvironment,
        config_path: Optional[str] = None
    ) -> EnvironmentReport:
        """
        生成环境报告
        
        Args:
            environment: 环境
            config_path: 配置文件路径
            
        Returns:
            EnvironmentReport: 环境报告
        """
        try:
            self.logger.info(f"生成环境报告: {environment.value}")
            
            # 1. 健康状态检查
            health = await self.check_environment_health(environment)
            
            # 2. API权限检查
            api_permissions = await self._check_api_permissions(environment)
            
            # 3. 配置验证
            configuration_valid = self._validate_configuration(config_path)
            
            # 4. 生成建议和警告
            recommendations, warnings = self._generate_recommendations_and_warnings(
                environment, health, api_permissions, configuration_valid
            )
            
            report = EnvironmentReport(
                environment=environment,
                health=health,
                api_permissions=api_permissions,
                configuration_valid=configuration_valid,
                recommendations=recommendations,
                warnings=warnings
            )
            
            self.logger.info(f"环境报告生成完成: {environment.value}")
            return report
            
        except Exception as e:
            self.logger.error(f"生成环境报告失败: {e}")
            raise ValidationError(f"环境报告生成失败: {e}")

    async def _check_api_permissions(self, environment: APIEnvironment) -> Dict[str, bool]:
        """检查API权限"""
        api_manager = self.api_managers[environment]
        
        try:
            # 尝试加载凭证
            api_manager.load_credentials_from_env()
            
            # 检查权限
            permissions = await api_manager.check_api_permissions()
            return permissions.get('futures', {})
            
        except Exception as e:
            self.logger.debug(f"API权限检查失败 {environment.value}: {e}")
            return {}

    def _validate_configuration(self, config_path: Optional[str]) -> bool:
        """验证配置"""
        if not config_path:
            return True  # 如果没有提供配置路径，跳过验证
        
        try:
            self.config_validator.validate_from_file(config_path)
            return True
        except Exception as e:
            self.logger.debug(f"配置验证失败: {e}")
            return False

    def _generate_recommendations_and_warnings(
        self,
        environment: APIEnvironment,
        health: EnvironmentHealth,
        api_permissions: Dict[str, bool],
        configuration_valid: bool
    ) -> Tuple[List[str], List[str]]:
        """生成建议和警告"""
        recommendations = []
        warnings = []
        
        # 基于健康状态的建议
        if health.status == EnvironmentStatus.UNAVAILABLE:
            recommendations.append("检查网络连接和防火墙设置")
            warnings.append(f"{environment.value}环境当前不可用")
        elif health.status == EnvironmentStatus.DEGRADED:
            recommendations.append("监控环境状态，考虑使用备用连接")
            warnings.append(f"{environment.value}环境性能降级")
        
        # 基于延迟的建议
        if health.latency_ms > 1000:
            recommendations.append("考虑优化网络连接以降低延迟")
            warnings.append(f"API延迟较高: {health.latency_ms:.1f}ms")
        
        # WebSocket相关建议
        if not health.websocket_available:
            recommendations.append("检查WebSocket连接配置")
            warnings.append("WebSocket连接不可用，实时数据功能可能受限")
        
        # API权限相关建议
        if not api_permissions.get('futures_trading', False):
            recommendations.append("确认API密钥具有期货交易权限")
            warnings.append("缺少期货交易权限")
        
        # 配置相关建议
        if not configuration_valid:
            recommendations.append("检查并修复配置文件错误")
            warnings.append("配置验证失败")
        
        # 环境特定建议
        if environment == APIEnvironment.MAINNET:
            recommendations.append("在生产环境中进行充分测试后再进行实盘交易")
            warnings.append("当前使用生产环境，请谨慎操作")
        else:
            recommendations.append("在测试环境中验证所有功能")
        
        return recommendations, warnings

    def detect_optimal_environment(self) -> APIEnvironment:
        """
        检测最优环境
        
        Returns:
            APIEnvironment: 推荐的环境
        """
        # 默认优先使用testnet确保安全
        return APIEnvironment.TESTNET

    async def auto_switch_environment(self) -> APIEnvironment:
        """
        自动切换到最佳环境
        
        Returns:
            APIEnvironment: 切换后的环境
        """
        try:
            # 检查两个环境的健康状态
            testnet_health = await self.check_environment_health(APIEnvironment.TESTNET)
            mainnet_health = await self.check_environment_health(APIEnvironment.MAINNET)
            
            # 优先使用健康的testnet
            if testnet_health.status in [EnvironmentStatus.HEALTHY, EnvironmentStatus.DEGRADED]:
                self.logger.info("选择testnet环境")
                return APIEnvironment.TESTNET
            elif mainnet_health.status in [EnvironmentStatus.HEALTHY, EnvironmentStatus.DEGRADED]:
                self.logger.warning("testnet不可用，切换到mainnet环境")
                return APIEnvironment.MAINNET
            else:
                self.logger.error("所有环境都不可用")
                raise ValidationError("无可用的API环境")
                
        except Exception as e:
            self.logger.error(f"自动环境切换失败: {e}")
            # 默认返回testnet
            return APIEnvironment.TESTNET

    def get_environment_config_template(self, environment: APIEnvironment) -> Dict[str, Any]:
        """
        获取环境配置模板
        
        Args:
            environment: 环境类型
            
        Returns:
            Dict[str, Any]: 配置模板
        """
        base_config = {
            'futures': {
                'environment': environment.value,
                'margin_type': 'CROSSED',
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'default_leverage': {
                    'BTCUSDT': 5 if environment == APIEnvironment.MAINNET else 10,
                    'ETHUSDT': 5 if environment == APIEnvironment.MAINNET else 10
                },
                'risk': {
                    'max_leverage': 10 if environment == APIEnvironment.MAINNET else 20,
                    'max_position_size': 500.0 if environment == APIEnvironment.MAINNET else 1000.0,
                    'stop_loss_percentage': 0.02,
                    'take_profit_percentage': 0.05,
                    'daily_loss_limit': 200.0 if environment == APIEnvironment.MAINNET else 500.0,
                    'max_open_positions': 3 if environment == APIEnvironment.MAINNET else 5,
                    'margin_requirement': 0.2 if environment == APIEnvironment.MAINNET else 0.1
                }
            }
        }
        
        return base_config

    def save_environment_report(self, report: EnvironmentReport, file_path: str) -> None:
        """
        保存环境报告到文件
        
        Args:
            report: 环境报告
            file_path: 文件路径
        """
        try:
            report_data = {
                'environment': report.environment.value,
                'timestamp': datetime.now().isoformat(),
                'health': {
                    'status': report.health.status.value,
                    'connectivity': report.health.connectivity.value,
                    'api_responsive': report.health.api_responsive,
                    'websocket_available': report.health.websocket_available,
                    'latency_ms': report.health.latency_ms,
                    'last_check': report.health.last_check.isoformat(),
                    'error_message': report.health.error_message
                },
                'api_permissions': report.api_permissions,
                'configuration_valid': report.configuration_valid,
                'recommendations': report.recommendations,
                'warnings': report.warnings
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(report_data, f, allow_unicode=True, default_flow_style=False)
            
            self.logger.info(f"环境报告已保存到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存环境报告失败: {e}")
            raise ValidationError(f"环境报告保存失败: {e}")

    async def comprehensive_environment_check(self) -> Dict[APIEnvironment, EnvironmentReport]:
        """
        全面环境检查
        
        Returns:
            Dict[APIEnvironment, EnvironmentReport]: 所有环境的报告
        """
        reports = {}
        
        for environment in APIEnvironment:
            try:
                report = await self.generate_environment_report(environment)
                reports[environment] = report
            except Exception as e:
                self.logger.error(f"环境检查失败 {environment.value}: {e}")
                # 创建错误报告
                error_health = EnvironmentHealth(
                    status=EnvironmentStatus.UNAVAILABLE,
                    connectivity=ConnectivityLevel.NONE,
                    api_responsive=False,
                    websocket_available=False,
                    latency_ms=float('inf'),
                    last_check=datetime.now(),
                    error_message=str(e)
                )
                
                reports[environment] = EnvironmentReport(
                    environment=environment,
                    health=error_health,
                    api_permissions={},
                    configuration_valid=False,
                    recommendations=['检查网络连接和配置'],
                    warnings=[f'环境检查失败: {str(e)}']
                )
        
        return reports