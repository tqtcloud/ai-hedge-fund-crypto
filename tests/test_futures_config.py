"""
期货配置系统完整性测试

测试期货配置的解析、验证、环境切换等功能。
包含配置管理器、热重载、安全性验证等测试用例。
"""

import os
import tempfile
import pytest
import yaml
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
from unittest.mock import patch, MagicMock, AsyncMock

from src.futures.config.config_validator import (
    FuturesConfigValidator, 
    FuturesConfig, 
    TradingMode, 
    MarginType,
    FuturesRiskConfig
)
from src.futures.config.api_manager import (
    FuturesAPIManager, 
    APIEnvironment, 
    APICredentials,
    APIPermission
)
from src.futures.config.environment_checker import (
    EnvironmentChecker, 
    EnvironmentStatus,
    ConnectivityLevel,
    EnvironmentHealth
)
from src.utils.exceptions import ValidationError


class TestFuturesConfigValidator:
    """期货配置验证器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.validator = FuturesConfigValidator()
    
    def test_validate_valid_config(self):
        """测试有效配置验证"""
        config_data = {
            'mode': 'backtest',
            'futures': {
                'environment': 'testnet',
                'margin_type': 'CROSSED',
                'symbols': ['BTCUSDT'],
                'default_leverage': {'BTCUSDT': 10},
                'allowed_order_types': ['MARKET', 'LIMIT'],
                'min_order_size': {'BTCUSDT': 0.001},
                'price_precision': {'BTCUSDT': 2},
                'quantity_precision': {'BTCUSDT': 3},
                'risk': {
                    'max_leverage': 20,
                    'max_position_size': 1000.0,
                    'stop_loss_percentage': 0.02,
                    'take_profit_percentage': 0.05,
                    'daily_loss_limit': 500.0,
                    'max_open_positions': 5,
                    'margin_requirement': 0.1
                },
                'websocket': {
                    'enable_user_data_stream': True,
                    'enable_market_data_stream': True
                }
            }
        }
        
        result = self.validator.validate_config(config_data)
        assert isinstance(result, FuturesConfig)
        assert result.trading_mode == TradingMode.BACKTEST
        assert result.environment == 'testnet'
        assert result.margin_type == MarginType.CROSSED
    
    def test_validate_invalid_mode(self):
        """测试无效交易模式"""
        config_data = {
            'mode': 'invalid_mode',
            'futures': {}
        }
        
        with pytest.raises(ValidationError, match="不支持的交易模式"):
            self.validator.validate_config(config_data)
    
    def test_validate_missing_futures_section(self):
        """测试缺少期货配置部分"""
        config_data = {
            'mode': 'backtest'
        }
        
        with pytest.raises(ValidationError, match="缺少期货配置部分"):
            self.validator.validate_config(config_data)
    
    def test_validate_invalid_environment(self):
        """测试无效环境"""
        config_data = {
            'mode': 'backtest',
            'futures': {
                'environment': 'invalid_env'
            }
        }
        
        with pytest.raises(ValidationError, match="不支持的环境"):
            self.validator.validate_config(config_data)
    
    def test_risk_config_validation(self):
        """测试风险配置验证"""
        # 测试无效杠杆
        with pytest.raises(ValidationError, match="max_leverage必须在1-125之间"):
            FuturesRiskConfig(
                max_leverage=0,
                max_position_size=1000.0,
                stop_loss_percentage=0.02,
                take_profit_percentage=0.05,
                daily_loss_limit=500.0,
                max_open_positions=5,
                margin_requirement=0.1
            )
        
        # 测试无效止损百分比
        with pytest.raises(ValidationError, match="stop_loss_percentage必须在0-1之间"):
            FuturesRiskConfig(
                max_leverage=10,
                max_position_size=1000.0,
                stop_loss_percentage=1.5,
                take_profit_percentage=0.05,
                daily_loss_limit=500.0,
                max_open_positions=5,
                margin_requirement=0.1
            )
    
    def test_generate_default_config(self):
        """测试生成默认配置"""
        default_config = self.validator.generate_default_config()
        
        assert default_config['mode'] == 'backtest'
        assert default_config['futures']['environment'] == 'testnet'
        assert 'BTCUSDT' in default_config['futures']['symbols']
        assert 'risk' in default_config['futures']
        assert 'websocket' in default_config['futures']
    
    def test_validate_from_file(self):
        """测试从文件验证配置"""
        config_data = self.validator.generate_default_config()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_file = f.name
        
        try:
            result = self.validator.validate_from_file(temp_file)
            assert isinstance(result, FuturesConfig)
        finally:
            os.unlink(temp_file)


class TestFuturesAPIManager:
    """期货API管理器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.api_manager = FuturesAPIManager(APIEnvironment.TESTNET)
    
    def test_init(self):
        """测试初始化"""
        assert self.api_manager.environment == APIEnvironment.TESTNET
        assert self.api_manager.endpoints.base_url == "https://testnet.binance.vision/api"
    
    def test_api_credentials_validation(self):
        """测试API凭证验证"""
        # 测试有效凭证
        credentials = APICredentials(
            api_key="test_api_key_12345678",
            api_secret="test_api_secret_12345678",
            environment=APIEnvironment.TESTNET,
            permissions=[APIPermission.FUTURES]
        )
        
        assert credentials.api_key == "test_api_key_12345678"
        assert credentials.environment == APIEnvironment.TESTNET
        
        # 测试无效凭证
        with pytest.raises(ValidationError, match="API密钥格式不正确"):
            APICredentials(
                api_key="short",
                api_secret="test_api_secret_12345678",
                environment=APIEnvironment.TESTNET,
                permissions=[APIPermission.FUTURES]
            )
    
    @patch.dict(os.environ, {
        'BINANCE_FUTURES_TESTNET_API_KEY': 'test_key_12345678',
        'BINANCE_FUTURES_TESTNET_API_SECRET': 'test_secret_12345678'
    })
    def test_load_credentials_from_env(self):
        """测试从环境变量加载凭证"""
        credentials = self.api_manager.load_credentials_from_env()
        
        assert credentials.api_key == 'test_key_12345678'
        assert credentials.api_secret == 'test_secret_12345678'
        assert credentials.environment == APIEnvironment.TESTNET
    
    def test_load_credentials_from_env_missing(self):
        """测试从环境变量加载凭证失败"""
        with pytest.raises(ValidationError, match="未找到API凭证环境变量"):
            self.api_manager.load_credentials_from_env()
    
    def test_switch_environment(self):
        """测试切换环境"""
        self.api_manager.switch_environment(APIEnvironment.MAINNET)
        
        assert self.api_manager.environment == APIEnvironment.MAINNET
        assert self.api_manager.endpoints.base_url == "https://api.binance.com/api"
        assert self.api_manager._credentials is None
    
    def test_build_query_string(self):
        """测试构建查询字符串"""
        params = {'symbol': 'BTCUSDT', 'timestamp': 1234567890}
        query_string = self.api_manager._build_query_string(params)
        
        assert query_string == 'symbol=BTCUSDT&timestamp=1234567890'
    
    def test_mask_sensitive_data(self):
        """测试掩码敏感数据"""
        data = {
            'api_key': 'test_api_key_12345678',
            'api_secret': 'test_api_secret_12345678',
            'signature': 'test_signature_12345678',
            'normal_field': 'normal_value'
        }
        
        masked = self.api_manager.mask_sensitive_data(data)
        
        assert masked['api_key'].startswith('test')
        assert masked['api_key'].endswith('5678')
        assert '*' in masked['api_key']
        assert masked['normal_field'] == 'normal_value'
    
    def test_get_credentials_status_no_credentials(self):
        """测试获取凭证状态（无凭证）"""
        status = self.api_manager.get_credentials_status()
        
        assert status['loaded'] is False
        assert status['environment'] == 'testnet'


class TestEnvironmentChecker:
    """环境检查器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.checker = EnvironmentChecker()
    
    def test_init(self):
        """测试初始化"""
        assert self.checker is not None
        assert len(self.checker.api_managers) == 2
    
    def test_determine_environment_status(self):
        """测试确定环境状态"""
        # 测试健康状态
        status = self.checker._determine_environment_status(
            ConnectivityLevel.FULL, True, True
        )
        assert status == EnvironmentStatus.HEALTHY
        
        # 测试降级状态
        status = self.checker._determine_environment_status(
            ConnectivityLevel.FULL, True, False
        )
        assert status == EnvironmentStatus.DEGRADED
        
        # 测试不可用状态
        status = self.checker._determine_environment_status(
            ConnectivityLevel.NONE, False, False
        )
        assert status == EnvironmentStatus.UNAVAILABLE
    
    def test_detect_optimal_environment(self):
        """测试检测最优环境"""
        optimal = self.checker.detect_optimal_environment()
        assert optimal == APIEnvironment.TESTNET  # 默认推荐testnet
    
    def test_get_environment_config_template(self):
        """测试获取环境配置模板"""
        testnet_template = self.checker.get_environment_config_template(APIEnvironment.TESTNET)
        mainnet_template = self.checker.get_environment_config_template(APIEnvironment.MAINNET)
        
        assert testnet_template['futures']['environment'] == 'testnet'
        assert mainnet_template['futures']['environment'] == 'mainnet'
        
        # testnet应该有更宽松的风险设置
        assert (testnet_template['futures']['risk']['max_leverage'] >= 
                mainnet_template['futures']['risk']['max_leverage'])
    
    @patch('src.futures.config.environment_checker.EnvironmentChecker._check_network_connectivity')
    @patch('src.futures.config.environment_checker.EnvironmentChecker._check_api_responsiveness')
    @patch('src.futures.config.environment_checker.EnvironmentChecker._check_websocket_availability')
    async def test_check_environment_health(self, mock_ws, mock_api, mock_network):
        """测试检查环境健康状态"""
        mock_network.return_value = ConnectivityLevel.FULL
        mock_api.return_value = (True, 100.0)
        mock_ws.return_value = True
        
        health = await self.checker.check_environment_health(APIEnvironment.TESTNET)
        
        assert health.status == EnvironmentStatus.HEALTHY
        assert health.connectivity == ConnectivityLevel.FULL
        assert health.api_responsive is True
        assert health.websocket_available is True
        assert health.latency_ms == 100.0
    
    def test_generate_recommendations_and_warnings(self):
        """测试生成建议和警告"""
        health = EnvironmentHealth(
            status=EnvironmentStatus.DEGRADED,
            connectivity=ConnectivityLevel.LIMITED,
            api_responsive=True,
            websocket_available=False,
            latency_ms=1500.0,
            last_check=None
        )
        
        recommendations, warnings = self.checker._generate_recommendations_and_warnings(
            APIEnvironment.TESTNET, health, {}, True
        )
        
        assert len(recommendations) > 0
        assert len(warnings) > 0
        assert any('网络连接' in rec for rec in recommendations)
        assert any('WebSocket' in warn for warn in warnings)


@pytest.mark.asyncio
async def test_integration_workflow():
    """测试集成工作流程"""
    # 这是一个简化的集成测试
    checker = EnvironmentChecker()
    validator = FuturesConfigValidator()
    
    # 1. 生成默认配置
    default_config = validator.generate_default_config()
    assert default_config['futures']['environment'] == 'testnet'
    
    # 2. 验证配置
    futures_config = validator.validate_config(default_config)
    assert isinstance(futures_config, FuturesConfig)
    
    # 3. 选择最优环境
    optimal_env = checker.detect_optimal_environment()
    assert optimal_env == APIEnvironment.TESTNET
    
    # 4. 检查环境配置模板
    template = checker.get_environment_config_template(optimal_env)
    assert template['futures']['environment'] == optimal_env.value


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])