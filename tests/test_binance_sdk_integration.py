"""
Binance SDK集成测试

测试重构后的期货交易数据类和SDK管理器的功能，
确保向后兼容性和新功能的正确性。
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

# 导入要测试的模块
from src.futures.models.data_models import (
    FuturesSignal,
    Position,
    MarginStatus,
    FuturesOrderRequest,
    TradingDirection,
    OperationType,
    PositionSide,
    RiskLevel,
    BinanceSDKCompatibility,
    SDK_AVAILABLE,
    create_position_from_binance_data,
    create_margin_status_from_binance_data,
    convert_signal_to_binance_order
)

from src.futures.config.sdk_manager import (
    BinanceSDKManager,
    SDKCredentials,
    get_global_sdk_manager,
    create_sdk_manager_from_config
)

from src.futures.constants import (
    get_sdk_environment_config,
    create_sdk_rest_config,
    create_sdk_websocket_config,
    BinanceSDKConfig,
    get_current_environment,
    is_testnet_environment
)


class TestBackwardCompatibility:
    """测试向后兼容性"""
    
    def test_futures_signal_basic_functionality(self):
        """测试FuturesSignal基本功能保持不变"""
        # 创建信号（原有方式）
        signal = FuturesSignal(
            ticker="BTCUSDT",
            direction=TradingDirection.LONG,
            operation_type=OperationType.OPEN,
            confidence=85.0,
            strength=0.7,
            suggested_leverage=10.0
        )
        
        # 验证基本属性
        assert signal.ticker == "BTCUSDT"
        assert signal.direction == TradingDirection.LONG
        assert signal.operation_type == OperationType.OPEN
        assert signal.confidence == 85.0
        assert signal.strength == 0.7
        assert signal.suggested_leverage == 10.0
        
        # 验证原有方法仍然工作
        assert signal.is_valid() is True
        signal_dict = signal.to_dict()
        assert isinstance(signal_dict, dict)
        assert signal_dict["ticker"] == "BTCUSDT"
    
    def test_position_basic_functionality(self):
        """测试Position基本功能保持不变"""
        position = Position(
            ticker="BTCUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=10.0,
            initial_margin=500.0,
            maintenance_margin=25.0
        )
        
        # 验证基本属性
        assert position.ticker == "BTCUSDT"
        assert position.side == PositionSide.LONG
        assert position.size == 0.1
        assert position.entry_price == 50000.0
        assert position.current_price == 51000.0
        
        # 验证计算功能
        assert position.unrealized_pnl == 100.0  # (51000 - 50000) * 0.1
        assert position.roe_percentage == 20.0  # 100 / 500 * 100
        
        # 验证原有方法
        position_dict = position.to_dict()
        assert isinstance(position_dict, dict)
        assert position_dict["ticker"] == "BTCUSDT"
    
    def test_margin_status_basic_functionality(self):
        """测试MarginStatus基本功能保持不变"""
        margin_status = MarginStatus(
            total_balance=10000.0,
            available_margin=8000.0,
            used_margin=2000.0,
            margin_ratio=0.2
        )
        
        # 验证基本属性
        assert margin_status.total_balance == 10000.0
        assert margin_status.available_margin == 8000.0
        assert margin_status.used_margin == 2000.0
        assert margin_status.margin_ratio == 0.2
        
        # 验证计算功能
        max_position = margin_status.get_max_position_size(
            leverage=10.0,
            price=50000.0
        )
        assert max_position > 0
        
        # 验证原有方法
        margin_dict = margin_status.to_dict()
        assert isinstance(margin_dict, dict)
        assert margin_dict["total_balance"] == 10000.0


class TestSDKIntegrationFeatures:
    """测试SDK集成功能"""
    
    def test_sdk_availability_check(self):
        """测试SDK可用性检查"""
        # 检查SDK可用性
        is_available = BinanceSDKCompatibility.is_sdk_available()
        assert isinstance(is_available, bool)
        
        # 验证与全局变量一致
        assert is_available == SDK_AVAILABLE
    
    def test_futures_signal_binance_conversion(self):
        """测试FuturesSignal与Binance格式转换"""
        signal = FuturesSignal(
            ticker="BTCUSDT",
            direction=TradingDirection.LONG,
            operation_type=OperationType.OPEN,
            confidence=85.0,
            strength=0.7,
            suggested_leverage=10.0,
            position_size=1000.0,
            entry_price=50000.0
        )
        
        # 测试转换为Binance订单格式
        order_params = signal.to_binance_order_request(quantity=0.02)
        
        assert isinstance(order_params, dict)
        assert order_params["symbol"] == "BTCUSDT"
        assert order_params["side"] in ["BUY", "SELL"]
        assert order_params["type"] == "MARKET"
        assert "quantity" in order_params
    
    def test_position_from_binance_data(self):
        """测试从Binance数据创建Position"""
        # 模拟Binance SDK响应数据
        mock_binance_data = Mock()
        mock_binance_data.symbol = "BTCUSDT"
        mock_binance_data.positionSide = "LONG"
        mock_binance_data.positionAmt = "0.1"
        mock_binance_data.entryPrice = "50000.0"
        mock_binance_data.markPrice = "51000.0"
        mock_binance_data.leverage = "10"
        mock_binance_data.unRealizedProfit = "100.0"
        mock_binance_data.liquidationPrice = "45000.0"
        mock_binance_data.isolatedWallet = "500.0"
        
        # 测试转换功能
        if SDK_AVAILABLE:
            position = Position.from_binance_position_response(mock_binance_data)
            
            assert position.ticker == "BTCUSDT"
            assert position.side == PositionSide.LONG
            assert position.size == 0.1
            assert position.entry_price == 50000.0
            assert position.current_price == 51000.0
            assert position.leverage == 10.0
            assert position.unrealized_pnl == 100.0
        else:
            with pytest.raises(ImportError):
                Position.from_binance_position_response(mock_binance_data)
    
    def test_margin_status_from_binance_data(self):
        """测试从Binance账户数据创建MarginStatus"""
        # 模拟Binance SDK账户响应数据
        mock_account_data = Mock()
        mock_account_data.totalWalletBalance = "10000.0"
        mock_account_data.totalUnrealizedProfit = "100.0"
        mock_account_data.totalMarginBalance = "10100.0"
        mock_account_data.totalInitialMargin = "2000.0"
        mock_account_data.totalMaintMargin = "500.0"
        
        # 测试转换功能
        if SDK_AVAILABLE:
            margin_status = MarginStatus.from_binance_account_response(mock_account_data)
            
            assert margin_status.total_balance == 10100.0  # wallet + unrealized
            assert margin_status.used_margin == 2000.0
            assert margin_status.maintenance_margin_requirement == 500.0
            assert margin_status.cross_margin == 10100.0
        else:
            with pytest.raises(ImportError):
                MarginStatus.from_binance_account_response(mock_account_data)
    
    def test_futures_order_request_from_signal(self):
        """测试从信号创建订单请求"""
        signal = FuturesSignal(
            ticker="BTCUSDT",
            direction=TradingDirection.LONG,
            operation_type=OperationType.OPEN,
            confidence=85.0,
            strength=0.7,
            suggested_leverage=10.0,
            entry_price=50000.0
        )
        
        order_request = FuturesOrderRequest.from_futures_signal(signal, quantity=0.02)
        
        assert order_request.ticker == "BTCUSDT"
        assert order_request.side == "BUY"
        assert order_request.quantity == 0.02
        assert order_request.price == 50000.0
        assert order_request.leverage == 10.0
    
    def test_binance_api_format_conversion(self):
        """测试Binance API格式转换"""
        order_request = FuturesOrderRequest(
            ticker="BTCUSDT",
            side="BUY",
            quantity=0.02,
            price=50000.0,
            position_side=PositionSide.LONG,
            leverage=10.0
        )
        
        api_format = order_request.to_binance_api_format()
        
        assert isinstance(api_format, dict)
        assert api_format["symbol"] == "BTCUSDT"
        assert api_format["side"] == "BUY"
        assert api_format["quantity"] == "0.02"
        assert api_format["positionSide"] == "LONG"


class TestSDKManager:
    """测试SDK管理器"""
    
    def test_sdk_manager_initialization(self):
        """测试SDK管理器初始化"""
        # 无认证初始化
        manager = BinanceSDKManager()
        
        assert manager.environment == "testnet"  # 默认环境
        assert isinstance(manager.get_environment_info(), dict)
    
    def test_sdk_manager_with_credentials(self):
        """测试带认证凭据的SDK管理器"""
        credentials = SDKCredentials(
            api_key="test_key",
            api_secret="test_secret"
        )
        
        manager = BinanceSDKManager(
            credentials=credentials,
            environment="testnet"
        )
        
        assert manager.credentials == credentials
        assert manager.environment == "testnet"
    
    def test_environment_switching(self):
        """测试环境切换"""
        manager = BinanceSDKManager(environment="testnet")
        
        assert manager.environment == "testnet"
        
        manager.switch_environment("mainnet")
        
        assert manager.environment == "mainnet"
    
    def test_global_sdk_manager(self):
        """测试全局SDK管理器"""
        # 重置全局管理器
        from src.futures.config.sdk_manager import reset_global_sdk_manager
        reset_global_sdk_manager()
        
        # 获取全局管理器
        manager = get_global_sdk_manager()
        
        assert isinstance(manager, BinanceSDKManager)
    
    @patch.dict('os.environ', {
        'BINANCE_API_KEY': 'test_key',
        'BINANCE_API_SECRET': 'test_secret',
        'BINANCE_ENVIRONMENT': 'testnet'
    })
    def test_sdk_manager_from_env_vars(self):
        """测试从环境变量创建SDK管理器"""
        manager = BinanceSDKManager.from_env_vars()
        
        assert manager.credentials.api_key == 'test_key'
        assert manager.credentials.api_secret == 'test_secret'
        assert manager.environment == 'testnet'
    
    def test_create_sdk_manager_from_config(self):
        """测试从配置创建SDK管理器"""
        manager = create_sdk_manager_from_config(
            api_key="test_key",
            api_secret="test_secret",
            environment="testnet"
        )
        
        assert manager.credentials.api_key == "test_key"
        assert manager.environment == "testnet"


class TestConstants:
    """测试常量和配置"""
    
    def test_environment_config(self):
        """测试环境配置"""
        testnet_config = get_sdk_environment_config("testnet")
        mainnet_config = get_sdk_environment_config("mainnet")
        
        assert isinstance(testnet_config, dict)
        assert isinstance(mainnet_config, dict)
        
        assert "rest_api_url" in testnet_config
        assert "ws_streams_url" in testnet_config
        assert "max_leverage" in testnet_config
        
        # 验证URL不同
        assert testnet_config["rest_api_url"] != mainnet_config["rest_api_url"]
    
    def test_sdk_rest_config_creation(self):
        """测试REST配置创建"""
        config = create_sdk_rest_config(
            api_key="test_key",
            api_secret="test_secret",
            environment="testnet"
        )
        
        assert isinstance(config, dict)
        assert config["api_key"] == "test_key"
        assert config["api_secret"] == "test_secret"
        assert "base_path" in config
        assert "timeout" in config
    
    def test_sdk_websocket_config_creation(self):
        """测试WebSocket配置创建"""
        streams_config = create_sdk_websocket_config(
            environment="testnet",
            connection_type="streams"
        )
        
        api_config = create_sdk_websocket_config(
            api_key="test_key",
            environment="testnet",
            connection_type="api"
        )
        
        assert isinstance(streams_config, dict)
        assert isinstance(api_config, dict)
        
        assert "stream_url" in streams_config
        assert "stream_url" in api_config
        assert "api_key" in api_config
    
    def test_environment_detection(self):
        """测试环境检测"""
        current_env = get_current_environment()
        is_testnet = is_testnet_environment()
        
        assert isinstance(current_env, str)
        assert isinstance(is_testnet, bool)
        
        # 验证一致性
        assert (current_env == "testnet") == is_testnet


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_create_position_from_binance_data(self):
        """测试从Binance数据创建Position的便捷函数"""
        mock_data = Mock()
        mock_data.symbol = "BTCUSDT"
        mock_data.positionSide = "LONG"
        mock_data.positionAmt = "0.1"
        mock_data.entryPrice = "50000.0"
        mock_data.markPrice = "51000.0"
        mock_data.leverage = "10"
        mock_data.unRealizedProfit = "100.0"
        mock_data.liquidationPrice = "45000.0"
        mock_data.isolatedWallet = "500.0"
        
        if SDK_AVAILABLE:
            position = create_position_from_binance_data(mock_data)
            assert isinstance(position, Position)
            assert position.ticker == "BTCUSDT"
        else:
            with pytest.raises(ImportError):
                create_position_from_binance_data(mock_data)
    
    def test_create_margin_status_from_binance_data(self):
        """测试从Binance数据创建MarginStatus的便捷函数"""
        mock_data = Mock()
        mock_data.totalWalletBalance = "10000.0"
        mock_data.totalUnrealizedProfit = "100.0"
        mock_data.totalMarginBalance = "10100.0"
        mock_data.totalInitialMargin = "2000.0"
        mock_data.totalMaintMargin = "500.0"
        
        if SDK_AVAILABLE:
            margin_status = create_margin_status_from_binance_data(mock_data)
            assert isinstance(margin_status, MarginStatus)
            assert margin_status.total_balance == 10100.0
        else:
            with pytest.raises(ImportError):
                create_margin_status_from_binance_data(mock_data)
    
    def test_convert_signal_to_binance_order(self):
        """测试信号转换为Binance订单的便捷函数"""
        signal = FuturesSignal(
            ticker="BTCUSDT",
            direction=TradingDirection.LONG,
            operation_type=OperationType.OPEN,
            confidence=85.0,
            strength=0.7
        )
        
        order_dict = convert_signal_to_binance_order(signal, quantity=0.02)
        
        assert isinstance(order_dict, dict)
        assert order_dict["symbol"] == "BTCUSDT"
        assert order_dict["side"] == "BUY"


class TestErrorHandling:
    """测试错误处理"""
    
    def test_invalid_futures_signal_parameters(self):
        """测试无效的FuturesSignal参数"""
        # 测试无效置信度
        with pytest.raises(ValueError):
            FuturesSignal(
                ticker="BTCUSDT",
                direction=TradingDirection.LONG,
                operation_type=OperationType.OPEN,
                confidence=150.0,  # 无效：> 100
                strength=0.7
            )
        
        # 测试无效强度
        with pytest.raises(ValueError):
            FuturesSignal(
                ticker="BTCUSDT",
                direction=TradingDirection.LONG,
                operation_type=OperationType.OPEN,
                confidence=85.0,
                strength=1.5  # 无效：> 1
            )
    
    def test_invalid_position_parameters(self):
        """测试无效的Position参数"""
        # 测试无效仓位大小
        with pytest.raises(Exception):  # PositionSizeError
            Position(
                ticker="BTCUSDT",
                side=PositionSide.LONG,
                size=-0.1,  # 无效：负数
                entry_price=50000.0,
                current_price=51000.0,
                leverage=10.0,
                initial_margin=500.0,
                maintenance_margin=25.0
            )
    
    def test_invalid_sdk_credentials(self):
        """测试无效的SDK认证凭据"""
        # 测试空API密钥
        with pytest.raises(ValueError):
            SDKCredentials(api_key="")
        
        # 测试既无api_secret也无private_key
        with pytest.raises(ValueError):
            SDKCredentials(api_key="test_key")
    
    def test_invalid_environment_config(self):
        """测试无效的环境配置"""
        with pytest.raises(ValueError):
            get_sdk_environment_config("invalid_env")


class TestAsyncOperations:
    """测试异步操作"""
    
    @pytest.mark.asyncio
    async def test_sdk_manager_client_creation(self):
        """测试SDK管理器客户端创建（模拟）"""
        credentials = SDKCredentials(
            api_key="test_key",
            api_secret="test_secret"
        )
        
        manager = BinanceSDKManager(
            credentials=credentials,
            environment="testnet"
        )
        
        # 如果SDK可用，测试客户端创建
        if manager.is_sdk_available():
            try:
                rest_client = manager.create_rest_client()
                assert rest_client is not None
            except Exception as e:
                # 可能由于网络或认证问题失败，这是正常的
                assert "SDK" in str(e) or "connection" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_websocket_client_creation(self):
        """测试WebSocket客户端创建（模拟）"""
        manager = BinanceSDKManager(environment="testnet")
        
        # 如果SDK可用，测试WebSocket客户端创建
        if manager.is_sdk_available():
            try:
                ws_client = manager.create_websocket_streams_client()
                assert ws_client is not None
            except Exception as e:
                # 可能由于网络问题失败，这是正常的
                assert "SDK" in str(e) or "connection" in str(e).lower()


if __name__ == "__main__":
    """运行测试"""
    pytest.main([__file__, "-v", "--tb=short"])