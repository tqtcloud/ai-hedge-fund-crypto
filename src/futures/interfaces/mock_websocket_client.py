"""
Mock WebSocket客户端实现

基于FuturesWebSocketInterface的完整Mock实现，用于：
- 开发阶段的数据模拟
- 单元测试和集成测试
- 演示和原型开发
- 离线环境的功能验证

该实现完全使用模拟数据，不需要真实的网络连接。
"""

import asyncio
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

from .websocket_interface import (
    FuturesWebSocketInterface, WebSocketConfig, ConnectionType,
    WSListenerState, SubscriptionInfo
)
from .mock_data_generator import (
    FuturesMockDataGenerator, MockWebSocketConnection,
    MockMarketConfig, MockAccountConfig
)
from .data_formats import EventType, ErrorMessage


class MockFuturesWebSocketClient(FuturesWebSocketInterface):
    """Mock期货WebSocket客户端实现"""
    
    def __init__(
        self,
        config: WebSocketConfig = None,
        symbols: List[str] = None,
        market_config: MockMarketConfig = None,
        account_config: MockAccountConfig = None
    ):
        """
        初始化Mock WebSocket客户端
        
        Args:
            config: WebSocket配置
            symbols: 支持的交易对列表
            market_config: 市场数据配置
            account_config: 账户数据配置
        """
        # 使用默认配置
        if config is None:
            config = WebSocketConfig(use_mock_data=True, testnet=True)
        
        # 强制使用Mock数据
        config.use_mock_data = True
        
        super().__init__(config)
        
        # 初始化Mock数据生成器
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        self.mock_generator = FuturesMockDataGenerator(
            symbols=self.symbols,
            market_config=market_config,
            account_config=account_config
        )
        
        self.logger.info(f"Mock WebSocket客户端初始化，支持交易对: {self.symbols}")
    
    async def _create_connection(self, stream_name: str, connection_type: ConnectionType) -> Optional[MockWebSocketConnection]:
        """
        创建Mock WebSocket连接
        
        Args:
            stream_name: 流名称
            connection_type: 连接类型
            
        Returns:
            Mock连接对象
        """
        try:
            self.logger.info(f"创建Mock连接: {stream_name} ({connection_type.value})")
            
            # 创建Mock连接
            connection = MockWebSocketConnection(stream_name, self.mock_generator)
            
            # 启动数据生成
            await connection.start_data_production()
            
            self.logger.info(f"Mock连接创建成功: {stream_name}")
            return connection
            
        except Exception as e:
            self.logger.error(f"创建Mock连接失败 {stream_name}: {e}")
            return None
    
    async def _send_message(self, connection: MockWebSocketConnection, message: Dict[str, Any]) -> None:
        """
        发送消息到Mock连接 (Mock实现不需要实际发送)
        
        Args:
            connection: Mock连接对象
            message: 要发送的消息
        """
        self.logger.debug(f"Mock发送消息: {message}")
        # Mock实现不需要实际发送消息
        pass
    
    async def _receive_message(self, connection: MockWebSocketConnection) -> Optional[Dict[str, Any]]:
        """
        从Mock连接接收消息
        
        Args:
            connection: Mock连接对象
            
        Returns:
            接收到的消息数据
        """
        try:
            return await connection.recv()
        except Exception as e:
            self.logger.error(f"接收Mock消息失败: {e}")
            return None
    
    async def _close_connection(self, connection: MockWebSocketConnection) -> None:
        """
        关闭Mock连接
        
        Args:
            connection: Mock连接对象
        """
        try:
            await connection.close()
            self.logger.debug("Mock连接已关闭")
        except Exception as e:
            self.logger.error(f"关闭Mock连接失败: {e}")
    
    def get_supported_symbols(self) -> List[str]:
        """获取支持的交易对列表"""
        return self.symbols.copy()
    
    def get_mock_generator(self) -> FuturesMockDataGenerator:
        """获取Mock数据生成器实例"""
        return self.mock_generator
    
    async def simulate_account_event(self) -> None:
        """模拟账户事件 (用于测试)"""
        try:
            account_update = await self.mock_generator.generate_account_update()
            await self._message_queue.put(("account_simulation", account_update))
            self.logger.info("模拟账户事件已生成")
        except Exception as e:
            self.logger.error(f"模拟账户事件失败: {e}")


# 便捷的创建函数
def create_mock_client(
    symbols: List[str] = None,
    use_realistic_data: bool = True,
    log_level: str = "INFO"
) -> MockFuturesWebSocketClient:
    """
    创建配置好的Mock WebSocket客户端
    
    Args:
        symbols: 交易对列表
        use_realistic_data: 是否使用更真实的数据参数
        log_level: 日志级别
        
    Returns:
        配置好的Mock客户端实例
    """
    # 基础配置
    config = WebSocketConfig(
        use_mock_data=True,
        testnet=True,
        timeout=10,
        max_reconnects=3,
        log_level=log_level,
        enable_message_logging=False
    )
    
    # 市场配置
    market_config = None
    account_config = None
    
    if use_realistic_data:
        # 使用更真实的市场参数
        market_config = MockMarketConfig(
            base_price=50000,  # BTC基础价格
            price_volatility=0.015,  # 1.5%波动率
            trend_strength=0.3,
            trend_change_prob=0.005,  # 0.5%趋势变化概率
            base_volume=50,
            volume_volatility=0.4,
            depth_levels=20,
            spread_percentage=0.0005,  # 0.05%价差
            trade_frequency=0.1,  # 更频繁的交易
            large_trade_prob=0.02,  # 2%大额交易概率
            large_trade_multiplier=8.0
        )
        
        # 账户配置
        account_config = MockAccountConfig(
            initial_usdt_balance=100000,  # 10万USDT
            initial_btc_balance=1.0,      # 1 BTC
            max_active_orders=10,
            order_fill_prob=0.15,
            partial_fill_prob=0.25
        )
    
    return MockFuturesWebSocketClient(
        config=config,
        symbols=symbols,
        market_config=market_config,
        account_config=account_config
    )


# 使用示例
async def example_usage():
    """使用示例"""
    print("=== Mock WebSocket客户端使用示例 ===")
    
    # 创建Mock客户端
    client = create_mock_client(
        symbols=["BTCUSDT", "ETHUSDT"],
        use_realistic_data=True,
        log_level="INFO"
    )
    
    # 设置消息回调
    def on_kline_data(message):
        if hasattr(message, 'symbol') and hasattr(message, 'close_price'):
            print(f"K线数据 - {message.symbol}: {message.close_price} (完结: {message.is_kline_closed})")
    
    def on_depth_update(message):
        if hasattr(message, 'symbol'):
            print(f"深度更新 - {message.symbol}: {len(message.bids)}档买单, {len(message.asks)}档卖单")
    
    def on_trade_data(message):
        if hasattr(message, 'symbol') and hasattr(message, 'price'):
            print(f"交易数据 - {message.symbol}: {message.price} x {message.quantity}")
    
    # 添加事件回调
    client.add_event_callback(EventType.KLINE, on_kline_data)
    client.add_event_callback(EventType.DEPTH_UPDATE, on_depth_update)
    client.add_event_callback(EventType.AGG_TRADE, on_trade_data)
    
    try:
        # 启动客户端
        await client.start()
        print("Mock WebSocket客户端已启动")
        
        # 订阅数据流
        from .websocket_interface import (
            create_kline_stream, create_depth_stream, create_trade_stream
        )
        
        await client.subscribe(create_kline_stream("BTCUSDT", "1m"))
        await client.subscribe(create_depth_stream("BTCUSDT", 10))
        await client.subscribe(create_trade_stream("BTCUSDT"))
        
        await client.subscribe(create_kline_stream("ETHUSDT", "1m"))
        
        print("数据流订阅成功，开始接收数据...")
        print("支持的交易对:", client.get_supported_symbols())
        
        # 运行一段时间观察数据
        await asyncio.sleep(30)
        
        # 获取连接状态
        status = client.get_connection_status()
        print(f"连接状态: 连接={status.is_connected}, 重连次数={status.reconnect_count}, 错误次数={status.error_count}")
        
        # 获取订阅信息
        subscriptions = client.get_subscriptions()
        print(f"当前订阅数量: {len(subscriptions)}")
        for sub_key, sub_info in subscriptions.items():
            print(f"  - {sub_info.stream_name} ({sub_info.connection_type.value})")
        
        # 模拟账户事件
        await client.simulate_account_event()
        
    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"运行错误: {e}")
    finally:
        # 停止客户端
        await client.stop()
        print("Mock WebSocket客户端已停止")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())