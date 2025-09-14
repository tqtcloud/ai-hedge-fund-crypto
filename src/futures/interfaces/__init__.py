"""
期货交易WebSocket接口模块

该模块提供期货交易WebSocket连接的完整解决方案，包括：

1. 数据格式定义 (data_formats.py):
   - 标准化的WebSocket消息格式
   - 支持市场数据、账户数据、交易数据
   - 使用dataclass和类型注解确保类型安全

2. WebSocket基础框架 (websocket_interface.py):
   - 统一的WebSocket接口抽象
   - 多种连接类型支持 (市场数据、用户数据、交易)
   - 自动重连、错误处理、状态管理
   - 事件回调机制

3. Mock数据生成器 (mock_data_generator.py):
   - 高质量的模拟数据生成
   - 支持各种市场数据类型
   - 便于开发测试和并行开发
   - 可配置的市场行为模拟

主要特性：
- 基于现有binance websocket架构设计
- 支持testnet和mainnet环境
- 默认使用mock数据，便于开发
- 预留真实WebSocket实现接口
- 完整的错误处理和日志记录
- 异步设计，高性能

使用示例：
    from src.futures.interfaces import (
        FuturesWebSocketInterface, WebSocketConfig,
        create_kline_stream, EventType
    )
    
    # 创建配置
    config = WebSocketConfig(testnet=True, use_mock_data=True)
    
    # 实现WebSocket接口
    class MyWebSocket(FuturesWebSocketInterface):
        async def _create_connection(self, stream_name, connection_type):
            # 实现连接逻辑
            pass
    
    # 使用接口
    ws = MyWebSocket(config)
    await ws.start()
    await ws.subscribe(create_kline_stream("BTCUSDT", "1m"))
"""

# 数据格式定义
from .data_formats import (
    # 事件类型和枚举
    EventType,
    PositionSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    
    # 基础消息类
    WebSocketMessage,
    ConnectionStatus,
    ErrorMessage,
    
    # 市场数据类
    KlineData,
    DepthUpdate,
    MarkPriceUpdate,
    AggTradeData,
    TickerData,
    
    # 账户数据类
    Position,
    Balance,
    OrderUpdate,
    AccountUpdate,
    OrderTradeUpdate,
    
    # 工具函数
    parse_websocket_message,
    MESSAGE_TYPE_MAPPING
)

# WebSocket接口框架
from .websocket_interface import (
    # 接口类
    FuturesWebSocketInterface,
    
    # 配置和枚举
    WebSocketConfig,
    ConnectionType,
    WSListenerState,
    SubscriptionInfo,
    
    # 便捷函数
    create_kline_stream,
    create_depth_stream,
    create_trade_stream,
    create_ticker_stream,
    create_mark_price_stream,
    create_multiplex_stream
)

# Mock数据生成器
from .mock_data_generator import (
    # 配置类
    MockMarketConfig,
    MockAccountConfig,
    
    # 引擎类
    MockPriceEngine,
    MockVolumeEngine,
    MockDepthEngine,
    
    # 主生成器
    FuturesMockDataGenerator,
    MockWebSocketConnection
)

__version__ = "1.0.0"

__all__ = [
    # 数据格式
    "EventType",
    "PositionSide", 
    "OrderType",
    "OrderStatus",
    "TimeInForce",
    "WebSocketMessage",
    "ConnectionStatus",
    "ErrorMessage",
    "KlineData",
    "DepthUpdate",
    "MarkPriceUpdate", 
    "AggTradeData",
    "TickerData",
    "Position",
    "Balance",
    "OrderUpdate",
    "AccountUpdate",
    "OrderTradeUpdate",
    "parse_websocket_message",
    "MESSAGE_TYPE_MAPPING",
    
    # WebSocket框架
    "FuturesWebSocketInterface",
    "WebSocketConfig",
    "ConnectionType",
    "WSListenerState", 
    "SubscriptionInfo",
    "create_kline_stream",
    "create_depth_stream",
    "create_trade_stream", 
    "create_ticker_stream",
    "create_mark_price_stream",
    "create_multiplex_stream",
    
    # Mock数据生成器
    "MockMarketConfig",
    "MockAccountConfig",
    "MockPriceEngine",
    "MockVolumeEngine", 
    "MockDepthEngine",
    "FuturesMockDataGenerator",
    "MockWebSocketConnection",
]