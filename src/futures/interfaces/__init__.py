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

3. 真实币安期货WebSocket实现:
   - Listen Key管理器 (listen_key_manager.py): 自动管理Listen Key生命周期
   - 用户数据流 (user_data_stream.py): 真实的币安期货用户数据流实现
   - 支持ACCOUNT_UPDATE、ORDER_TRADE_UPDATE、MARGIN_CALL事件
   - 基于官方币安SDK，完整的错误处理和重连机制

4. Mock数据生成器 (mock_data_generator.py):
   - 高质量的模拟数据生成
   - 支持各种市场数据类型
   - 便于开发测试和并行开发
   - 可配置的市场行为模拟

主要特性：
- 基于现有binance websocket架构设计
- 支持testnet和mainnet环境
- 提供mock数据和真实API两种模式
- 完整的错误处理和日志记录
- 异步设计，高性能
- 自动Listen Key刷新和重连

使用示例：
    # 使用真实的币安期货用户数据流
    from src.futures.interfaces import create_user_data_stream

    async with create_user_data_stream(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True
    ) as stream:
        # 添加事件处理器
        stream.add_account_update_handler(handle_account_update)
        stream.add_order_update_handler(handle_order_update)

        # 流会自动处理所有事件
        await asyncio.sleep(60)
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

# 真实币安期货WebSocket实现
from .listen_key_manager import (
    # 管理器类
    BinanceFuturesListenKeyManager,
    ListenKeyInfo,

    # 便捷函数
    create_futures_listen_key_manager
)

from .user_data_stream import (
    # 主实现类
    BinanceFuturesUserDataStream,
    UserDataStreamConfig,

    # 便捷函数
    create_user_data_stream
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

    # 真实币安期货WebSocket实现
    "BinanceFuturesListenKeyManager",
    "ListenKeyInfo",
    "create_futures_listen_key_manager",
    "BinanceFuturesUserDataStream",
    "UserDataStreamConfig",
    "create_user_data_stream",
]