# 期货交易WebSocket接口框架

这是一个为期货交易设计的WebSocket接口基础框架，提供了完整的数据格式定义、连接管理和Mock数据生成功能。

## 🚀 特性

- **标准化数据格式**: 使用dataclass和类型注解定义所有WebSocket消息格式
- **抽象接口设计**: 基于现有binance websocket架构的统一接口
- **Mock数据支持**: 高质量的模拟数据生成，便于开发测试
- **多连接类型**: 支持市场数据、用户数据流、交易接口
- **自动重连机制**: 内置连接管理和错误恢复
- **事件回调系统**: 灵活的消息处理和事件监听
- **异步设计**: 高性能的异步实现

## 📁 文件结构

```
src/futures/interfaces/
├── __init__.py                 # 模块导入定义
├── data_formats.py            # 标准数据格式定义
├── websocket_interface.py     # WebSocket基础框架
├── mock_data_generator.py     # Mock数据生成器
├── mock_websocket_client.py   # Mock客户端实现
├── test_framework.py          # 框架测试脚本
└── README.md                  # 本文档
```

## 🛠️ 安装和依赖

框架基于Python 3.8+，主要依赖：

```python
# 核心依赖
asyncio      # 异步编程
dataclasses  # 数据类定义
typing       # 类型注解
decimal      # 精确数值计算
enum         # 枚举类型
logging      # 日志记录
```

## 📖 快速开始

### 1. 基本使用

```python
import asyncio
from src.futures.interfaces import (
    create_mock_client, create_kline_stream, 
    create_depth_stream, EventType
)

async def main():
    # 创建Mock客户端
    client = create_mock_client(
        symbols=["BTCUSDT", "ETHUSDT"],
        use_realistic_data=True
    )
    
    # 设置消息回调
    def on_kline_data(message):
        print(f"K线: {message.symbol} - {message.close_price}")
    
    client.add_event_callback(EventType.KLINE, on_kline_data)
    
    # 启动并订阅数据
    async with client.connection_context():
        await client.subscribe(create_kline_stream("BTCUSDT", "1m"))
        await client.subscribe(create_depth_stream("BTCUSDT", 20))
        
        # 运行30秒
        await asyncio.sleep(30)

# 运行
asyncio.run(main())
```

### 2. 自定义WebSocket实现

```python
from src.futures.interfaces import (
    FuturesWebSocketInterface, WebSocketConfig, ConnectionType
)

class MyWebSocketClient(FuturesWebSocketInterface):
    async def _create_connection(self, stream_name: str, connection_type: ConnectionType):
        # 实现真实的WebSocket连接逻辑
        # 例如：连接到币安期货WebSocket API
        pass
    
    async def _send_message(self, connection, message):
        # 实现消息发送
        pass
    
    async def _receive_message(self, connection):
        # 实现消息接收
        pass

# 使用自定义实现
config = WebSocketConfig(testnet=False, use_mock_data=False)
client = MyWebSocketClient(config)
```

### 3. 数据格式处理

```python
from src.futures.interfaces import parse_websocket_message, KlineData

# 解析WebSocket原始数据
raw_data = {
    "e": "kline",
    "E": 1640995200000,
    "s": "BTCUSDT",
    "k": {
        "t": 1640995140000,
        "T": 1640995199999,
        # ... 其他K线字段
    }
}

# 自动解析为对应的数据类
parsed = parse_websocket_message(raw_data)
if isinstance(parsed, KlineData):
    print(f"收到K线数据: {parsed.symbol} - {parsed.close_price}")
```

## 🔧 配置选项

### WebSocketConfig

```python
from src.futures.interfaces import WebSocketConfig

config = WebSocketConfig(
    testnet=True,                    # 是否使用测试网
    timeout=10,                      # 连接超时时间
    max_reconnects=5,               # 最大重连次数
    max_queue_size=100,             # 消息队列大小
    use_mock_data=True,             # 是否使用Mock数据
    mock_data_interval=1.0,         # Mock数据推送间隔
    log_level="INFO",               # 日志级别
    enable_message_logging=False    # 是否记录消息日志
)
```

### MockMarketConfig

```python
from src.futures.interfaces import MockMarketConfig

market_config = MockMarketConfig(
    base_price=50000,               # 基础价格
    price_volatility=0.02,          # 价格波动率 (2%)
    trend_strength=0.5,             # 趋势强度
    base_volume=100,                # 基础成交量
    depth_levels=20,                # 深度档位数
    spread_percentage=0.001,        # 价差百分比
    trade_frequency=0.2             # 交易频率
)
```

## 📊 支持的数据类型

### 市场数据
- **K线数据** (KlineData): 开高低收价格、成交量等
- **深度数据** (DepthUpdate): 买卖盘深度更新
- **标记价格** (MarkPriceUpdate): 标记价格和资金费率
- **归集交易** (AggTradeData): 聚合交易数据
- **价格统计** (TickerData): 24小时价格变动统计

### 账户数据
- **余额信息** (Balance): 账户资产余额
- **持仓信息** (Position): 期货持仓详情
- **订单更新** (OrderUpdate): 订单状态变化
- **账户更新** (AccountUpdate): 账户信息变化
- **交易更新** (OrderTradeUpdate): 订单成交信息

## 🧪 测试

运行测试脚本验证框架功能：

```bash
cd src/futures/interfaces
python test_framework.py
```

测试覆盖：
- 数据格式解析测试
- Mock数据生成测试  
- WebSocket客户端功能测试
- 工具函数测试

## 🔌 集成示例

### 与现有系统集成

```python
from src.futures.interfaces import MockFuturesWebSocketClient
from src.futures.interfaces import EventType, KlineData

class TradingBot:
    def __init__(self):
        self.client = MockFuturesWebSocketClient()
        self.client.add_event_callback(EventType.KLINE, self.on_kline_data)
    
    async def on_kline_data(self, kline: KlineData):
        if kline.is_kline_closed:
            # K线完结，执行交易逻辑
            await self.check_trading_signals(kline)
    
    async def check_trading_signals(self, kline: KlineData):
        # 实现交易信号检查
        print(f"检查 {kline.symbol} 交易信号，价格: {kline.close_price}")
    
    async def start(self):
        await self.client.start()
        await self.client.subscribe("btcusdt@kline_1m")
```

## 🛣️ 路线图

- [x] 基础框架和数据格式定义
- [x] Mock数据生成器
- [x] 测试框架
- [ ] 真实WebSocket连接实现
- [ ] 连接池管理
- [ ] 数据持久化支持
- [ ] 性能监控和指标
- [ ] 更多交易对支持

## 📝 贡献指南

1. 遵循现有的代码风格和类型注解
2. 添加适当的错误处理和日志记录
3. 为新功能编写测试用例
4. 更新文档说明

## ⚠️ 注意事项

1. **Mock数据**: 默认使用模拟数据，生产环境需要实现真实连接
2. **精度处理**: 使用Decimal类型处理价格和数量，避免浮点数精度问题
3. **错误处理**: 框架提供基础错误处理，具体业务需要额外处理
4. **性能考虑**: Mock模式下性能较高，真实连接需要考虑网络延迟

## 📞 支持

如有问题或建议，请查看测试用例或查阅相关文档。框架设计参考了币安WebSocket API的最佳实践。