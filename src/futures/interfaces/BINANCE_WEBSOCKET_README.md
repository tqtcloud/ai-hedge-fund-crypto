# 币安期货WebSocket API实现

这个模块提供了基于官方币安SDK的完整期货WebSocket API实现，支持用户数据流的实时监听和处理。

## 主要功能

### 1. Listen Key管理器 (`listen_key_manager.py`)
- **自动创建**：获取新的Listen Key
- **自动刷新**：每30分钟自动刷新，确保连接不断开
- **自动删除**：程序结束时清理Listen Key
- **错误处理**：完整的异常处理和重试机制
- **回调支持**：支持创建、刷新、过期等事件回调

### 2. 用户数据流 (`user_data_stream.py`)
- **继承WebSocket框架**：基于`FuturesWebSocketInterface`实现
- **事件处理**：支持ACCOUNT_UPDATE、ORDER_TRADE_UPDATE、MARGIN_CALL事件
- **自动重连**：Listen Key过期时自动重新连接
- **多处理器**：支持注册多个事件处理器
- **与现有架构集成**：无缝集成到现有的WebSocket框架中

## 快速开始

### 基本使用

```python
import asyncio
from src.futures.interfaces import create_user_data_stream

async def main():
    # 使用上下文管理器自动处理启动和关闭
    async with create_user_data_stream(
        api_key="your_binance_api_key",
        api_secret="your_binance_api_secret",
        testnet=True  # 使用testnet环境
    ) as stream:

        # 添加事件处理器
        def handle_account_update(update):
            print(f"账户更新: {len(update.balances)} 余额, {len(update.positions)} 持仓")

        def handle_order_update(update):
            order = update.order
            print(f"订单更新: {order.symbol} {order.order_status.value}")

        stream.add_account_update_handler(handle_account_update)
        stream.add_order_update_handler(handle_order_update)

        # 保持连接运行
        await asyncio.sleep(60)  # 运行60秒

if __name__ == "__main__":
    asyncio.run(main())
```

### 高级使用 - 账户监控器

```python
import asyncio
import logging
from src.futures.interfaces import (
    BinanceFuturesUserDataStream,
    UserDataStreamConfig
)

class AccountMonitor:
    def __init__(self, api_key, api_secret):
        # 创建配置
        self.config = UserDataStreamConfig(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
            reconnect_delay=5000,
            max_reconnect_attempts=10,
            enable_compression=True
        )

        self.stream = None
        self.balances = {}
        self.positions = {}

    async def start(self):
        # 创建用户数据流
        self.stream = BinanceFuturesUserDataStream(self.config)

        # 注册事件处理器
        self.stream.add_account_update_handler(self._handle_account_update)
        self.stream.add_order_update_handler(self._handle_order_update)
        self.stream.add_margin_call_handler(self._handle_margin_call)

        # 启动流
        await self.stream.start()

    async def stop(self):
        if self.stream:
            await self.stream.stop()

    def _handle_account_update(self, update):
        # 更新余额信息
        for balance in update.balances:
            self.balances[balance.asset] = balance
            logging.info(f"余额更新 - {balance.asset}: {balance.wallet_balance}")

        # 更新持仓信息
        for position in update.positions:
            key = f"{position.symbol}_{position.position_side.value}"
            self.positions[key] = position

            if position.position_amount != 0:
                logging.info(f"持仓更新 - {position.symbol}: {position.position_amount}")

    def _handle_order_update(self, update):
        order = update.order
        logging.info(f"订单更新 - {order.symbol}: {order.order_status.value}")

    def _handle_margin_call(self, data):
        logging.warning(f"保证金追加通知: {data}")
        # 在这里实现风险管理措施

# 使用监控器
async def main():
    monitor = AccountMonitor("your_api_key", "your_api_secret")

    try:
        await monitor.start()
        await asyncio.sleep(3600)  # 运行1小时
    finally:
        await monitor.stop()
```

### 只使用Listen Key管理器

```python
import asyncio
from src.futures.interfaces import create_futures_listen_key_manager

async def main():
    # 创建Listen Key管理器
    manager = await create_futures_listen_key_manager(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True
    )

    # 设置回调
    def on_key_created(key):
        print(f"新Listen Key创建: {key}")

    def on_key_refreshed(key):
        print(f"Listen Key已刷新: {key}")

    manager.set_on_key_created(on_key_created)
    manager.set_on_key_refreshed(on_key_refreshed)

    try:
        # 获取当前key
        current_key = await manager.get_current_key()
        print(f"当前Listen Key: {current_key}")

        # 强制刷新
        await manager.force_refresh()

        await asyncio.sleep(60)
    finally:
        await manager.stop()
```

## 配置选项

### UserDataStreamConfig

```python
config = UserDataStreamConfig(
    api_key="your_api_key",           # 必需：Binance API Key
    api_secret="your_api_secret",     # 必需：Binance API Secret
    testnet=True,                     # 是否使用testnet环境
    reconnect_delay=5000,             # 重连延迟（毫秒）
    max_reconnect_attempts=10,        # 最大重连次数
    listen_key_auto_refresh=True,     # 是否自动刷新Listen Key
    enable_compression=True           # 是否启用压缩
)
```

### WebSocket事件类型

支持的事件类型：

- `ACCOUNT_UPDATE`：账户更新（余额、持仓变化）
- `ORDER_TRADE_UPDATE`：订单交易更新（下单、成交、撤销等）
- `MARGIN_CALL`：保证金追加通知
- `LISTEN_KEY_EXPIRED`：Listen Key过期通知

### 事件处理器

每个事件类型都可以注册多个处理器：

```python
# 添加处理器
stream.add_account_update_handler(handler1)
stream.add_account_update_handler(handler2)

# 移除处理器
stream.remove_account_update_handler(handler1)
```

## 错误处理

### 自动重连
- Listen Key过期时自动重新创建并重新连接
- 网络断开时自动重连（最多10次）
- 重连间隔：5秒递增

### 异常处理
```python
try:
    async with create_user_data_stream(api_key, api_secret) as stream:
        # 添加错误处理器
        def handle_error(error):
            logging.error(f"用户数据流错误: {error}")

        stream.set_on_error(handle_error)

        # 正常使用...
        await asyncio.sleep(60)

except Exception as e:
    logging.error(f"用户数据流启动失败: {e}")
```

## 环境要求

### Python包依赖

确保已安装以下包：

```bash
# 已在项目中安装的包
uv add binance-sdk-derivatives-trading-usds-futures
uv add binance-common
```

### 环境变量设置

为了安全，建议使用环境变量存储API密钥：

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

```python
import os

api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
```

## 测试

### 运行测试

```bash
# 运行单元测试
cd src/futures/interfaces
python -m pytest test_user_data_stream.py -v

# 运行手动测试（需要有效的API密钥）
python test_user_data_stream.py
```

### 运行使用示例

```bash
# 设置环境变量
export BINANCE_API_KEY="your_testnet_api_key"
export BINANCE_API_SECRET="your_testnet_api_secret"

# 运行示例
python usage_example.py
```

## 安全提示

1. **永远不要在代码中硬编码API密钥**
2. **使用testnet环境进行开发和测试**
3. **生产环境使用专用的API密钥，并设置适当的权限**
4. **定期轮换API密钥**
5. **监控API使用量，避免超过限制**

## 与现有架构的集成

这个实现完全兼容现有的WebSocket框架：

```python
from src.futures.interfaces import (
    BinanceFuturesUserDataStream,
    WebSocketConfig,
    ConnectionType
)

# 可以像使用其他WebSocket实现一样使用
config = WebSocketConfig(testnet=True)
stream = BinanceFuturesUserDataStream(config)

# 使用标准的WebSocket接口方法
await stream.start()
await stream.subscribe("user_data", ConnectionType.USER_DATA)

# 也可以使用专门的事件处理器
stream.add_account_update_handler(my_handler)
```

## 故障排除

### 常见问题

1. **API密钥无效**
   ```
   错误: 创建Listen Key失败: HTTP 401
   解决: 检查API密钥是否正确，是否有期货权限
   ```

2. **Listen Key过期**
   ```
   日志: Listen Key已过期，重新创建
   解决: 这是正常行为，系统会自动处理
   ```

3. **网络连接问题**
   ```
   错误: 连接WebSocket失败
   解决: 检查网络连接，确认testnet/mainnet URL正确
   ```

### 调试技巧

启用详细日志：

```python
import logging

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 或只为特定模块启用
logging.getLogger('src.futures.interfaces').setLevel(logging.DEBUG)
```

查看连接状态：

```python
# 获取状态信息
status = stream.get_connection_status()
print(f"连接状态: {status}")

# 获取当前Listen Key
key = stream.get_current_listen_key()
print(f"当前Listen Key: {key}")

# 获取重连次数
count = stream.get_reconnect_count()
print(f"重连次数: {count}")
```