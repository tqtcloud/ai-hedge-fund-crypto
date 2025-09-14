"""
币安期货用户数据流WebSocket实现

基于FuturesWebSocketInterface实现真实的币安期货用户数据流，包括：
- ACCOUNT_UPDATE：账户更新事件
- ORDER_TRADE_UPDATE：订单交易更新事件
- MARGIN_CALL：保证金追加通知事件
- LISTEN_KEY_EXPIRED：监听密钥过期事件

使用官方币安SDK (binance-connector-python) 实现，支持testnet环境。
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager

from binance_common.configuration import ConfigurationWebSocketStreams
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import DerivativesTradingUsdsFutures

from .websocket_interface import (
    FuturesWebSocketInterface,
    ConnectionType,
    WebSocketConfig,
    SubscriptionInfo
)
from .data_formats import (
    EventType,
    AccountUpdate,
    OrderTradeUpdate,
    ErrorMessage,
    parse_websocket_message
)
from .listen_key_manager import (
    BinanceFuturesListenKeyManager,
    create_futures_listen_key_manager
)


@dataclass
class UserDataStreamConfig:
    """用户数据流配置"""
    api_key: str                               # API Key
    api_secret: str                           # API Secret
    testnet: bool = True                      # 是否使用testnet
    reconnect_delay: int = 5000              # 重连延迟（毫秒）
    max_reconnect_attempts: int = 10         # 最大重连次数
    listen_key_auto_refresh: bool = True     # 是否自动刷新Listen Key
    enable_compression: bool = True          # 是否启用压缩

    def to_websocket_config(self) -> WebSocketConfig:
        """转换为WebSocket配置"""
        return WebSocketConfig(
            testnet=self.testnet,
            timeout=10,
            max_reconnects=self.max_reconnect_attempts,
            use_mock_data=False,  # 用户数据流不使用模拟数据
            log_level="INFO",
            enable_message_logging=False
        )


class BinanceFuturesUserDataStream(FuturesWebSocketInterface):
    """币安期货用户数据流WebSocket实现"""

    def __init__(self, config: UserDataStreamConfig):
        """
        初始化用户数据流

        Args:
            config: 用户数据流配置
        """
        # 转换配置格式
        ws_config = config.to_websocket_config()
        super().__init__(ws_config)

        self.stream_config = config

        # 币安SDK客户端
        self._binance_client: Optional[DerivativesTradingUsdsFutures] = None
        self._ws_connection: Optional[Any] = None
        self._user_stream: Optional[Any] = None

        # Listen Key管理器
        self._listen_key_manager: Optional[BinanceFuturesListenKeyManager] = None
        self._current_listen_key: Optional[str] = None

        # 连接管理
        self._reconnect_count = 0
        self._last_message_time: Optional[int] = None

        # 事件处理器
        self._account_update_handlers: List[Callable[[AccountUpdate], None]] = []
        self._order_update_handlers: List[Callable[[OrderTradeUpdate], None]] = []
        self._margin_call_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self._listen_key_expired_handlers: List[Callable[[str], None]] = []

    async def initialize(self) -> None:
        """初始化用户数据流组件"""
        try:
            self.logger.info("初始化币安期货用户数据流...")

            # 创建币安SDK客户端
            await self._create_binance_client()

            # 创建Listen Key管理器
            await self._create_listen_key_manager()

            # 设置事件回调
            self._setup_event_callbacks()

            self.logger.info("用户数据流初始化完成")

        except Exception as e:
            self.logger.error(f"初始化用户数据流失败: {e}")
            raise

    async def _create_binance_client(self) -> None:
        """创建币安SDK客户端"""
        try:
            # 根据testnet设置选择URL
            if self.stream_config.testnet:
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
                stream_url = DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_TESTNET_URL
            else:
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL
                stream_url = DERIVATIVES_TRADING_USDS_FUTURES_WS_STREAMS_PROD_URL

            # 创建WebSocket流配置
            ws_streams_config = ConfigurationWebSocketStreams(
                stream_url=stream_url,
                reconnect_delay=self.stream_config.reconnect_delay,
                compression=1 if self.stream_config.enable_compression else 0
            )

            # 创建币安客户端
            self._binance_client = DerivativesTradingUsdsFutures(
                config_ws_streams=ws_streams_config
            )

            self.logger.debug("币安SDK客户端创建成功")

        except Exception as e:
            self.logger.error(f"创建币安SDK客户端失败: {e}")
            raise

    async def _create_listen_key_manager(self) -> None:
        """创建Listen Key管理器"""
        try:
            self._listen_key_manager = await create_futures_listen_key_manager(
                api_key=self.stream_config.api_key,
                api_secret=self.stream_config.api_secret,
                testnet=self.stream_config.testnet,
                auto_start=True
            )

            # 获取当前Listen Key
            self._current_listen_key = await self._listen_key_manager.get_current_key()
            if not self._current_listen_key:
                raise Exception("无法获取有效的Listen Key")

            self.logger.debug(f"Listen Key管理器创建成功: {self._current_listen_key[:10]}...")

        except Exception as e:
            self.logger.error(f"创建Listen Key管理器失败: {e}")
            raise

    def _setup_event_callbacks(self) -> None:
        """设置事件回调"""
        if self._listen_key_manager:
            # Listen Key创建回调
            self._listen_key_manager.set_on_key_created(self._on_listen_key_created)
            # Listen Key刷新回调
            self._listen_key_manager.set_on_key_refreshed(self._on_listen_key_refreshed)
            # Listen Key过期回调
            self._listen_key_manager.set_on_key_expired(self._on_listen_key_expired)
            # 错误回调
            self._listen_key_manager.set_on_error(self._on_listen_key_error)

    def _on_listen_key_created(self, listen_key: str) -> None:
        """Listen Key创建回调"""
        self.logger.info(f"新Listen Key已创建: {listen_key[:10]}...")
        self._current_listen_key = listen_key

        # 如果有活跃连接，需要重新连接
        if self._ws_connection and self._user_stream:
            asyncio.create_task(self._reconnect_with_new_key(listen_key))

    def _on_listen_key_refreshed(self, listen_key: str) -> None:
        """Listen Key刷新回调"""
        self.logger.debug(f"Listen Key已刷新: {listen_key[:10]}...")

    def _on_listen_key_expired(self, listen_key: str) -> None:
        """Listen Key过期回调"""
        self.logger.warning(f"Listen Key已过期: {listen_key[:10]}...")

        # 调用过期处理器
        for handler in self._listen_key_expired_handlers:
            try:
                handler(listen_key)
            except Exception as e:
                self.logger.error(f"Listen Key过期处理器执行失败: {e}")

    def _on_listen_key_error(self, error: Exception) -> None:
        """Listen Key错误回调"""
        self.logger.error(f"Listen Key管理器错误: {error}")

        # 创建错误消息
        error_msg = ErrorMessage.from_exception(error)
        asyncio.create_task(self._handle_error_message(error_msg))

    async def _reconnect_with_new_key(self, new_listen_key: str) -> None:
        """使用新的Listen Key重新连接"""
        try:
            self.logger.info(f"使用新Listen Key重新连接: {new_listen_key[:10]}...")

            # 关闭当前连接
            if self._user_stream:
                try:
                    await self._user_stream.unsubscribe()
                except Exception as e:
                    self.logger.warning(f"取消订阅失败: {e}")

            if self._ws_connection:
                try:
                    await self._ws_connection.close_connection(close_session=False)
                except Exception as e:
                    self.logger.warning(f"关闭连接失败: {e}")

            # 使用新Key重新建立连接
            await self._establish_user_data_stream(new_listen_key)

        except Exception as e:
            self.logger.error(f"使用新Listen Key重新连接失败: {e}")

    async def _create_connection(self, stream_name: str, connection_type: ConnectionType) -> Any:
        """创建WebSocket连接（实现抽象方法）"""
        try:
            if connection_type != ConnectionType.USER_DATA:
                raise ValueError(f"用户数据流不支持连接类型: {connection_type}")

            # 确保有有效的Listen Key
            if not self._current_listen_key:
                raise Exception("没有有效的Listen Key")

            # 建立用户数据流连接
            await self._establish_user_data_stream(self._current_listen_key)

            return self._ws_connection

        except Exception as e:
            self.logger.error(f"创建WebSocket连接失败: {e}")
            raise

    async def _establish_user_data_stream(self, listen_key: str) -> None:
        """建立用户数据流连接"""
        try:
            self.logger.debug(f"建立用户数据流连接: {listen_key[:10]}...")

            # 创建WebSocket连接
            self._ws_connection = await self._binance_client.websocket_streams.create_connection()

            # 订阅用户数据流
            self._user_stream = self._ws_connection.user_data(listenKey=listen_key)

            # 设置消息处理回调
            self._user_stream.on("message", self._handle_user_data_message)

            self.logger.info(f"用户数据流连接建立成功: {listen_key[:10]}...")

        except Exception as e:
            self.logger.error(f"建立用户数据流连接失败: {e}")
            raise

    def _handle_user_data_message(self, raw_data: Dict[str, Any]) -> None:
        """处理用户数据消息"""
        try:
            # 更新最后消息时间
            self._last_message_time = int(time.time() * 1000)
            self._status.update_message_time()

            # 解析消息
            parsed_message = parse_websocket_message(raw_data)

            # 根据事件类型分发处理
            if hasattr(parsed_message, 'event_type') and parsed_message.event_type:
                asyncio.create_task(self._dispatch_user_event(parsed_message))
            else:
                self.logger.warning(f"未知的用户数据消息: {raw_data}")

        except Exception as e:
            self.logger.error(f"处理用户数据消息失败: {e}")
            error_msg = ErrorMessage.from_exception(e)
            asyncio.create_task(self._handle_error_message(error_msg))

    async def _dispatch_user_event(self, message: Any) -> None:
        """分发用户事件"""
        try:
            event_type = message.event_type

            if event_type == EventType.ACCOUNT_UPDATE:
                await self._handle_account_update(message)

            elif event_type == EventType.ORDER_TRADE_UPDATE:
                await self._handle_order_trade_update(message)

            elif event_type == EventType.LISTEN_KEY_EXPIRED:
                await self._handle_listen_key_expired(message)

            else:
                # 处理其他事件类型（如MARGIN_CALL等）
                await self._handle_other_events(message)

        except Exception as e:
            self.logger.error(f"分发用户事件失败: {e}")

    async def _handle_account_update(self, message: AccountUpdate) -> None:
        """处理账户更新事件"""
        try:
            self.logger.debug(f"收到账户更新事件: {message.event_time}")

            # 调用注册的处理器
            for handler in self._account_update_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    self.logger.error(f"账户更新处理器执行失败: {e}")

            # 调用父类的事件回调
            await self._handle_message("user_data", message)

        except Exception as e:
            self.logger.error(f"处理账户更新事件失败: {e}")

    async def _handle_order_trade_update(self, message: OrderTradeUpdate) -> None:
        """处理订单交易更新事件"""
        try:
            self.logger.debug(f"收到订单交易更新事件: {message.event_time}")

            # 调用注册的处理器
            for handler in self._order_update_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    self.logger.error(f"订单更新处理器执行失败: {e}")

            # 调用父类的事件回调
            await self._handle_message("user_data", message)

        except Exception as e:
            self.logger.error(f"处理订单交易更新事件失败: {e}")

    async def _handle_listen_key_expired(self, message: Dict[str, Any]) -> None:
        """处理Listen Key过期事件"""
        try:
            self.logger.warning("收到Listen Key过期事件")

            # 调用过期处理器
            for handler in self._listen_key_expired_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(self._current_listen_key or "")
                    else:
                        handler(self._current_listen_key or "")
                except Exception as e:
                    self.logger.error(f"Listen Key过期处理器执行失败: {e}")

            # 触发重新创建Listen Key
            if self._listen_key_manager:
                try:
                    new_key = await self._listen_key_manager.start()
                    await self._reconnect_with_new_key(new_key)
                except Exception as e:
                    self.logger.error(f"重新创建Listen Key失败: {e}")

        except Exception as e:
            self.logger.error(f"处理Listen Key过期事件失败: {e}")

    async def _handle_other_events(self, message: Dict[str, Any]) -> None:
        """处理其他事件"""
        try:
            # 检查是否是保证金追加通知
            if message.get('e') == 'MARGIN_CALL':
                self.logger.warning(f"收到保证金追加通知: {message}")

                # 调用保证金追加处理器
                for handler in self._margin_call_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        self.logger.error(f"保证金追加处理器执行失败: {e}")
            else:
                self.logger.debug(f"收到其他用户事件: {message}")

            # 调用父类的事件回调
            await self._handle_message("user_data", message)

        except Exception as e:
            self.logger.error(f"处理其他事件失败: {e}")

    async def _handle_error_message(self, error_msg: ErrorMessage) -> None:
        """处理错误消息"""
        self.logger.error(f"用户数据流错误: {error_msg.error_msg}")

        # 调用父类的错误处理
        await self._handle_message("user_data_error", error_msg)

    async def _send_message(self, connection: Any, message: Dict[str, Any]) -> None:
        """发送消息到WebSocket连接（实现抽象方法）"""
        # 用户数据流是只读的，不支持发送消息
        raise NotImplementedError("用户数据流不支持发送消息")

    async def _receive_message(self, connection: Any) -> Optional[Dict[str, Any]]:
        """从WebSocket连接接收消息（实现抽象方法）"""
        # 消息接收通过币安SDK的回调机制处理，这里不需要实现
        return None

    async def _close_connection(self, connection: Any) -> None:
        """关闭WebSocket连接（实现抽象方法）"""
        try:
            if self._user_stream:
                await self._user_stream.unsubscribe()
                self._user_stream = None

            if self._ws_connection:
                await self._ws_connection.close_connection(close_session=True)
                self._ws_connection = None

        except Exception as e:
            self.logger.error(f"关闭连接失败: {e}")

    async def start(self) -> None:
        """启动用户数据流"""
        try:
            # 初始化组件
            await self.initialize()

            # 启动父类
            await super().start()

            # 订阅用户数据流
            success = await self.subscribe(
                stream_name="user_data",
                connection_type=ConnectionType.USER_DATA
            )

            if not success:
                raise Exception("订阅用户数据流失败")

            self.logger.info("用户数据流启动成功")

        except Exception as e:
            self.logger.error(f"启动用户数据流失败: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """停止用户数据流"""
        try:
            self.logger.info("停止用户数据流...")

            # 停止Listen Key管理器
            if self._listen_key_manager:
                await self._listen_key_manager.stop()
                self._listen_key_manager = None

            # 停止父类
            await super().stop()

            # 清理资源
            self._binance_client = None
            self._ws_connection = None
            self._user_stream = None
            self._current_listen_key = None

            self.logger.info("用户数据流已停止")

        except Exception as e:
            self.logger.error(f"停止用户数据流失败: {e}")

    # 事件处理器注册方法
    def add_account_update_handler(self, handler: Callable[[AccountUpdate], None]) -> None:
        """添加账户更新事件处理器"""
        self._account_update_handlers.append(handler)
        self.logger.debug("添加账户更新事件处理器")

    def add_order_update_handler(self, handler: Callable[[OrderTradeUpdate], None]) -> None:
        """添加订单更新事件处理器"""
        self._order_update_handlers.append(handler)
        self.logger.debug("添加订单更新事件处理器")

    def add_margin_call_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """添加保证金追加通知处理器"""
        self._margin_call_handlers.append(handler)
        self.logger.debug("添加保证金追加通知处理器")

    def add_listen_key_expired_handler(self, handler: Callable[[str], None]) -> None:
        """添加Listen Key过期处理器"""
        self._listen_key_expired_handlers.append(handler)
        self.logger.debug("添加Listen Key过期处理器")

    def remove_account_update_handler(self, handler: Callable[[AccountUpdate], None]) -> None:
        """移除账户更新事件处理器"""
        try:
            self._account_update_handlers.remove(handler)
            self.logger.debug("移除账户更新事件处理器")
        except ValueError:
            self.logger.warning("账户更新处理器不存在")

    def remove_order_update_handler(self, handler: Callable[[OrderTradeUpdate], None]) -> None:
        """移除订单更新事件处理器"""
        try:
            self._order_update_handlers.remove(handler)
            self.logger.debug("移除订单更新事件处理器")
        except ValueError:
            self.logger.warning("订单更新处理器不存在")

    def remove_margin_call_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """移除保证金追加通知处理器"""
        try:
            self._margin_call_handlers.remove(handler)
            self.logger.debug("移除保证金追加通知处理器")
        except ValueError:
            self.logger.warning("保证金追加通知处理器不存在")

    def remove_listen_key_expired_handler(self, handler: Callable[[str], None]) -> None:
        """移除Listen Key过期处理器"""
        try:
            self._listen_key_expired_handlers.remove(handler)
            self.logger.debug("移除Listen Key过期处理器")
        except ValueError:
            self.logger.warning("Listen Key过期处理器不存在")

    # 状态查询方法
    def get_current_listen_key(self) -> Optional[str]:
        """获取当前Listen Key"""
        return self._current_listen_key

    def get_last_message_time(self) -> Optional[int]:
        """获取最后消息时间"""
        return self._last_message_time

    def get_reconnect_count(self) -> int:
        """获取重连次数"""
        return self._reconnect_count

    @asynccontextmanager
    async def user_data_context(self):
        """用户数据流上下文管理器"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# 便捷函数
async def create_user_data_stream(
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    auto_start: bool = True,
    **kwargs
) -> BinanceFuturesUserDataStream:
    """
    创建币安期货用户数据流

    Args:
        api_key: Binance API Key
        api_secret: Binance API Secret
        testnet: 是否使用testnet环境
        auto_start: 是否自动启动
        **kwargs: 其他配置参数

    Returns:
        配置好的用户数据流实例
    """
    config = UserDataStreamConfig(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
        **kwargs
    )

    stream = BinanceFuturesUserDataStream(config)

    if auto_start:
        await stream.start()

    return stream