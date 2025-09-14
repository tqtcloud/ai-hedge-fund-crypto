"""
期货交易WebSocket接口基础框架

提供期货交易WebSocket连接的统一接口，包括：
- 多种WebSocket连接类型支持 (市场数据、用户数据流、交易接口)
- 统一的消息处理接口和事件回调机制
- 连接管理、自动重连、错误处理
- Mock数据支持，便于开发测试
- 支持testnet和mainnet环境配置

基于现有的binance websocket实现架构，提供期货交易专用的WebSocket接口。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any, Union
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .data_formats import (
    WebSocketMessage, EventType, ConnectionStatus, ErrorMessage,
    parse_websocket_message, KlineData, DepthUpdate, MarkPriceUpdate,
    AggTradeData, TickerData, AccountUpdate, OrderTradeUpdate
)


class ConnectionType(Enum):
    """WebSocket连接类型枚举"""
    MARKET_DATA = "market_data"        # 市场数据流
    USER_DATA = "user_data"            # 用户数据流 
    TRADING = "trading"                # 交易接口


class WSListenerState(Enum):
    """WebSocket监听器状态枚举"""
    INITIALISING = "initialising"      # 初始化中
    STREAMING = "streaming"            # 数据流中
    RECONNECTING = "reconnecting"      # 重连中
    EXITING = "exiting"                # 退出中


@dataclass
class WebSocketConfig:
    """WebSocket配置"""
    # 基础配置
    testnet: bool = True                              # 是否使用测试网
    timeout: int = 10                                 # 连接超时时间（秒）
    max_reconnects: int = 5                          # 最大重连次数
    max_queue_size: int = 100                        # 消息队列最大大小
    
    # URL配置
    futures_base_url: str = "wss://fstream.binance.{}/"              # 期货基础URL
    futures_testnet_url: str = "wss://stream.binancefuture.com/"     # 期货测试网URL
    
    # Mock数据配置
    use_mock_data: bool = True                       # 是否使用模拟数据
    mock_data_interval: float = 1.0                  # 模拟数据推送间隔（秒）
    
    # 日志配置
    log_level: str = "INFO"                          # 日志级别
    enable_message_logging: bool = False             # 是否记录消息日志
    
    def get_futures_url(self) -> str:
        """获取期货WebSocket URL"""
        if self.testnet:
            return self.futures_testnet_url
        return self.futures_base_url.format("com")


@dataclass 
class SubscriptionInfo:
    """订阅信息"""
    stream_name: str                    # 流名称
    connection_type: ConnectionType     # 连接类型
    callback: Optional[Callable] = None # 消息回调函数
    is_active: bool = True              # 是否激活
    subscribe_time: int = field(default_factory=lambda: int(time.time() * 1000))


class FuturesWebSocketInterface(ABC):
    """期货WebSocket接口抽象基类"""
    
    def __init__(self, config: WebSocketConfig):
        """
        初始化WebSocket接口
        
        Args:
            config: WebSocket配置对象
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # 连接管理
        self._connections: Dict[str, Any] = {}
        self._subscriptions: Dict[str, SubscriptionInfo] = {}
        self._status = ConnectionStatus()
        
        # 消息处理
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)
        self._global_callback: Optional[Callable] = None
        self._event_callbacks: Dict[EventType, List[Callable]] = {}
        
        # 异步任务管理
        self._tasks: List[asyncio.Task] = []
        self._is_running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    async def _create_connection(self, stream_name: str, connection_type: ConnectionType) -> Any:
        """
        创建WebSocket连接（抽象方法）
        
        Args:
            stream_name: 流名称
            connection_type: 连接类型
            
        Returns:
            WebSocket连接对象
        """
        pass
    
    @abstractmethod
    async def _send_message(self, connection: Any, message: Dict[str, Any]) -> None:
        """
        发送消息到WebSocket连接（抽象方法）
        
        Args:
            connection: WebSocket连接对象
            message: 要发送的消息
        """
        pass
    
    @abstractmethod
    async def _receive_message(self, connection: Any) -> Optional[Dict[str, Any]]:
        """
        从WebSocket连接接收消息（抽象方法）
        
        Args:
            connection: WebSocket连接对象
            
        Returns:
            接收到的消息数据
        """
        pass
    
    async def start(self) -> None:
        """启动WebSocket接口"""
        if self._is_running:
            self.logger.warning("WebSocket接口已在运行中")
            return
        
        self.logger.info("启动期货WebSocket接口")
        self._is_running = True
        self._loop = asyncio.get_event_loop()
        
        # 启动消息处理任务
        self._tasks.append(
            asyncio.create_task(self._message_processor())
        )
        
        # 启动状态监控任务
        self._tasks.append(
            asyncio.create_task(self._status_monitor())
        )
        
        self.logger.info("期货WebSocket接口启动完成")
    
    async def stop(self) -> None:
        """停止WebSocket接口"""
        if not self._is_running:
            return
        
        self.logger.info("停止期货WebSocket接口")
        self._is_running = False
        
        # 取消所有异步任务
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # 关闭所有连接
        for conn_key, connection in self._connections.items():
            try:
                await self._close_connection(connection)
                self.logger.debug(f"关闭连接: {conn_key}")
            except Exception as e:
                self.logger.error(f"关闭连接失败 {conn_key}: {e}")
        
        self._connections.clear()
        self._subscriptions.clear()
        self._tasks.clear()
        
        self.logger.info("期货WebSocket接口已停止")
    
    async def subscribe(
        self, 
        stream_name: str, 
        connection_type: ConnectionType = ConnectionType.MARKET_DATA,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        订阅WebSocket流
        
        Args:
            stream_name: 流名称 (如: "btcusdt@kline_1m")
            connection_type: 连接类型
            callback: 消息回调函数
            
        Returns:
            是否订阅成功
        """
        try:
            conn_key = f"{connection_type.value}_{stream_name}"
            
            if conn_key in self._subscriptions:
                self.logger.warning(f"流已订阅: {stream_name}")
                return True
            
            # 创建连接
            connection = await self._create_connection(stream_name, connection_type)
            if connection is None:
                self.logger.error(f"创建连接失败: {stream_name}")
                return False
            
            # 保存连接和订阅信息
            self._connections[conn_key] = connection
            self._subscriptions[conn_key] = SubscriptionInfo(
                stream_name=stream_name,
                connection_type=connection_type,
                callback=callback
            )
            
            # 启动消息监听任务
            task = asyncio.create_task(
                self._connection_listener(conn_key, connection)
            )
            self._tasks.append(task)
            
            self.logger.info(f"订阅成功: {stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅失败 {stream_name}: {e}")
            return False
    
    async def unsubscribe(self, stream_name: str, connection_type: ConnectionType = ConnectionType.MARKET_DATA) -> bool:
        """
        取消订阅WebSocket流
        
        Args:
            stream_name: 流名称
            connection_type: 连接类型
            
        Returns:
            是否取消订阅成功
        """
        try:
            conn_key = f"{connection_type.value}_{stream_name}"
            
            if conn_key not in self._subscriptions:
                self.logger.warning(f"流未订阅: {stream_name}")
                return True
            
            # 关闭连接
            if conn_key in self._connections:
                await self._close_connection(self._connections[conn_key])
                del self._connections[conn_key]
            
            # 移除订阅信息
            del self._subscriptions[conn_key]
            
            self.logger.info(f"取消订阅成功: {stream_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"取消订阅失败 {stream_name}: {e}")
            return False
    
    def set_global_callback(self, callback: Callable[[Any], None]) -> None:
        """
        设置全局消息回调函数
        
        Args:
            callback: 全局回调函数
        """
        self._global_callback = callback
        self.logger.info("设置全局消息回调函数")
    
    def add_event_callback(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        添加特定事件类型的回调函数
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        
        self._event_callbacks[event_type].append(callback)
        self.logger.info(f"添加 {event_type.value} 事件回调函数")
    
    def remove_event_callback(self, event_type: EventType, callback: Callable[[Any], None]) -> None:
        """
        移除特定事件类型的回调函数
        
        Args:
            event_type: 事件类型 
            callback: 回调函数
        """
        if event_type in self._event_callbacks:
            try:
                self._event_callbacks[event_type].remove(callback)
                if not self._event_callbacks[event_type]:
                    del self._event_callbacks[event_type]
                self.logger.info(f"移除 {event_type.value} 事件回调函数")
            except ValueError:
                self.logger.warning(f"回调函数不存在于 {event_type.value} 事件中")
    
    def get_connection_status(self) -> ConnectionStatus:
        """获取连接状态"""
        return self._status
    
    def get_subscriptions(self) -> Dict[str, SubscriptionInfo]:
        """获取当前订阅信息"""
        return self._subscriptions.copy()
    
    async def _connection_listener(self, conn_key: str, connection: Any) -> None:
        """
        连接监听器，负责接收和处理来自特定连接的消息
        
        Args:
            conn_key: 连接键
            connection: WebSocket连接对象
        """
        self.logger.debug(f"启动连接监听器: {conn_key}")
        
        try:
            while self._is_running and conn_key in self._connections:
                try:
                    # 接收消息
                    raw_message = await self._receive_message(connection)
                    if raw_message is None:
                        continue
                    
                    # 更新状态
                    self._status.update_message_time()
                    
                    # 解析消息
                    parsed_message = parse_websocket_message(raw_message)
                    
                    # 记录消息（如果启用）
                    if self.config.enable_message_logging:
                        self.logger.debug(f"收到消息 [{conn_key}]: {raw_message}")
                    
                    # 加入消息队列处理
                    try:
                        await self._message_queue.put((conn_key, parsed_message))
                    except asyncio.QueueFull:
                        self.logger.error(f"消息队列已满，丢弃消息: {conn_key}")
                        self._status.increment_error()
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"连接监听器错误 [{conn_key}]: {e}")
                    self._status.increment_error()
                    
                    # 创建错误消息
                    error_msg = ErrorMessage.from_exception(e)
                    await self._message_queue.put((conn_key, error_msg))
                    
                    # 如果是连接错误，尝试重连
                    await asyncio.sleep(1.0)
        
        except Exception as e:
            self.logger.error(f"连接监听器异常 [{conn_key}]: {e}")
        
        finally:
            self.logger.debug(f"连接监听器结束: {conn_key}")
    
    async def _message_processor(self) -> None:
        """消息处理器，负责处理消息队列中的消息"""
        self.logger.debug("启动消息处理器")
        
        try:
            while self._is_running:
                try:
                    # 获取消息 (设置超时避免无限等待)
                    conn_key, message = await asyncio.wait_for(
                        self._message_queue.get(), timeout=1.0
                    )
                    
                    # 处理消息
                    await self._handle_message(conn_key, message)
                    
                except asyncio.TimeoutError:
                    # 超时是正常的，继续循环
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"消息处理器错误: {e}")
        
        except Exception as e:
            self.logger.error(f"消息处理器异常: {e}")
        
        finally:
            self.logger.debug("消息处理器结束")
    
    async def _handle_message(self, conn_key: str, message: Any) -> None:
        """
        处理单个消息
        
        Args:
            conn_key: 连接键
            message: 消息对象
        """
        try:
            # 获取订阅信息
            subscription = self._subscriptions.get(conn_key)
            
            # 调用订阅特定的回调函数
            if subscription and subscription.callback:
                try:
                    if asyncio.iscoroutinefunction(subscription.callback):
                        await subscription.callback(message)
                    else:
                        subscription.callback(message)
                except Exception as e:
                    self.logger.error(f"订阅回调函数执行失败 [{conn_key}]: {e}")
            
            # 调用全局回调函数
            if self._global_callback:
                try:
                    if asyncio.iscoroutinefunction(self._global_callback):
                        await self._global_callback(message)
                    else:
                        self._global_callback(message)
                except Exception as e:
                    self.logger.error(f"全局回调函数执行失败: {e}")
            
            # 调用事件类型特定的回调函数
            if hasattr(message, 'event_type') and message.event_type:
                event_callbacks = self._event_callbacks.get(message.event_type, [])
                for callback in event_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(message)
                        else:
                            callback(message)
                    except Exception as e:
                        self.logger.error(f"事件回调函数执行失败 [{message.event_type.value}]: {e}")
        
        except Exception as e:
            self.logger.error(f"消息处理失败 [{conn_key}]: {e}")
    
    async def _status_monitor(self) -> None:
        """状态监控器，定期检查连接状态"""
        self.logger.debug("启动状态监控器")
        
        try:
            while self._is_running:
                try:
                    # 更新连接状态
                    self._status.update_connection(len(self._connections) > 0)
                    
                    # 定期状态日志
                    if self.config.enable_message_logging:
                        self.logger.debug(f"连接状态: {self._status}")
                    
                    # 等待下次检查
                    await asyncio.sleep(10.0)  # 每10秒检查一次
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"状态监控器错误: {e}")
                    await asyncio.sleep(5.0)
        
        except Exception as e:
            self.logger.error(f"状态监控器异常: {e}")
        
        finally:
            self.logger.debug("状态监控器结束")
    
    async def _close_connection(self, connection: Any) -> None:
        """
        关闭WebSocket连接
        
        Args:
            connection: WebSocket连接对象
        """
        # 子类实现具体的连接关闭逻辑
        pass
    
    @asynccontextmanager
    async def connection_context(self):
        """连接上下文管理器，确保连接正确启动和关闭"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# 便捷的订阅辅助函数
def create_kline_stream(symbol: str, interval: str) -> str:
    """创建K线流名称"""
    return f"{symbol.lower()}@kline_{interval}"


def create_depth_stream(symbol: str, levels: int = 20, speed: str = "100ms") -> str:
    """创建深度流名称"""
    if speed == "100ms":
        return f"{symbol.lower()}@depth{levels}@100ms"
    return f"{symbol.lower()}@depth{levels}"


def create_trade_stream(symbol: str) -> str:
    """创建交易流名称"""
    return f"{symbol.lower()}@aggTrade"


def create_ticker_stream(symbol: str) -> str:
    """创建价格统计流名称"""
    return f"{symbol.lower()}@ticker"


def create_mark_price_stream(symbol: str, speed: str = "1s") -> str:
    """创建标记价格流名称"""
    if speed == "1s":
        return f"{symbol.lower()}@markPrice@1s"
    return f"{symbol.lower()}@markPrice"


def create_multiplex_stream(streams: List[str]) -> str:
    """创建多路复用流名称"""
    return "/".join(streams)