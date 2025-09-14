"""
增强版期货市场数据分析器 - 集成官方Binance SDK

完全集成binance-connector-python官方SDK，支持：
1. 实时WebSocket数据流获取和处理
2. REST API历史数据获取  
3. 高性能数据缓存和处理管道
4. 真实数据和模拟数据双模式
5. 多时间框架实时分析
6. 完整的错误处理和自动重连
"""

import asyncio
import logging
import math
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import numpy as np
import pandas as pd

# 导入SDK相关类型
try:
    from ...utils.settings import BinanceSDKSettings
    from ...utils.binance_sdk_factory import create_usds_futures_client_from_settings
    SDK_AVAILABLE = True
except ImportError:
    BinanceSDKSettings = None
    SDK_AVAILABLE = False

# 导入原有模型和类型
try:
    from .market_analyzer import (
        MarketAnalyzer, VolatilityMetrics, TechnicalIndicators, 
        MarketConditionAnalysis, RiskMetrics, VolatilityRegime,
        TradingDirection, RiskLevel, MarketCondition, LiquidityLevel
    )
    from ..utils.technical_indicators import (
        calculate_ema, calculate_rsi, calculate_bollinger_bands, 
        calculate_atr, calculate_adx
    )
except ImportError:
    # 如果导入失败，使用简化版本
    logging.warning("无法导入完整的技术分析模块，将使用简化功能")


class EnhancedMarketAnalyzer(MarketAnalyzer):
    """
    增强版市场分析器 - 集成官方Binance SDK
    
    功能特性：
    1. 实时WebSocket数据流
    2. REST API历史数据
    3. 高性能缓存系统
    4. 真实/模拟数据双模式
    5. 多时间框架分析
    6. 自动重连和错误恢复
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        sdk_config: Optional[BinanceSDKSettings] = None,
        enable_real_time: bool = True
    ):
        """
        初始化增强版市场分析器
        
        Args:
            config: 分析器配置参数
            sdk_config: Binance SDK配置 (可选，如果提供则使用真实数据)
            enable_real_time: 是否启用实时数据流
        """
        # 首先调用父类初始化
        super().__init__(config)
        
        self.sdk_config = sdk_config
        self.enable_real_time = enable_real_time
        
        # SDK客户端
        self._usds_client = None
        self._websocket_connection = None
        self._active_streams: Dict[str, Any] = {}  # 活跃的WebSocket流
        self._stream_handlers: Dict[str, callable] = {}  # 数据流处理器
        
        # 实时数据缓存
        self._realtime_cache: Dict[str, Any] = {}
        self._kline_buffers: Dict[str, List[Dict]] = {}  # K线数据缓冲区
        self._depth_cache: Dict[str, Dict] = {}  # 深度数据缓存
        self._trade_cache: Dict[str, List[Dict]] = {}  # 交易数据缓存
        
        # 数据源模式
        self._use_real_data = (
            SDK_AVAILABLE and 
            sdk_config is not None and 
            sdk_config.validate_credentials()
        )
        self._mock_mode = not self._use_real_data
        
        # 性能监控
        self._stream_stats: Dict[str, Dict] = {}
        self._last_update_times: Dict[str, datetime] = {}
        
        # 连接状态
        self._is_connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = config.get("max_reconnect_attempts", 10) if config else 10
        
        # 数据质量控制
        self._data_quality_threshold = config.get("data_quality_threshold", 0.8) if config else 0.8
        
        # 初始化SDK客户端
        if self._use_real_data:
            asyncio.create_task(self._initialize_sdk_client())
        
        mode_str = "真实数据" if self._use_real_data else "模拟数据"
        self.logger.info(f"增强版市场分析器初始化完成 - 数据模式: {mode_str}")
    
    async def _initialize_sdk_client(self) -> None:
        """初始化Binance SDK客户端"""
        try:
            if not SDK_AVAILABLE:
                self.logger.error("Binance SDK不可用，切换到模拟模式")
                self._use_real_data = False
                self._mock_mode = True
                return
            
            self._usds_client = create_usds_futures_client_from_settings(self.sdk_config)
            self.logger.info("Binance SDK客户端初始化成功")
            
            # 测试连接
            if await self._test_connection():
                self._is_connected = True
                self.logger.info("SDK连接测试成功")
            else:
                self.logger.warning("SDK连接测试失败，但客户端已初始化")
            
        except Exception as e:
            self.logger.error(f"Binance SDK客户端初始化失败: {e}")
            self._use_real_data = False
            self._mock_mode = True
    
    async def _test_connection(self) -> bool:
        """测试SDK连接"""
        try:
            if not self._usds_client:
                return False
            
            # 尝试获取服务器时间
            response = await self._usds_client.rest_api.server_time()
            if response and hasattr(response, 'data'):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"连接测试失败: {e}")
            return False
    
    async def start_real_time_analysis(
        self, 
        symbols: List[str], 
        intervals: Optional[List[str]] = None
    ) -> None:
        """
        启动实时数据分析
        
        Args:
            symbols: 交易对符号列表
            intervals: K线时间间隔列表
        """
        if not self._use_real_data:
            self.logger.warning("当前为模拟模式，启动模拟数据生成")
            await self._start_mock_data_generation(symbols, intervals)
            return
        
        if not self.enable_real_time:
            self.logger.info("实时数据流已禁用")
            return
            
        try:
            # 创建WebSocket连接
            await self._establish_websocket_connection()
            
            intervals = intervals or ["1m", "5m", "1h"]
            
            # 启动数据流
            tasks = []
            for symbol in symbols:
                # 启动K线数据流
                for interval in intervals:
                    tasks.append(self._start_kline_stream(symbol, interval))
                
                # 启动深度数据流
                tasks.append(self._start_depth_stream(symbol))
                
                # 启动交易数据流
                tasks.append(self._start_trade_stream(symbol))
            
            # 并发启动所有数据流
            await asyncio.gather(*tasks, return_exceptions=True)
            
            self.logger.info(f"已启动{len(symbols)}个交易对的实时数据流，时间框架: {intervals}")
            
            # 启动数据质量监控
            asyncio.create_task(self._monitor_data_quality())
            
        except Exception as e:
            self.logger.error(f"启动实时数据分析失败: {e}")
            await self._handle_connection_error()
    
    async def _establish_websocket_connection(self) -> None:
        """建立WebSocket连接"""
        try:
            if self._websocket_connection:
                return  # 连接已存在
            
            if not self._usds_client:
                raise ValueError("SDK客户端未初始化")
            
            self._websocket_connection = await self._usds_client.websocket_streams.create_connection()
            self._is_connected = True
            self._reconnect_attempts = 0
            
            self.logger.info("WebSocket连接已建立")
            
        except Exception as e:
            self.logger.error(f"建立WebSocket连接失败: {e}")
            await self._handle_connection_error()
            raise
    
    async def _start_kline_stream(self, symbol: str, interval: str) -> None:
        """启动K线数据流"""
        try:
            stream_key = f"kline_{symbol}_{interval}"
            
            if stream_key in self._active_streams:
                return  # 流已存在
            
            # 根据官方SDK文档创建K线流
            stream = await self._websocket_connection.kline(
                symbol=symbol.lower(),
                interval=interval
            )
            
            # 设置消息处理器
            stream.on("message", lambda data: self._handle_kline_message(symbol, interval, data))
            stream.on("error", lambda error: self._handle_stream_error(stream_key, error))
            
            self._active_streams[stream_key] = stream
            self._stream_stats[stream_key] = {
                "messages_received": 0,
                "last_message_time": None,
                "errors": 0
            }
            
            # 初始化数据缓冲区
            if stream_key not in self._kline_buffers:
                self._kline_buffers[stream_key] = []
            
            self.logger.info(f"K线数据流已启动: {symbol} {interval}")
            
        except Exception as e:
            self.logger.error(f"启动K线数据流失败 {symbol} {interval}: {e}")
            raise
    
    async def _start_depth_stream(self, symbol: str, level: int = 20) -> None:
        """启动深度数据流"""
        try:
            stream_key = f"depth_{symbol}_{level}"
            
            if stream_key in self._active_streams:
                return
            
            # 创建深度数据流
            stream = await self._websocket_connection.partial_book_depth(
                symbol=symbol.lower(),
                level=level,
                speed="100ms"  # 100ms更新频率
            )
            
            stream.on("message", lambda data: self._handle_depth_message(symbol, data))
            stream.on("error", lambda error: self._handle_stream_error(stream_key, error))
            
            self._active_streams[stream_key] = stream
            self._stream_stats[stream_key] = {
                "messages_received": 0,
                "last_message_time": None,
                "errors": 0
            }
            
            self.logger.info(f"深度数据流已启动: {symbol}")
            
        except Exception as e:
            self.logger.error(f"启动深度数据流失败 {symbol}: {e}")
    
    async def _start_trade_stream(self, symbol: str) -> None:
        """启动交易数据流"""
        try:
            stream_key = f"trade_{symbol}"
            
            if stream_key in self._active_streams:
                return
            
            # 创建聚合交易数据流
            stream = await self._websocket_connection.agg_trade(
                symbol=symbol.lower()
            )
            
            stream.on("message", lambda data: self._handle_trade_message(symbol, data))
            stream.on("error", lambda error: self._handle_stream_error(stream_key, error))
            
            self._active_streams[stream_key] = stream
            self._stream_stats[stream_key] = {
                "messages_received": 0,
                "last_message_time": None,
                "errors": 0
            }
            
            # 初始化交易缓存
            if symbol not in self._trade_cache:
                self._trade_cache[symbol] = []
            
            self.logger.info(f"交易数据流已启动: {symbol}")
            
        except Exception as e:
            self.logger.error(f"启动交易数据流失败 {symbol}: {e}")
    
    def _handle_kline_message(self, symbol: str, interval: str, data: Dict[str, Any]) -> None:
        """处理K线消息"""
        try:
            stream_key = f"kline_{symbol}_{interval}"
            
            # 更新统计信息
            self._update_stream_stats(stream_key)
            
            kline = data.get("k", {})
            if not kline:
                return
            
            # 提取K线数据
            kline_data = {
                "timestamp": kline["t"],
                "open_time": kline["t"],
                "close_time": kline["T"],
                "symbol": kline["s"],
                "interval": kline["i"],
                "open": float(kline["o"]),
                "high": float(kline["h"]),
                "low": float(kline["l"]),
                "close": float(kline["c"]),
                "volume": float(kline["v"]),
                "quote_volume": float(kline["q"]),
                "trades": kline["n"],
                "taker_buy_volume": float(kline["V"]),
                "taker_buy_quote_volume": float(kline["Q"]),
                "is_closed": kline["x"]  # K线是否完成
            }
            
            # 更新缓冲区
            buffer = self._kline_buffers[stream_key]
            
            if kline_data["is_closed"]:
                # 完成的K线添加到缓冲区
                buffer.append(kline_data)
                
                # 保持缓冲区大小（最多1000条）
                if len(buffer) > 1000:
                    buffer[:] = buffer[-1000:]
            
            # 更新当前价格缓存
            self._realtime_cache[f"current_price_{symbol}"] = kline_data["close"]
            self._realtime_cache[f"latest_kline_{symbol}_{interval}"] = kline_data
            
            # 触发实时分析（如果需要）
            if kline_data["is_closed"]:
                asyncio.create_task(self._trigger_real_time_analysis(symbol, interval))
                
        except Exception as e:
            self.logger.error(f"处理K线消息失败: {e}")
    
    def _handle_depth_message(self, symbol: str, data: Dict[str, Any]) -> None:
        """处理深度消息"""
        try:
            stream_key = f"depth_{symbol}_20"
            self._update_stream_stats(stream_key)
            
            # 提取买卖盘数据
            bids = [[float(bid[0]), float(bid[1])] for bid in data.get("bids", [])]
            asks = [[float(ask[0]), float(ask[1])] for ask in data.get("asks", [])]
            
            depth_data = {
                "timestamp": data.get("E", int(datetime.now().timestamp() * 1000)),
                "last_update_id": data.get("lastUpdateId", 0),
                "bids": bids,
                "asks": asks,
                "best_bid": bids[0][0] if bids else 0,
                "best_ask": asks[0][0] if asks else 0,
                "best_bid_qty": bids[0][1] if bids else 0,
                "best_ask_qty": asks[0][1] if asks else 0
            }
            
            # 计算买卖价差
            if depth_data["best_bid"] and depth_data["best_ask"]:
                depth_data["spread"] = depth_data["best_ask"] - depth_data["best_bid"]
                depth_data["spread_percent"] = (depth_data["spread"] / depth_data["best_ask"]) * 100
                depth_data["mid_price"] = (depth_data["best_bid"] + depth_data["best_ask"]) / 2
            
            # 计算深度指标
            depth_data.update(self._calculate_depth_metrics(bids, asks))
            
            # 更新缓存
            self._depth_cache[symbol] = depth_data
            self._realtime_cache[f"depth_{symbol}"] = depth_data
            
        except Exception as e:
            self.logger.error(f"处理深度消息失败: {e}")
    
    def _handle_trade_message(self, symbol: str, data: Dict[str, Any]) -> None:
        """处理交易消息"""
        try:
            stream_key = f"trade_{symbol}"
            self._update_stream_stats(stream_key)
            
            trade_data = {
                "timestamp": data.get("E", int(datetime.now().timestamp() * 1000)),
                "symbol": data.get("s", symbol),
                "trade_id": data.get("t", 0),
                "price": float(data.get("p", 0)),
                "quantity": float(data.get("q", 0)),
                "buyer_order_id": data.get("b", 0),
                "seller_order_id": data.get("a", 0),
                "trade_time": data.get("T", 0),
                "is_buyer_maker": data.get("m", False)
            }
            
            # 更新交易缓存
            if symbol not in self._trade_cache:
                self._trade_cache[symbol] = []
            
            self._trade_cache[symbol].append(trade_data)
            
            # 保持缓存大小（最多500条最近交易）
            if len(self._trade_cache[symbol]) > 500:
                self._trade_cache[symbol] = self._trade_cache[symbol][-500:]
            
            # 更新最新交易价格
            self._realtime_cache[f"latest_trade_{symbol}"] = trade_data
            
        except Exception as e:
            self.logger.error(f"处理交易消息失败: {e}")
    
    def _handle_stream_error(self, stream_key: str, error: Any) -> None:
        """处理流错误"""
        try:
            self.logger.error(f"数据流错误 {stream_key}: {error}")
            
            # 更新错误统计
            if stream_key in self._stream_stats:
                self._stream_stats[stream_key]["errors"] += 1
            
            # 如果错误过多，尝试重连
            if (stream_key in self._stream_stats and 
                self._stream_stats[stream_key]["errors"] > 5):
                asyncio.create_task(self._reconnect_stream(stream_key))
                
        except Exception as e:
            self.logger.error(f"处理流错误失败: {e}")
    
    async def _reconnect_stream(self, stream_key: str) -> None:
        """重连特定数据流"""
        try:
            self.logger.info(f"尝试重连数据流: {stream_key}")
            
            # 移除旧流
            if stream_key in self._active_streams:
                try:
                    await self._active_streams[stream_key].unsubscribe()
                except:
                    pass
                del self._active_streams[stream_key]
            
            # 等待一段时间后重连
            await asyncio.sleep(5)
            
            # 解析流信息并重新启动
            parts = stream_key.split("_")
            if len(parts) >= 2:
                stream_type = parts[0]
                symbol = parts[1]
                
                if stream_type == "kline" and len(parts) >= 3:
                    interval = parts[2]
                    await self._start_kline_stream(symbol, interval)
                elif stream_type == "depth":
                    await self._start_depth_stream(symbol)
                elif stream_type == "trade":
                    await self._start_trade_stream(symbol)
            
        except Exception as e:
            self.logger.error(f"重连数据流失败 {stream_key}: {e}")
    
    def _update_stream_stats(self, stream_key: str) -> None:
        """更新流统计信息"""
        try:
            if stream_key not in self._stream_stats:
                self._stream_stats[stream_key] = {
                    "messages_received": 0,
                    "last_message_time": None,
                    "errors": 0
                }
            
            stats = self._stream_stats[stream_key]
            stats["messages_received"] += 1
            stats["last_message_time"] = datetime.now()
            
            self._last_update_times[stream_key] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"更新流统计信息失败: {e}")
    
    def _calculate_depth_metrics(self, bids: List[List[float]], asks: List[List[float]]) -> Dict[str, float]:
        """计算深度指标"""
        try:
            metrics = {}
            
            if not bids or not asks:
                return metrics
            
            # 计算各档位总量
            bid_volumes = {
                "bid_volume_5": sum([bid[1] for bid in bids[:5]]),
                "bid_volume_10": sum([bid[1] for bid in bids[:10]]),
                "bid_volume_20": sum([bid[1] for bid in bids[:20]]),
                "total_bid_volume": sum([bid[1] for bid in bids])
            }
            
            ask_volumes = {
                "ask_volume_5": sum([ask[1] for ask in asks[:5]]),
                "ask_volume_10": sum([ask[1] for ask in asks[:10]]),
                "ask_volume_20": sum([ask[1] for ask in asks[:20]]),
                "total_ask_volume": sum([ask[1] for ask in asks])
            }
            
            metrics.update(bid_volumes)
            metrics.update(ask_volumes)
            
            # 计算流动性比率
            if ask_volumes["ask_volume_5"] > 0:
                metrics["liquidity_ratio_5"] = bid_volumes["bid_volume_5"] / ask_volumes["ask_volume_5"]
            
            # 计算深度不平衡
            total_volume = bid_volumes["total_bid_volume"] + ask_volumes["total_ask_volume"]
            if total_volume > 0:
                metrics["depth_imbalance"] = (
                    bid_volumes["total_bid_volume"] - ask_volumes["total_ask_volume"]
                ) / total_volume
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算深度指标失败: {e}")
            return {}
    
    async def _trigger_real_time_analysis(self, symbol: str, interval: str) -> None:
        """触发实时分析"""
        try:
            # 这里可以添加实时分析逻辑
            # 例如：检测价格突破、计算技术指标等
            
            # 获取最新的K线数据
            kline_data = await self.get_real_time_klines(symbol, interval, limit=100)
            
            if kline_data is not None and len(kline_data) >= 20:
                # 执行快速技术分析
                analysis_result = await self._perform_quick_analysis(symbol, kline_data)
                
                # 缓存分析结果
                cache_key = f"analysis_{symbol}_{interval}"
                self._realtime_cache[cache_key] = {
                    "result": analysis_result,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"触发实时分析失败: {e}")
    
    async def _perform_quick_analysis(self, symbol: str, kline_data: pd.DataFrame) -> Dict[str, Any]:
        """执行快速技术分析"""
        try:
            # 计算基础技术指标
            close_prices = kline_data['close']
            
            # RSI
            rsi = self._calculate_simple_rsi(close_prices, 14)
            
            # EMA
            ema_short = close_prices.ewm(span=9).mean().iloc[-1]
            ema_long = close_prices.ewm(span=21).mean().iloc[-1]
            
            # 价格变化
            price_change = (close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]
            
            # 成交量变化（如果有）
            volume_change = 0
            if 'volume' in kline_data.columns:
                volumes = kline_data['volume']
                if len(volumes) >= 2:
                    volume_change = (volumes.iloc[-1] - volumes.iloc[-2]) / volumes.iloc[-2]
            
            return {
                "symbol": symbol,
                "current_price": float(close_prices.iloc[-1]),
                "rsi": float(rsi),
                "ema_short": float(ema_short),
                "ema_long": float(ema_long),
                "price_change_pct": float(price_change * 100),
                "volume_change_pct": float(volume_change * 100),
                "trend_signal": "bullish" if ema_short > ema_long else "bearish"
            }
            
        except Exception as e:
            self.logger.error(f"快速分析失败: {e}")
            return {}
    
    def _calculate_simple_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算简单RSI"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
            
        except Exception:
            return 50.0
    
    async def get_real_time_klines(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        获取实时K线数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            limit: 返回条数限制
            
        Returns:
            pd.DataFrame: K线数据或None
        """
        try:
            if self._use_real_data:
                # 从实时缓存获取
                buffer_key = f"kline_{symbol}_{interval}"
                cached_data = self._kline_buffers.get(buffer_key, [])
                
                if cached_data and len(cached_data) >= limit:
                    # 转换为DataFrame
                    df_data = cached_data[-limit:]
                    df = pd.DataFrame(df_data)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # 只返回需要的列
                    columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                    available_columns = [col for col in columns if col in df.columns]
                    
                    return df[available_columns]
                
                # 如果缓存不足，从REST API获取
                return await self._fetch_historical_klines(symbol, interval, limit)
            else:
                # 生成模拟数据
                return self._generate_mock_klines(symbol, interval, limit)
                
        except Exception as e:
            self.logger.error(f"获取实时K线数据失败 {symbol} {interval}: {e}")
            return None
    
    async def _fetch_historical_klines(
        self, 
        symbol: str, 
        interval: str, 
        limit: int = 100
    ) -> Optional[pd.DataFrame]:
        """从REST API获取历史K线数据"""
        try:
            if not self._usds_client:
                return None
            
            # 使用REST API获取历史K线
            response = await self._usds_client.rest_api.klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if response and hasattr(response, 'data'):
                klines = response.data()
                
                # 转换为DataFrame
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
                ])
                
                # 数据类型转换
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('open_time', inplace=True)
                
                return df[numeric_columns]  # 只返回需要的列
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取历史K线数据失败: {e}")
            return None
    
    def _generate_mock_klines(self, symbol: str, interval: str, limit: int = 100) -> pd.DataFrame:
        """生成模拟K线数据（重用父类方法）"""
        try:
            return super()._generate_mock_klines(symbol, interval, limit)
        except Exception as e:
            self.logger.error(f"生成模拟K线数据失败: {e}")
            return pd.DataFrame()
    
    async def analyze_market_depth(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """分析实时市场深度"""
        try:
            if self._use_real_data:
                # 从实时缓存获取深度数据
                depth_data = self._depth_cache.get(symbol)
                
                if not depth_data:
                    # 如果缓存中没有，尝试从REST API获取
                    depth_data = await self._fetch_order_book(symbol, limit)
                
                if depth_data:
                    return self._analyze_depth_data(depth_data)
            else:
                # 生成模拟深度数据
                depth_data = self._generate_mock_depth(symbol, limit)
                return self._analyze_depth_data(depth_data)
            
            return None
            
        except Exception as e:
            self.logger.error(f"分析市场深度失败 {symbol}: {e}")
            return None
    
    async def _fetch_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """从REST API获取订单簿数据"""
        try:
            if not self._usds_client:
                return None
            
            response = await self._usds_client.rest_api.depth(
                symbol=symbol,
                limit=limit
            )
            
            if response and hasattr(response, 'data'):
                data = response.data()
                
                bids = [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])]
                
                return {
                    'timestamp': data.get('lastUpdateId', 0),
                    'bids': bids,
                    'asks': asks,
                    'best_bid': bids[0][0] if bids else 0,
                    'best_ask': asks[0][0] if asks else 0
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"获取订单簿数据失败: {e}")
            return None
    
    async def _start_mock_data_generation(self, symbols: List[str], intervals: List[str]) -> None:
        """启动模拟数据生成"""
        try:
            self.logger.info("启动模拟数据生成器")
            
            # 为每个交易对生成模拟数据
            for symbol in symbols:
                # 生成初始价格
                base_price = self._get_base_price(symbol)
                self._realtime_cache[f"current_price_{symbol}"] = base_price
                
                # 生成模拟深度数据
                depth_data = self._generate_mock_depth(symbol, 20)
                self._depth_cache[symbol] = depth_data
                
                # 启动模拟K线数据生成
                for interval in intervals:
                    asyncio.create_task(self._generate_mock_kline_stream(symbol, interval))
                
                # 启动模拟交易数据生成
                asyncio.create_task(self._generate_mock_trade_stream(symbol))
            
        except Exception as e:
            self.logger.error(f"启动模拟数据生成失败: {e}")
    
    async def _generate_mock_kline_stream(self, symbol: str, interval: str) -> None:
        """生成模拟K线数据流"""
        try:
            buffer_key = f"kline_{symbol}_{interval}"
            
            # 初始化缓冲区
            if buffer_key not in self._kline_buffers:
                self._kline_buffers[buffer_key] = []
            
            interval_seconds = self._get_interval_seconds(interval)
            current_price = self._realtime_cache.get(f"current_price_{symbol}", 50000.0)
            
            while True:
                # 生成新的K线数据
                timestamp = int(datetime.now().timestamp() * 1000)
                
                # 模拟价格变动
                price_change = random.uniform(-0.02, 0.02)  # ±2%变动
                new_price = current_price * (1 + price_change)
                
                # 生成OHLC
                high = new_price * random.uniform(1.0, 1.01)
                low = new_price * random.uniform(0.99, 1.0)
                volume = random.uniform(100, 1000)
                
                kline_data = {
                    "timestamp": timestamp,
                    "open": current_price,
                    "high": high,
                    "low": low,
                    "close": new_price,
                    "volume": volume,
                    "quote_volume": volume * (high + low) / 2,
                    "trades": random.randint(50, 200),
                    "is_closed": True
                }
                
                # 更新缓冲区
                self._kline_buffers[buffer_key].append(kline_data)
                
                # 保持缓冲区大小
                if len(self._kline_buffers[buffer_key]) > 1000:
                    self._kline_buffers[buffer_key] = self._kline_buffers[buffer_key][-1000:]
                
                # 更新当前价格
                current_price = new_price
                self._realtime_cache[f"current_price_{symbol}"] = current_price
                
                # 等待下一个周期
                await asyncio.sleep(interval_seconds)
                
        except Exception as e:
            self.logger.error(f"生成模拟K线流失败: {e}")
    
    async def _generate_mock_trade_stream(self, symbol: str) -> None:
        """生成模拟交易数据流"""
        try:
            if symbol not in self._trade_cache:
                self._trade_cache[symbol] = []
            
            while True:
                current_price = self._realtime_cache.get(f"current_price_{symbol}", 50000.0)
                
                # 生成模拟交易
                price_variation = random.uniform(-0.001, 0.001)  # ±0.1%变动
                trade_price = current_price * (1 + price_variation)
                
                trade_data = {
                    "timestamp": int(datetime.now().timestamp() * 1000),
                    "symbol": symbol,
                    "price": trade_price,
                    "quantity": random.uniform(0.1, 10.0),
                    "is_buyer_maker": random.choice([True, False])
                }
                
                # 更新缓存
                self._trade_cache[symbol].append(trade_data)
                
                # 保持缓存大小
                if len(self._trade_cache[symbol]) > 500:
                    self._trade_cache[symbol] = self._trade_cache[symbol][-500:]
                
                # 随机间隔（模拟真实交易频率）
                await asyncio.sleep(random.uniform(0.1, 2.0))
                
        except Exception as e:
            self.logger.error(f"生成模拟交易流失败: {e}")
    
    def _get_base_price(self, symbol: str) -> float:
        """获取交易对的基础价格"""
        symbol_upper = symbol.upper()
        if 'BTC' in symbol_upper:
            return 50000.0
        elif 'ETH' in symbol_upper:
            return 3000.0
        elif 'BNB' in symbol_upper:
            return 300.0
        else:
            return 100.0
    
    def _get_interval_seconds(self, interval: str) -> float:
        """获取时间间隔的秒数"""
        interval_map = {
            '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600,
            '8h': 28800, '12h': 43200, '1d': 86400
        }
        return interval_map.get(interval, 300)  # 默认5分钟
    
    async def _monitor_data_quality(self) -> None:
        """监控数据质量"""
        try:
            while True:
                await asyncio.sleep(60)  # 每分钟检查一次
                
                current_time = datetime.now()
                stale_streams = []
                
                # 检查数据流是否存在延迟
                for stream_key, last_update in self._last_update_times.items():
                    time_diff = (current_time - last_update).total_seconds()
                    
                    if time_diff > 300:  # 5分钟无数据更新
                        stale_streams.append(stream_key)
                
                if stale_streams:
                    self.logger.warning(f"发现延迟数据流: {stale_streams}")
                
                # 检查连接状态
                if self._use_real_data and not self._is_connected:
                    self.logger.warning("WebSocket连接已断开，尝试重连")
                    await self._handle_connection_error()
                
        except Exception as e:
            self.logger.error(f"数据质量监控失败: {e}")
    
    async def _handle_connection_error(self) -> None:
        """处理连接错误"""
        try:
            if self._reconnect_attempts >= self._max_reconnect_attempts:
                self.logger.error("达到最大重连次数，切换到模拟模式")
                self._use_real_data = False
                self._mock_mode = True
                return
            
            self._reconnect_attempts += 1
            wait_time = min(60, 5 * self._reconnect_attempts)  # 指数退避
            
            self.logger.info(f"等待{wait_time}秒后重连... (尝试: {self._reconnect_attempts})")
            await asyncio.sleep(wait_time)
            
            # 重新建立连接
            if self._websocket_connection:
                try:
                    await self._websocket_connection.close_connection(close_session=True)
                except:
                    pass
                self._websocket_connection = None
            
            await self._establish_websocket_connection()
            
        except Exception as e:
            self.logger.error(f"处理连接错误失败: {e}")
    
    async def stop_all_streams(self) -> None:
        """停止所有数据流"""
        try:
            # 停止所有活跃的流
            for stream_key, stream in self._active_streams.items():
                try:
                    await stream.unsubscribe()
                    self.logger.info(f"已停止数据流: {stream_key}")
                except Exception as e:
                    self.logger.error(f"停止数据流失败 {stream_key}: {e}")
            
            self._active_streams.clear()
            
            # 关闭WebSocket连接
            if self._websocket_connection:
                try:
                    await self._websocket_connection.close_connection(close_session=True)
                    self._websocket_connection = None
                    self.logger.info("WebSocket连接已关闭")
                except Exception as e:
                    self.logger.error(f"关闭WebSocket连接失败: {e}")
            
            self._is_connected = False
            
        except Exception as e:
            self.logger.error(f"停止数据流失败: {e}")
    
    # 覆盖父类方法以使用增强功能
    async def analyze_market_comprehensive(
        self,
        ticker: str,
        price_data: Optional[pd.DataFrame] = None,
        volume_data: Optional[pd.DataFrame] = None,
        timeframes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行全面的市场分析（增强版）
        
        如果没有提供price_data，将尝试从实时数据获取
        """
        try:
            # 如果没有提供数据，从实时源获取
            if price_data is None:
                price_data = await self.get_real_time_klines(ticker, "1h", 200)
                
                if price_data is None or len(price_data) < 50:
                    self.logger.warning(f"无法获取足够的价格数据: {ticker}")
                    return self._generate_empty_analysis(ticker)
            
            # 调用父类的全面分析
            analysis_result = await super().analyze_market_comprehensive(
                ticker, price_data, volume_data, timeframes
            )
            
            # 添加实时数据增强
            if self._use_real_data:
                real_time_data = self._get_real_time_enhancements(ticker)
                analysis_result["real_time_data"] = real_time_data
                
                # 更新分析置信度
                analysis_result["analysis_confidence"] *= self._calculate_real_time_confidence(ticker)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"全面市场分析失败 {ticker}: {e}")
            return self._generate_empty_analysis(ticker)
    
    def _get_real_time_enhancements(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据增强"""
        try:
            enhancements = {
                "current_price": self._realtime_cache.get(f"current_price_{symbol}"),
                "latest_trade": self._realtime_cache.get(f"latest_trade_{symbol}"),
                "market_depth": self._depth_cache.get(symbol),
                "stream_status": self._get_stream_status(symbol),
                "data_freshness": self._calculate_data_freshness(symbol)
            }
            
            return {k: v for k, v in enhancements.items() if v is not None}
            
        except Exception as e:
            self.logger.error(f"获取实时数据增强失败: {e}")
            return {}
    
    def _get_stream_status(self, symbol: str) -> Dict[str, Any]:
        """获取数据流状态"""
        try:
            status = {}
            
            for stream_key, stats in self._stream_stats.items():
                if symbol in stream_key:
                    status[stream_key] = {
                        "messages_received": stats["messages_received"],
                        "last_message_time": stats["last_message_time"].isoformat() if stats["last_message_time"] else None,
                        "errors": stats["errors"]
                    }
            
            return status
            
        except Exception as e:
            self.logger.error(f"获取流状态失败: {e}")
            return {}
    
    def _calculate_data_freshness(self, symbol: str) -> Dict[str, float]:
        """计算数据新鲜度"""
        try:
            freshness = {}
            current_time = datetime.now()
            
            for stream_key, last_update in self._last_update_times.items():
                if symbol in stream_key:
                    age_seconds = (current_time - last_update).total_seconds()
                    freshness[stream_key] = age_seconds
            
            return freshness
            
        except Exception as e:
            self.logger.error(f"计算数据新鲜度失败: {e}")
            return {}
    
    def _calculate_real_time_confidence(self, symbol: str) -> float:
        """计算实时数据置信度"""
        try:
            if not self._use_real_data:
                return 0.8  # 模拟数据置信度较低
            
            confidence_factors = []
            
            # 检查数据流活跃度
            active_streams = sum(1 for key in self._active_streams.keys() if symbol in key)
            confidence_factors.append(min(1.0, active_streams / 3))  # 期望至少3个流
            
            # 检查数据新鲜度
            freshness = self._calculate_data_freshness(symbol)
            if freshness:
                avg_age = sum(freshness.values()) / len(freshness)
                freshness_factor = max(0.5, 1.0 - avg_age / 300)  # 5分钟内为满分
                confidence_factors.append(freshness_factor)
            
            # 检查错误率
            total_messages = sum(
                stats["messages_received"] 
                for key, stats in self._stream_stats.items() 
                if symbol in key
            )
            total_errors = sum(
                stats["errors"] 
                for key, stats in self._stream_stats.items() 
                if symbol in key
            )
            
            if total_messages > 0:
                error_rate = total_errors / total_messages
                error_factor = max(0.3, 1.0 - error_rate * 10)  # 10%错误率为0.3分
                confidence_factors.append(error_factor)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"计算实时数据置信度失败: {e}")
            return 0.5
    
    def _generate_empty_analysis(self, ticker: str) -> Dict[str, Any]:
        """生成空分析结果"""
        return {
            "ticker": ticker,
            "analysis_timestamp": datetime.now().isoformat(),
            "error": "数据不足或分析失败",
            "analysis_confidence": 0.0,
            "data_mode": "real_data" if self._use_real_data else "mock_data"
        }
    
    # 公共接口方法
    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格"""
        return self._realtime_cache.get(f"current_price_{symbol}")
    
    def get_market_depth_summary(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取市场深度摘要"""
        depth_data = self._depth_cache.get(symbol)
        if depth_data:
            return {
                'best_bid': depth_data.get('best_bid'),
                'best_ask': depth_data.get('best_ask'),
                'spread': depth_data.get('spread'),
                'spread_percent': depth_data.get('spread_percent'),
                'timestamp': depth_data.get('timestamp')
            }
        return None
    
    def get_latest_trades(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最新交易"""
        trades = self._trade_cache.get(symbol, [])
        return trades[-limit:] if trades else []
    
    def is_using_real_data(self) -> bool:
        """是否使用真实数据"""
        return self._use_real_data
    
    def get_active_streams(self) -> List[str]:
        """获取活跃的数据流列表"""
        return list(self._active_streams.keys())
    
    def get_stream_statistics(self) -> Dict[str, Dict[str, Any]]:
        """获取数据流统计信息"""
        return dict(self._stream_stats)
    
    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态"""
        return {
            'realtime_cache_size': len(self._realtime_cache),
            'kline_buffers': {k: len(v) for k, v in self._kline_buffers.items()},
            'depth_cache_size': len(self._depth_cache),
            'trade_cache': {k: len(v) for k, v in self._trade_cache.items()},
            'active_streams': len(self._active_streams),
            'is_connected': self._is_connected,
            'reconnect_attempts': self._reconnect_attempts
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.stop_all_streams()