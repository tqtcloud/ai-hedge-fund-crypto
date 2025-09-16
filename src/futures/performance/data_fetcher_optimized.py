"""
优化的数据获取器 - 并行获取和智能缓存

主要优化：
1. 并行数据获取
2. 多级缓存机制
3. 连接池管理
4. 增量数据更新
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging
from dataclasses import dataclass
import hashlib
import json
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    data: pd.DataFrame
    timestamp: datetime
    hit_count: int = 0
    last_access: datetime = None


class OptimizedDataFetcher:
    """
    优化的数据获取器

    特性：
    - 并行获取多时间框架数据
    - 智能缓存管理
    - 连接池复用
    - 增量数据更新
    """

    def __init__(self,
                 api_client: Any,
                 cache_size: int = 100,
                 enable_compression: bool = True):
        """
        初始化优化数据获取器

        Args:
            api_client: API客户端
            cache_size: 缓存大小
            enable_compression: 是否启用压缩
        """
        self.api_client = api_client
        self.cache_size = cache_size
        self.enable_compression = enable_compression

        # 多级缓存
        self.l1_cache: Dict[str, CacheEntry] = {}  # 内存缓存
        self.l2_cache: Dict[str, CacheEntry] = {}  # 二级缓存

        # 缓存TTL配置（秒）
        self.cache_ttl = {
            '1m': 30,      # 30秒
            '5m': 150,     # 2.5分钟
            '15m': 450,    # 7.5分钟
            '30m': 900,    # 15分钟
            '1h': 1800,    # 30分钟
            '4h': 7200,    # 2小时
            '1d': 43200,   # 12小时
        }

        # 性能统计
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'total_fetch_time': 0,
            'parallel_fetches': 0
        }

        # 线程池用于CPU密集型操作
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 连接池
        self.session = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300
            )
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def fetch_all_timeframes_parallel(
        self,
        symbol: str,
        timeframes: List[str],
        limit: int = 1000,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        并行获取所有时间框架数据

        Args:
            symbol: 交易对
            timeframes: 时间框架列表
            limit: 数据条数限制
            use_cache: 是否使用缓存

        Returns:
            时间框架到数据的映射
        """
        start_time = datetime.now()

        try:
            # 1. 检查缓存
            if use_cache:
                cached_data = self._check_multi_cache(symbol, timeframes)
                if len(cached_data) == len(timeframes):
                    logger.info(f"全部数据从缓存获取: {symbol}")
                    return cached_data

                # 找出需要获取的时间框架
                missing_timeframes = [
                    tf for tf in timeframes
                    if tf not in cached_data
                ]
            else:
                missing_timeframes = timeframes
                cached_data = {}

            # 2. 并行获取缺失的数据
            if missing_timeframes:
                logger.info(f"并行获取数据: {symbol} - {missing_timeframes}")

                # 创建异步任务
                tasks = []
                for timeframe in missing_timeframes:
                    task = self._fetch_single_timeframe_async(
                        symbol, timeframe, limit
                    )
                    tasks.append(task)

                # 并行执行
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # 处理结果
                for timeframe, result in zip(missing_timeframes, results):
                    if isinstance(result, Exception):
                        logger.error(f"获取数据失败 {symbol}:{timeframe} - {result}")
                        continue

                    # 更新缓存
                    self._update_cache(symbol, timeframe, result)
                    cached_data[timeframe] = result

                self.stats['parallel_fetches'] += 1

            # 3. 记录性能统计
            elapsed = (datetime.now() - start_time).total_seconds()
            self.stats['total_fetch_time'] += elapsed

            logger.info(
                f"数据获取完成: {symbol}, "
                f"耗时: {elapsed:.2f}秒, "
                f"缓存命中率: {self._get_cache_hit_rate():.1f}%"
            )

            return cached_data

        except Exception as e:
            logger.error(f"并行获取数据失败: {e}")
            raise

    async def _fetch_single_timeframe_async(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """
        异步获取单个时间框架数据

        Args:
            symbol: 交易对
            timeframe: 时间框架
            limit: 数据条数

        Returns:
            价格数据DataFrame
        """
        try:
            # 记录API调用
            self.stats['api_calls'] += 1

            # 调用API获取数据
            response = await self.api_client.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=limit
            )

            # 转换为DataFrame
            df = self._convert_to_dataframe(response)

            # 数据验证
            if not self._validate_data(df):
                raise ValueError(f"数据验证失败: {symbol}:{timeframe}")

            return df

        except Exception as e:
            logger.error(f"获取数据失败 {symbol}:{timeframe} - {e}")
            raise

    def _convert_to_dataframe(self, data: List) -> pd.DataFrame:
        """
        转换API响应为DataFrame

        Args:
            data: API响应数据

        Returns:
            DataFrame
        """
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])

        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # 确保数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性

        Args:
            df: 数据DataFrame

        Returns:
            是否有效
        """
        if df.empty:
            return False

        # 检查必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return False

        # 检查数据合理性
        if (df['high'] < df['low']).any():
            return False

        if (df['close'] <= 0).any():
            return False

        return True

    def _check_multi_cache(
        self,
        symbol: str,
        timeframes: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        检查多个时间框架的缓存

        Args:
            symbol: 交易对
            timeframes: 时间框架列表

        Returns:
            缓存的数据
        """
        cached_data = {}

        for timeframe in timeframes:
            cache_key = self._get_cache_key(symbol, timeframe)

            # 检查L1缓存
            if cache_entry := self.l1_cache.get(cache_key):
                if self._is_cache_valid(cache_entry, timeframe):
                    cached_data[timeframe] = cache_entry.data
                    cache_entry.hit_count += 1
                    cache_entry.last_access = datetime.now()
                    self.stats['cache_hits'] += 1
                    continue

            # 检查L2缓存
            if cache_entry := self.l2_cache.get(cache_key):
                if self._is_cache_valid(cache_entry, timeframe):
                    # 提升到L1缓存
                    self._promote_to_l1(cache_key, cache_entry)
                    cached_data[timeframe] = cache_entry.data
                    cache_entry.hit_count += 1
                    cache_entry.last_access = datetime.now()
                    self.stats['cache_hits'] += 1
                    continue

            self.stats['cache_misses'] += 1

        return cached_data

    def _is_cache_valid(self, entry: CacheEntry, timeframe: str) -> bool:
        """
        检查缓存是否有效

        Args:
            entry: 缓存条目
            timeframe: 时间框架

        Returns:
            是否有效
        """
        if not entry or entry.data.empty:
            return False

        ttl = self.cache_ttl.get(timeframe, 300)
        age = (datetime.now() - entry.timestamp).total_seconds()

        return age < ttl

    def _update_cache(
        self,
        symbol: str,
        timeframe: str,
        data: pd.DataFrame
    ):
        """
        更新缓存

        Args:
            symbol: 交易对
            timeframe: 时间框架
            data: 数据
        """
        cache_key = self._get_cache_key(symbol, timeframe)

        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            last_access=datetime.now()
        )

        # 更新L1缓存
        if len(self.l1_cache) >= self.cache_size:
            self._evict_cache()

        self.l1_cache[cache_key] = entry

    def _promote_to_l1(self, cache_key: str, entry: CacheEntry):
        """
        将缓存条目提升到L1

        Args:
            cache_key: 缓存键
            entry: 缓存条目
        """
        if len(self.l1_cache) >= self.cache_size:
            self._evict_cache()

        self.l1_cache[cache_key] = entry
        del self.l2_cache[cache_key]

    def _evict_cache(self):
        """
        缓存淘汰策略 - LRU
        """
        if not self.l1_cache:
            return

        # 找出最久未访问的条目
        oldest_key = min(
            self.l1_cache.keys(),
            key=lambda k: self.l1_cache[k].last_access or datetime.min
        )

        # 移动到L2缓存
        self.l2_cache[oldest_key] = self.l1_cache[oldest_key]
        del self.l1_cache[oldest_key]

        # L2缓存大小控制
        if len(self.l2_cache) > self.cache_size * 2:
            # 删除最老的一半
            sorted_keys = sorted(
                self.l2_cache.keys(),
                key=lambda k: self.l2_cache[k].last_access or datetime.min
            )
            for key in sorted_keys[:len(sorted_keys)//2]:
                del self.l2_cache[key]

    def _get_cache_key(self, symbol: str, timeframe: str) -> str:
        """
        生成缓存键

        Args:
            symbol: 交易对
            timeframe: 时间框架

        Returns:
            缓存键
        """
        return f"{symbol}_{timeframe}"

    def _get_cache_hit_rate(self) -> float:
        """
        获取缓存命中率

        Returns:
            命中率百分比
        """
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        if total == 0:
            return 0.0

        return (self.stats['cache_hits'] / total) * 100

    async def update_incremental_data(
        self,
        symbol: str,
        timeframe: str,
        since_timestamp: Optional[int] = None
    ) -> pd.DataFrame:
        """
        增量更新数据

        Args:
            symbol: 交易对
            timeframe: 时间框架
            since_timestamp: 起始时间戳

        Returns:
            更新后的完整数据
        """
        cache_key = self._get_cache_key(symbol, timeframe)

        # 获取现有缓存数据
        existing_data = None
        if cache_entry := self.l1_cache.get(cache_key):
            existing_data = cache_entry.data

        if existing_data is None or existing_data.empty:
            # 没有缓存，获取全量数据
            return await self._fetch_single_timeframe_async(
                symbol, timeframe, 1000
            )

        # 计算需要获取的新数据时间范围
        last_timestamp = existing_data.index[-1]

        # 获取增量数据
        new_data = await self._fetch_single_timeframe_async(
            symbol, timeframe, 100  # 只获取最新的100条
        )

        # 合并数据
        if not new_data.empty:
            # 过滤出新数据
            new_data = new_data[new_data.index > last_timestamp]

            if not new_data.empty:
                # 合并并去重
                combined_data = pd.concat([existing_data, new_data])
                combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

                # 限制数据大小
                if len(combined_data) > 5000:
                    combined_data = combined_data.iloc[-5000:]

                # 更新缓存
                self._update_cache(symbol, timeframe, combined_data)

                return combined_data

        return existing_data

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取性能统计

        Returns:
            统计信息
        """
        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'cache_hit_rate': self._get_cache_hit_rate(),
            'api_calls': self.stats['api_calls'],
            'total_fetch_time': self.stats['total_fetch_time'],
            'parallel_fetches': self.stats['parallel_fetches'],
            'l1_cache_size': len(self.l1_cache),
            'l2_cache_size': len(self.l2_cache)
        }

    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        清理缓存

        Args:
            symbol: 交易对（可选）
            timeframe: 时间框架（可选）
        """
        if symbol and timeframe:
            # 清理特定缓存
            cache_key = self._get_cache_key(symbol, timeframe)
            self.l1_cache.pop(cache_key, None)
            self.l2_cache.pop(cache_key, None)
        elif symbol:
            # 清理特定交易对的所有缓存
            keys_to_remove = [
                k for k in self.l1_cache.keys()
                if k.startswith(symbol)
            ]
            for key in keys_to_remove:
                del self.l1_cache[key]

            keys_to_remove = [
                k for k in self.l2_cache.keys()
                if k.startswith(symbol)
            ]
            for key in keys_to_remove:
                del self.l2_cache[key]
        else:
            # 清理所有缓存
            self.l1_cache.clear()
            self.l2_cache.clear()

        logger.info(f"缓存已清理: symbol={symbol}, timeframe={timeframe}")


async def example_usage():
    """使用示例"""
    from some_api import APIClient  # 假设的API客户端

    api_client = APIClient()

    async with OptimizedDataFetcher(api_client) as fetcher:
        # 并行获取多个时间框架数据
        timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        data = await fetcher.fetch_all_timeframes_parallel(
            symbol='BTCUSDT',
            timeframes=timeframes
        )

        # 打印统计信息
        stats = fetcher.get_statistics()
        print(f"性能统计: {stats}")

        # 增量更新
        updated_data = await fetcher.update_incremental_data(
            symbol='BTCUSDT',
            timeframe='1m'
        )

        print(f"更新后数据条数: {len(updated_data)}")


if __name__ == "__main__":
    asyncio.run(example_usage())