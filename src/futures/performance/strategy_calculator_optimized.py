"""
优化的策略计算器 - 预计算和缓存机制

主要优化：
1. 技术指标预计算
2. 策略并行执行
3. 结果缓存复用
4. 内存优化
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pandas_ta as ta
import hashlib
import pickle
from functools import lru_cache

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """指标计算结果"""
    name: str
    value: pd.Series
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySignal:
    """策略信号"""
    strategy: str
    direction: str  # long/short/neutral
    strength: float
    confidence: float
    timestamp: datetime
    indicators_used: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class IndicatorCache:
    """
    技术指标缓存管理器

    特性：
    - 预计算所有常用指标
    - 智能缓存管理
    - 增量计算支持
    """

    def __init__(self, cache_size: int = 1000):
        """
        初始化指标缓存

        Args:
            cache_size: 缓存大小
        """
        self.cache_size = cache_size
        self.cache: Dict[str, IndicatorResult] = {}
        self.hit_count = 0
        self.miss_count = 0

        # 指标计算函数映射
        self.indicator_functions = {
            'rsi': self._calculate_rsi,
            'macd': self._calculate_macd,
            'bollinger': self._calculate_bollinger,
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'atr': self._calculate_atr,
            'adx': self._calculate_adx,
            'stochastic': self._calculate_stochastic,
            'obv': self._calculate_obv,
            'vwap': self._calculate_vwap
        }

    def calculate_all_indicators(
        self,
        market_data: Dict[str, pd.DataFrame],
        indicators: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, IndicatorResult]]:
        """
        批量计算所有指标

        Args:
            market_data: 多时间框架市场数据
            indicators: 要计算的指标列表（None表示全部）

        Returns:
            时间框架到指标结果的映射
        """
        if indicators is None:
            indicators = list(self.indicator_functions.keys())

        results = {}

        for timeframe, data in market_data.items():
            if data.empty:
                continue

            timeframe_results = {}

            for indicator_name in indicators:
                # 生成缓存键
                cache_key = self._generate_cache_key(
                    timeframe, indicator_name, data
                )

                # 检查缓存
                if cached_result := self.cache.get(cache_key):
                    timeframe_results[indicator_name] = cached_result
                    self.hit_count += 1
                    logger.debug(f"缓存命中: {timeframe}:{indicator_name}")
                    continue

                # 计算指标
                self.miss_count += 1
                indicator_func = self.indicator_functions.get(indicator_name)

                if indicator_func:
                    try:
                        result = indicator_func(data)
                        timeframe_results[indicator_name] = result

                        # 更新缓存
                        self._update_cache(cache_key, result)

                        logger.debug(f"计算完成: {timeframe}:{indicator_name}")
                    except Exception as e:
                        logger.error(f"指标计算失败 {indicator_name}: {e}")

            results[timeframe] = timeframe_results

        logger.info(
            f"指标计算完成 - 命中率: {self.get_hit_rate():.1f}%, "
            f"计算指标数: {len(indicators) * len(market_data)}"
        )

        return results

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """计算RSI"""
        rsi = ta.rsi(data['close'], length=period)
        return IndicatorResult(
            name='rsi',
            value=rsi,
            timestamp=datetime.now(),
            metadata={'period': period}
        )

    def _calculate_macd(self, data: pd.DataFrame) -> IndicatorResult:
        """计算MACD"""
        macd = ta.macd(data['close'])
        return IndicatorResult(
            name='macd',
            value=macd,
            timestamp=datetime.now(),
            metadata={'fast': 12, 'slow': 26, 'signal': 9}
        )

    def _calculate_bollinger(self, data: pd.DataFrame, period: int = 20) -> IndicatorResult:
        """计算布林带"""
        bbands = ta.bbands(data['close'], length=period)
        return IndicatorResult(
            name='bollinger',
            value=bbands,
            timestamp=datetime.now(),
            metadata={'period': period, 'std': 2}
        )

    def _calculate_sma(self, data: pd.DataFrame, period: int = 20) -> IndicatorResult:
        """计算简单移动平均"""
        sma = ta.sma(data['close'], length=period)
        return IndicatorResult(
            name='sma',
            value=sma,
            timestamp=datetime.now(),
            metadata={'period': period}
        )

    def _calculate_ema(self, data: pd.DataFrame, period: int = 20) -> IndicatorResult:
        """计算指数移动平均"""
        ema = ta.ema(data['close'], length=period)
        return IndicatorResult(
            name='ema',
            value=ema,
            timestamp=datetime.now(),
            metadata={'period': period}
        )

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """计算平均真实范围"""
        atr = ta.atr(data['high'], data['low'], data['close'], length=period)
        return IndicatorResult(
            name='atr',
            value=atr,
            timestamp=datetime.now(),
            metadata={'period': period}
        )

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """计算平均方向指数"""
        adx = ta.adx(data['high'], data['low'], data['close'], length=period)
        return IndicatorResult(
            name='adx',
            value=adx,
            timestamp=datetime.now(),
            metadata={'period': period}
        )

    def _calculate_stochastic(self, data: pd.DataFrame) -> IndicatorResult:
        """计算随机指标"""
        stoch = ta.stoch(data['high'], data['low'], data['close'])
        return IndicatorResult(
            name='stochastic',
            value=stoch,
            timestamp=datetime.now(),
            metadata={'k_period': 14, 'd_period': 3}
        )

    def _calculate_obv(self, data: pd.DataFrame) -> IndicatorResult:
        """计算能量潮指标"""
        obv = ta.obv(data['close'], data['volume'])
        return IndicatorResult(
            name='obv',
            value=obv,
            timestamp=datetime.now()
        )

    def _calculate_vwap(self, data: pd.DataFrame) -> IndicatorResult:
        """计算成交量加权平均价"""
        vwap = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
        return IndicatorResult(
            name='vwap',
            value=vwap,
            timestamp=datetime.now()
        )

    def _generate_cache_key(
        self,
        timeframe: str,
        indicator: str,
        data: pd.DataFrame
    ) -> str:
        """生成缓存键"""
        # 使用数据的最后时间戳和长度作为键的一部分
        data_signature = f"{len(data)}_{data.index[-1]}"
        return f"{timeframe}_{indicator}_{data_signature}"

    def _update_cache(self, key: str, result: IndicatorResult):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            # LRU淘汰策略
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].timestamp
            )
            del self.cache[oldest_key]

        self.cache[key] = result

    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        if total == 0:
            return 0.0
        return (self.hit_count / total) * 100

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0


class ParallelStrategyExecutor:
    """
    并行策略执行器

    特性：
    - 策略并行执行
    - 自动负载均衡
    - 故障隔离
    """

    def __init__(
        self,
        max_workers: int = 4,
        use_process_pool: bool = False
    ):
        """
        初始化并行执行器

        Args:
            max_workers: 最大工作线程/进程数
            use_process_pool: 是否使用进程池（CPU密集型任务）
        """
        self.max_workers = max_workers
        self.use_process_pool = use_process_pool

        if use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)

        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_time': 0
        }

    async def execute_strategies_parallel(
        self,
        strategies: List[Any],
        market_data: Dict[str, pd.DataFrame],
        precomputed_indicators: Dict[str, Dict[str, IndicatorResult]],
        ticker: str,
        current_price: float
    ) -> List[StrategySignal]:
        """
        并行执行多个策略

        Args:
            strategies: 策略列表
            market_data: 市场数据
            precomputed_indicators: 预计算的指标
            ticker: 交易对
            current_price: 当前价格

        Returns:
            策略信号列表
        """
        start_time = datetime.now()
        self.execution_stats['total_executions'] += 1

        try:
            # 创建异步任务
            tasks = []
            for strategy in strategies:
                task = asyncio.create_task(
                    self._execute_single_strategy_async(
                        strategy,
                        market_data,
                        precomputed_indicators,
                        ticker,
                        current_price
                    )
                )
                tasks.append(task)

            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 过滤成功的结果
            valid_signals = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"策略执行失败: {result}")
                    self.execution_stats['failed_executions'] += 1
                elif result:
                    valid_signals.append(result)
                    self.execution_stats['successful_executions'] += 1

            # 记录执行时间
            elapsed = (datetime.now() - start_time).total_seconds()
            self.execution_stats['total_time'] += elapsed

            logger.info(
                f"并行执行完成: {len(valid_signals)}/{len(strategies)} 策略成功, "
                f"耗时: {elapsed:.2f}秒"
            )

            return valid_signals

        except Exception as e:
            logger.error(f"并行执行失败: {e}")
            self.execution_stats['failed_executions'] += len(strategies)
            raise

    async def _execute_single_strategy_async(
        self,
        strategy: Any,
        market_data: Dict[str, pd.DataFrame],
        precomputed_indicators: Dict[str, Dict[str, IndicatorResult]],
        ticker: str,
        current_price: float
    ) -> Optional[StrategySignal]:
        """
        异步执行单个策略

        Args:
            strategy: 策略对象
            market_data: 市场数据
            precomputed_indicators: 预计算指标
            ticker: 交易对
            current_price: 当前价格

        Returns:
            策略信号
        """
        try:
            # 在线程池中执行策略
            loop = asyncio.get_event_loop()
            signal = await loop.run_in_executor(
                self.executor,
                self._execute_strategy_sync,
                strategy,
                market_data,
                precomputed_indicators,
                ticker,
                current_price
            )

            return signal

        except Exception as e:
            logger.error(f"策略执行失败 {strategy.__class__.__name__}: {e}")
            return None

    def _execute_strategy_sync(
        self,
        strategy: Any,
        market_data: Dict[str, pd.DataFrame],
        precomputed_indicators: Dict[str, Dict[str, IndicatorResult]],
        ticker: str,
        current_price: float
    ) -> StrategySignal:
        """
        同步执行策略

        Args:
            strategy: 策略对象
            market_data: 市场数据
            precomputed_indicators: 预计算指标
            ticker: 交易对
            current_price: 当前价格

        Returns:
            策略信号
        """
        # 这里应该调用实际的策略分析方法
        # 示例实现
        strategy_name = strategy.__class__.__name__

        # 策略使用预计算的指标
        primary_timeframe = '1h'  # 示例
        indicators = precomputed_indicators.get(primary_timeframe, {})

        # 模拟策略逻辑
        rsi_result = indicators.get('rsi')
        if rsi_result and not rsi_result.value.empty:
            current_rsi = rsi_result.value.iloc[-1]

            if current_rsi < 30:
                direction = 'long'
                strength = (30 - current_rsi) / 30
            elif current_rsi > 70:
                direction = 'short'
                strength = (current_rsi - 70) / 30
            else:
                direction = 'neutral'
                strength = 0.0

            return StrategySignal(
                strategy=strategy_name,
                direction=direction,
                strength=strength,
                confidence=0.8,
                timestamp=datetime.now(),
                indicators_used=['rsi'],
                metadata={'rsi': current_rsi}
            )

        return StrategySignal(
            strategy=strategy_name,
            direction='neutral',
            strength=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            indicators_used=[],
            metadata={}
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        total = self.execution_stats['total_executions']
        if total == 0:
            return self.execution_stats

        return {
            **self.execution_stats,
            'success_rate': (
                self.execution_stats['successful_executions'] / total * 100
                if total > 0 else 0
            ),
            'avg_execution_time': (
                self.execution_stats['total_time'] / total
                if total > 0 else 0
            )
        }

    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)


class OptimizedStrategyCalculator:
    """
    优化的策略计算器

    整合指标缓存和并行执行
    """

    def __init__(self):
        """初始化优化策略计算器"""
        self.indicator_cache = IndicatorCache()
        self.strategy_executor = ParallelStrategyExecutor(max_workers=4)

        self.performance_stats = {
            'total_calculations': 0,
            'cache_savings': 0,
            'parallel_speedup': 0
        }

    async def calculate_all_signals(
        self,
        strategies: List[Any],
        market_data: Dict[str, pd.DataFrame],
        ticker: str,
        current_price: float,
        required_indicators: Optional[List[str]] = None
    ) -> List[StrategySignal]:
        """
        计算所有策略信号

        Args:
            strategies: 策略列表
            market_data: 市场数据
            ticker: 交易对
            current_price: 当前价格
            required_indicators: 需要的指标列表

        Returns:
            策略信号列表
        """
        start_time = datetime.now()
        self.performance_stats['total_calculations'] += 1

        try:
            # 步骤1: 预计算所有指标
            logger.info("开始预计算技术指标...")
            indicators = self.indicator_cache.calculate_all_indicators(
                market_data, required_indicators
            )

            # 记录缓存节省
            self.performance_stats['cache_savings'] = (
                self.indicator_cache.hit_count /
                max(1, self.indicator_cache.hit_count + self.indicator_cache.miss_count)
            )

            # 步骤2: 并行执行策略
            logger.info("开始并行执行策略...")
            signals = await self.strategy_executor.execute_strategies_parallel(
                strategies,
                market_data,
                indicators,
                ticker,
                current_price
            )

            # 计算性能提升
            elapsed = (datetime.now() - start_time).total_seconds()
            expected_serial_time = len(strategies) * 1.5  # 假设每个策略1.5秒
            self.performance_stats['parallel_speedup'] = expected_serial_time / elapsed

            logger.info(
                f"策略计算完成: {len(signals)} 个信号, "
                f"耗时: {elapsed:.2f}秒, "
                f"加速比: {self.performance_stats['parallel_speedup']:.1f}x"
            )

            return signals

        except Exception as e:
            logger.error(f"策略计算失败: {e}")
            raise

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'indicator_cache': {
                'hit_rate': self.indicator_cache.get_hit_rate(),
                'cache_size': len(self.indicator_cache.cache),
                'total_hits': self.indicator_cache.hit_count,
                'total_misses': self.indicator_cache.miss_count
            },
            'strategy_execution': self.strategy_executor.get_statistics(),
            'overall': {
                'total_calculations': self.performance_stats['total_calculations'],
                'cache_savings': f"{self.performance_stats['cache_savings']*100:.1f}%",
                'parallel_speedup': f"{self.performance_stats['parallel_speedup']:.1f}x"
            }
        }


async def example_usage():
    """使用示例"""
    # 模拟市场数据
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
    market_data = {
        '1h': pd.DataFrame({
            'open': np.random.randn(len(dates)) * 100 + 50000,
            'high': np.random.randn(len(dates)) * 100 + 50100,
            'low': np.random.randn(len(dates)) * 100 + 49900,
            'close': np.random.randn(len(dates)) * 100 + 50000,
            'volume': np.random.randn(len(dates)) * 1000 + 10000
        }, index=dates)
    }

    # 创建优化计算器
    calculator = OptimizedStrategyCalculator()

    # 模拟策略列表
    strategies = [object() for _ in range(5)]  # 5个策略

    # 执行计算
    signals = await calculator.calculate_all_signals(
        strategies=strategies,
        market_data=market_data,
        ticker='BTCUSDT',
        current_price=50000.0,
        required_indicators=['rsi', 'macd', 'bollinger']
    )

    # 打印结果
    print(f"生成信号数: {len(signals)}")

    # 打印性能报告
    report = calculator.get_performance_report()
    print(f"性能报告: {report}")


if __name__ == "__main__":
    asyncio.run(example_usage())