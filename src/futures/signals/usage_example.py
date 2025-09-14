"""
期货策略集成机制使用示例

展示如何使用统一的策略接口、工厂模式和管理器来
进行期货策略分析和信号生成。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict
import asyncio
import logging

from .strategy_factory import strategy_factory
from .strategy_manager import strategy_manager, AggregationMethod, ValidationLevel
from .base_strategy import StrategyConfig, StrategyType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data(ticker: str, timeframes: list) -> Dict[str, pd.DataFrame]:
    """
    创建示例市场数据

    Args:
        ticker: 交易对
        timeframes: 时间框架列表

    Returns:
        多时间框架数据字典
    """
    data = {}

    for timeframe in timeframes:
        # 生成100个数据点
        np.random.seed(42)  # 确保可重复
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

        # 生成OHLC数据
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)  # BTC价格模拟
        high_prices = close_prices + np.random.uniform(50, 200, 100)
        low_prices = close_prices - np.random.uniform(50, 200, 100)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]

        volumes = np.random.uniform(1000, 5000, 100)

        df = pd.DataFrame({
            'datetime': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })

        df.set_index('datetime', inplace=True)
        data[timeframe] = df

    return data


async def demonstrate_factory_usage():
    """演示策略工厂的使用"""
    print("\n" + "="*60)
    print("期货策略工厂使用示例")
    print("="*60)

    # 获取可用策略列表
    available_strategies = strategy_factory.get_available_strategies()
    print(f"可用策略: {available_strategies}")

    # 获取工厂统计信息
    factory_stats = strategy_factory.get_factory_stats()
    print(f"工厂统计: {factory_stats}")

    # 创建MACD策略实例
    try:
        macd_strategy = strategy_factory.create_strategy("futures_macd_strategy")
        print(f"成功创建MACD策略: {macd_strategy.name}")
        print(f"策略类型: {macd_strategy.strategy_type.value}")
        print(f"支持时间框架: {macd_strategy.get_supported_timeframes()}")
        print(f"需要指标: {macd_strategy.get_required_indicators()}")
    except Exception as e:
        print(f"创建MACD策略失败: {e}")

    # 创建RSI策略实例
    try:
        rsi_strategy = strategy_factory.create_strategy("futures_rsi_strategy")
        print(f"成功创建RSI策略: {rsi_strategy.name}")
        print(f"策略类型: {rsi_strategy.strategy_type.value}")
        print(f"支持时间框架: {rsi_strategy.get_supported_timeframes()}")
    except Exception as e:
        print(f"创建RSI策略失败: {e}")

    # 使用自定义配置创建策略
    try:
        custom_config = {
            "name": "custom_macd_strategy",
            "strategy_type": "trend_following",
            "default_leverage": 15.0,
            "max_leverage": 25.0,
            "min_confidence": 65.0,
            "timeframes": ["1h", "4h"],
            "custom_params": {
                "fast_period": 10,
                "slow_period": 30
            }
        }

        custom_macd = strategy_factory.create_strategy(
            "futures_macd_strategy",
            config=custom_config
        )
        print(f"成功创建自定义MACD策略: {custom_macd.config.default_leverage}x杠杆")
    except Exception as e:
        print(f"创建自定义策略失败: {e}")


async def demonstrate_strategy_manager():
    """演示策略管理器的使用"""
    print("\n" + "="*60)
    print("期货策略管理器使用示例")
    print("="*60)

    # 添加策略到管理器
    success1 = strategy_manager.add_strategy(
        "futures_macd_strategy",
        weight=0.6,
        confidence_multiplier=1.1
    )

    success2 = strategy_manager.add_strategy(
        "futures_rsi_strategy",
        weight=0.4,
        confidence_multiplier=0.9
    )

    print(f"添加MACD策略: {'成功' if success1 else '失败'}")
    print(f"添加RSI策略: {'成功' if success2 else '失败'}")

    # 获取管理器状态
    manager_status = strategy_manager.get_manager_status()
    print(f"管理器状态: {manager_status['total_strategies']}个策略，{manager_status['active_strategies']}个活跃")
    print(f"活跃策略: {manager_status['active_strategy_names']}")

    return success1 and success2


async def demonstrate_multi_strategy_analysis():
    """演示多策略分析"""
    print("\n" + "="*60)
    print("多策略分析示例")
    print("="*60)

    # 创建示例数据
    ticker = "BTCUSDT"
    timeframes = ["1h", "4h", "1d"]
    data = create_sample_data(ticker, timeframes)
    current_price = 51500.0

    print(f"分析交易对: {ticker}")
    print(f"当前价格: ${current_price}")
    print(f"数据时间框架: {timeframes}")

    # 执行多策略分析
    try:
        aggregated_signal = strategy_manager.analyze_ticker(
            ticker=ticker,
            data=data,
            current_price=current_price,
            aggregation_method=AggregationMethod.CONFIDENCE_WEIGHTED,
            validation_level=ValidationLevel.STANDARD
        )

        if aggregated_signal:
            print("\n多策略分析结果:")
            print(f"最终信号: {aggregated_signal.final_signal.direction.value if aggregated_signal.final_signal else 'None'}")

            if aggregated_signal.final_signal:
                signal = aggregated_signal.final_signal
                print(f"置信度: {signal.confidence:.1f}%")
                print(f"建议杠杆: {signal.suggested_leverage:.1f}x")
                print(f"仓位大小: ${signal.position_size:.1f}")
                print(f"止损价格: ${signal.stop_loss_price:.2f}" if signal.stop_loss_price else "止损价格: 未设置")
                print(f"止盈价格: ${signal.take_profit_price:.2f}" if signal.take_profit_price else "止盈价格: 未设置")
                print(f"风险等级: {signal.risk_level.value}")

            print(f"\n参与策略数量: {len(aggregated_signal.contributing_signals)}")
            print("各策略置信度:")
            for strategy, confidence in aggregated_signal.confidence_scores.items():
                print(f"  {strategy}: {confidence:.1f}%")

            print(f"\n聚合方法: {aggregated_signal.aggregation_metadata['method']}")
            print(f"处理时间: {aggregated_signal.aggregation_metadata['processing_time']:.3f}秒")

            # 验证结果
            if aggregated_signal.validation_results:
                print("\n验证结果:")
                for validation in aggregated_signal.validation_results:
                    if not validation.is_valid:
                        print(f"  ⚠️  {validation.field_name}: {validation.message}")
            else:
                print("✅ 信号验证通过")

        else:
            print("❌ 多策略分析未产生有效信号")

    except Exception as e:
        print(f"❌ 多策略分析失败: {e}")


async def demonstrate_different_aggregation_methods():
    """演示不同的信号聚合方法"""
    print("\n" + "="*60)
    print("不同聚合方法对比")
    print("="*60)

    ticker = "ETHUSDT"
    timeframes = ["1h", "4h"]
    data = create_sample_data(ticker, timeframes)
    current_price = 2800.0

    aggregation_methods = [
        AggregationMethod.CONFIDENCE_WEIGHTED,
        AggregationMethod.MAJORITY_VOTE,
        AggregationMethod.HIGHEST_CONFIDENCE,
        AggregationMethod.WEIGHTED_AVERAGE
    ]

    results = {}

    for method in aggregation_methods:
        try:
            result = strategy_manager.analyze_ticker(
                ticker=ticker,
                data=data,
                current_price=current_price,
                aggregation_method=method,
                validation_level=ValidationLevel.BASIC
            )

            if result and result.final_signal:
                results[method.value] = {
                    "direction": result.final_signal.direction.value,
                    "confidence": result.final_signal.confidence,
                    "leverage": result.final_signal.suggested_leverage
                }
            else:
                results[method.value] = {"direction": "无信号", "confidence": 0, "leverage": 0}

        except Exception as e:
            results[method.value] = {"error": str(e)}

    # 展示结果
    print(f"交易对: {ticker}, 当前价格: ${current_price}")
    print("\n各聚合方法结果:")
    for method, result in results.items():
        if "error" in result:
            print(f"{method:20} | ❌ {result['error']}")
        else:
            direction = result['direction']
            confidence = result['confidence']
            leverage = result['leverage']
            print(f"{method:20} | {direction:8} | 置信度: {confidence:5.1f}% | 杠杆: {leverage:4.1f}x")


async def demonstrate_performance_monitoring():
    """演示性能监控"""
    print("\n" + "="*60)
    print("策略性能监控示例")
    print("="*60)

    # 获取管理器状态
    status = strategy_manager.get_manager_status()

    print("性能统计:")
    perf_stats = status['performance_stats']
    print(f"总分析次数: {perf_stats['total_analyses']}")
    print(f"成功分析: {perf_stats['successful_analyses']}")
    print(f"失败分析: {perf_stats['failed_analyses']}")

    if perf_stats['total_analyses'] > 0:
        success_rate = perf_stats['successful_analyses'] / perf_stats['total_analyses'] * 100
        print(f"成功率: {success_rate:.1f}%")
        print(f"平均处理时间: {perf_stats['avg_processing_time']:.3f}秒")

    # 获取各策略状态
    print("\n策略状态:")
    for strategy_name in status['active_strategy_names']:
        if strategy_name in strategy_manager.strategies:
            strategy = strategy_manager.strategies[strategy_name]
            strategy_status = strategy.get_status()
            print(f"  {strategy_name}:")
            print(f"    信号数量: {strategy_status['signal_count']}")
            print(f"    最后信号时间: {strategy_status['last_signal_time']}")


async def main():
    """主函数"""
    print("期货策略集成机制演示")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 演示策略工厂
        await demonstrate_factory_usage()

        # 演示策略管理器
        manager_ready = await demonstrate_strategy_manager()

        if manager_ready:
            # 演示多策略分析
            await demonstrate_multi_strategy_analysis()

            # 演示不同聚合方法
            await demonstrate_different_aggregation_methods()

            # 演示性能监控
            await demonstrate_performance_monitoring()
        else:
            print("策略管理器初始化失败，跳过后续演示")

    except Exception as e:
        logger.error(f"演示过程出错: {e}")
        raise
    finally:
        # 清理资源
        strategy_manager.shutdown()

    print(f"\n演示完成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())