#!/usr/bin/env python3
"""
止盈止损计算器使用示例

展示如何使用TpSlCalculator计算智能止盈止损价格
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

from tp_sl_calculator import (
    TpSlCalculator,
    TpSlCalculationParams,
    TpSlCalculationMethod,
    RiskRewardStrategy,
    TrailingStopType
)
from ..models.data_models import TradingDirection


async def basic_usage_example():
    """基础使用示例"""
    print("=== 基础止盈止损计算示例 ===\n")

    # 创建计算器
    calculator = TpSlCalculator()

    # 设置计算参数
    params = TpSlCalculationParams(
        ticker="BTCUSDT",
        direction=TradingDirection.LONG,
        current_price=50000.0,
        leverage=10.0,
        volatility=0.03,  # 3% 波动率
        risk_reward_strategy=RiskRewardStrategy.BALANCED
    )

    # 计算止盈止损
    result = await calculator.calculate_tp_sl(params)

    if result:
        print(f"交易对: {params.ticker}")
        print(f"方向: {params.direction.value}")
        print(f"当前价格: ${params.current_price:,.2f}")
        print(f"杠杆: {params.leverage:.1f}x")
        print()
        print(f"止盈价格: ${result.take_profit_price:,.2f}")
        print(f"止损价格: ${result.stop_loss_price:,.2f}")
        print(f"风险收益比: 1:{result.risk_reward_ratio:.2f}")
        print(f"止损距离: {result.stop_loss_pct:.2f}%")
        print(f"止盈距离: {result.take_profit_pct:.2f}%")
        print(f"置信度评分: {result.confidence_score:.2f}")

        if result.warnings:
            print("\n⚠️ 警告:")
            for warning in result.warnings:
                print(f"  - {warning}")

    else:
        print("❌ 计算失败")


async def advanced_usage_example():
    """高级功能示例"""
    print("\n=== 高级功能示例 ===\n")

    calculator = TpSlCalculator()

    # 创建模拟市场数据
    dates = pd.date_range(start='2024-01-01', periods=50, freq='1h')
    prices = [50000 * (1 + np.random.normal(0, 0.02)) for _ in range(50)]

    market_data = {
        '1h': pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 50
        })
    }

    # 高级参数配置
    params = TpSlCalculationParams(
        ticker="ETHUSDT",
        direction=TradingDirection.LONG,
        current_price=3000.0,
        leverage=5.0,
        market_data=market_data,
        volatility=0.04,
        calculation_method=TpSlCalculationMethod.ATR_BASED,
        risk_reward_strategy=RiskRewardStrategy.DYNAMIC,

        # 期货特有参数
        liquidation_price=2400.0,  # 强平价格
        min_liquidation_buffer=0.15,  # 15% 强平缓冲

        # 技术位参数
        support_levels=[2900.0, 2850.0],
        resistance_levels=[3100.0, 3200.0],

        # 移动止损配置
        enable_trailing_stop=True,
        trailing_stop_type=TrailingStopType.PERCENTAGE_BASED,
        trailing_activation_pct=0.02,

        # 分级止盈配置
        enable_tiered_tp=True,
        tp_levels=[(0.5, 0.3), (0.8, 0.4), (1.0, 0.3)],

        # 时间参数
        max_holding_hours=48.0,
        enable_time_exit=True
    )

    # 执行高级计算
    result = await calculator.calculate_tp_sl(params)

    if result:
        print(f"高级计算结果:")
        print(f"计算方法: {result.calculation_method.value}")
        print(f"风险策略: {result.risk_reward_strategy.value}")
        print(f"止盈价格: ${result.take_profit_price:,.2f}")
        print(f"止损价格: ${result.stop_loss_price:,.2f}")
        print(f"强平缓冲: {result.liquidation_buffer:.1f}%")
        print(f"最大持仓: {result.max_holding_hours:.0f}小时")

        # 移动止损配置
        if result.trailing_stop_config:
            print(f"\n📈 移动止损配置:")
            config = result.trailing_stop_config
            print(f"  类型: {config.get('type', 'unknown')}")
            print(f"  激活阈值: {config.get('activation_threshold_pct', 0):.1f}%")

        # 分级止盈配置
        if result.tiered_take_profit:
            print(f"\n🎯 分级止盈配置:")
            for tier in result.tiered_take_profit:
                print(f"  第{tier['tier']}级: ${tier['price']:,.2f} ({tier['quantity_ratio']*100:.0f}%)")

        # 时间退出
        if result.time_based_exit:
            print(f"\n⏰ 时间退出: {result.time_based_exit.strftime('%Y-%m-%d %H:%M')}")

    else:
        print("❌ 高级计算失败")


async def risk_strategy_comparison():
    """风险策略对比示例"""
    print("\n=== 风险策略对比 ===\n")

    calculator = TpSlCalculator()
    base_params = TpSlCalculationParams(
        ticker="BTCUSDT",
        direction=TradingDirection.LONG,
        current_price=50000.0,
        leverage=8.0,
        volatility=0.03
    )

    strategies = [
        RiskRewardStrategy.CONSERVATIVE,
        RiskRewardStrategy.BALANCED,
        RiskRewardStrategy.AGGRESSIVE,
        RiskRewardStrategy.MAXIMUM
    ]

    print("策略对比结果:")
    print("策略\t\t风险收益比\t止盈价格\t止损价格")
    print("-" * 60)

    for strategy in strategies:
        base_params.risk_reward_strategy = strategy
        result = await calculator.calculate_tp_sl(base_params)

        if result:
            print(f"{strategy.value:<12}\t1:{result.risk_reward_ratio:.2f}\t\t"
                  f"${result.take_profit_price:,.0f}\t${result.stop_loss_price:,.0f}")


async def leverage_impact_analysis():
    """杠杆影响分析"""
    print("\n=== 杠杆影响分析 ===\n")

    calculator = TpSlCalculator()
    base_params = TpSlCalculationParams(
        ticker="BTCUSDT",
        direction=TradingDirection.LONG,
        current_price=50000.0,
        leverage=1.0,  # 将被覆盖
        volatility=0.025,
        risk_reward_strategy=RiskRewardStrategy.BALANCED
    )

    leverages = [2, 5, 10, 15, 20]

    print("杠杆影响分析:")
    print("杠杆\t止损距离\t止盈距离\t强平风险")
    print("-" * 50)

    for leverage in leverages:
        base_params.leverage = leverage
        # 估算强平价格
        base_params.liquidation_price = 50000.0 * (1 - 0.9/leverage)

        result = await calculator.calculate_tp_sl(base_params)

        if result:
            risk_level = "低" if leverage <= 5 else "中" if leverage <= 10 else "高"
            print(f"{leverage}x\t{result.stop_loss_pct:.2f}%\t\t"
                  f"{result.take_profit_pct:.2f}%\t\t{risk_level}")


async def main():
    """主函数"""
    print("🚀 止盈止损计算器使用示例")
    print("=" * 50)

    await basic_usage_example()
    await advanced_usage_example()
    await risk_strategy_comparison()
    await leverage_impact_analysis()

    print("\n" + "=" * 50)
    print("✅ 所有示例运行完成")


if __name__ == "__main__":
    asyncio.run(main())