"""
杠杆控制器使用示例

演示如何使用智能杠杆控制系统，包括基本使用、高级配置、
集成市场分析器等场景。
"""
import asyncio
import logging
from datetime import datetime
from pathlib import Path

# 导入杠杆控制模块
from .leverage_controller import LeverageController
from .leverage_models import (
    LeverageConfig,
    LeverageStrategy,
    LeverageLimits,
    MarketCondition,
    MarketRegime,
    LiquidityLevel
)
from .config_manager import LeverageConfigManager

# 导入相关数据模型
from ..models.data_models import MarginStatus, Position, PositionSide, RiskLevel
from ..market.market_analyzer import MarketAnalyzer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def basic_usage_example():
    """基本使用示例"""
    print("=== 基本使用示例 ===")

    # 1. 创建杠杆控制器（使用默认配置）
    controller = LeverageController()

    # 2. 基本杠杆计算
    result = await controller.calculate_optimal_leverage(
        ticker='BTCUSDT',
        strategy=LeverageStrategy.MODERATE
    )

    print(f"计算结果:")
    print(f"  交易对: {result.ticker}")
    print(f"  基础杠杆: {result.base_leverage}")
    print(f"  计算杠杆: {result.calculated_leverage:.2f}")
    print(f"  应用杠杆: {result.applied_leverage:.2f}")
    print(f"  风险等级: {result.risk_level.value}")
    print(f"  置信度: {result.confidence_score:.2f}")
    print(f"  建议: {result.recommendation}")

    # 3. 同步计算（适用于非异步环境）
    sync_result = controller.calculate_leverage_sync(
        ticker='ETHUSDT',
        strategy=LeverageStrategy.CONSERVATIVE
    )
    print(f"\n同步计算结果 - ETHUSDT: {sync_result.applied_leverage:.2f}")

    return controller


async def advanced_configuration_example():
    """高级配置示例"""
    print("\n=== 高级配置示例 ===")

    # 1. 创建自定义配置
    config = LeverageConfig()
    config.default_leverage = 8.0
    config.emergency_leverage_cap = 3.0
    config.high_volatility_threshold = 0.35
    config.enable_emergency_reduction = True

    # 2. 设置交易对特定限制
    btc_limits = LeverageLimits(ticker='BTCUSDT', max_leverage=100.0)
    btc_limits.strategy_based_limits[LeverageStrategy.CONSERVATIVE] = 20.0
    btc_limits.strategy_based_limits[LeverageStrategy.AGGRESSIVE] = 80.0
    config.add_symbol_limits('BTCUSDT', btc_limits)

    # 3. 创建控制器
    controller = LeverageController(config=config)

    # 4. 计算杠杆
    result = await controller.calculate_optimal_leverage(
        ticker='BTCUSDT',
        strategy=LeverageStrategy.AGGRESSIVE,
        requested_leverage=75.0
    )

    print(f"高级配置结果:")
    print(f"  请求杠杆: {result.requested_leverage}")
    print(f"  最终杠杆: {result.applied_leverage:.2f}")
    print(f"  有效限制: {result.effective_limit:.2f}")

    return controller


async def market_integration_example():
    """市场分析器集成示例"""
    print("\n=== 市场分析器集成示例 ===")

    # 1. 创建模拟市场分析器
    class MockMarketAnalyzer:
        async def analyze_market_condition(self, ticker: str):
            """模拟市场分析"""
            # 模拟不同的市场条件
            if ticker == 'BTCUSDT':
                return {
                    'volatility': 0.15,  # 低波动
                    'liquidity_score': 0.9,  # 高流动性
                    'trend_strength': 0.7,  # 强趋势
                    'atr_percentage': 0.03,
                    'volume_ratio': 1.5,
                    'spread_bps': 1.2
                }
            elif ticker == 'ALTUSDT':  # 山寨币
                return {
                    'volatility': 0.45,  # 高波动
                    'liquidity_score': 0.4,  # 低流动性
                    'trend_strength': 0.3,  # 弱趋势
                    'atr_percentage': 0.08,
                    'volume_ratio': 0.8,
                    'spread_bps': 5.0
                }

    # 2. 创建控制器
    mock_analyzer = MockMarketAnalyzer()
    controller = LeverageController(market_analyzer=mock_analyzer)

    # 3. 比较不同市场条件下的杠杆
    btc_result = await controller.calculate_optimal_leverage(
        ticker='BTCUSDT',
        strategy=LeverageStrategy.MODERATE
    )

    alt_result = await controller.calculate_optimal_leverage(
        ticker='ALTUSDT',
        strategy=LeverageStrategy.MODERATE
    )

    print(f"市场条件影响比较:")
    print(f"  BTC (低波动/高流动性): {btc_result.applied_leverage:.2f}")
    print(f"    波动率倍数: {btc_result.adjustment_factors.volatility_multiplier:.3f}")
    print(f"    流动性倍数: {btc_result.adjustment_factors.liquidity_multiplier:.3f}")

    print(f"  ALT (高波动/低流动性): {alt_result.applied_leverage:.2f}")
    print(f"    波动率倍数: {alt_result.adjustment_factors.volatility_multiplier:.3f}")
    print(f"    流动性倍数: {alt_result.adjustment_factors.liquidity_multiplier:.3f}")


async def risk_management_example():
    """风险管理示例"""
    print("\n=== 风险管理示例 ===")

    controller = LeverageController()

    # 1. 模拟不同风险等级的保证金状态
    risk_scenarios = [
        {
            'name': '低风险',
            'margin': MarginStatus(
                total_balance=10000.0,
                available_margin=9000.0,
                used_margin=1000.0,
                margin_ratio=0.1,
                risk_level=RiskLevel.LOW
            )
        },
        {
            'name': '中等风险',
            'margin': MarginStatus(
                total_balance=10000.0,
                available_margin=6000.0,
                used_margin=4000.0,
                margin_ratio=0.4,
                risk_level=RiskLevel.MEDIUM
            )
        },
        {
            'name': '高风险',
            'margin': MarginStatus(
                total_balance=10000.0,
                available_margin=2000.0,
                used_margin=8000.0,
                margin_ratio=0.8,
                risk_level=RiskLevel.HIGH
            )
        },
        {
            'name': '危急风险',
            'margin': MarginStatus(
                total_balance=10000.0,
                available_margin=500.0,
                used_margin=9500.0,
                margin_ratio=0.95,
                risk_level=RiskLevel.CRITICAL
            )
        }
    ]

    print(f"不同风险等级下的杠杆调整:")
    for scenario in risk_scenarios:
        result = await controller.calculate_optimal_leverage(
            ticker='BTCUSDT',
            strategy=LeverageStrategy.MODERATE,
            margin_status=scenario['margin']
        )

        print(f"  {scenario['name']}:")
        print(f"    保证金率: {scenario['margin'].margin_ratio:.1%}")
        print(f"    风险调整: {result.adjustment_factors.risk_adjustment:.3f}")
        print(f"    最终杠杆: {result.applied_leverage:.2f}")
        print(f"    风险等级: {result.risk_level.value}")


async def position_concentration_example():
    """仓位集中度管理示例"""
    print("\n=== 仓位集中度管理示例 ===")

    controller = LeverageController()

    # 1. 创建仓位组合 - 高集中度
    high_concentration_positions = [
        Position(
            ticker='BTCUSDT',
            side=PositionSide.LONG,
            size=1.0,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=10.0,
            initial_margin=5100.0,
            maintenance_margin=2550.0,
            notional_value=51000.0
        ),
        Position(
            ticker='BTCUSDT',
            side=PositionSide.SHORT,
            size=0.5,
            entry_price=52000.0,
            current_price=51000.0,
            leverage=5.0,
            initial_margin=5100.0,
            maintenance_margin=2550.0,
            notional_value=25500.0
        )
    ]

    # 2. 创建仓位组合 - 分散持仓
    diversified_positions = [
        Position(
            ticker='BTCUSDT',
            side=PositionSide.LONG,
            size=0.3,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=10.0,
            initial_margin=1530.0,
            maintenance_margin=765.0,
            notional_value=15300.0
        ),
        Position(
            ticker='ETHUSDT',
            side=PositionSide.LONG,
            size=5.0,
            entry_price=3000.0,
            current_price=3100.0,
            leverage=5.0,
            initial_margin=3100.0,
            maintenance_margin=1550.0,
            notional_value=15500.0
        ),
        Position(
            ticker='ADAUSDT',
            side=PositionSide.LONG,
            size=10000.0,
            entry_price=0.5,
            current_price=0.52,
            leverage=3.0,
            initial_margin=1733.33,
            maintenance_margin=866.67,
            notional_value=5200.0
        )
    ]

    # 3. 比较集中度影响
    concentrated_result = await controller.calculate_optimal_leverage(
        ticker='BTCUSDT',
        strategy=LeverageStrategy.MODERATE,
        current_positions=high_concentration_positions
    )

    diversified_result = await controller.calculate_optimal_leverage(
        ticker='BTCUSDT',
        strategy=LeverageStrategy.MODERATE,
        current_positions=diversified_positions
    )

    print(f"仓位集中度影响:")
    print(f"  高集中度:")
    print(f"    仓位调整: {concentrated_result.adjustment_factors.position_adjustment:.3f}")
    print(f"    最终杠杆: {concentrated_result.applied_leverage:.2f}")

    print(f"  分散持仓:")
    print(f"    仓位调整: {diversified_result.adjustment_factors.position_adjustment:.3f}")
    print(f"    最终杠杆: {diversified_result.applied_leverage:.2f}")


async def config_management_example():
    """配置管理示例"""
    print("\n=== 配置管理示例 ===")

    # 1. 创建配置管理器
    config_manager = LeverageConfigManager()

    # 2. 获取配置模板
    template = config_manager.get_config_template()
    print(f"配置模板包含 {len(template)} 个字段")

    # 3. 创建自定义配置
    custom_config_data = {
        'default_leverage': 6.0,
        'base_leverage_by_strategy': {
            'conservative': 3.0,
            'moderate': 6.0,
            'aggressive': 12.0,
            'expert': 20.0
        },
        'max_adjustment_percentage': 40.0,
        'enable_emergency_reduction': True,
        'symbol_specific_limits': {
            'BTCUSDT': {
                'max_leverage': 125.0,
                'volatility_threshold': 0.25,
                'risk_based_limits': {
                    'low': 125.0,
                    'medium': 100.0,
                    'high': 75.0,
                    'critical': 25.0
                }
            }
        }
    }

    # 4. 解析配置
    config = config_manager._parse_config_data(custom_config_data)

    # 5. 验证配置
    validation_errors = config_manager.validate_config(config)
    if validation_errors:
        print(f"配置验证发现问题: {validation_errors}")
    else:
        print("配置验证通过")

    # 6. 使用自定义配置创建控制器
    controller = LeverageController(config=config)
    result = await controller.calculate_optimal_leverage(
        ticker='BTCUSDT',
        strategy=LeverageStrategy.EXPERT
    )

    print(f"自定义配置结果:")
    print(f"  基础杠杆: {result.base_leverage}")
    print(f"  最终杠杆: {result.applied_leverage:.2f}")


async def real_time_monitoring_example():
    """实时监控示例"""
    print("\n=== 实时监控示例 ===")

    controller = LeverageController()

    # 模拟实时价格更新和杠杆重计算
    prices = [50000, 50500, 51000, 49000, 48000]  # 模拟价格变化

    print(f"模拟实时杠杆调整:")
    for i, price in enumerate(prices):
        # 创建当前市场数据
        market_data = {
            'volatility': 0.1 + (abs(price - prices[0]) / prices[0]),  # 基于价格变化计算波动率
            'liquidity_score': 0.8,
            'market_regime': 'volatile' if abs(price - prices[0]) / prices[0] > 0.05 else 'calm',
            'liquidity_level': 'high'
        }

        result = await controller.calculate_optimal_leverage(
            ticker='BTCUSDT',
            strategy=LeverageStrategy.MODERATE,
            market_data=market_data
        )

        print(f"  时间 {i+1}: 价格={price}, 波动率={market_data['volatility']:.3f}, "
              f"杠杆={result.applied_leverage:.2f}")

    # 获取统计信息
    stats = controller.get_leverage_statistics()
    print(f"\n控制器统计:")
    print(f"  计算次数: {stats['calculation_count']}")
    print(f"  调整统计: {stats['adjustment_stats']}")


async def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")

    controller = LeverageController()

    # 1. 测试无效参数
    try:
        result = await controller.calculate_optimal_leverage(
            ticker='',  # 空交易对
            strategy=LeverageStrategy.MODERATE
        )
        if result.has_errors():
            print(f"处理无效参数: {result.errors}")
    except Exception as e:
        print(f"捕获异常: {e}")

    # 2. 测试极端市场条件
    extreme_market_data = {
        'volatility': 0.9,  # 极高波动
        'liquidity_score': 0.1,  # 极低流动性
        'market_regime': 'high_volatility',
        'liquidity_level': 'very_low'
    }

    result = await controller.calculate_optimal_leverage(
        ticker='ALTUSDT',
        strategy=LeverageStrategy.AGGRESSIVE,
        market_data=extreme_market_data
    )

    print(f"极端市场条件:")
    print(f"  计算杠杆: {result.calculated_leverage:.2f}")
    print(f"  最终杠杆: {result.applied_leverage:.2f}")
    print(f"  综合调整倍数: {result.adjustment_factors.get_combined_multiplier():.3f}")
    if result.has_warnings():
        print(f"  警告: {result.warnings}")


async def main():
    """主函数 - 运行所有示例"""
    print("杠杆控制器使用示例")
    print("=" * 50)

    try:
        await basic_usage_example()
        await advanced_configuration_example()
        await market_integration_example()
        await risk_management_example()
        await position_concentration_example()
        await config_management_example()
        await real_time_monitoring_example()
        await error_handling_example()

        print("\n" + "=" * 50)
        print("所有示例运行完成")

    except Exception as e:
        logger.error(f"运行示例时发生错误: {e}")


def sync_main():
    """同步主函数"""
    asyncio.run(main())


if __name__ == '__main__':
    sync_main()