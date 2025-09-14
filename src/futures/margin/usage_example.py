"""
保证金管理器使用示例

该示例展示了如何使用MarginManager进行：
- 基本的保证金计算
- 强平价格计算
- 保证金充足性检查
- 风险评估和监控
"""

import asyncio
import logging
from datetime import datetime

from .margin_manager import MarginManager, MarginConfig
from ..models.data_models import RiskLevel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def basic_usage_example():
    """基本使用示例"""
    logger.info("=== 保证金管理器基本使用示例 ===")

    # 1. 创建配置
    config = MarginConfig(
        initial_cash=10000.0,        # 初始资金10,000 USDT
        initial_margin_rate=0.1,     # 初始保证金率10%
        maintenance_margin_rate=0.05, # 维持保证金率5%
        margin_call_threshold=0.3,   # 追保阈值30%
        liquidation_threshold=0.8,   # 强平阈值80%
        max_leverage=20              # 最大杠杆20倍
    )

    # 2. 创建保证金管理器
    margin_manager = MarginManager(config)

    try:
        # 3. 初始化
        await margin_manager.initialize()
        logger.info(f"初始可用保证金: {margin_manager.get_available_margin()} USDT")

        # 4. 保证金计算示例
        position_value = 5000.0  # 5000 USDT 仓位价值
        leverage = 10           # 10倍杠杆

        initial_margin = margin_manager.calculate_initial_margin(position_value, leverage)
        maintenance_margin = margin_manager.calculate_maintenance_margin(position_value, leverage)

        logger.info(f"仓位价值: {position_value} USDT")
        logger.info(f"杠杆倍数: {leverage}x")
        logger.info(f"所需初始保证金: {initial_margin} USDT")
        logger.info(f"维持保证金: {maintenance_margin} USDT")

        # 5. 强平价格计算示例
        entry_price = 50000.0    # BTC入场价格
        position_size = 0.1      # 0.1 BTC
        margin = initial_margin

        long_liquidation = margin_manager.calculate_liquidation_price(
            entry_price, position_size, margin, "long"
        )
        short_liquidation = margin_manager.calculate_liquidation_price(
            entry_price, -position_size, margin, "short"
        )

        logger.info(f"多头强平价格: {long_liquidation:.2f} USDT")
        logger.info(f"空头强平价格: {short_liquidation:.2f} USDT")

        # 6. 保证金充足性检查
        margin_status = margin_manager.check_margin_sufficiency(initial_margin)
        logger.info(f"保证金充足性: {margin_status.can_open_position}")
        logger.info(f"账户状态: {margin_status.account_status}")
        logger.info(f"风险等级: {margin_status.risk_level}")

        # 7. 风险评估
        risk_level = await margin_manager.assess_risk_level()
        logger.info(f"当前风险等级: {risk_level}")

    finally:
        await margin_manager.cleanup()


async def monitoring_example():
    """保证金监控示例"""
    logger.info("=== 保证金监控示例 ===")

    config = MarginConfig(
        initial_cash=5000.0,
        initial_margin_rate=0.1,
        maintenance_margin_rate=0.05,
        margin_call_threshold=0.3,
        liquidation_threshold=0.8,
        max_leverage=20
    )

    margin_manager = MarginManager(config)

    try:
        await margin_manager.initialize()

        # 启动保证金健康监控
        await margin_manager.start_monitoring()
        logger.info("保证金监控已启动")

        # 模拟一些交易场景
        logger.info("模拟高风险场景...")

        # 模拟账户使用了大量保证金（高风险）
        margin_manager.account_data['totalPositionInitialMargin'] = 1500.0  # 30%使用率
        await asyncio.sleep(6)  # 等待监控检测

        # 模拟极高风险场景
        margin_manager.account_data['totalPositionInitialMargin'] = 4000.0  # 80%使用率
        await asyncio.sleep(6)  # 等待监控检测

        # 恢复正常
        margin_manager.account_data['totalPositionInitialMargin'] = 500.0  # 10%使用率
        await asyncio.sleep(3)

        # 停止监控
        await margin_manager.stop_monitoring()
        logger.info("保证金监控已停止")

    finally:
        await margin_manager.cleanup()


async def advanced_scenarios():
    """高级应用场景"""
    logger.info("=== 高级应用场景 ===")

    config = MarginConfig(
        initial_cash=50000.0,
        initial_margin_rate=0.08,    # 8%初始保证金率
        maintenance_margin_rate=0.04, # 4%维持保证金率
        margin_call_threshold=0.25,  # 25%追保阈值
        liquidation_threshold=0.75,  # 75%强平阈值
        max_leverage=25              # 25倍最大杠杆
    )

    margin_manager = MarginManager(config)

    try:
        await margin_manager.initialize()

        # 场景1: 多个仓位的保证金管理
        logger.info("场景1: 多个仓位保证金计算")

        positions = [
            {"symbol": "BTCUSDT", "value": 10000, "leverage": 10},
            {"symbol": "ETHUSDT", "value": 8000, "leverage": 15},
            {"symbol": "BNBUSDT", "value": 5000, "leverage": 20},
        ]

        total_margin_required = 0
        for pos in positions:
            margin = margin_manager.calculate_initial_margin(pos["value"], pos["leverage"])
            total_margin_required += margin
            logger.info(f"{pos['symbol']}: 价值{pos['value']} USDT, "
                       f"{pos['leverage']}x杠杆, 需要保证金{margin:.2f} USDT")

        logger.info(f"总保证金需求: {total_margin_required:.2f} USDT")

        # 检查总体保证金充足性
        overall_status = margin_manager.check_margin_sufficiency(total_margin_required)
        logger.info(f"可同时开启所有仓位: {overall_status.can_open_position}")

        # 场景2: 动态风险调整
        logger.info("场景2: 动态风险调整")

        # 模拟市场波动导致的保证金变化
        risk_scenarios = [
            ("正常市场", 0.1),    # 10%保证金使用率
            ("波动市场", 0.3),    # 30%保证金使用率
            ("高风险市场", 0.6),  # 60%保证金使用率
            ("极端市场", 0.85),   # 85%保证金使用率
        ]

        for scenario_name, usage_ratio in risk_scenarios:
            # 设置保证金使用率
            used_margin = margin_manager.account_data['totalWalletBalance'] * usage_ratio
            margin_manager.account_data['totalPositionInitialMargin'] = used_margin

            risk_level = await margin_manager.assess_risk_level()
            margin_ratio = margin_manager.get_margin_ratio()

            logger.info(f"{scenario_name}: 使用率{margin_ratio:.1%}, 风险等级{risk_level}")

        # 场景3: 状态摘要和报告
        logger.info("场景3: 详细状态报告")

        status_summary = margin_manager.get_status_summary()
        logger.info("=== 账户状态摘要 ===")
        for key, value in status_summary.items():
            if key != 'risk_level':  # 跳过异步任务
                logger.info(f"{key}: {value}")

    finally:
        await margin_manager.cleanup()


async def main():
    """主函数 - 运行所有示例"""
    logger.info("🚀 开始运行保证金管理器使用示例")

    try:
        # 运行基本使用示例
        await basic_usage_example()
        await asyncio.sleep(1)

        # 运行监控示例
        await monitoring_example()
        await asyncio.sleep(1)

        # 运行高级场景
        await advanced_scenarios()

        logger.info("✅ 所有示例运行完成")

    except Exception as e:
        logger.error(f"❌ 示例运行出错: {e}")
        raise


if __name__ == "__main__":
    # 运行所有示例
    asyncio.run(main())