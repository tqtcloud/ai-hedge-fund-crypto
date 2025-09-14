"""
保证金管理器基本测试功能

提供基本的测试用例来验证MarginManager的核心功能
"""

import asyncio
import logging
from typing import Dict, Any

from .margin_manager import MarginManager, MarginConfig, WebSocketTradingInterface
from ..models.data_models import RiskLevel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestMarginManager:
    """保证金管理器测试类"""

    def __init__(self):
        """初始化测试环境"""
        # 创建测试配置
        self.config = MarginConfig(
            initial_cash=1000.0,
            initial_margin_rate=0.1,
            maintenance_margin_rate=0.05,
            margin_call_threshold=0.3,
            liquidation_threshold=0.8,
            emergency_threshold=0.9,
            max_leverage=20
        )

        # 创建Mock WebSocket接口
        self.ws_interface = WebSocketTradingInterface()

        # 创建保证金管理器
        self.margin_manager = MarginManager(self.config, self.ws_interface)

    async def test_initialization(self) -> bool:
        """测试保证金管理器初始化"""
        try:
            logger.info("=== 测试保证金管理器初始化 ===")

            await self.margin_manager.initialize()

            # 验证初始化状态
            available = self.margin_manager.get_available_margin()
            ratio = self.margin_manager.get_margin_ratio()

            logger.info(f"初始可用保证金: {available}")
            logger.info(f"初始保证金使用率: {ratio:.2%}")

            assert available > 0, "可用保证金应大于0"
            assert ratio == 0.0, "初始保证金使用率应为0"

            logger.info("✅ 初始化测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 初始化测试失败: {e}")
            return False

    async def test_margin_calculation(self) -> bool:
        """测试保证金计算功能"""
        try:
            logger.info("=== 测试保证金计算功能 ===")

            # 测试参数
            position_value = 1000.0  # 1000 USDT仓位价值
            leverage = 10           # 10倍杠杆

            # 计算初始保证金
            initial_margin = self.margin_manager.calculate_initial_margin(
                position_value, leverage
            )
            expected_initial = (position_value / leverage) * self.config.initial_margin_rate

            logger.info(f"仓位价值: {position_value} USDT")
            logger.info(f"杠杆倍数: {leverage}x")
            logger.info(f"计算得出初始保证金: {initial_margin} USDT")
            logger.info(f"预期初始保证金: {expected_initial} USDT")

            assert abs(initial_margin - expected_initial) < 0.01, \
                f"初始保证金计算错误: {initial_margin} != {expected_initial}"

            # 计算维持保证金
            maintenance_margin = self.margin_manager.calculate_maintenance_margin(
                position_value, leverage
            )
            expected_maintenance = (position_value / leverage) * self.config.maintenance_margin_rate

            logger.info(f"计算得出维持保证金: {maintenance_margin} USDT")
            logger.info(f"预期维持保证金: {expected_maintenance} USDT")

            assert abs(maintenance_margin - expected_maintenance) < 0.01, \
                f"维持保证金计算错误: {maintenance_margin} != {expected_maintenance}"

            logger.info("✅ 保证金计算测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 保证金计算测试失败: {e}")
            return False

    async def test_liquidation_price(self) -> bool:
        """测试强平价格计算"""
        try:
            logger.info("=== 测试强平价格计算 ===")

            # 测试多头强平价格
            entry_price = 50000.0
            position_size = 0.01
            margin = 100.0
            side = "long"

            liquidation_price = self.margin_manager.calculate_liquidation_price(
                entry_price, position_size, margin, side
            )

            logger.info(f"多头 - 入场价格: {entry_price} USDT")
            logger.info(f"仓位大小: {position_size} BTC")
            logger.info(f"保证金: {margin} USDT")
            logger.info(f"强平价格: {liquidation_price} USDT")

            assert liquidation_price > 0, "强平价格必须大于0"
            assert liquidation_price < entry_price, "多头强平价格应低于入场价格"

            # 测试空头强平价格
            side = "short"
            liquidation_price_short = self.margin_manager.calculate_liquidation_price(
                entry_price, -position_size, margin, side
            )

            logger.info(f"空头强平价格: {liquidation_price_short} USDT")

            assert liquidation_price_short > entry_price, "空头强平价格应高于入场价格"

            logger.info("✅ 强平价格计算测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 强平价格计算测试失败: {e}")
            return False

    async def test_margin_sufficiency_check(self) -> bool:
        """测试保证金充足性检查"""
        try:
            logger.info("=== 测试保证金充足性检查 ===")

            await self.margin_manager.initialize()

            # 测试充足保证金
            required_margin = 100.0
            margin_status = self.margin_manager.check_margin_sufficiency(required_margin)

            logger.info(f"所需保证金: {required_margin} USDT")
            logger.info(f"可用保证金: {margin_status.available_margin} USDT")
            logger.info(f"能否开仓: {margin_status.can_open_position}")

            assert margin_status.can_open_position, "应该能够开仓"

            # 测试不足保证金
            excessive_margin = 2000.0  # 超过初始资金
            margin_status_insufficient = self.margin_manager.check_margin_sufficiency(excessive_margin)

            logger.info(f"过高保证金需求: {excessive_margin} USDT")
            logger.info(f"可用保证金: {margin_status_insufficient.available_margin} USDT")
            logger.info(f"能否开仓: {margin_status_insufficient.can_open_position}")

            # 验证当需求超过可用时不能开仓
            assert not margin_status_insufficient.can_open_position, f"不应该能够开仓，可用: {margin_status_insufficient.available_margin}, 需要: {excessive_margin}"

            logger.info("✅ 保证金充足性检查测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 保证金充足性检查测试失败: {e}")
            return False

    async def test_risk_assessment(self) -> bool:
        """测试风险评估功能"""
        try:
            logger.info("=== 测试风险评估功能 ===")

            await self.margin_manager.initialize()

            # 初始风险等级（应该是低风险）
            initial_risk = await self.margin_manager.assess_risk_level()
            logger.info(f"初始风险等级: {initial_risk}")
            assert initial_risk == RiskLevel.LOW, "初始风险等级应为LOW"

            # 模拟高风险场景
            # 通过修改账户数据来模拟高保证金使用率
            original_used = self.margin_manager.account_data.get('totalPositionInitialMargin', 0.0)

            # 设置为20%使用率（应为MEDIUM风险）
            self.margin_manager.account_data['totalPositionInitialMargin'] = 200.0
            margin_ratio = self.margin_manager.get_margin_ratio()
            logger.info(f"保证金使用率: {margin_ratio:.2%}")

            medium_risk = await self.margin_manager.assess_risk_level()
            logger.info(f"中等风险等级: {medium_risk}")
            assert medium_risk == RiskLevel.MEDIUM, f"应为中等风险，实际为: {medium_risk}"

            # 模拟更高风险 - 设置为85%使用率
            self.margin_manager.account_data['totalPositionInitialMargin'] = 850.0
            margin_ratio_high = self.margin_manager.get_margin_ratio()
            logger.info(f"高风险保证金使用率: {margin_ratio_high:.2%}")

            high_risk = await self.margin_manager.assess_risk_level()
            logger.info(f"高风险等级: {high_risk}")
            assert high_risk == RiskLevel.CRITICAL, f"应为关键风险，实际为: {high_risk}"

            # 恢复原始状态
            self.margin_manager.account_data['totalPositionInitialMargin'] = original_used

            logger.info("✅ 风险评估测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 风险评估测试失败: {e}")
            return False

    async def test_monitoring_functionality(self) -> bool:
        """测试监控功能"""
        try:
            logger.info("=== 测试监控功能 ===")

            await self.margin_manager.initialize()

            # 启动监控
            await self.margin_manager.start_monitoring()
            assert self.margin_manager.is_monitoring, "监控应已启动"
            logger.info("✅ 监控启动成功")

            # 等待几秒观察监控
            await asyncio.sleep(3)

            # 停止监控
            await self.margin_manager.stop_monitoring()
            assert not self.margin_manager.is_monitoring, "监控应已停止"
            logger.info("✅ 监控停止成功")

            logger.info("✅ 监控功能测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 监控功能测试失败: {e}")
            return False

    async def test_status_summary(self) -> bool:
        """测试状态摘要功能"""
        try:
            logger.info("=== 测试状态摘要功能 ===")

            await self.margin_manager.initialize()

            summary = self.margin_manager.get_status_summary()

            required_keys = [
                'total_balance', 'available_margin', 'used_margin',
                'margin_ratio', 'is_monitoring', 'currency', 'last_update'
            ]

            for key in required_keys:
                assert key in summary, f"摘要缺少键: {key}"

            logger.info("状态摘要:")
            for key, value in summary.items():
                if key != 'risk_level':  # 跳过异步任务
                    logger.info(f"  {key}: {value}")

            logger.info("✅ 状态摘要测试通过")
            return True

        except Exception as e:
            logger.error(f"❌ 状态摘要测试失败: {e}")
            return False

    async def run_all_tests(self) -> Dict[str, bool]:
        """运行所有测试"""
        logger.info("🚀 开始运行所有保证金管理器测试")

        test_results = {}

        # 运行所有测试
        test_functions = [
            ("初始化测试", self.test_initialization),
            ("保证金计算测试", self.test_margin_calculation),
            ("强平价格测试", self.test_liquidation_price),
            ("保证金充足性测试", self.test_margin_sufficiency_check),
            ("风险评估测试", self.test_risk_assessment),
            ("监控功能测试", self.test_monitoring_functionality),
            ("状态摘要测试", self.test_status_summary),
        ]

        for test_name, test_func in test_functions:
            try:
                result = await test_func()
                test_results[test_name] = result
            except Exception as e:
                logger.error(f"测试 {test_name} 执行出错: {e}")
                test_results[test_name] = False

            # 测试间稍作延迟
            await asyncio.sleep(0.5)

        # 输出测试结果摘要
        logger.info("\n" + "="*50)
        logger.info("📊 测试结果摘要:")
        logger.info("="*50)

        passed_count = 0
        for test_name, result in test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{test_name}: {status}")
            if result:
                passed_count += 1

        total_tests = len(test_results)
        logger.info("="*50)
        logger.info(f"总计: {passed_count}/{total_tests} 通过")

        if passed_count == total_tests:
            logger.info("🎉 所有测试均通过！")
        else:
            logger.warning(f"⚠️  {total_tests - passed_count} 个测试失败")

        # 清理资源
        await self.margin_manager.cleanup()

        return test_results


async def main():
    """主测试入口"""
    test_runner = TestMarginManager()
    results = await test_runner.run_all_tests()
    return results


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())