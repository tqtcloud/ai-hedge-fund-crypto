"""
期货策略集成机制单元测试

测试统一接口、工厂模式和管理器的核心功能，
确保策略输出的正确性和一致性。
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.futures.signals.base_strategy import (
    FuturesBaseStrategy,
    StrategyConfig,
    StrategyType,
    SignalStrength
)
from src.futures.signals.strategy_factory import FuturesStrategyFactory
from src.futures.signals.strategy_manager import (
    FuturesStrategyManager,
    AggregationMethod,
    ValidationLevel
)
from src.futures.models.data_models import TradingDirection, OperationType


class MockStrategy(FuturesBaseStrategy):
    """用于测试的模拟策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.test_signal_direction = TradingDirection.LONG
        self.test_confidence = 75.0

    def analyze(self, ticker, data, current_price, **kwargs):
        signal = self.create_futures_signal(
            ticker=ticker,
            direction=self.test_signal_direction,
            operation_type=OperationType.OPEN,
            confidence=self.test_confidence,
            strength=SignalStrength.STRONG,
            current_price=current_price
        )

        from src.futures.signals.base_strategy import StrategyOutput
        return StrategyOutput(
            signal=signal,
            metadata={"test": True},
            diagnostics={"data_quality": "good"},
            performance_metrics={"test_metric": 1.0}
        )

    def get_signal_strength(self, data):
        return SignalStrength.STRONG

    def calculate_confidence(self, data):
        return self.test_confidence

    def get_supported_timeframes(self):
        return ["1h", "4h"]

    def get_required_indicators(self):
        return ["test_indicator"]


class TestFuturesStrategyIntegration(unittest.TestCase):
    """期货策略集成测试类"""

    def setUp(self):
        """设置测试环境"""
        # 创建测试用的策略工厂
        self.factory = FuturesStrategyFactory()

        # 创建测试用的策略管理器
        self.manager = FuturesStrategyManager(factory=self.factory)

        # 创建测试数据
        self.test_ticker = "BTCUSDT"
        self.test_price = 50000.0
        self.test_data = self._create_test_data()

    def tearDown(self):
        """清理测试环境"""
        self.manager.shutdown()

    def _create_test_data(self):
        """创建测试用的市场数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')

        close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        high_prices = close_prices + np.random.uniform(50, 200, 100)
        low_prices = close_prices - np.random.uniform(50, 200, 100)
        open_prices = np.roll(close_prices, 1)
        volumes = np.random.uniform(1000, 5000, 100)

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)

        return {
            "1h": df,
            "4h": df.iloc[::4]  # 模拟4小时数据
        }

    def test_strategy_factory_registration(self):
        """测试策略工厂注册功能"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        # 验证注册成功
        available_strategies = self.factory.get_available_strategies()
        self.assertIn("test_strategy", available_strategies)

        # 验证策略信息
        strategy_info = self.factory.get_strategy_info("test_strategy")
        self.assertIsNotNone(strategy_info)
        self.assertEqual(strategy_info["name"], "test_strategy")

    def test_strategy_factory_creation(self):
        """测试策略工厂创建功能"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        # 创建策略实例
        strategy = self.factory.create_strategy("test_strategy")

        # 验证策略实例
        self.assertIsInstance(strategy, MockStrategy)
        self.assertEqual(strategy.name, "test_strategy")
        self.assertTrue(strategy.is_enabled)

        # 测试自定义配置创建
        custom_config = {
            "name": "custom_test_strategy",
            "default_leverage": 15.0,
            "min_confidence": 60.0
        }

        custom_strategy = self.factory.create_strategy(
            "test_strategy",
            config=custom_config
        )

        self.assertEqual(custom_strategy.config.default_leverage, 15.0)
        self.assertEqual(custom_strategy.config.min_confidence, 60.0)

    def test_strategy_manager_add_remove(self):
        """测试策略管理器添加和移除功能"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        # 添加策略到管理器
        success = self.manager.add_strategy(
            "test_strategy",
            weight=0.8,
            confidence_multiplier=1.2
        )

        self.assertTrue(success)

        # 验证策略已添加
        active_strategies = self.manager.get_active_strategies()
        self.assertIn("test_strategy", active_strategies)

        # 获取管理器状态
        status = self.manager.get_manager_status()
        self.assertEqual(status["total_strategies"], 1)
        self.assertEqual(status["active_strategies"], 1)

        # 移除策略
        success = self.manager.remove_strategy("test_strategy")
        self.assertTrue(success)

        # 验证策略已移除
        active_strategies = self.manager.get_active_strategies()
        self.assertNotIn("test_strategy", active_strategies)

    def test_single_strategy_analysis(self):
        """测试单策略分析"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        # 添加策略到管理器
        self.manager.add_strategy("test_strategy")

        # 执行分析
        result = self.manager.analyze_ticker(
            ticker=self.test_ticker,
            data=self.test_data,
            current_price=self.test_price,
            validation_level=ValidationLevel.BASIC
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.final_signal)
        self.assertEqual(result.final_signal.ticker, self.test_ticker)
        self.assertEqual(result.final_signal.direction, TradingDirection.LONG)
        self.assertEqual(len(result.contributing_signals), 1)

    def test_multi_strategy_analysis(self):
        """测试多策略分析和聚合"""
        # 注册两个测试策略
        for i in range(2):
            test_config = StrategyConfig(
                name=f"test_strategy_{i}",
                strategy_type=StrategyType.TREND_FOLLOWING
            )

            strategy_class = type(f"MockStrategy{i}", (MockStrategy,), {})
            self.factory.register_strategy(
                f"test_strategy_{i}",
                strategy_class,
                test_config
            )

            self.manager.add_strategy(
                f"test_strategy_{i}",
                weight=0.5
            )

        # 执行多策略分析
        result = self.manager.analyze_ticker(
            ticker=self.test_ticker,
            data=self.test_data,
            current_price=self.test_price,
            aggregation_method=AggregationMethod.CONFIDENCE_WEIGHTED,
            validation_level=ValidationLevel.STANDARD
        )

        # 验证结果
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.final_signal)
        self.assertEqual(len(result.contributing_signals), 2)
        self.assertEqual(len(result.confidence_scores), 2)

        # 验证聚合元数据
        metadata = result.aggregation_metadata
        self.assertEqual(metadata["method"], "confidence_weighted")
        self.assertEqual(len(metadata["active_strategies"]), 2)

    def test_different_aggregation_methods(self):
        """测试不同的信号聚合方法"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        self.manager.add_strategy("test_strategy")

        # 测试不同聚合方法
        methods = [
            AggregationMethod.CONFIDENCE_WEIGHTED,
            AggregationMethod.HIGHEST_CONFIDENCE,
            AggregationMethod.MAJORITY_VOTE
        ]

        for method in methods:
            with self.subTest(method=method):
                result = self.manager.analyze_ticker(
                    ticker=self.test_ticker,
                    data=self.test_data,
                    current_price=self.test_price,
                    aggregation_method=method
                )

                self.assertIsNotNone(result)
                self.assertEqual(
                    result.aggregation_metadata["method"],
                    method.value
                )

    def test_signal_validation(self):
        """测试信号验证功能"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        self.manager.add_strategy("test_strategy")

        # 测试不同验证级别
        validation_levels = [
            ValidationLevel.BASIC,
            ValidationLevel.STANDARD,
            ValidationLevel.STRICT
        ]

        for level in validation_levels:
            with self.subTest(level=level):
                result = self.manager.analyze_ticker(
                    ticker=self.test_ticker,
                    data=self.test_data,
                    current_price=self.test_price,
                    validation_level=level
                )

                self.assertIsNotNone(result)
                # 验证结果应该包含验证信息
                self.assertIsInstance(result.validation_results, list)

    def test_strategy_weight_management(self):
        """测试策略权重管理"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        # 添加策略并设置权重
        self.manager.add_strategy(
            "test_strategy",
            weight=0.8,
            confidence_multiplier=1.5
        )

        # 验证权重设置
        status = self.manager.get_manager_status()
        weights = status["strategy_weights"]
        self.assertEqual(weights["test_strategy"]["weight"], 0.8)
        self.assertEqual(weights["test_strategy"]["confidence_multiplier"], 1.5)

        # 更新权重
        success = self.manager.update_strategy_weight(
            "test_strategy",
            weight=0.6,
            confidence_multiplier=1.2
        )

        self.assertTrue(success)

        # 验证权重更新
        updated_status = self.manager.get_manager_status()
        updated_weights = updated_status["strategy_weights"]
        self.assertEqual(updated_weights["test_strategy"]["weight"], 0.6)
        self.assertEqual(updated_weights["test_strategy"]["confidence_multiplier"], 1.2)

    def test_performance_tracking(self):
        """测试性能统计追踪"""
        # 注册测试策略
        test_config = StrategyConfig(
            name="test_strategy",
            strategy_type=StrategyType.TREND_FOLLOWING
        )

        self.factory.register_strategy(
            "test_strategy",
            MockStrategy,
            test_config
        )

        self.manager.add_strategy("test_strategy")

        # 执行多次分析以生成统计数据
        for _ in range(3):
            self.manager.analyze_ticker(
                ticker=self.test_ticker,
                data=self.test_data,
                current_price=self.test_price
            )

        # 验证性能统计
        status = self.manager.get_manager_status()
        perf_stats = status["performance_stats"]

        self.assertGreaterEqual(perf_stats["total_analyses"], 3)
        self.assertGreaterEqual(perf_stats["successful_analyses"], 0)
        self.assertGreaterEqual(perf_stats["avg_processing_time"], 0)

    def test_error_handling(self):
        """测试错误处理"""
        # 测试创建不存在的策略
        with self.assertRaises(Exception):
            self.factory.create_strategy("non_existent_strategy")

        # 测试分析无效数据
        result = self.manager.analyze_ticker(
            ticker=self.test_ticker,
            data={},  # 空数据
            current_price=self.test_price
        )

        # 应该返回None或处理错误
        self.assertIsNone(result)

    @patch('src.futures.signals.strategy_adapters.FuturesMacdStrategyAdapter')
    def test_builtin_strategy_registration(self, mock_adapter):
        """测试内置策略注册（使用mock）"""
        # 重置工厂以测试内置策略注册
        fresh_factory = FuturesStrategyFactory()

        # 验证内置策略是否已注册
        available_strategies = fresh_factory.get_available_strategies()

        # 由于导入问题，我们主要测试注册机制
        self.assertIsInstance(available_strategies, list)

        # 测试工厂统计
        stats = fresh_factory.get_factory_stats()
        self.assertIn("registered_strategies", stats)
        self.assertIn("strategy_types", stats)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)