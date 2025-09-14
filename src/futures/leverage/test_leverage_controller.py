"""
杠杆控制器单元测试

测试杠杆控制器的各项功能，包括杠杆计算、配置管理、
风险评估等核心功能。
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

# 导入测试目标
from .leverage_controller import LeverageController, PortfolioRiskMetrics
from .leverage_models import (
    LeverageConfig,
    LeverageCalculationResult,
    LeverageAdjustmentFactor,
    LeverageLimits,
    MarketCondition,
    LeverageStrategy,
    MarketRegime,
    LiquidityLevel
)
from .config_manager import LeverageConfigManager
from ..models.data_models import RiskLevel, MarginStatus, Position, PositionSide
from ..market.market_analyzer import MarketAnalyzer


class TestLeverageController:
    """杠杆控制器测试类"""

    @pytest.fixture
    def mock_market_analyzer(self):
        """模拟市场分析器"""
        analyzer = Mock(spec=MarketAnalyzer)
        analyzer.analyze_market_condition = AsyncMock(return_value={
            'volatility': 0.2,
            'liquidity_score': 0.7,
            'trend_strength': 0.5,
            'atr_percentage': 0.05,
            'volume_ratio': 1.2,
            'spread_bps': 2.5
        })
        return analyzer

    @pytest.fixture
    def sample_config(self):
        """示例配置"""
        config = LeverageConfig()
        # 添加BTCUSDT的特定限制
        btc_limits = LeverageLimits(ticker='BTCUSDT', max_leverage=125.0)
        config.symbol_specific_limits['BTCUSDT'] = btc_limits
        return config

    @pytest.fixture
    def sample_margin_status(self):
        """示例保证金状态"""
        return MarginStatus(
            total_balance=10000.0,
            available_margin=8000.0,
            used_margin=2000.0,
            margin_ratio=0.2,
            risk_level=RiskLevel.LOW
        )

    @pytest.fixture
    def sample_position(self):
        """示例仓位"""
        return Position(
            ticker='BTCUSDT',
            side=PositionSide.LONG,
            size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            leverage=10.0,
            initial_margin=500.0,
            maintenance_margin=250.0,
            distance_to_liquidation=15.0
        )

    @pytest.fixture
    def leverage_controller(self, sample_config, mock_market_analyzer):
        """杠杆控制器实例"""
        return LeverageController(
            config=sample_config,
            market_analyzer=mock_market_analyzer
        )

    def test_controller_initialization(self, sample_config, mock_market_analyzer):
        """测试控制器初始化"""
        controller = LeverageController(
            config=sample_config,
            market_analyzer=mock_market_analyzer
        )

        assert controller.config is not None
        assert controller.market_analyzer is not None
        assert controller._calculation_count == 0
        assert len(controller._market_conditions_cache) == 0

    def test_default_configuration(self):
        """测试默认配置"""
        controller = LeverageController()

        assert controller.config is not None
        assert controller.config.default_leverage == 5.0
        assert LeverageStrategy.CONSERVATIVE in controller.config.base_leverage_by_strategy
        assert LeverageStrategy.MODERATE in controller.config.base_leverage_by_strategy

    @pytest.mark.asyncio
    async def test_basic_leverage_calculation(
        self,
        leverage_controller,
        sample_margin_status,
        sample_position
    ):
        """测试基本杠杆计算"""
        result = await leverage_controller.calculate_optimal_leverage(
            ticker='BTCUSDT',
            strategy=LeverageStrategy.MODERATE,
            current_positions=[sample_position],
            margin_status=sample_margin_status
        )

        assert isinstance(result, LeverageCalculationResult)
        assert result.ticker == 'BTCUSDT'
        assert result.applied_leverage > 0
        assert result.base_leverage > 0
        assert result.is_valid()
        assert not result.has_errors()

    def test_sync_leverage_calculation(
        self,
        leverage_controller,
        sample_margin_status
    ):
        """测试同步杠杆计算"""
        result = leverage_controller.calculate_leverage_sync(
            ticker='BTCUSDT',
            strategy=LeverageStrategy.CONSERVATIVE
        )

        assert isinstance(result, LeverageCalculationResult)
        assert result.ticker == 'BTCUSDT'
        assert result.applied_leverage > 0

    @pytest.mark.asyncio
    async def test_leverage_calculation_with_requested_leverage(
        self,
        leverage_controller
    ):
        """测试指定杠杆的计算"""
        requested_leverage = 20.0

        result = await leverage_controller.calculate_optimal_leverage(
            ticker='BTCUSDT',
            strategy=LeverageStrategy.MODERATE,
            requested_leverage=requested_leverage
        )

        assert result.requested_leverage == requested_leverage
        # 应该应用请求的杠杆（在限制范围内）
        assert result.applied_leverage <= result.effective_limit

    @pytest.mark.asyncio
    async def test_leverage_calculation_with_high_risk(
        self,
        leverage_controller
    ):
        """测试高风险情况下的杠杆计算"""
        high_risk_margin = MarginStatus(
            total_balance=1000.0,
            available_margin=100.0,
            used_margin=900.0,
            margin_ratio=0.9,  # 高风险
            risk_level=RiskLevel.CRITICAL
        )

        result = await leverage_controller.calculate_optimal_leverage(
            ticker='BTCUSDT',
            strategy=LeverageStrategy.MODERATE,
            margin_status=high_risk_margin
        )

        assert result.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.EMERGENCY]
        assert result.applied_leverage < result.base_leverage  # 应该降低杠杆
        assert result.has_warnings() or result.adjustment_factors.risk_adjustment < 0.8

    def test_get_leverage_limits(self, leverage_controller):
        """测试获取杠杆限制"""
        # 测试已配置的交易对
        btc_limits = leverage_controller._get_leverage_limits('BTCUSDT')
        assert btc_limits.ticker == 'BTCUSDT'
        assert btc_limits.max_leverage == 125.0

        # 测试未配置的交易对
        eth_limits = leverage_controller._get_leverage_limits('ETHUSDT')
        assert eth_limits.ticker == 'ETHUSDT'
        assert eth_limits.max_leverage > 0  # 应该有默认限制

    def test_risk_level_assessment(self, leverage_controller, sample_position):
        """测试风险等级评估"""
        # 低风险保证金状态
        low_risk_margin = MarginStatus(
            total_balance=10000.0,
            available_margin=9000.0,
            used_margin=1000.0,
            margin_ratio=0.1,
            risk_level=RiskLevel.LOW
        )

        risk_level = leverage_controller._assess_risk_level(
            margin_status=low_risk_margin,
            positions=[sample_position],
            market_condition=None
        )

        assert isinstance(risk_level, RiskLevel)

        # 高风险保证金状态
        high_risk_margin = MarginStatus(
            total_balance=1000.0,
            available_margin=100.0,
            used_margin=900.0,
            margin_ratio=0.9,
            risk_level=RiskLevel.CRITICAL
        )

        risk_level = leverage_controller._assess_risk_level(
            margin_status=high_risk_margin,
            positions=[sample_position],
            market_condition=None
        )

        assert risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EMERGENCY]

    def test_market_regime_adjustment(self, leverage_controller):
        """测试市场状态调整"""
        # 测试不同市场状态的调整因子
        calm_adjustment = leverage_controller._get_market_regime_adjustment(MarketRegime.CALM)
        assert calm_adjustment == 1.0

        volatile_adjustment = leverage_controller._get_market_regime_adjustment(MarketRegime.VOLATILE)
        assert volatile_adjustment < 1.0

        trending_adjustment = leverage_controller._get_market_regime_adjustment(MarketRegime.TRENDING)
        assert trending_adjustment >= 1.0

    def test_position_adjustment_calculation(self, leverage_controller):
        """测试仓位调整计算"""
        # 创建多个仓位，测试集中度
        positions = [
            Position(
                ticker='BTCUSDT',
                side=PositionSide.LONG,
                size=0.5,
                entry_price=50000.0,
                current_price=51000.0,
                leverage=10.0,
                initial_margin=2500.0,
                maintenance_margin=1250.0,
                notional_value=25500.0
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
            )
        ]

        adjustment = leverage_controller._calculate_position_adjustment(
            ticker='BTCUSDT',
            positions=positions
        )

        assert 0.0 < adjustment <= 1.0

    def test_create_default_limits(self, leverage_controller):
        """测试创建默认限制"""
        # 主流币种
        btc_limits = leverage_controller._create_default_limits('BTCUSDT')
        assert btc_limits.max_leverage == 125.0

        # 二线币种
        ada_limits = leverage_controller._create_default_limits('ADAUSDT')
        assert ada_limits.max_leverage == 75.0

        # 其他币种
        other_limits = leverage_controller._create_default_limits('XYZUSDT')
        assert other_limits.max_leverage == 50.0

        # 非USDT交易对
        non_usdt_limits = leverage_controller._create_default_limits('BTCETH')
        assert non_usdt_limits.max_leverage == 20.0

    def test_leverage_statistics(self, leverage_controller):
        """测试统计信息"""
        initial_stats = leverage_controller.get_leverage_statistics()

        assert 'calculation_count' in initial_stats
        assert 'adjustment_stats' in initial_stats
        assert 'cache_size' in initial_stats

        assert initial_stats['calculation_count'] == 0
        assert isinstance(initial_stats['adjustment_stats'], dict)

    def test_cache_management(self, leverage_controller):
        """测试缓存管理"""
        # 初始缓存应该是空的
        stats = leverage_controller.get_leverage_statistics()
        assert stats['cache_size']['market_conditions'] == 0

        # 清空缓存
        leverage_controller.clear_cache()
        stats = leverage_controller.get_leverage_statistics()
        assert stats['cache_size']['market_conditions'] == 0

    def test_config_update(self, leverage_controller):
        """测试配置更新"""
        new_config = LeverageConfig()
        new_config.default_leverage = 8.0

        old_default = leverage_controller.config.default_leverage
        leverage_controller.update_config(new_config)

        assert leverage_controller.config.default_leverage == 8.0
        assert leverage_controller.config.default_leverage != old_default

    def test_add_symbol_limits(self, leverage_controller):
        """测试添加交易对限制"""
        new_limits = LeverageLimits(ticker='DOGEUSDT', max_leverage=30.0)

        leverage_controller.add_symbol_limits('DOGEUSDT', new_limits)

        # 验证限制已添加
        retrieved_limits = leverage_controller._get_leverage_limits('DOGEUSDT')
        assert retrieved_limits.ticker == 'DOGEUSDT'
        assert retrieved_limits.max_leverage == 30.0


class TestLeverageModels:
    """杠杆模型测试类"""

    def test_market_condition_creation(self):
        """测试市场条件创建"""
        condition = MarketCondition(
            ticker='BTCUSDT',
            volatility=0.25,
            liquidity_score=0.8,
            market_regime=MarketRegime.VOLATILE,
            liquidity_level=LiquidityLevel.HIGH
        )

        assert condition.ticker == 'BTCUSDT'
        assert condition.volatility == 0.25
        assert condition.liquidity_score == 0.8
        assert condition.market_regime == MarketRegime.VOLATILE
        assert condition.liquidity_level == LiquidityLevel.HIGH

    def test_market_condition_multipliers(self):
        """测试市场条件倍数计算"""
        condition = MarketCondition(
            ticker='BTCUSDT',
            volatility=0.25,
            liquidity_score=0.8,
            market_regime=MarketRegime.VOLATILE,
            liquidity_level=LiquidityLevel.HIGH
        )

        vol_multiplier = condition.get_volatility_multiplier()
        liq_multiplier = condition.get_liquidity_multiplier()

        assert isinstance(vol_multiplier, float)
        assert isinstance(liq_multiplier, float)
        assert 0.0 < vol_multiplier <= 2.0
        assert 0.0 < liq_multiplier <= 2.0

    def test_leverage_limits_creation(self):
        """测试杠杆限制创建"""
        limits = LeverageLimits(ticker='BTCUSDT', max_leverage=100.0)

        assert limits.ticker == 'BTCUSDT'
        assert limits.max_leverage == 100.0
        assert limits.min_leverage == 1.0
        assert len(limits.risk_based_limits) > 0
        assert len(limits.strategy_based_limits) > 0

    def test_leverage_limits_effective_limit(self):
        """测试有效杠杆限制计算"""
        limits = LeverageLimits(ticker='BTCUSDT', max_leverage=100.0)

        effective = limits.get_effective_limit(
            risk_level=RiskLevel.HIGH,
            strategy=LeverageStrategy.CONSERVATIVE,
            market_regime=MarketRegime.VOLATILE
        )

        assert isinstance(effective, float)
        assert effective > 0
        assert effective <= limits.max_leverage

    def test_adjustment_factors(self):
        """测试调整因子"""
        factors = LeverageAdjustmentFactor(
            volatility_multiplier=0.8,
            liquidity_multiplier=0.9,
            market_adjustment=1.0,
            position_adjustment=0.95,
            risk_adjustment=0.9
        )

        combined = factors.get_combined_multiplier()
        expected = 0.8 * 0.9 * 1.0 * 0.95 * 0.9

        assert abs(combined - expected) < 0.001

    def test_leverage_calculation_result(self):
        """测试杠杆计算结果"""
        result = LeverageCalculationResult(
            ticker='BTCUSDT',
            calculated_leverage=10.5,
            applied_leverage=10.0,
            base_leverage=12.0,
            requested_leverage=11.0
        )

        assert result.ticker == 'BTCUSDT'
        assert result.calculated_leverage == 10.5
        assert result.applied_leverage == 10.0
        assert result.base_leverage == 12.0
        assert result.requested_leverage == 11.0

        # 测试调整百分比计算
        adjustment_pct = result.get_adjustment_percentage()
        expected_pct = ((10.0 - 11.0) / 11.0) * 100
        assert abs(adjustment_pct - expected_pct) < 0.001

    def test_leverage_calculation_result_validity(self):
        """测试杠杆计算结果有效性"""
        result = LeverageCalculationResult(
            ticker='BTCUSDT',
            calculated_leverage=10.0,
            applied_leverage=10.0,
            base_leverage=10.0
        )

        # 初始状态应该有效
        assert result.is_valid()

        # 添加错误后应该无效
        result.add_error("测试错误")
        assert not result.is_valid()
        assert result.has_errors()

        # 重置错误，添加过期时间
        result.errors.clear()
        result.valid_until = datetime.now() - timedelta(minutes=1)  # 已过期
        assert not result.is_valid()

    def test_leverage_config(self):
        """测试杠杆配置"""
        config = LeverageConfig()

        assert config.default_leverage > 0
        assert len(config.base_leverage_by_strategy) > 0
        assert config.max_adjustment_percentage > 0
        assert config.min_adjustment_percentage >= 0

        # 测试获取基础杠杆
        conservative_leverage = config.get_base_leverage(LeverageStrategy.CONSERVATIVE)
        moderate_leverage = config.get_base_leverage(LeverageStrategy.MODERATE)

        assert conservative_leverage > 0
        assert moderate_leverage > 0
        assert conservative_leverage <= moderate_leverage  # 保守策略应该杠杆更低

    def test_leverage_config_serialization(self):
        """测试杠杆配置序列化"""
        config = LeverageConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'default_leverage' in config_dict
        assert 'base_leverage_by_strategy' in config_dict
        assert isinstance(config_dict['base_leverage_by_strategy'], dict)


class TestLeverageConfigManager:
    """杠杆配置管理器测试类"""

    @pytest.fixture
    def config_manager(self):
        """配置管理器实例"""
        return LeverageConfigManager()

    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        manager = LeverageConfigManager()

        assert manager.config_path is None
        assert manager._config_cache is None

    def test_create_default_config(self, config_manager):
        """测试创建默认配置"""
        config = config_manager._create_default_config()

        assert isinstance(config, LeverageConfig)
        assert config.default_leverage > 0
        assert len(config.symbol_specific_limits) > 0

    def test_validate_config(self, config_manager):
        """测试配置验证"""
        # 有效配置
        valid_config = LeverageConfig()
        errors = config_manager.validate_config(valid_config)
        assert len(errors) == 0

        # 无效配置
        invalid_config = LeverageConfig()
        invalid_config.default_leverage = -1.0  # 无效杠杆
        errors = config_manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("杠杆必须大于0" in error for error in errors)

    def test_config_template(self, config_manager):
        """测试配置模板"""
        template = config_manager.get_config_template()

        assert isinstance(template, dict)
        assert 'default_leverage' in template
        assert 'base_leverage_by_strategy' in template
        assert 'symbol_specific_limits' in template

        # 验证模板结构
        assert isinstance(template['base_leverage_by_strategy'], dict)
        assert isinstance(template['symbol_specific_limits'], dict)

    def test_parse_config_data(self, config_manager):
        """测试解析配置数据"""
        config_data = {
            'default_leverage': 8.0,
            'base_leverage_by_strategy': {
                'conservative': 4.0,
                'moderate': 8.0
            },
            'max_adjustment_percentage': 60.0
        }

        config = config_manager._parse_config_data(config_data)

        assert config.default_leverage == 8.0
        assert config.max_adjustment_percentage == 60.0
        assert LeverageStrategy.CONSERVATIVE in config.base_leverage_by_strategy
        assert config.base_leverage_by_strategy[LeverageStrategy.CONSERVATIVE] == 4.0

    def test_parse_symbol_limits(self, config_manager):
        """测试解析交易对限制"""
        limits_data = {
            'max_leverage': 50.0,
            'min_leverage': 2.0,
            'risk_based_limits': {
                'low': 50.0,
                'high': 25.0
            },
            'strategy_based_limits': {
                'conservative': 10.0,
                'aggressive': 40.0
            }
        }

        limits = config_manager._parse_symbol_limits('TESTUSDT', limits_data)

        assert limits.ticker == 'TESTUSDT'
        assert limits.max_leverage == 50.0
        assert limits.min_leverage == 2.0
        assert RiskLevel.LOW in limits.risk_based_limits
        assert limits.risk_based_limits[RiskLevel.LOW] == 50.0


# 运行测试的主函数
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])