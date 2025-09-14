"""
杠杆控制器集成演示

展示如何将杠杆控制器集成到现有的期货交易系统中，
包括与RiskManagement、MarketAnalyzer等组件的协作。
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional

from .leverage_controller import LeverageController
from .leverage_models import LeverageStrategy
from ..models.data_models import (
    FuturesSignal,
    MarginStatus,
    Position,
    RiskLevel,
    TradingDirection,
    OperationType
)

logger = logging.getLogger(__name__)


class EnhancedRiskManagementDemo:
    """
    增强风险管理演示类

    展示如何在风险管理流程中集成杠杆控制器
    """

    def __init__(self):
        self.leverage_controller = LeverageController()
        self.user_strategy = LeverageStrategy.MODERATE

    async def process_trading_signal(
        self,
        signal: FuturesSignal,
        current_positions: List[Position],
        margin_status: MarginStatus
    ) -> FuturesSignal:
        """
        处理交易信号，集成杠杆控制

        Args:
            signal: 原始交易信号
            current_positions: 当前持仓
            margin_status: 保证金状态

        Returns:
            经过杠杆优化的交易信号
        """
        print(f"处理交易信号: {signal.ticker} {signal.direction.value}")

        # 1. 计算最优杠杆
        leverage_result = await self.leverage_controller.calculate_optimal_leverage(
            ticker=signal.ticker,
            strategy=self.user_strategy,
            requested_leverage=signal.suggested_leverage,
            current_positions=current_positions,
            margin_status=margin_status
        )

        # 2. 应用杠杆建议
        original_leverage = signal.suggested_leverage
        signal.suggested_leverage = leverage_result.applied_leverage

        # 3. 调整仓位大小（如果杠杆发生显著变化）
        if leverage_result.is_adjustment_significant(threshold=0.2):
            # 基于杠杆调整重新计算仓位大小
            if signal.position_size and original_leverage:
                leverage_ratio = leverage_result.applied_leverage / original_leverage
                signal.position_size = signal.position_size * leverage_ratio

                print(f"  杠杆显著调整: {original_leverage:.2f} -> {leverage_result.applied_leverage:.2f}")
                print(f"  仓位大小调整: {signal.position_size:.2f} USDT")

        # 4. 更新信号风险等级
        signal.risk_level = leverage_result.risk_level

        # 5. 添加杠杆相关元数据
        signal.metadata.update({
            'leverage_calculation': {
                'original_leverage': original_leverage,
                'calculated_leverage': leverage_result.calculated_leverage,
                'applied_leverage': leverage_result.applied_leverage,
                'adjustment_factors': leverage_result.adjustment_factors.to_dict(),
                'risk_assessment': leverage_result.risk_level.value,
                'confidence_score': leverage_result.confidence_score,
                'warnings': leverage_result.warnings,
                'recommendation': leverage_result.recommendation
            }
        })

        # 6. 风险检查
        if leverage_result.has_warnings():
            print(f"  ⚠️  杠杆计算警告: {'; '.join(leverage_result.warnings)}")

        if leverage_result.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
            print(f"  🚨 高风险警告: {leverage_result.risk_level.value}")
            # 在高风险情况下，可以选择拒绝信号
            signal.confidence = min(signal.confidence, 30.0)  # 降低信号置信度

        print(f"  最终杠杆: {signal.suggested_leverage:.2f}x")
        print(f"  风险等级: {signal.risk_level.value}")
        print(f"  建议: {leverage_result.recommendation}")

        return signal


class PositionSizingDemo:
    """
    仓位大小计算演示类

    展示如何使用杠杆控制器优化仓位大小计算
    """

    def __init__(self):
        self.leverage_controller = LeverageController()

    async def calculate_position_size(
        self,
        ticker: str,
        account_balance: float,
        risk_percentage: float,
        entry_price: float,
        stop_loss_price: float,
        strategy: LeverageStrategy = LeverageStrategy.MODERATE,
        current_positions: Optional[List[Position]] = None,
        margin_status: Optional[MarginStatus] = None
    ) -> Dict[str, float]:
        """
        计算最优仓位大小

        Args:
            ticker: 交易对
            account_balance: 账户余额
            risk_percentage: 风险百分比
            entry_price: 入场价格
            stop_loss_price: 止损价格
            strategy: 杠杆策略
            current_positions: 当前持仓
            margin_status: 保证金状态

        Returns:
            仓位计算结果
        """
        print(f"\n计算最优仓位大小: {ticker}")

        # 1. 获取最优杠杆
        leverage_result = await self.leverage_controller.calculate_optimal_leverage(
            ticker=ticker,
            strategy=strategy,
            current_positions=current_positions,
            margin_status=margin_status
        )

        optimal_leverage = leverage_result.applied_leverage

        # 2. 计算风险金额
        risk_amount = account_balance * (risk_percentage / 100)

        # 3. 计算价格风险（每合约的潜在损失）
        price_risk = abs(entry_price - stop_loss_price)
        price_risk_percentage = price_risk / entry_price

        # 4. 计算基础仓位大小（不考虑杠杆）
        base_position_size = risk_amount / price_risk

        # 5. 应用杠杆优化
        # 杠杆影响保证金需求，但不直接影响风险金额计算
        margin_required = (base_position_size * entry_price) / optimal_leverage

        # 6. 检查可用保证金限制
        if margin_status and margin_required > margin_status.available_margin:
            # 如果保证金不足，调整仓位大小
            max_affordable_position = (margin_status.available_margin * optimal_leverage) / entry_price
            base_position_size = min(base_position_size, max_affordable_position)
            margin_required = (base_position_size * entry_price) / optimal_leverage
            print(f"  ⚠️  保证金限制，仓位大小已调整")

        # 7. 最终仓位验证
        notional_value = base_position_size * entry_price
        effective_leverage = notional_value / margin_required if margin_required > 0 else 0

        result = {
            'position_size': base_position_size,
            'notional_value': notional_value,
            'margin_required': margin_required,
            'optimal_leverage': optimal_leverage,
            'effective_leverage': effective_leverage,
            'risk_amount': risk_amount,
            'price_risk_percentage': price_risk_percentage * 100,
            'leverage_confidence': leverage_result.confidence_score
        }

        print(f"  仓位大小: {result['position_size']:.4f} 合约")
        print(f"  名义价值: {result['notional_value']:.2f} USDT")
        print(f"  保证金需求: {result['margin_required']:.2f} USDT")
        print(f"  最优杠杆: {result['optimal_leverage']:.2f}x")
        print(f"  有效杠杆: {result['effective_leverage']:.2f}x")
        print(f"  价格风险: {result['price_risk_percentage']:.2f}%")

        return result


async def comprehensive_demo():
    """综合演示"""
    print("=" * 60)
    print("杠杆控制器系统集成演示")
    print("=" * 60)

    # 创建演示数据
    sample_signal = FuturesSignal(
        ticker='BTCUSDT',
        direction=TradingDirection.LONG,
        operation_type=OperationType.OPEN,
        confidence=75.0,
        strength=0.8,
        suggested_leverage=10.0,
        entry_price=50000.0,
        current_price=50000.0,
        take_profit_price=52000.0,
        stop_loss_price=48000.0,
        strategy_source='MACD_Strategy'
    )

    current_positions = [
        Position(
            ticker='ETHUSDT',
            side='LONG',
            size=2.0,
            entry_price=3000.0,
            current_price=3100.0,
            leverage=5.0,
            initial_margin=1240.0,
            maintenance_margin=620.0,
            notional_value=6200.0
        )
    ]

    margin_status = MarginStatus(
        total_balance=15000.0,
        available_margin=12000.0,
        used_margin=3000.0,
        margin_ratio=0.2,
        risk_level=RiskLevel.LOW
    )

    # 演示1: 风险管理集成
    print("\n1. 风险管理集成演示")
    print("-" * 30)

    risk_manager = EnhancedRiskManagementDemo()
    enhanced_signal = await risk_manager.process_trading_signal(
        signal=sample_signal,
        current_positions=current_positions,
        margin_status=margin_status
    )

    # 演示2: 仓位大小计算
    print("\n2. 仓位大小计算演示")
    print("-" * 30)

    position_sizer = PositionSizingDemo()
    position_result = await position_sizer.calculate_position_size(
        ticker='BTCUSDT',
        account_balance=15000.0,
        risk_percentage=2.0,  # 2% 风险
        entry_price=50000.0,
        stop_loss_price=48000.0,
        strategy=LeverageStrategy.MODERATE,
        current_positions=current_positions,
        margin_status=margin_status
    )

    # 演示3: 不同策略对比
    print("\n3. 不同杠杆策略对比")
    print("-" * 30)

    strategies = [
        LeverageStrategy.CONSERVATIVE,
        LeverageStrategy.MODERATE,
        LeverageStrategy.AGGRESSIVE,
        LeverageStrategy.EXPERT
    ]

    controller = LeverageController()

    for strategy in strategies:
        result = await controller.calculate_optimal_leverage(
            ticker='BTCUSDT',
            strategy=strategy,
            current_positions=current_positions,
            margin_status=margin_status
        )

        print(f"  {strategy.value:>12}: {result.applied_leverage:>6.2f}x "
              f"(调整: {result.adjustment_factors.get_combined_multiplier():.3f})")

    # 演示4: 系统性能统计
    print("\n4. 系统性能统计")
    print("-" * 30)

    stats = controller.get_leverage_statistics()
    print(f"  计算次数: {stats['calculation_count']}")
    print(f"  总调整次数: {stats['adjustment_stats']['total_adjustments']}")
    print(f"  紧急降杠杆次数: {stats['adjustment_stats']['emergency_reductions']}")
    print(f"  波动率调整次数: {stats['adjustment_stats']['volatility_adjustments']}")
    print(f"  缓存大小: {stats['cache_size']}")

    print("\n" + "=" * 60)
    print("演示完成 - 杠杆控制器已成功集成到交易系统中")
    print("=" * 60)


if __name__ == '__main__':
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行综合演示
    asyncio.run(comprehensive_demo())