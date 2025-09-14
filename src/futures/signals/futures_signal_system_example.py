"""
期货信号系统使用示例

演示如何集成和使用 FuturesSignalSystem 生成完整的期货交易信号
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# 导入必要模块
from .futures_signal_system import FuturesSignalSystem
from .strategy_manager import FuturesStrategyManager
from ..leverage.leverage_controller import LeverageController, LeverageStrategy
from ..market.market_analyzer import MarketAnalyzer
from ..models.data_models import (
    FuturesSignal, Position, MarginStatus, TradingDirection,
    PositionSide, RiskLevel
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturesSignalSystemDemo:
    """期货信号系统演示类"""

    def __init__(self):
        """初始化演示系统"""
        # 创建核心组件
        self.strategy_manager = FuturesStrategyManager()
        self.leverage_controller = LeverageController()
        self.market_analyzer = None  # 可选组件

        # 创建信号系统
        self.signal_system = FuturesSignalSystem(
            strategy_manager=self.strategy_manager,
            leverage_controller=self.leverage_controller,
            market_analyzer=self.market_analyzer
        )

    def setup_strategies(self) -> bool:
        """设置交易策略"""
        try:
            # 添加MACD策略
            success1 = self.strategy_manager.add_strategy(
                strategy_name="futures_macd",
                config={
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                },
                weight=0.4,
                confidence_multiplier=1.0
            )

            # 添加RSI策略
            success2 = self.strategy_manager.add_strategy(
                strategy_name="futures_rsi",
                config={
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30
                },
                weight=0.35,
                confidence_multiplier=1.1
            )

            if success1 and success2:
                logger.info("策略设置完成")
                return True
            else:
                logger.error("策略设置失败")
                return False

        except Exception as e:
            logger.error(f"设置策略时发生错误: {e}")
            return False

    def create_sample_market_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """创建示例市场数据"""
        try:
            # 生成多时间框架的示例数据
            timeframes = ['5m', '15m', '30m', '1h', '4h']
            market_data = {}

            base_price = 45000.0 if ticker == 'BTCUSDT' else 3000.0  # BTC或ETH的基础价格

            for tf in timeframes:
                # 根据时间框架调整数据点数量
                if tf == '5m':
                    periods = 288  # 24小时的5分钟数据
                elif tf == '15m':
                    periods = 96   # 24小时的15分钟数据
                elif tf == '30m':
                    periods = 48   # 24小时的30分钟数据
                elif tf == '1h':
                    periods = 120  # 5天的小时数据
                else:  # 4h
                    periods = 60   # 10天的4小时数据

                # 生成价格数据（简化的随机游走）
                np.random.seed(42)  # 确保可重复性
                returns = np.random.normal(0, 0.02, periods)  # 2%日波动率
                prices = [base_price]

                for i in range(1, periods):
                    new_price = prices[-1] * (1 + returns[i])
                    prices.append(new_price)

                # 生成OHLCV数据
                opens = prices[:-1]
                closes = prices[1:]

                # 简化的高低价生成
                highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.005))) for o, c in zip(opens, closes)]
                lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.005))) for o, c in zip(opens, closes)]

                # 生成成交量
                volumes = np.random.lognormal(10, 0.5, len(opens))

                # 创建DataFrame
                df = pd.DataFrame({
                    'timestamp': pd.date_range(
                        start=datetime.now() - timedelta(hours=periods),
                        periods=len(opens),
                        freq='5min' if tf == '5m' else tf
                    ),
                    'open': opens,
                    'high': highs,
                    'low': lows,
                    'close': closes,
                    'volume': volumes
                })

                market_data[tf] = df

            logger.info(f"生成{ticker}的示例市场数据，时间框架: {list(market_data.keys())}")
            return market_data

        except Exception as e:
            logger.error(f"创建示例市场数据失败: {e}")
            return {}

    def create_sample_positions(self, ticker: str) -> List[Position]:
        """创建示例持仓"""
        try:
            current_price = 45000.0 if ticker == 'BTCUSDT' else 3000.0

            positions = [
                Position(
                    ticker=ticker,
                    side=PositionSide.LONG,
                    size=0.1,  # 0.1 BTC或 ETH
                    entry_price=current_price * 0.98,  # 2%盈利
                    current_price=current_price,
                    leverage=5.0,
                    initial_margin=current_price * 0.1 / 5.0,  # 初始保证金
                    maintenance_margin=current_price * 0.1 * 0.004,  # 维持保证金
                    liquidation_price=current_price * 0.8,  # 强平价格
                    position_id=f"pos_{ticker}_001"
                )
            ]

            logger.info(f"创建{ticker}的示例持仓")
            return positions

        except Exception as e:
            logger.error(f"创建示例持仓失败: {e}")
            return []

    def create_sample_margin_status(self) -> MarginStatus:
        """创建示例保证金状态"""
        try:
            margin_status = MarginStatus(
                total_balance=10000.0,  # 总余额10000 USDT
                available_margin=5000.0,  # 可用保证金5000 USDT
                used_margin=3000.0,  # 已用保证金3000 USDT
                margin_ratio=0.3,  # 保证金率30%
                initial_margin_requirement=2500.0,
                maintenance_margin_requirement=1000.0,
                risk_level=RiskLevel.MEDIUM,
                can_trade=True,
                can_open_position=True,
                account_status="normal"
            )

            logger.info("创建示例保证金状态")
            return margin_status

        except Exception as e:
            logger.error(f"创建示例保证金状态失败: {e}")
            return MarginStatus(
                total_balance=1000.0,
                available_margin=500.0,
                used_margin=300.0,
                margin_ratio=0.3
            )

    async def generate_signal_demo(self, ticker: str = "BTCUSDT") -> Optional[FuturesSignal]:
        """生成信号演示"""
        try:
            logger.info(f"开始生成{ticker}的期货信号演示")

            # 1. 准备市场数据
            market_data = self.create_sample_market_data(ticker)
            if not market_data:
                logger.error("无法创建市场数据")
                return None

            current_price = market_data['1h']['close'].iloc[-1]

            # 2. 准备持仓和保证金数据
            current_positions = self.create_sample_positions(ticker)
            margin_status = self.create_sample_margin_status()

            # 3. 生成信号
            signal = await self.signal_system.generate_signal(
                ticker=ticker,
                market_data=market_data,
                current_price=current_price,
                current_positions=current_positions,
                margin_status=margin_status,
                leverage_strategy=LeverageStrategy.MODERATE
            )

            if signal:
                logger.info("期货信号生成成功")
                self._print_signal_details(signal)
                return signal
            else:
                logger.warning("期货信号生成失败")
                return None

        except Exception as e:
            logger.error(f"信号生成演示失败: {e}")
            return None

    def _print_signal_details(self, signal: FuturesSignal):
        """打印信号详细信息"""
        print("\n" + "="*80)
        print(f"期货交易信号详情 - {signal.ticker}")
        print("="*80)

        print(f"📊 基础信息:")
        print(f"   交易方向: {signal.direction.value.upper()}")
        print(f"   操作类型: {signal.operation_type.value.upper()}")
        print(f"   信号强度: {signal.strength:.3f}")
        print(f"   置信度: {signal.confidence:.1f}%")
        print(f"   风险等级: {signal.risk_level.value.upper()}")

        print(f"\n💰 价格信息:")
        print(f"   当前价格: ${signal.current_price:,.2f}")
        if signal.entry_price:
            print(f"   入场价格: ${signal.entry_price:,.2f}")
        if signal.take_profit_price:
            print(f"   止盈价格: ${signal.take_profit_price:,.2f}")
        if signal.stop_loss_price:
            print(f"   止损价格: ${signal.stop_loss_price:,.2f}")

        print(f"\n⚡ 杠杆和仓位:")
        print(f"   建议杠杆: {signal.suggested_leverage:.1f}x")
        if signal.position_size:
            print(f"   仓位大小: ${signal.position_size:,.2f} USDT")

        print(f"\n📈 风险收益:")
        risk_reward_ratio = signal.calculate_risk_reward_ratio()
        if risk_reward_ratio:
            print(f"   风险收益比: 1:{risk_reward_ratio:.2f}")

        print(f"\n🕐 时间信息:")
        print(f"   生成时间: {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if signal.expiry_time:
            print(f"   过期时间: {signal.expiry_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if signal.metadata:
            print(f"\n📋 详细信息:")
            metadata = signal.metadata
            if 'expected_return' in metadata:
                print(f"   预期收益: {metadata['expected_return']:.2f}%")
            if 'expected_duration_hours' in metadata:
                print(f"   预期持仓: {metadata['expected_duration_hours']:.1f} 小时")
            if 'risk_score' in metadata:
                print(f"   风险评分: {metadata['risk_score']:.1f}/100")
            if 'processing_time_ms' in metadata:
                print(f"   处理时间: {metadata['processing_time_ms']:.1f} ms")

        print("="*80 + "\n")

    async def run_full_demo(self):
        """运行完整演示"""
        try:
            print("\n🚀 期货信号系统完整演示")
            print("=" * 50)

            # 1. 初始化系统
            print("\n1. 初始化信号系统...")
            if not self.signal_system.initialize():
                print("❌ 信号系统初始化失败")
                return

            # 2. 设置策略
            print("\n2. 设置交易策略...")
            if not self.setup_strategies():
                print("❌ 策略设置失败")
                return

            # 3. 生成多个交易对的信号
            tickers = ["BTCUSDT", "ETHUSDT"]

            for ticker in tickers:
                print(f"\n3. 生成 {ticker} 交易信号...")
                signal = await self.generate_signal_demo(ticker)

                if signal:
                    print(f"✅ {ticker} 信号生成成功")
                else:
                    print(f"❌ {ticker} 信号生成失败")

            # 4. 显示系统状态
            print("\n4. 系统状态汇总:")
            system_status = self.signal_system.get_system_status()
            print(f"   总信号数: {system_status['performance_stats']['total_signals']}")
            print(f"   成功信号数: {system_status['performance_stats']['successful_signals']}")
            print(f"   平均处理时间: {system_status['performance_stats']['avg_processing_time']:.3f}s")
            print(f"   活跃策略数: {system_status['active_strategies']}")

            print("\n✅ 演示完成!")

        except Exception as e:
            logger.error(f"完整演示失败: {e}")
            print(f"❌ 演示过程中发生错误: {e}")


# 独立运行示例
async def main():
    """主函数"""
    try:
        # 导入numpy（用于数据生成）
        global np
        import numpy as np

        demo = FuturesSignalSystemDemo()
        await demo.run_full_demo()

    except Exception as e:
        print(f"演示运行失败: {e}")
        logger.error(f"主函数执行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())