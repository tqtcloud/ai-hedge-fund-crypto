import os
import sys
import asyncio
import logging
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dotenv import load_dotenv
from src.utils import settings
from datetime import datetime
from src.agent import Agent
from src.agent.futures_agent import FuturesAgent
from src.backtest.backtester import Backtester

# 导入市场分析器
from src.futures.market import MarketAnalyzer
from src.futures.market.market_analyzer_demo import generate_mock_price_data
from src.utils.binance_data_provider import BinanceDataProvider
from datetime import timedelta
import pandas as pd

# 导入保证金管理器
from src.futures.margin.margin_manager import MarginManager, MarginConfig, WebSocketTradingInterface
from src.futures.interfaces.user_data_stream import BinanceFuturesUserDataStream, UserDataStreamConfig

# 导入期货信号系统
from src.futures.signals.futures_signal_system import FuturesSignalSystem, SignalConfidenceMetrics
from src.futures.signals.strategy_manager import FuturesStrategyManager
from src.futures.leverage.leverage_controller import LeverageController, LeverageStrategy
from src.futures.models.data_models import TradingDirection, OperationType, RiskLevel, FuturesSignal

# 导入期货配置
from src.futures.constants import RiskParameters
from src.futures.config.futures_config import FuturesConfig, TradingMode

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_margin_manager_config() -> MarginConfig:
    """创建保证金管理器配置

    从config.yaml和环境变量中读取配置参数
    """
    # 从settings或环境变量获取配置
    initial_cash = getattr(settings, 'initial_cash', 1000.0)

    # 从futures配置中获取风险参数（如果有的话）
    futures_config = getattr(settings, 'futures', {})
    risk_config = futures_config.get('risk', {}) if isinstance(futures_config, dict) else {}

    # 支持环境变量覆盖配置
    config = MarginConfig(
        initial_cash=float(os.getenv('MARGIN_INITIAL_CASH', initial_cash)),
        initial_margin_rate=float(os.getenv('MARGIN_INITIAL_RATE',
                                          risk_config.get('margin_requirement', 0.1))),
        maintenance_margin_rate=float(os.getenv('MARGIN_MAINTENANCE_RATE', 0.05)),
        margin_call_threshold=float(os.getenv('MARGIN_CALL_THRESHOLD', 0.3)),
        liquidation_threshold=float(os.getenv('MARGIN_LIQUIDATION_THRESHOLD', 0.8)),
        emergency_threshold=float(os.getenv('MARGIN_EMERGENCY_THRESHOLD', 0.9)),
        max_leverage=int(os.getenv('MARGIN_MAX_LEVERAGE',
                                 risk_config.get('max_leverage', 20))),
        currency=os.getenv('MARGIN_CURRENCY', 'USDT')
    )

    logger.info(f"保证金管理器配置创建完成: 初始资金={config.initial_cash} {config.currency}")
    logger.info(f"环境设置: testnet模式，风险参数已优化")
    return config


def create_margin_manager() -> Optional[MarginManager]:
    """创建保证金管理器实例

    Returns:
        保证金管理器实例，如果创建失败返回None
    """
    try:
        # 创建配置
        config = create_margin_manager_config()

        # 创建WebSocket接口（Mock实现，用于开发和测试）
        ws_interface = WebSocketTradingInterface()

        # 创建保证金管理器
        margin_manager = MarginManager(config=config, ws_interface=ws_interface)

        logger.info("✅ 保证金管理器创建成功")
        return margin_manager

    except Exception as e:
        logger.error(f"❌ 保证金管理器创建失败: {e}")
        return None


def create_futures_signal_system(
    margin_manager: Optional[MarginManager] = None,
    market_analyzer: Optional[MarketAnalyzer] = None
) -> Optional[FuturesSignalSystem]:
    """创建期货信号系统实例

    Args:
        margin_manager: 保证金管理器
        market_analyzer: 市场分析器

    Returns:
        期货信号系统实例，如果创建失败返回None
    """
    try:
        logger.info("🔧 开始创建期货信号系统...")

        # 创建策略管理器
        strategy_manager = FuturesStrategyManager()

        # 添加策略（使用正确的期货策略名称）
        strategy_manager.add_strategy('futures_macd_strategy', weight=0.6)
        strategy_manager.add_strategy('futures_rsi_strategy', weight=0.4)

        logger.info(f"✅ 策略管理器创建完成，活跃策略数：{len(strategy_manager.get_active_strategies())}")

        # 创建杠杆控制器
        leverage_controller = LeverageController()

        # 风险参数配置
        risk_params = {
            'max_position_ratio': RiskParameters.MAX_POSITION_RATIO,
            'min_position_size': 10.0,
            'max_position_size': 10000.0
        }

        # 创建期货信号系统
        signal_system = FuturesSignalSystem(
            strategy_manager=strategy_manager,
            leverage_controller=leverage_controller,
            market_analyzer=market_analyzer,
            risk_params=risk_params
        )

        # 初始化系统
        if signal_system.initialize():
            logger.info("✅ 期货信号系统创建和初始化成功")
            return signal_system
        else:
            logger.error("❌ 期货信号系统初始化失败")
            return None

    except Exception as e:
        logger.error(f"❌ 期货信号系统创建失败: {e}")
        return None


async def display_margin_status(margin_manager: MarginManager):
    """显示保证金状态信息

    Args:
        margin_manager: 保证金管理器实例
    """
    try:
        # 获取状态摘要
        status = margin_manager.get_status_summary()

        # 获取风险等级（异步）
        risk_level = await margin_manager.assess_risk_level()

        print("\n" + "="*50)
        print("📊 保证金管理器状态")
        print("="*50)
        print(f"💰 总余额: {status['total_balance']:.2f} {status['currency']}")
        print(f"💵 可用保证金: {status['available_margin']:.2f} {status['currency']}")
        print(f"🔒 已用保证金: {status['used_margin']:.2f} {status['currency']}")
        print(f"📈 保证金比例: {status['margin_ratio']:.3f}")
        print(f"⚠️  风险等级: {risk_level.value}")
        print(f"🔍 监控状态: {'启用' if status['is_monitoring'] else '禁用'}")
        print(f"🕐 更新时间: {status['last_update']}")
        print("="*50)

    except Exception as e:
        logger.error(f"❌ 显示保证金状态失败: {e}")


async def test_margin_calculations(margin_manager: MarginManager):
    """测试保证金计算功能

    Args:
        margin_manager: 保证金管理器实例
    """
    try:
        logger.info("🧪 开始保证金计算验证...")

        # 模拟仓位数据
        test_position_size = 0.1  # BTC
        test_entry_price = 50000.0  # USDT
        test_current_price = 52000.0  # USDT
        test_leverage = 10  # 10倍杠杆

        # 计算仓位价值
        position_value = test_position_size * test_entry_price
        current_position_value = test_position_size * test_current_price

        # 测试基础保证金计算
        initial_margin = margin_manager.calculate_initial_margin(
            position_value=position_value,
            leverage=test_leverage
        )

        maintenance_margin = margin_manager.calculate_maintenance_margin(
            position_value=current_position_value,
            leverage=test_leverage
        )

        print("\n🧮 保证金计算测试结果:")
        print(f"   仓位大小: {test_position_size} BTC")
        print(f"   入场价格: {test_entry_price:.2f} USDT")
        print(f"   当前价格: {test_current_price:.2f} USDT")
        print(f"   杠杆倍数: {test_leverage}x")
        print(f"   仓位价值: {position_value:.2f} USDT")
        print(f"   初始保证金: {initial_margin:.2f} USDT")
        print(f"   维持保证金: {maintenance_margin:.2f} USDT")

        # 测试强平价格计算
        try:
            liquidation_price = margin_manager.calculate_liquidation_price(
                entry_price=test_entry_price,
                position_size=test_position_size,
                margin=initial_margin,
                side='long'
            )
            print(f"   强平价格: {liquidation_price:.2f} USDT")
        except Exception as e:
            print(f"   强平价格计算: ⚠️  {str(e)}")

        # 测试保证金充足性检查
        try:
            sufficiency_result = margin_manager.check_margin_sufficiency(
                required_margin=initial_margin
            )

            print(f"   保证金充足性: {'✅ 充足' if sufficiency_result.can_open_position else '❌ 不足'}")
            print(f"   所需保证金: {sufficiency_result.initial_margin_requirement:.2f} USDT")
            print(f"   可用保证金: {sufficiency_result.available_margin:.2f} USDT")
            print(f"   账户状态: {sufficiency_result.account_status}")
        except Exception as e:
            print(f"   保证金充足性检查: ⚠️  {str(e)}")

        logger.info("✅ 保证金计算验证完成")

    except Exception as e:
        logger.error(f"❌ 保证金计算验证失败: {e}")


async def run_margin_manager_validation():
    """运行保证金管理器功能验证"""
    logger.info("🔍 开始保证金管理器集成验证...")

    try:
        # 创建保证金管理器
        margin_manager = create_margin_manager()
        if not margin_manager:
            logger.error("❌ 保证金管理器创建失败，跳过验证")
            return False

        # 初始化保证金管理器
        await margin_manager.initialize()
        logger.info("✅ 保证金管理器初始化完成")

        # 显示状态
        await display_margin_status(margin_manager)

        # 测试计算功能
        await test_margin_calculations(margin_manager)

        # 清理资源
        await margin_manager.cleanup()

        logger.info("🎉 保证金管理器集成验证成功")
        return True

    except Exception as e:
        logger.error(f"💥 保证金管理器验证失败: {e}")
        return False


async def test_futures_signal_system(
    signal_system: FuturesSignalSystem,
    margin_manager: Optional[MarginManager] = None,
    market_analyzer: Optional[MarketAnalyzer] = None,
    use_real_data: bool = True
) -> bool:
    """测试期货信号系统功能

    Args:
        signal_system: 期货信号系统
        margin_manager: 保证金管理器
        market_analyzer: 市场分析器
        use_real_data: 是否使用真实数据

    Returns:
        测试是否成功
    """
    try:
        logger.info("🧪 开始期货信号系统功能测试...")

        # 测试参数 - 使用配置文件中的第一个ticker
        from src.utils import settings
        test_ticker = settings.signals.tickers[0] if settings.signals.tickers else "BTCUSDT"
        test_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']  # 匹配期货系统期望的时间框架

        # 获取市场数据
        if use_real_data:
            logger.info("🌐 使用真实市场数据进行测试")
            # 使用真实数据（不同时间框架）
            market_data = {}
            failed_timeframes = []

            for timeframe in test_timeframes:
                try:
                    logger.info(f"正在获取 {test_ticker} {timeframe} 时间框架数据...")
                    # 根据时间框架调整数据天数以避免超时
                    days_map = {'1m': 3, '5m': 7, '15m': 15, '30m': 30, '1h': 50, '4h': 100, '1d': 250}
                    days = days_map.get(timeframe, 50)
                    data = get_real_market_data(test_ticker, days=days, timeframe=timeframe, use_cache=False)
                    if data.empty:
                        raise ValueError(f"{timeframe} 数据为空")
                    market_data[timeframe] = data
                    logger.info(f"✅ 成功获取 {timeframe} 数据，共 {len(data)} 条记录")
                except Exception as e:
                    logger.warning(f"❌ 获取{timeframe}数据失败: {e}")
                    failed_timeframes.append(timeframe)

            # 如果有任何时间框架数据获取失败，为所有时间框架生成模拟数据确保一致性
            if failed_timeframes:
                logger.warning(f"⚠️ 以下时间框架数据获取失败: {failed_timeframes}")
                logger.info("🔄 为确保数据一致性，将为所有时间框架生成模拟数据")
                market_data = {}
                for timeframe in test_timeframes:
                    market_data[timeframe] = generate_mock_price_data(days=100, start_price=50000.0)
                    logger.info(f"📊 已生成 {timeframe} 模拟数据")
            else:
                logger.info("✅ 所有时间框架的真实数据获取成功")
        else:
            logger.info("📊 使用模拟数据进行测试")
            # 使用模拟数据
            market_data = {}
            for timeframe in test_timeframes:
                market_data[timeframe] = generate_mock_price_data(days=100, start_price=50000.0)

        # 获取当前价格
        current_price = float(market_data['1h']['close'].iloc[-1])
        logger.info(f"💰 当前价格: {current_price:.2f} USDT")

        # 模拟保证金状态
        margin_status = None
        if margin_manager:
            await margin_manager.initialize()
            margin_status = margin_manager.get_margin_status()
            logger.info(f"📊 保证金状态: 可用={margin_status.available_margin:.2f} USDT")

        # 生成信号
        logger.info(f"🕰️ 开始生成期货交易信号: {test_ticker}")

        signal = await signal_system.generate_signal(
            ticker=test_ticker,
            market_data=market_data,
            current_price=current_price,
            current_positions=None,  # 无当前持仓
            margin_status=margin_status,
            leverage_strategy=LeverageStrategy.MODERATE
        )

        if signal:
            # 显示信号结果
            print("\n" + "="*60)
            print("📊 期货交易信号结果")
            print("="*60)
            print(f"📈 交易对: {signal.ticker}")
            print(f"🧭 交易方向: {signal.direction.value}")
            print(f"⚙️ 操作类型: {signal.operation_type.value}")
            print(f"🔎 置信度: {signal.confidence:.1f}%")
            print(f"💪 信号强度: {signal.strength:.3f}")
            print(f"🎯 建议杠杆: {signal.suggested_leverage:.1f}x")
            print(f"💰 仓位大小: {signal.position_size:.2f} USDT")
            print(f"🟢 入场价格: {signal.entry_price:.2f}")

            if signal.take_profit_price:
                print(f"🎆 止盈价格: {signal.take_profit_price:.2f}")
            if signal.stop_loss_price:
                print(f"🛑 止损价格: {signal.stop_loss_price:.2f}")

            print(f"⚠️ 风险等级: {signal.risk_level.value}")
            print(f"⏰ 信号有效期: {signal.expiry_time.strftime('%H:%M:%S')}")

            # 显示元数据
            if signal.metadata:
                print("\n📊 信号元数据:")
                metadata = signal.metadata
                print(f"   时间框架信号数: {metadata.get('timeframe_signals', 0)}")
                print(f"   预期收益率: {metadata.get('expected_return', 0):.2f}%")
                print(f"   预期持仓时间: {metadata.get('expected_duration_hours', 0):.1f}小时")
                print(f"   风险评分: {metadata.get('risk_score', 0):.1f}/100")
                print(f"   处理时间: {metadata.get('processing_time_ms', 0):.1f}ms")

            print("="*60)

            # 验证信号有效性
            if signal.is_valid():
                logger.info("✅ 信号验证通过")

                # 检查是否正确使用期货语义
                if signal.direction in [TradingDirection.LONG, TradingDirection.SHORT, TradingDirection.NEUTRAL]:
                    logger.info("✅ 正确使用期货交易语义 (long/short/neutral)")
                else:
                    logger.warning(f"⚠️ 信号使用了错误的方向: {signal.direction}")

                # 检查杠杆限制
                if 1 <= signal.suggested_leverage <= 25:
                    logger.info(f"✅ 杠杆在合理范围内: {signal.suggested_leverage:.1f}x")
                else:
                    logger.warning(f"⚠️ 杠杆超出安全范围: {signal.suggested_leverage:.1f}x")

                return True
            else:
                logger.error("❌ 信号验证失败")
                return False
        else:
            logger.error(f"❌ 未能生成有效信号: {test_ticker}")
            return False

    except Exception as e:
        logger.error(f"❌ 期货信号系统测试失败: {e}")
        return False
    finally:
        # 清理资源
        if margin_manager:
            try:
                await margin_manager.cleanup()
            except Exception as e:
                logger.warning(f"清理保证金管理器资源失败: {e}")


async def run_futures_signal_validation(
    use_real_data: bool = True
) -> bool:
    """运行期货信号系统集成验证

    Args:
        use_real_data: 是否使用真实数据

    Returns:
        验证是否成功
    """
    try:
        logger.info("🔍 开始期货信号系统集成验证...")

        # 创建市场分析器
        market_analyzer = None
        try:
            analyzer_config = {
                "cache_timeout": 300,
                "enable_async": True,
                "volatility_periods": [21, 63, 252],
                "technical_periods": {
                    "rsi": [14, 28],
                    "ema": [9, 21, 50, 200],
                    "bb": 20,
                    "atr": 14,
                    "adx": 14
                }
            }
            market_analyzer = MarketAnalyzer(analyzer_config)
            logger.info("✅ 市场分析器创建成功")
        except Exception as e:
            logger.warning(f"市场分析器创建失败，将继续使用基础功能: {e}")

        # 创建保证金管理器
        margin_manager = create_margin_manager()

        # 创建期货信号系统
        signal_system = create_futures_signal_system(
            margin_manager=margin_manager,
            market_analyzer=market_analyzer
        )

        if not signal_system:
            logger.error("❌ 期货信号系统创建失败")
            return False

        # 运行测试
        success = await test_futures_signal_system(
            signal_system=signal_system,
            margin_manager=margin_manager,
            market_analyzer=market_analyzer,
            use_real_data=use_real_data
        )

        if success:
            logger.info("🎉 期货信号系统集成验证成功")
        else:
            logger.error("💥 期货信号系统集成验证失败")

        return success

    except Exception as e:
        logger.error(f"💥 期货信号系统验证过程出错: {e}")
        return False


def get_real_market_data(symbol: str = "BTCUSDT", days: int = 250, timeframe: str = "1h", use_cache: bool = False) -> pd.DataFrame:
    """
    从Binance获取真实市场数据

    Args:
        symbol: 交易对符号
        days: 获取的天数
        timeframe: 时间框架 (1m, 5m, 15m, 30m, 1h, 4h, 1d等)
        use_cache: 是否使用缓存数据

    Returns:
        包含OHLCV数据的DataFrame
    """
    try:
        # 创建数据提供者实例（公共API不需要密钥）
        provider = BinanceDataProvider()

        # 计算开始和结束时间
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"正在从Binance获取 {symbol} 的真实数据 ({timeframe})...")
        logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")

        # 获取历史数据（使用指定的时间框架）
        df = provider.get_historical_klines(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache  # 根据参数决定是否使用缓存
        )
        
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 的数据")
        
        # 转换为市场分析器期望的格式
        # 创建时间索引
        df = df.set_index('open_time')
        
        # 选择需要的列并创建DataFrame
        market_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # 确保所有数据都是数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            market_data[col] = pd.to_numeric(market_data[col], errors='coerce')
        
        # 删除任何包含NaN的行
        market_data = market_data.dropna()
        
        logger.info(f"✅ 成功获取 {len(market_data)} 条真实数据记录")
        logger.info(f"数据时间范围: {market_data.index[0]} 到 {market_data.index[-1]}")
        
        return market_data
        
    except Exception as e:
        logger.error(f"❌ 获取真实数据失败: {e}")
        logger.warning("⚠️  将使用模拟数据作为备选方案")
        # 如果获取真实数据失败，使用模拟数据
        return generate_mock_price_data(days=days, start_price=50000.0)


async def test_market_analyzer(use_real_data: bool = True):
    """测试市场分析器功能"""
    data_type = "真实数据" if use_real_data else "模拟数据"
    logger.info(f"=== 市场分析器功能验证 (使用{data_type}) ===")
    
    try:
        # 创建市场分析器实例
        analyzer_config = {
            "cache_timeout": 300,
            "enable_async": True,
            "volatility_periods": [21, 63, 252],
            "technical_periods": {
                "rsi": [14, 28],
                "ema": [9, 21, 50, 200],
                "bb": 20,
                "atr": 14,
                "adx": 14
            }
        }
        
        analyzer = MarketAnalyzer(analyzer_config)
        logger.info("✅ 市场分析器初始化成功")
        
        # 获取市场数据 (使用250天确保有足够数据进行技术分析)
        from src.utils import settings
        test_symbol = settings.signals.tickers[0] if settings.signals.tickers else "BTCUSDT"

        if use_real_data:
            test_data = get_real_market_data(symbol=test_symbol, days=250, timeframe="1h")
        else:
            test_data = generate_mock_price_data(days=250, start_price=50000.0)
        logger.info("✅ 市场数据获取成功")
        
        # 测试核心功能
        logger.info("🔍 测试波动率计算...")
        vol_metrics = await analyzer.calculate_volatility_metrics(test_symbol, test_data)
        logger.info(f"   历史波动率: {vol_metrics.historical_volatility:.3f}")
        logger.info(f"   波动率状态: {vol_metrics.volatility_regime.value}")

        logger.info("🔍 测试技术指标计算...")
        tech_indicators = await analyzer.calculate_technical_indicators(test_symbol, test_data)
        trend_signal, confidence = tech_indicators.get_trend_signal()
        logger.info(f"   技术信号: {trend_signal.value}, 置信度: {confidence:.2f}")
        logger.info(f"   RSI(14): {tech_indicators.rsi_14:.1f}")

        logger.info("🔍 测试市场条件分析...")
        market_conditions = await analyzer.analyze_market_conditions(test_symbol, test_data)
        logger.info(f"   主要状态: {market_conditions.primary_condition.value}")
        logger.info(f"   趋势强度: {market_conditions.trend_strength:.2f}")

        logger.info("🔍 测试风险指标计算...")
        risk_metrics = await analyzer.calculate_risk_metrics(test_symbol, test_data)
        logger.info(f"   风险等级: {risk_metrics.risk_level.value}")
        logger.info(f"   最大回撤: {risk_metrics.max_drawdown:.3f}")

        logger.info("🔍 测试综合分析...")
        comprehensive_analysis = await analyzer.analyze_market_comprehensive(
            ticker=test_symbol,
            price_data=test_data
        )
        logger.info(f"   分析置信度: {comprehensive_analysis['analysis_confidence']:.2f}")
        logger.info(f"   数据质量: {comprehensive_analysis['data_quality']:.2f}")
        
        # 测试交易建议
        recommendation = comprehensive_analysis.get('trading_recommendation', {})
        if recommendation:
            logger.info(f"   交易建议: {recommendation['action']}")
            logger.info(f"   建议置信度: {recommendation['confidence']:.2f}")
            logger.info(f"   最大仓位比例: {recommendation['max_position_ratio']:.3f}")
        
        logger.info("✅ 所有市场分析器功能测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 市场分析器测试失败: {e}")
        return False


def run_market_analyzer_validation(use_real_data: bool = True):
    """运行市场分析器验证"""
    try:
        result = asyncio.run(test_market_analyzer(use_real_data=use_real_data))
        if result:
            logger.info("🎉 市场分析器集成验证成功")
        else:
            logger.error("💥 市场分析器集成验证失败")
        return result
    except Exception as e:
        logger.error(f"💥 市场分析器验证过程出错: {e}")
        return False


if __name__ == "__main__":
    
    # 检查命令行参数
    use_real_data = True  # 默认使用真实数据

    if len(sys.argv) > 1:
        # 检查是否有保证金管理器测试参数
        if "--test-margin-manager" in sys.argv:
            logger.info("🧪 运行保证金管理器独立测试")
            success = asyncio.run(run_margin_manager_validation())
            sys.exit(0 if success else 1)

        # 检查是否有市场分析器测试参数
        if "--test-market-analyzer" in sys.argv:
            logger.info("🧪 运行市场分析器独立测试")

            # 检查是否指定使用模拟数据
            if "--mock-data" in sys.argv:
                use_real_data = False
                logger.info("📊 使用模拟数据进行测试")
            else:
                logger.info("🌐 使用真实数据进行测试")

            success = run_market_analyzer_validation(use_real_data=use_real_data)
            sys.exit(0 if success else 1)

        # 检查是否有期货信号系统测试参数
        if "--test-futures-signals" in sys.argv:
            logger.info("🧪 运行期货信号系统独立测试")

            # 检查是否指定使用模拟数据
            if "--mock-data" in sys.argv:
                use_real_data = False
                logger.info("📊 使用模拟数据进行测试")
            else:
                logger.info("🌐 使用真实数据进行测试")

            success = asyncio.run(run_futures_signal_validation(use_real_data=use_real_data))
            sys.exit(0 if success else 1)

        # 检查是否指定使用模拟数据
        if "--mock-data" in sys.argv:
            use_real_data = False
    
    # 在主要流程开始前，运行验证测试
    data_type = "真实数据" if use_real_data else "模拟数据"

    # 验证市场分析器
    logger.info(f"🔍 验证市场分析器集成... (使用{data_type})")
    analyzer_validation_success = run_market_analyzer_validation(use_real_data=use_real_data)

    if not analyzer_validation_success:
        logger.warning("⚠️  市场分析器验证失败，但继续执行主要流程...")

    # 验证期货信号系统
    logger.info(f"🔍 验证期货信号系统集成... (使用{data_type})")
    futures_validation_success = asyncio.run(run_futures_signal_validation(use_real_data=use_real_data))

    if not futures_validation_success:
        logger.warning("⚠️  期货信号系统验证失败，但继续执行主要流程...")
    else:
        logger.info("✅ 期货信号系统验证成功！")
    
    if settings.mode == "backtest":
        backtester = Backtester(
            primary_interval=settings.primary_interval,
            intervals=settings.signals.intervals,
            tickers=settings.signals.tickers,
            start_date=settings.start_date,
            end_date=settings.end_date,
            initial_capital=settings.initial_cash,
            strategies=settings.signals.strategies,
            show_agent_graph=settings.show_agent_graph,
            show_reasoning=settings.show_reasoning,
            model_name=settings.model.name,
            model_provider=settings.model.provider,
            model_base_url=settings.model.base_url,
        )
        print("Starting backtest...")
        performance_metrics = backtester.run_backtest()
        performance_df = backtester.analyze_performance()

    else:
        # 实时交易模式 - 集成期货交易系统
        logger.info("🚀 启动实时期货交易模式...")

        # 首先验证保证金管理器功能
        logger.info("🔍 验证保证金管理器集成...")
        margin_validation_success = asyncio.run(run_margin_manager_validation())

        if not margin_validation_success:
            logger.warning("⚠️  保证金管理器验证失败，但继续执行主要流程...")

        # 创建市场分析器
        market_analyzer = None
        try:
            analyzer_config = {
                "cache_timeout": 300,
                "enable_async": True,
                "volatility_periods": [21, 63, 252],
                "technical_periods": {
                    "rsi": [14, 28],
                    "ema": [9, 21, 50, 200],
                    "bb": 20,
                    "atr": 14,
                    "adx": 14
                }
            }
            market_analyzer = MarketAnalyzer(analyzer_config)
            logger.info("✅ 市场分析器初始化成功")
        except Exception as e:
            logger.warning(f"市场分析器初始化失败: {e}")

        # 创建保证金管理器实例
        margin_manager = create_margin_manager()

        # 创建期货信号系统
        futures_signal_system = create_futures_signal_system(
            margin_manager=margin_manager,
            market_analyzer=market_analyzer
        )

        if not futures_signal_system:
            logger.error("❌ 期货信号系统创建失败，退出")
            sys.exit(1)

        logger.info("✅ 期货信号系统创建成功")

        portfolio = {
            "cash": settings.initial_cash,  # Initial cash amount
            "margin_requirement": settings.margin_requirement,  # Initial margin requirement
            "margin_used": 0.0,  # total margin usage across all short positions
            "positions": {
                ticker: {
                    "long": 0.0,  # Number of shares held long
                    "short": 0.0,  # Number of shares held short
                    "long_cost_basis": 0.0,  # Average cost basis for long positions
                    "short_cost_basis": 0.0,  # Average price at which shares were sold short
                    "short_margin_used": 0.0,  # Dollars of margin used for this ticker's short
                }
                for ticker in settings.signals.tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,  # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                }
                for ticker in settings.signals.tickers
            },
            "margin_manager": margin_manager,  # 添加保证金管理器到portfolio
            "futures_signal_system": futures_signal_system,  # 添加期货信号系统
            "market_analyzer": market_analyzer,  # 添加市场分析器
        }

        # 如果保证金管理器创建成功，初始化并显示状态
        if margin_manager:
            try:
                asyncio.run(margin_manager.initialize())
                asyncio.run(display_margin_status(margin_manager))
                logger.info("✅ 保证金管理器已集成到交易系统")
            except Exception as e:
                logger.error(f"❌ 保证金管理器初始化失败: {e}")

        # 使用期货代理替代传统Agent
        try:
            agent = FuturesAgent(
                intervals=settings.signals.intervals,
                show_agent_graph=settings.show_agent_graph,
                risk_level="moderate",  # 可以从配置中读取
                futures_signal_system=futures_signal_system,
                margin_manager=margin_manager,
                market_analyzer=market_analyzer
            )
            logger.info("✅ 期货代理创建成功")

            # 初始化期货组件
            initialization_success = asyncio.run(agent.initialize_futures_components())
            if not initialization_success:
                logger.warning("⚠️ 期货组件初始化部分失败")

        except Exception as e:
            logger.error(f"❌ 期货代理创建失败，回退到传统Agent: {e}")
            # 回退到传统Agent
            agent = Agent(
                intervals=settings.signals.intervals,
                strategies=settings.signals.strategies,
                show_agent_graph=settings.show_agent_graph,
            )

        try:
            result = agent.run(
                primary_interval=settings.primary_interval,
                tickers=settings.signals.tickers,
                end_date=datetime.now(),
                portfolio=portfolio,
                show_reasoning=settings.show_reasoning,
                model_name=settings.model.name,
                model_provider=settings.model.provider,
                model_base_url=settings.model.base_url
            )

            # 显示交易决策结果
            print(result.get('decisions'))

        finally:
            # 清理期货代理和保证金管理器资源
            try:
                # 如果是期货代理，清理期货组件
                if isinstance(agent, FuturesAgent):
                    asyncio.run(agent.cleanup_futures_components())
                    logger.info("✅ 期货代理资源清理完成")

                # 清理保证金管理器资源
                if margin_manager:
                    asyncio.run(margin_manager.cleanup())
                    logger.info("✅ 保证金管理器资源清理完成")

            except Exception as e:
                logger.error(f"❌ 资源清理失败: {e}")
