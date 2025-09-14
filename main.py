import os
import sys
import asyncio
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from dotenv import load_dotenv
from src.utils import settings
from datetime import datetime
from src.agent import Agent
from src.backtest.backtester import Backtester

# 导入市场分析器
from src.futures.market import MarketAnalyzer
from src.futures.market.market_analyzer_demo import generate_mock_price_data
from src.utils.binance_data_provider import BinanceDataProvider
from datetime import timedelta
import pandas as pd

load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_real_market_data(symbol: str = "BTCUSDT", days: int = 250) -> pd.DataFrame:
    """
    从Binance获取真实市场数据
    
    Args:
        symbol: 交易对符号
        days: 获取的天数
        
    Returns:
        包含OHLCV数据的DataFrame
    """
    try:
        # 创建数据提供者实例（公共API不需要密钥）
        provider = BinanceDataProvider()
        
        # 计算开始和结束时间
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"正在从Binance获取 {symbol} 的真实数据...")
        logger.info(f"时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        
        # 获取历史数据（使用1小时时间框架）
        df = provider.get_historical_klines(
            symbol=symbol,
            timeframe='1h',
            start_date=start_date,
            end_date=end_date,
            use_cache=True  # 使用缓存以提高性能
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
        if use_real_data:
            test_data = get_real_market_data(symbol="BTCUSDT", days=250)
        else:
            test_data = generate_mock_price_data(days=250, start_price=50000.0)
        logger.info("✅ 市场数据获取成功")
        
        # 测试核心功能
        logger.info("🔍 测试波动率计算...")
        vol_metrics = await analyzer.calculate_volatility_metrics("BTCUSDT", test_data)
        logger.info(f"   历史波动率: {vol_metrics.historical_volatility:.3f}")
        logger.info(f"   波动率状态: {vol_metrics.volatility_regime.value}")
        
        logger.info("🔍 测试技术指标计算...")
        tech_indicators = await analyzer.calculate_technical_indicators("BTCUSDT", test_data)
        trend_signal, confidence = tech_indicators.get_trend_signal()
        logger.info(f"   技术信号: {trend_signal.value}, 置信度: {confidence:.2f}")
        logger.info(f"   RSI(14): {tech_indicators.rsi_14:.1f}")
        
        logger.info("🔍 测试市场条件分析...")
        market_conditions = await analyzer.analyze_market_conditions("BTCUSDT", test_data)
        logger.info(f"   主要状态: {market_conditions.primary_condition.value}")
        logger.info(f"   趋势强度: {market_conditions.trend_strength:.2f}")
        
        logger.info("🔍 测试风险指标计算...")
        risk_metrics = await analyzer.calculate_risk_metrics("BTCUSDT", test_data)
        logger.info(f"   风险等级: {risk_metrics.risk_level.value}")
        logger.info(f"   最大回撤: {risk_metrics.max_drawdown:.3f}")
        
        logger.info("🔍 测试综合分析...")
        comprehensive_analysis = await analyzer.analyze_market_comprehensive(
            ticker="BTCUSDT",
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
        
        # 检查是否指定使用模拟数据  
        if "--mock-data" in sys.argv:
            use_real_data = False
    
    # 在主要流程开始前，运行市场分析器验证
    data_type = "真实数据" if use_real_data else "模拟数据"
    logger.info(f"🔍 验证市场分析器集成... (使用{data_type})")
    analyzer_validation_success = run_market_analyzer_validation(use_real_data=use_real_data)
    
    if not analyzer_validation_success:
        logger.warning("⚠️  市场分析器验证失败，但继续执行主要流程...")
    
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
        }

        agent = Agent(
            intervals=settings.signals.intervals,
            strategies=settings.signals.strategies,
            show_agent_graph=settings.show_agent_graph,
        )

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
        # print(result)
        print(result.get('decisions'))
