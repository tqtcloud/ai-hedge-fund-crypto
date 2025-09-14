"""
市场分析器演示脚本

展示如何使用MarketAnalyzer进行市场分析，包括模拟数据生成和分析结果展示。
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import json

from .market_analyzer import MarketAnalyzer, MarketCondition, VolatilityRegime, LiquidityLevel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_mock_price_data(
    days: int = 252,
    start_price: float = 50000.0,
    volatility: float = 0.02,
    trend: float = 0.0001
) -> pd.DataFrame:
    """
    生成模拟价格数据
    
    Args:
        days: 数据天数
        start_price: 起始价格
        volatility: 日波动率
        trend: 趋势系数
        
    Returns:
        pd.DataFrame: 包含OHLCV数据的DataFrame
    """
    np.random.seed(42)  # 确保可重现
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # 生成价格路径
    returns = np.random.normal(trend, volatility, days)
    prices = [start_price]
    
    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    # 生成OHLC数据
    data = []
    for i in range(days):
        # 简单模拟日内波动
        daily_vol = volatility * 0.5
        high = prices[i] * (1 + abs(np.random.normal(0, daily_vol)))
        low = prices[i] * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i] * (1 + np.random.normal(0, daily_vol * 0.3))
        close_price = prices[i]
        volume = np.random.normal(1000000, 200000)
        
        data.append({
            'date': dates[i],
            'open': max(low, open_price),
            'high': max(high, open_price, close_price),
            'low': min(low, open_price, close_price),
            'close': close_price,
            'volume': max(100000, volume)
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df


async def demo_market_analyzer():
    """演示市场分析器功能"""
    
    logger.info("=== 市场分析器演示开始 ===")
    
    # 1. 创建市场分析器实例
    config = {
        "cache_timeout": 300,
        "enable_async": True,
        "volatility_periods": [21, 63, 252],
        "technical_periods": {
            "rsi": [14, 28],
            "ema": [9, 21, 50, 200],
            "bb": 20,
            "atr": 14,
            "adx": 14
        },
        "var_confidence_levels": [0.95, 0.99],
        "risk_periods": [1, 5, 21]
    }
    
    analyzer = MarketAnalyzer(config)
    logger.info("市场分析器初始化完成")
    
    # 2. 生成模拟数据
    logger.info("生成模拟价格数据...")
    
    # 场景1：正常市场条件
    normal_data = generate_mock_price_data(days=300, volatility=0.02, trend=0.0005)
    
    # 场景2：高波动市场
    volatile_data = generate_mock_price_data(days=300, volatility=0.05, trend=-0.001)
    
    # 场景3：趋势市场
    trending_data = generate_mock_price_data(days=300, volatility=0.015, trend=0.002)
    
    scenarios = {
        "正常市场": ("BTCUSDT", normal_data),
        "高波动市场": ("ETHUSDT", volatile_data),
        "趋势市场": ("BNBUSDT", trending_data)
    }
    
    # 3. 执行各场景分析
    for scenario_name, (ticker, price_data) in scenarios.items():
        logger.info(f"\n--- {scenario_name} 分析 ({ticker}) ---")
        
        try:
            # 执行全面分析
            analysis_result = await analyzer.analyze_market_comprehensive(
                ticker=ticker,
                price_data=price_data,
                timeframes=["1h", "4h", "1d"]
            )
            
            # 展示分析结果
            await display_analysis_results(scenario_name, analysis_result)
            
        except Exception as e:
            logger.error(f"{scenario_name} 分析失败: {e}")
    
    # 4. 演示单独功能模块
    logger.info("\n=== 单独功能模块演示 ===")
    await demo_individual_features(analyzer, normal_data)
    
    logger.info("\n=== 市场分析器演示完成 ===")


async def display_analysis_results(scenario_name: str, analysis: Dict[str, Any]):
    """展示分析结果"""
    
    logger.info(f"📊 {scenario_name} 综合分析结果:")
    logger.info(f"分析时间: {analysis['analysis_timestamp']}")
    logger.info(f"整体置信度: {analysis['analysis_confidence']:.2f}")
    logger.info(f"数据质量评分: {analysis['data_quality']:.2f}")
    
    # 波动率指标
    if analysis['volatility_metrics']:
        vol_metrics = analysis['volatility_metrics']
        logger.info(f"\n📈 波动率分析:")
        logger.info(f"  历史波动率: {vol_metrics['historical_volatility']:.3f}")
        logger.info(f"  波动率状态: {vol_metrics['volatility_regime']}")
        logger.info(f"  波动率百分位: {vol_metrics['volatility_percentile']:.2f}")
        logger.info(f"  波动率Z分数: {vol_metrics['volatility_z_score']:.2f}")
    
    # 技术指标
    if analysis['technical_indicators']:
        tech_indicators = analysis['technical_indicators']
        logger.info(f"\n🔧 技术指标:")
        logger.info(f"  RSI(14): {tech_indicators['rsi_14']:.1f}")
        logger.info(f"  ADX: {tech_indicators['adx']:.1f}")
        logger.info(f"  布林带位置: {tech_indicators['bb_percent']:.2f}")
        logger.info(f"  价格位置: {tech_indicators['price_position']:.2f}")
    
    # 市场条件
    if analysis['market_conditions']:
        market_cond = analysis['market_conditions']
        logger.info(f"\n🌍 市场条件:")
        logger.info(f"  主要状态: {market_cond['primary_condition']}")
        logger.info(f"  趋势强度: {market_cond['trend_strength']:.2f}")
        logger.info(f"  趋势方向: {market_cond['trend_direction']}")
        logger.info(f"  流动性水平: {market_cond['liquidity_level']}")
        logger.info(f"  市场情绪: {market_cond['market_sentiment']:.2f}")
        logger.info(f"  市场压力: {market_cond['market_stress_level']:.2f}")
    
    # 风险指标
    if analysis['risk_metrics']:
        risk_metrics = analysis['risk_metrics']
        logger.info(f"\n⚠️  风险指标:")
        logger.info(f"  1日95% VaR: {risk_metrics['var_1d_95']:.4f}")
        logger.info(f"  最大回撤: {risk_metrics['max_drawdown']:.3f}")
        logger.info(f"  当前回撤: {risk_metrics['current_drawdown']:.3f}")
        logger.info(f"  风险等级: {risk_metrics['risk_level']}")
        logger.info(f"  综合风险评分: {risk_metrics['overall_risk_score']:.1f}")
    
    # 交易建议
    if analysis['trading_recommendation']:
        recommendation = analysis['trading_recommendation']
        logger.info(f"\n💡 交易建议:")
        logger.info(f"  建议操作: {recommendation['action']}")
        logger.info(f"  置信度: {recommendation['confidence']:.2f}")
        logger.info(f"  风险调整系数: {recommendation['risk_adjustment']:.2f}")
        logger.info(f"  最大仓位比例: {recommendation['max_position_ratio']:.3f}")
        if recommendation['reasons']:
            logger.info(f"  理由: {'; '.join(recommendation['reasons'])}")


async def demo_individual_features(analyzer: MarketAnalyzer, price_data: pd.DataFrame):
    """演示单独功能模块"""
    
    ticker = "DEMO"
    
    # 1. 波动率分析
    logger.info("\n📈 波动率分析演示:")
    vol_metrics = await analyzer.calculate_volatility_metrics(ticker, price_data)
    logger.info(f"Parkinson波动率: {vol_metrics.parkinson_volatility:.3f}")
    logger.info(f"Garman-Klass波动率: {vol_metrics.garman_klass_volatility:.3f}")
    logger.info(f"Yang-Zhang波动率: {vol_metrics.yang_zhang_volatility:.3f}")
    
    # 2. 技术指标计算
    logger.info("\n🔧 技术指标计算演示:")
    tech_indicators = await analyzer.calculate_technical_indicators(ticker, price_data)
    trend_signal, confidence = tech_indicators.get_trend_signal()
    logger.info(f"技术指标趋势信号: {trend_signal.value}, 置信度: {confidence:.2f}")
    logger.info(f"EMA排列: 9({tech_indicators.ema_9:.1f}) vs 21({tech_indicators.ema_21:.1f}) vs 50({tech_indicators.ema_50:.1f})")
    
    # 3. 市场条件分析
    logger.info("\n🌍 市场条件分析演示:")
    market_conditions = await analyzer.analyze_market_conditions(ticker, price_data)
    logger.info(f"市场是否适合交易(低风险): {market_conditions.is_favorable_for_trading('low')}")
    logger.info(f"市场是否适合交易(中风险): {market_conditions.is_favorable_for_trading('medium')}")
    logger.info(f"市场是否适合交易(高风险): {market_conditions.is_favorable_for_trading('high')}")
    
    # 4. 风险指标计算
    logger.info("\n⚠️  风险指标计算演示:")
    risk_metrics = await analyzer.calculate_risk_metrics(ticker, price_data)
    logger.info(f"风险调整系数: {risk_metrics.get_risk_adjustment_factor():.2f}")
    logger.info(f"是否应该减少敞口: {risk_metrics.should_reduce_exposure()}")
    logger.info(f"收益分布偏度: {risk_metrics.skewness:.3f}")
    logger.info(f"收益分布峰度: {risk_metrics.kurtosis:.3f}")


def create_sample_config() -> Dict[str, Any]:
    """创建示例配置"""
    return {
        "market_analyzer": {
            "cache_timeout": 300,
            "enable_async": True,
            "volatility_periods": [21, 63, 252],
            "technical_periods": {
                "rsi": [14, 28],
                "ema": [9, 21, 50, 200],
                "bb": 20,
                "atr": 14,
                "adx": 14
            },
            "var_confidence_levels": [0.95, 0.99],
            "risk_periods": [1, 5, 21],
            "market_thresholds": {
                "trend_strength_min": 0.6,
                "volatility_regime_boundaries": [0.8, 1.2],
                "liquidity_thresholds": [0.2, 0.5, 1.5, 2.5],
                "sentiment_extremes": [0.2, 0.8]
            }
        }
    }


async def main():
    """主函数"""
    try:
        await demo_market_analyzer()
    except Exception as e:
        logger.error(f"演示过程中出错: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())