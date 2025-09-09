#!/usr/bin/env python3
"""
杠杆分析功能测试脚本
测试RiskManagementNode类的analyze_leverage方法
"""

import sys
import os
import pandas as pd
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from graph.risk_management_node import RiskManagementNode
from graph.state import AgentState
from utils import Interval


def create_mock_price_data() -> pd.DataFrame:
    """创建模拟价格数据"""
    import numpy as np
    
    # 生成100个交易日的模拟OHLC数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    base_price = 50000  # BTC基础价格
    
    # 生成随机价格走势
    np.random.seed(42)  # 确保结果可重现
    price_changes = np.random.normal(0, 0.02, 100)  # 2%的日波动率
    
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # 生成OHLC数据
    ohlc_data = []
    for i, close_price in enumerate(prices):
        high = close_price * (1 + abs(np.random.normal(0, 0.01)))
        low = close_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close_price
        
        ohlc_data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': np.random.uniform(100, 1000)
        })
    
    df = pd.DataFrame(ohlc_data, index=dates)
    return df


def create_mock_technical_analysis() -> Dict[str, Any]:
    """创建模拟技术分析数据"""
    return {
        "BTCUSDT": {
            "1h": {
                "signal": "HOLD",
                "confidence": 75,
                "atr_values": {
                    "atr_14": 1200.5,
                    "atr_28": 1350.2,
                    "atr_percentile": 65.0
                },
                "volatility_analysis": {
                    "volatility_percentile": 70.0,
                    "volatility_trend": "increasing",
                    "volatility_forecast": 0.025,
                    "regime_probability": 0.6
                }
            },
            "cross_timeframe_analysis": {
                "timeframe_consensus": 0.7,
                "dominant_timeframe": "1h",
                "conflict_areas": [],
                "trend_alignment": "bullish",
                "overall_signal_strength": "strong"
            }
        }
    }


def create_mock_portfolio(scenario: str = "normal") -> Dict[str, Any]:
    """创建不同场景的模拟投资组合数据"""
    portfolios = {
        "normal": {
            "cash": 10000.0,
            "cost_basis": {"BTCUSDT": 5000.0}
        },
        "high_cash": {
            "cash": 50000.0,
            "cost_basis": {"BTCUSDT": 10000.0}
        },
        "low_cash": {
            "cash": 1000.0,
            "cost_basis": {"BTCUSDT": 15000.0}
        },
        "no_position": {
            "cash": 20000.0,
            "cost_basis": {}
        }
    }
    return portfolios.get(scenario, portfolios["normal"])


def create_mock_state(portfolio_scenario: str = "normal") -> AgentState:
    """创建模拟的AgentState"""
    price_df = create_mock_price_data()
    technical_analysis = create_mock_technical_analysis()
    portfolio = create_mock_portfolio(portfolio_scenario)
    
    return AgentState({
        'data': {
            'tickers': ['BTCUSDT'],
            'primary_interval': Interval.HOUR_1,
            'portfolio': portfolio,
            'BTCUSDT_1h': price_df,
            'analyst_signals': {
                'technical_analyst_agent': technical_analysis
            }
        },
        'metadata': {
            'show_reasoning': False
        }
    })


def test_leverage_analysis_scenarios():
    """测试不同场景下的杠杆分析"""
    risk_node = RiskManagementNode()
    
    scenarios = ["normal", "high_cash", "low_cash", "no_position"]
    
    print("=" * 80)
    print("杠杆分析功能测试")
    print("=" * 80)
    
    for scenario in scenarios:
        print(f"\n【测试场景：{scenario}】")
        print("-" * 50)
        
        try:
            # 创建模拟状态
            state = create_mock_state(scenario)
            
            # 执行杠杆分析
            leverage_results = risk_node.analyze_leverage(state)
            
            # 显示结果
            for ticker, result in leverage_results.items():
                print(f"\n币种：{ticker}")
                print(f"当前价格：${result['current_price']:.2f}")
                
                leverage_analysis = result.get('leverage_analysis', {})
                print(f"推荐杠杆：{leverage_analysis.get('recommended_leverage', 'N/A')}x")
                print(f"最大安全杠杆：{leverage_analysis.get('max_safe_leverage', 'N/A')}x")
                print(f"杠杆选项：{leverage_analysis.get('leverage_options', 'N/A')}")
                print(f"杠杆风险评分：{leverage_analysis.get('leverage_risk_score', 'N/A')}")
                print(f"波动率调整杠杆：{leverage_analysis.get('volatility_adjusted_leverage', 'N/A')}x")
                
                reasoning = result.get('reasoning', {})
                print(f"\n分析基础：")
                print(f"  ATR (14期)：{reasoning.get('atr_14', 'N/A')}")
                print(f"  ATR百分位：{reasoning.get('atr_percentile', 'N/A')}%")
                print(f"  波动率百分位：{reasoning.get('volatility_percentile', 'N/A')}%")
                print(f"  波动率趋势：{reasoning.get('volatility_trend', 'N/A')}")
                print(f"  时间框架共识：{reasoning.get('timeframe_consensus', 'N/A')}")
                print(f"  投资组合价值：${reasoning.get('portfolio_value', 'N/A'):.2f}")
                print(f"  可用现金：${reasoning.get('available_cash', 'N/A'):.2f}")
                
        except Exception as e:
            print(f"测试场景 {scenario} 失败：{e}")
            import traceback
            traceback.print_exc()


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("边界情况测试")
    print("=" * 80)
    
    risk_node = RiskManagementNode()
    
    # 测试空数据情况
    print("\n【测试：空技术分析数据】")
    try:
        empty_state = AgentState({
            'data': {
                'tickers': ['BTCUSDT'],
                'primary_interval': Interval.HOUR_1,
                'portfolio': create_mock_portfolio(),
                'BTCUSDT_1h': create_mock_price_data(),
                'analyst_signals': {}  # 空的技术分析数据
            },
            'metadata': {'show_reasoning': False}
        })
        
        result = risk_node.analyze_leverage(empty_state)
        print(f"空数据测试通过，返回默认值：{result}")
        
    except Exception as e:
        print(f"空数据测试失败：{e}")
    
    # 测试极端波动率情况
    print("\n【测试：极端波动率情况】")
    try:
        extreme_volatility_analysis = create_mock_technical_analysis()
        # 设置极端高波动率
        extreme_volatility_analysis["BTCUSDT"]["1h"]["atr_values"]["atr_percentile"] = 95.0
        extreme_volatility_analysis["BTCUSDT"]["1h"]["volatility_analysis"]["volatility_percentile"] = 98.0
        extreme_volatility_analysis["BTCUSDT"]["1h"]["volatility_analysis"]["volatility_trend"] = "increasing"
        
        extreme_state = AgentState({
            'data': {
                'tickers': ['BTCUSDT'],
                'primary_interval': Interval.HOUR_1,
                'portfolio': create_mock_portfolio(),
                'BTCUSDT_1h': create_mock_price_data(),
                'analyst_signals': {'technical_analyst_agent': extreme_volatility_analysis}
            },
            'metadata': {'show_reasoning': False}
        })
        
        result = risk_node.analyze_leverage(extreme_state)
        leverage_data = result["BTCUSDT"]["leverage_analysis"]
        print(f"极端波动率测试通过:")
        print(f"  推荐杠杆：{leverage_data['recommended_leverage']}x（应该较低）")
        print(f"  风险评分：{leverage_data['leverage_risk_score']}（应该较高）")
        
    except Exception as e:
        print(f"极端波动率测试失败：{e}")


def validate_leverage_constraints():
    """验证杠杆约束条件"""
    print("\n" + "=" * 80)
    print("杠杆约束验证")
    print("=" * 80)
    
    risk_node = RiskManagementNode()
    scenarios = ["normal", "high_cash", "low_cash"]
    
    for scenario in scenarios:
        state = create_mock_state(scenario)
        result = risk_node.analyze_leverage(state)
        
        for ticker, data in result.items():
            leverage_analysis = data.get('leverage_analysis', {})
            
            recommended = leverage_analysis.get('recommended_leverage', 0)
            max_safe = leverage_analysis.get('max_safe_leverage', 0)
            volatility_adjusted = leverage_analysis.get('volatility_adjusted_leverage', 0)
            leverage_options = leverage_analysis.get('leverage_options', [])
            risk_score = leverage_analysis.get('leverage_risk_score', 0)
            
            print(f"\n【{scenario} - {ticker}】约束验证:")
            
            # 验证杠杆倍数范围 (1-125)
            constraints_passed = True
            
            if not (1 <= recommended <= 125):
                print(f"  ❌ 推荐杠杆超出范围：{recommended}")
                constraints_passed = False
            
            if not (1 <= max_safe <= 125):
                print(f"  ❌ 最大安全杠杆超出范围：{max_safe}")
                constraints_passed = False
                
            if not (1 <= volatility_adjusted <= 125):
                print(f"  ❌ 波动率调整杠杆超出范围：{volatility_adjusted}")
                constraints_passed = False
            
            # 验证杠杆选项数量和范围
            if len(leverage_options) != 3:
                print(f"  ❌ 杠杆选项数量错误：{len(leverage_options)} (应为3个)")
                constraints_passed = False
            
            for option in leverage_options:
                if not (1 <= option <= 125):
                    print(f"  ❌ 杠杆选项超出范围：{option}")
                    constraints_passed = False
            
            # 验证推荐杠杆不超过最大安全杠杆
            if recommended > max_safe:
                print(f"  ❌ 推荐杠杆超过最大安全杠杆：{recommended} > {max_safe}")
                constraints_passed = False
            
            # 验证风险评分范围 (0-1)
            if not (0 <= risk_score <= 1):
                print(f"  ❌ 风险评分超出范围：{risk_score}")
                constraints_passed = False
            
            if constraints_passed:
                print(f"  ✅ 所有约束验证通过")
                print(f"     推荐：{recommended}x, 最大：{max_safe}x, 选项：{leverage_options}")
                print(f"     风险评分：{risk_score}, 波动率调整：{volatility_adjusted}x")


if __name__ == "__main__":
    try:
        # 运行所有测试
        test_leverage_analysis_scenarios()
        test_edge_cases()
        validate_leverage_constraints()
        
        print("\n" + "=" * 80)
        print("✅ 杠杆分析功能测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)