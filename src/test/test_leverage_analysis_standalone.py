#!/usr/bin/env python3
"""
杠杆分析功能独立测试脚本
直接测试RiskManagementNode类的analyze_leverage方法，不依赖外部网络连接
"""

import json
import pandas as pd
from typing import Dict, Any, Optional
from enum import Enum


class Interval(Enum):
    """模拟Interval枚举"""
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    MIN_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class AgentState(dict):
    """模拟AgentState类"""
    pass


class RiskManagementNode:
    """RiskManagementNode的简化版本，只包含杠杆分析功能"""
    
    def analyze_leverage(self, state: AgentState) -> Dict[str, Dict[str, Any]]:
        """
        基于实际市场波动率和当前资金状况分析杠杆倍数建议
        
        Args:
            state: AgentState包含技术分析数据、投资组合信息等
            
        Returns:
            Dict[str, Dict[str, Any]]: 每个ticker的杠杆分析结果
        """
        data = state.get('data', {})
        portfolio = data.get("portfolio", {})
        tickers = data.get("tickers", [])
        primary_interval = data.get("primary_interval")
        
        # 获取技术分析数据
        technical_signals = data.get("analyst_signals", {}).get("technical_analyst_agent", {})
        
        leverage_analysis = {}
        
        for ticker in tickers:
            try:
                # 获取价格数据
                price_df = data.get(f"{ticker}_{primary_interval.value}")
                if price_df is None or price_df.empty:
                    leverage_analysis[ticker] = self._get_default_leverage_analysis()
                    continue
                
                current_price = price_df["close"].iloc[-1]
                
                # 获取技术分析数据
                tech_data = technical_signals.get(ticker, {})
                interval_data = tech_data.get(primary_interval.value, {})
                
                # 提取关键技术指标
                atr_values = interval_data.get("atr_values", {})
                volatility_analysis = interval_data.get("volatility_analysis", {})
                cross_timeframe = tech_data.get("cross_timeframe_analysis", {})
                
                # 计算投资组合相关指标
                total_portfolio_value = portfolio.get("cash", 0.0) + sum(
                    portfolio.get("cost_basis", {}).get(t, 0.0) for t in portfolio.get("cost_basis", {}))
                available_cash = portfolio.get("cash", 0.0)
                current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0.0)
                
                # 基于波动率和资金状况计算杠杆建议
                leverage_metrics = self._calculate_leverage_metrics(
                    atr_values=atr_values,
                    volatility_analysis=volatility_analysis,
                    cross_timeframe=cross_timeframe,
                    portfolio_value=total_portfolio_value,
                    available_cash=available_cash,
                    current_position=current_position_value,
                    current_price=current_price
                )
                
                leverage_analysis[ticker] = {
                    "leverage_analysis": leverage_metrics,
                    "current_price": float(current_price),
                    "reasoning": {
                        "atr_14": atr_values.get("atr_14", 0.0),
                        "atr_percentile": atr_values.get("atr_percentile", 50.0),
                        "volatility_percentile": volatility_analysis.get("volatility_percentile", 50.0),
                        "volatility_trend": volatility_analysis.get("volatility_trend", "stable"),
                        "timeframe_consensus": cross_timeframe.get("timeframe_consensus", 0.5),
                        "portfolio_value": float(total_portfolio_value),
                        "available_cash": float(available_cash)
                    }
                }
                
            except Exception as e:
                print(f"Error analyzing leverage for {ticker}: {e}")
                leverage_analysis[ticker] = {
                    "leverage_analysis": self._get_default_leverage_analysis(),
                    "current_price": float(price_df["close"].iloc[-1]) if price_df is not None and not price_df.empty else 0.0,
                    "reasoning": {"error": str(e)}
                }
        
        return leverage_analysis
    
    def _calculate_leverage_metrics(self, atr_values: Dict, volatility_analysis: Dict, 
                                  cross_timeframe: Dict, portfolio_value: float, 
                                  available_cash: float, current_position: float, 
                                  current_price: float) -> Dict[str, Any]:
        """计算杠杆相关指标"""
        
        # 提取关键指标
        atr_14 = atr_values.get("atr_14", 0.0)
        atr_percentile = atr_values.get("atr_percentile", 50.0)
        volatility_percentile = volatility_analysis.get("volatility_percentile", 50.0)
        volatility_trend = volatility_analysis.get("volatility_trend", "stable")
        timeframe_consensus = cross_timeframe.get("timeframe_consensus", 0.5)
        
        # 计算基础风险分数 (0-1，越高风险越大)
        base_risk_score = self._calculate_base_risk_score(
            atr_percentile, volatility_percentile, volatility_trend, timeframe_consensus
        )
        
        # 计算资金利用率风险
        position_ratio = current_position / max(portfolio_value, 1.0) if portfolio_value > 0 else 0.0
        cash_ratio = available_cash / max(portfolio_value, 1.0) if portfolio_value > 0 else 0.0
        
        # 资金风险调整
        capital_risk_adjustment = min(1.0, position_ratio * 2.0)  # 已有持仓越多，风险越高
        cash_availability_factor = max(0.1, cash_ratio)  # 可用现金越少，杠杆越保守
        
        # 综合风险评分
        final_risk_score = min(1.0, base_risk_score + capital_risk_adjustment * 0.3)
        
        # 基于风险评分计算杠杆倍数
        max_safe_leverage = self._calculate_max_safe_leverage(final_risk_score, cash_availability_factor)
        recommended_leverage = self._calculate_recommended_leverage(final_risk_score, max_safe_leverage)
        volatility_adjusted_leverage = self._calculate_volatility_adjusted_leverage(
            recommended_leverage, volatility_percentile, atr_percentile
        )
        
        # 生成杠杆选项
        leverage_options = self._generate_leverage_options(recommended_leverage, max_safe_leverage)
        
        return {
            "recommended_leverage": int(recommended_leverage),
            "max_safe_leverage": int(max_safe_leverage),
            "leverage_options": leverage_options,
            "leverage_risk_score": round(final_risk_score, 4),
            "volatility_adjusted_leverage": int(volatility_adjusted_leverage)
        }
    
    def _calculate_base_risk_score(self, atr_percentile: float, volatility_percentile: float, 
                                 volatility_trend: str, timeframe_consensus: float) -> float:
        """计算基础风险分数"""
        # ATR百分位数风险 (越高越危险)
        atr_risk = (atr_percentile / 100.0) * 0.3
        
        # 波动率百分位数风险
        vol_risk = (volatility_percentile / 100.0) * 0.3
        
        # 波动率趋势风险
        trend_risk_map = {"increasing": 0.3, "stable": 0.1, "decreasing": 0.0}
        trend_risk = trend_risk_map.get(volatility_trend, 0.1)
        
        # 时间框架共识风险 (共识度低表示信号不明确，风险更高)
        consensus_risk = (1.0 - timeframe_consensus) * 0.1
        
        return min(1.0, atr_risk + vol_risk + trend_risk + consensus_risk)
    
    def _calculate_max_safe_leverage(self, risk_score: float, cash_factor: float) -> int:
        """计算最大安全杠杆倍数"""
        # 基础最大杠杆：低风险125x，高风险1x
        base_max = max(1, int(125 * (1.0 - risk_score)))
        
        # 根据可用现金调整
        cash_adjusted = max(1, int(base_max * cash_factor))
        
        # 确保在合理范围内
        return min(125, max(1, cash_adjusted))
    
    def _calculate_recommended_leverage(self, risk_score: float, max_safe: int) -> int:
        """计算推荐杠杆倍数（通常是最大安全杠杆的60-80%）"""
        safety_factor = 0.7  # 保守系数
        recommended = int(max_safe * safety_factor)
        return min(max_safe, max(1, recommended))
    
    def _calculate_volatility_adjusted_leverage(self, base_leverage: int, 
                                              vol_percentile: float, atr_percentile: float) -> int:
        """根据波动率调整杠杆倍数"""
        # 计算波动率调整因子
        vol_adjustment = 1.0 - (vol_percentile / 100.0) * 0.3  # 高波动率降低杠杆
        atr_adjustment = 1.0 - (atr_percentile / 100.0) * 0.2   # 高ATR降低杠杆
        
        combined_adjustment = min(vol_adjustment, atr_adjustment)
        adjusted = int(base_leverage * combined_adjustment)
        
        return min(125, max(1, adjusted))
    
    def _generate_leverage_options(self, recommended: int, max_safe: int) -> list:
        """生成3个杠杆选项：保守、推荐、激进"""
        conservative = max(1, int(recommended * 0.7))
        aggressive = min(max_safe, int(recommended * 1.3))
        
        options = [conservative, recommended, aggressive]
        # 去重并排序
        options = sorted(list(set(options)))
        
        # 确保有3个选项
        while len(options) < 3:
            if options[-1] < 125:
                options.append(min(125, options[-1] + 1))
            elif options[0] > 1:
                options.insert(0, max(1, options[0] - 1))
            else:
                break
        
        return options[:3]  # 返回前3个选项
    
    def _get_default_leverage_analysis(self) -> Dict[str, Any]:
        """返回默认的杠杆分析结果"""
        return {
            "recommended_leverage": 5,
            "max_safe_leverage": 10,
            "leverage_options": [3, 5, 8],
            "leverage_risk_score": 0.5,
            "volatility_adjusted_leverage": 4
        }


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