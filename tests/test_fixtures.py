"""
双向交易测试数据工厂和公共测试工具

提供标准化的测试数据、断言辅助函数和Mock对象
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
from enum import Enum
import random
from datetime import datetime, timedelta

from src.utils.signal_analyzer import TradingDirection, SignalStrength
from src.utils.exceptions import (
    LeverageExceedsLimitError, 
    MarginInsufficientError, 
    PositionSizeError, 
    RiskLimitExceededError
)


class TestDataFactory:
    """测试数据工厂类"""
    
    @staticmethod
    def create_technical_signals(
        ticker: str = "BTCUSDT",
        timeframes: List[str] = None,
        signal_type: str = "mixed",
        confidence_range: tuple = (50, 90)
    ) -> Dict[str, Any]:
        """
        创建技术信号测试数据
        
        Args:
            ticker: 交易对标识
            timeframes: 时间框架列表
            signal_type: 信号类型 (bullish, bearish, mixed, neutral)
            confidence_range: 置信度范围
            
        Returns:
            技术信号数据字典
        """
        if timeframes is None:
            timeframes = ["5m", "15m", "30m", "1h", "4h"]
        
        signals = {}
        
        for timeframe in timeframes:
            # 根据信号类型生成主要信号
            if signal_type == "bullish":
                main_signal = "bullish"
                strategy_signals = {
                    "trend_following": {"signal": "bullish", "confidence": random.randint(70, 95)},
                    "momentum": {"signal": "bullish", "confidence": random.randint(65, 85)},
                    "volatility": {"signal": "neutral", "confidence": random.randint(50, 70)}
                }
            elif signal_type == "bearish":
                main_signal = "bearish"
                strategy_signals = {
                    "trend_following": {"signal": "bearish", "confidence": random.randint(70, 95)},
                    "momentum": {"signal": "bearish", "confidence": random.randint(65, 85)},
                    "mean_reversion": {"signal": "bearish", "confidence": random.randint(60, 80)}
                }
            elif signal_type == "neutral":
                main_signal = "neutral"
                strategy_signals = {
                    "trend_following": {"signal": "neutral", "confidence": random.randint(40, 60)},
                    "momentum": {"signal": "neutral", "confidence": random.randint(45, 65)},
                    "volatility": {"signal": "neutral", "confidence": random.randint(50, 70)}
                }
            else:  # mixed
                signals_options = ["bullish", "bearish", "neutral"]
                main_signal = random.choice(signals_options)
                strategy_signals = {
                    "trend_following": {"signal": random.choice(signals_options), "confidence": random.randint(60, 85)},
                    "momentum": {"signal": random.choice(signals_options), "confidence": random.randint(55, 80)},
                    "mean_reversion": {"signal": random.choice(signals_options), "confidence": random.randint(50, 75)},
                    "volatility": {"signal": random.choice(signals_options), "confidence": random.randint(45, 70)}
                }
            
            # 为主信号生成置信度
            main_confidence = random.randint(confidence_range[0], confidence_range[1])
            
            # 添加价格水平数据
            current_price = 50000.0  # 假设BTC价格
            price_levels = {
                "support_levels": [current_price * 0.95, current_price * 0.92, current_price * 0.88],
                "resistance_levels": [current_price * 1.05, current_price * 1.08, current_price * 1.12]
            }
            
            # 添加ATR数据
            atr_values = {
                "atr_14": current_price * 0.03,  # 3% ATR
                "atr_percentage": 0.03
            }
            
            signals[timeframe] = {
                "signal": main_signal,
                "confidence": main_confidence,
                "strategy_signals": strategy_signals,
                "price_levels": price_levels,
                "atr_values": atr_values,
                "timestamp": datetime.now().isoformat()
            }
        
        # 添加跨时间框架分析
        if signal_type == "bullish":
            trend_alignment = "aligned"
            overall_signal_strength = "strong"
            dominant_timeframe = "4h"
        elif signal_type == "bearish":
            trend_alignment = "aligned"
            overall_signal_strength = "strong"
            dominant_timeframe = "4h"
        elif signal_type == "neutral":
            trend_alignment = "mixed"
            overall_signal_strength = "weak"
            dominant_timeframe = "1h"
        else:  # mixed
            trend_alignment = "mixed"
            overall_signal_strength = "moderate"
            dominant_timeframe = random.choice(timeframes)
        
        signals["cross_timeframe_analysis"] = {
            "trend_alignment": trend_alignment,
            "overall_signal_strength": overall_signal_strength,
            "dominant_timeframe": dominant_timeframe,
            "signal_consistency": 0.7 if signal_type in ["bullish", "bearish"] else 0.4
        }
        
        return {ticker: signals}
    
    @staticmethod
    def create_risk_data(
        portfolio_cash: float = 100000.0,
        risk_level: str = "moderate",
        margin_scenario: str = "normal"
    ) -> Dict[str, Any]:
        """
        创建风险管理测试数据
        
        Args:
            portfolio_cash: 组合现金
            risk_level: 风险级别 (conservative, moderate, aggressive)
            margin_scenario: 保证金场景 (normal, tight, critical)
            
        Returns:
            风险管理数据字典
        """
        # 根据风险级别设置杠杆
        if risk_level == "conservative":
            recommended_leverage = 2
            max_safe_leverage = 3
            volatility_adjusted_leverage = 2
        elif risk_level == "aggressive":
            recommended_leverage = 8
            max_safe_leverage = 12
            volatility_adjusted_leverage = 6
        else:  # moderate
            recommended_leverage = 5
            max_safe_leverage = 8
            volatility_adjusted_leverage = 4
        
        # 根据保证金场景设置保证金参数
        if margin_scenario == "tight":
            available_margin = portfolio_cash * 0.6
            margin_utilization = 0.7
        elif margin_scenario == "critical":
            available_margin = portfolio_cash * 0.3
            margin_utilization = 0.9
        else:  # normal
            available_margin = portfolio_cash * 0.8
            margin_utilization = 0.5
        
        return {
            "leverage_analysis": {
                "recommended_leverage": recommended_leverage,
                "max_safe_leverage": max_safe_leverage,
                "volatility_adjusted_leverage": volatility_adjusted_leverage
            },
            "position_risk_control": {
                "max_position_size": portfolio_cash * 0.2,
                "position_sizing_factor": 0.02,
                "risk_per_trade": portfolio_cash * 0.02
            },
            "margin_management": {
                "available_margin": available_margin,
                "margin_utilization": margin_utilization,
                "liquidation_buffer": 0.1
            }
        }
    
    @staticmethod
    def create_current_positions(
        ticker: str = "BTCUSDT",
        position_type: str = "empty"
    ) -> Dict[str, Any]:
        """
        创建当前持仓测试数据
        
        Args:
            ticker: 交易对标识
            position_type: 持仓类型 (empty, long, short, both)
            
        Returns:
            当前持仓数据字典
        """
        positions = {}
        
        if position_type == "empty":
            positions[ticker] = {"long": 0.0, "short": 0.0}
        elif position_type == "long":
            positions[ticker] = {"long": 5000.0, "short": 0.0}
        elif position_type == "short":
            positions[ticker] = {"long": 0.0, "short": 3000.0}
        elif position_type == "both":
            positions[ticker] = {"long": 4000.0, "short": 2000.0}  # 净多头2000
        
        return positions
    
    @staticmethod
    def create_market_context(
        volatility: float = 0.03,
        liquidity_score: float = 0.8,
        market_trend: str = "neutral",
        trend_alignment: float = 0.0
    ) -> Dict[str, Any]:
        """
        创建市场环境测试数据
        
        Returns:
            市场环境数据字典
        """
        return {
            "volatility": volatility,
            "liquidity_score": liquidity_score,
            "market_trend": market_trend,
            "trend_alignment": trend_alignment,
            "market_phase": "normal"
        }


class MockDataProvider:
    """模拟数据提供者"""
    
    @staticmethod
    def mock_signal_analyzer():
        """创建SignalAnalyzer的Mock对象"""
        mock_analyzer = Mock()
        mock_analyzer.analyze_signals.return_value = (
            TradingDirection.LONG,
            0.6,
            {
                "signal_strength": "moderate",
                "decision_confidence": 75.0,
                "processed_signals_count": 5,
                "score_breakdown": {"timeframe_averages": {"1h": 0.5, "4h": 0.7}},
                "market_adjustments": {}
            }
        )
        return mock_analyzer
    
    @staticmethod
    def mock_portfolio_calculator():
        """创建PortfolioCalculator的Mock对象"""
        mock_calculator = Mock()
        mock_calculator.determine_direction_and_operation.return_value = ("long", "open")
        mock_calculator.calculate_leverage.return_value = 5
        mock_calculator.calculate_position_size.return_value = (10000.0, 0.1)
        mock_calculator.determine_entry_strategy.return_value = (50000.0, "market")
        return mock_calculator


class TestAssertions:
    """测试断言辅助类"""
    
    @staticmethod
    def assert_valid_trading_direction(direction: str):
        """断言交易方向有效"""
        assert direction in ["long", "short", "hold"], f"无效的交易方向: {direction}"
    
    @staticmethod
    def assert_valid_operation_type(operation: str):
        """断言操作类型有效"""
        assert operation in ["open", "close", "add", "reduce", "hold"], f"无效的操作类型: {operation}"
    
    @staticmethod
    def assert_leverage_within_bounds(leverage: int, min_leverage: int = 1, max_leverage: int = 125):
        """断言杠杆倍数在合理范围内"""
        assert min_leverage <= leverage <= max_leverage, \
            f"杠杆倍数 {leverage} 超出范围 [{min_leverage}, {max_leverage}]"
    
    @staticmethod
    def assert_position_size_positive(position_size: float):
        """断言仓位大小为正数"""
        assert position_size > 0, f"仓位大小必须为正数，当前值: {position_size}"
    
    @staticmethod
    def assert_position_ratio_valid(position_ratio: float):
        """断言仓位比例有效"""
        assert 0 <= position_ratio <= 1, f"仓位比例 {position_ratio} 超出范围 [0, 1]"
    
    @staticmethod
    def assert_confidence_valid(confidence: float):
        """断言置信度有效"""
        assert 0 <= confidence <= 100, f"置信度 {confidence} 超出范围 [0, 100]"
    
    @staticmethod
    def assert_risk_management_logic_correct(
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ):
        """断言风险管理逻辑正确"""
        if direction == "long":
            assert stop_loss < entry_price, \
                f"多头止损价 {stop_loss} 应低于入场价 {entry_price}"
            assert take_profit > entry_price, \
                f"多头止盈价 {take_profit} 应高于入场价 {entry_price}"
        elif direction == "short":
            assert stop_loss > entry_price, \
                f"空头止损价 {stop_loss} 应高于入场价 {entry_price}"
            assert take_profit < entry_price, \
                f"空头止盈价 {take_profit} 应低于入场价 {entry_price}"
    
    @staticmethod
    def assert_bidirectional_consistency(
        long_decision: Dict[str, Any],
        short_decision: Dict[str, Any]
    ):
        """断言双向决策的一致性"""
        # 两个方向的决策应该有相反的方向偏好
        long_direction = long_decision.get("basic_params", {}).get("direction")
        short_direction = short_decision.get("basic_params", {}).get("direction")
        
        if long_direction == "long" and short_direction == "short":
            # 检查风险参数的一致性
            long_leverage = long_decision.get("basic_params", {}).get("leverage", 1)
            short_leverage = short_decision.get("basic_params", {}).get("leverage", 1)
            
            # 杠杆应该在合理范围内
            assert abs(long_leverage - short_leverage) <= 2, \
                f"多空杠杆差异过大: long={long_leverage}, short={short_leverage}"


class TestScenarios:
    """测试场景生成器"""
    
    @staticmethod
    def generate_signal_strength_scenarios() -> List[Dict[str, Any]]:
        """生成不同信号强度的测试场景"""
        scenarios = []
        
        # 强烈看多信号
        scenarios.append({
            "name": "strong_bullish",
            "description": "强烈看多信号，多个时间框架一致",
            "technical_signals": TestDataFactory.create_technical_signals(
                signal_type="bullish", confidence_range=(80, 95)
            ),
            "expected_direction": "long",
            "expected_strength": "strong"
        })
        
        # 强烈看空信号
        scenarios.append({
            "name": "strong_bearish",
            "description": "强烈看空信号，多个时间框架一致",
            "technical_signals": TestDataFactory.create_technical_signals(
                signal_type="bearish", confidence_range=(80, 95)
            ),
            "expected_direction": "short",
            "expected_strength": "strong"
        })
        
        # 混合信号
        scenarios.append({
            "name": "mixed_signals",
            "description": "混合信号，不同时间框架存在分歧",
            "technical_signals": TestDataFactory.create_technical_signals(
                signal_type="mixed", confidence_range=(50, 75)
            ),
            "expected_direction": "hold",
            "expected_strength": "weak"
        })
        
        # 中性信号
        scenarios.append({
            "name": "neutral_signals",
            "description": "中性信号，市场横盘整理",
            "technical_signals": TestDataFactory.create_technical_signals(
                signal_type="neutral", confidence_range=(40, 60)
            ),
            "expected_direction": "hold",
            "expected_strength": "weak"
        })
        
        return scenarios
    
    @staticmethod
    def generate_position_scenarios() -> List[Dict[str, Any]]:
        """生成不同持仓情况的测试场景"""
        scenarios = []
        
        positions = [
            ("empty", "空仓状态，准备建仓"),
            ("long", "持有多头仓位"),
            ("short", "持有空头仓位"),
            ("both", "同时持有多空仓位")
        ]
        
        for position_type, description in positions:
            scenarios.append({
                "name": f"position_{position_type}",
                "description": description,
                "current_positions": TestDataFactory.create_current_positions(
                    position_type=position_type
                ),
                "position_type": position_type
            })
        
        return scenarios
    
    @staticmethod
    def generate_risk_scenarios() -> List[Dict[str, Any]]:
        """生成不同风险环境的测试场景"""
        scenarios = []
        
        # 低风险环境
        scenarios.append({
            "name": "low_risk",
            "description": "低风险环境，低波动率，高流动性",
            "market_context": TestDataFactory.create_market_context(
                volatility=0.015, liquidity_score=0.9, trend_alignment=0.8
            ),
            "risk_data": TestDataFactory.create_risk_data(risk_level="conservative")
        })
        
        # 高风险环境
        scenarios.append({
            "name": "high_risk",
            "description": "高风险环境，高波动率，低流动性",
            "market_context": TestDataFactory.create_market_context(
                volatility=0.08, liquidity_score=0.4, trend_alignment=0.2
            ),
            "risk_data": TestDataFactory.create_risk_data(risk_level="aggressive")
        })
        
        # 保证金紧张
        scenarios.append({
            "name": "margin_tight",
            "description": "保证金使用率高，资金紧张",
            "market_context": TestDataFactory.create_market_context(),
            "risk_data": TestDataFactory.create_risk_data(margin_scenario="tight")
        })
        
        return scenarios


@pytest.fixture
def test_data_factory():
    """测试数据工厂fixture"""
    return TestDataFactory()


@pytest.fixture
def mock_data_provider():
    """模拟数据提供者fixture"""
    return MockDataProvider()


@pytest.fixture
def test_assertions():
    """测试断言辅助fixture"""
    return TestAssertions()


@pytest.fixture
def test_scenarios():
    """测试场景fixture"""
    return TestScenarios()


@pytest.fixture
def sample_technical_signals():
    """示例技术信号fixture"""
    return TestDataFactory.create_technical_signals()


@pytest.fixture
def sample_risk_data():
    """示例风险数据fixture"""
    return TestDataFactory.create_risk_data()


@pytest.fixture
def sample_current_positions():
    """示例当前持仓fixture"""
    return TestDataFactory.create_current_positions()


@pytest.fixture
def sample_market_context():
    """示例市场环境fixture"""
    return TestDataFactory.create_market_context()