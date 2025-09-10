"""
SignalAnalyzer智能信号分析器测试

测试信号分析的核心功能，包括：
- 多时间框架信号分析
- 信号强度评估
- 双向交易决策逻辑
- 市场环境调整
- 边界情况处理
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from src.utils.signal_analyzer import SignalAnalyzer, TradingDirection, SignalStrength, analyze_trading_signals
from tests.test_fixtures import (
    TestDataFactory, TestAssertions, TestScenarios, 
    test_data_factory, test_assertions, test_scenarios
)


class TestSignalAnalyzer:
    """SignalAnalyzer核心功能测试"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.analyzer = SignalAnalyzer()
        self.test_ticker = "BTCUSDT"
        self.assertions = TestAssertions()
    
    def test_initialization(self):
        """测试SignalAnalyzer初始化"""
        # 测试默认初始化
        analyzer = SignalAnalyzer()
        assert analyzer.config is not None
        assert "timeframe_weights" in analyzer.config
        assert "signal_thresholds" in analyzer.config
        
        # 测试自定义配置初始化
        custom_config = {
            "timeframe_weights": {"1h": 0.5, "4h": 0.5},
            "signal_thresholds": {"strong_bullish": 0.8}
        }
        analyzer = SignalAnalyzer(custom_config)
        assert analyzer.config["timeframe_weights"]["1h"] == 0.5
        assert analyzer.config["signal_thresholds"]["strong_bullish"] == 0.8
    
    def test_strong_bullish_signal_analysis(self):
        """测试强烈看多信号分析"""
        # 创建强烈看多信号
        technical_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker,
            signal_type="bullish",
            confidence_range=(85, 95)
        )
        
        direction, score, details = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker
        )
        
        # 验证结果
        assert direction == TradingDirection.LONG
        assert score > 0.7  # 强烈信号应该有高分数
        assert details["signal_strength"] in ["strong", "moderate"]
        assert details["decision_confidence"] > 70
        assert details["processed_signals_count"] > 0
        
        # 验证分析详情
        assert "score_breakdown" in details
        assert "timeframe_averages" in details["score_breakdown"]
        assert details["score_breakdown"]["signal_count"] > 0
    
    def test_strong_bearish_signal_analysis(self):
        """测试强烈看空信号分析"""
        # 创建强烈看空信号
        technical_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker,
            signal_type="bearish",
            confidence_range=(85, 95)
        )
        
        direction, score, details = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker
        )
        
        # 验证结果
        assert direction == TradingDirection.SHORT
        assert score < -0.7  # 强烈空头信号应该有低分数
        assert details["signal_strength"] in ["strong", "moderate"]
        assert details["decision_confidence"] > 70
        
        # 验证空头信号的分数为负数
        assert details["adjusted_score"] < 0
        assert details["signal_score"] < 0
    
    def test_mixed_signal_analysis(self):
        """测试混合信号分析"""
        # 创建混合信号
        technical_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker,
            signal_type="mixed",
            confidence_range=(50, 75)
        )
        
        direction, score, details = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker
        )
        
        # 混合信号通常导致观望或弱信号
        assert direction in [TradingDirection.HOLD, TradingDirection.LONG, TradingDirection.SHORT]
        assert abs(score) < 0.7  # 混合信号分数应该不会太极端
        assert details["signal_strength"] in ["weak", "moderate"]
        assert details["decision_confidence"] < 80
    
    def test_neutral_signal_analysis(self):
        """测试中性信号分析"""
        # 创建中性信号
        technical_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker,
            signal_type="neutral",
            confidence_range=(40, 60)
        )
        
        direction, score, details = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker
        )
        
        # 中性信号应该导致观望
        assert direction == TradingDirection.HOLD
        assert abs(score) < 0.3  # 中性信号分数应该接近0
        assert details["signal_strength"] == "weak"
        assert details["decision_confidence"] < 70
    
    def test_market_context_adjustments(self):
        """测试市场环境调整"""
        # 创建基础看多信号
        technical_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker,
            signal_type="bullish",
            confidence_range=(75, 85)
        )
        
        # 测试高波动率环境调整
        high_volatility_context = {
            "volatility": 0.08,  # 8%高波动率
            "liquidity_score": 0.8,
            "trend_alignment": 0.3
        }
        
        direction, score, details = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker, high_volatility_context
        )
        
        # 高波动率应该降低信号权重
        assert "market_adjustments" in details
        if "volatility" in details["market_adjustments"]:
            assert details["market_adjustments"]["volatility"]["factor"] < 1.0
        
        # 测试低流动性环境调整
        low_liquidity_context = {
            "volatility": 0.03,
            "liquidity_score": 0.3,  # 低流动性
            "trend_alignment": 0.5
        }
        
        direction2, score2, details2 = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker, low_liquidity_context
        )
        
        # 低流动性应该降低信号权重
        if "liquidity" in details2["market_adjustments"]:
            assert details2["market_adjustments"]["liquidity"]["factor"] < 1.0
        
        # 测试趋势一致性奖励
        trend_aligned_context = {
            "volatility": 0.03,
            "liquidity_score": 0.8,
            "trend_alignment": 0.8  # 强趋势一致性
        }
        
        direction3, score3, details3 = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker, trend_aligned_context
        )
        
        # 趋势一致性应该提升信号权重
        if "trend_alignment" in details3["market_adjustments"]:
            assert details3["market_adjustments"]["trend_alignment"]["bonus"] > 0
    
    def test_cross_timeframe_analysis(self):
        """测试跨时间框架分析"""
        # 创建包含跨时间框架分析的信号
        technical_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker,
            timeframes=["5m", "15m", "30m", "1h", "4h"],
            signal_type="bullish"
        )
        
        direction, score, details = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker
        )
        
        # 验证不同时间框架的权重被正确应用
        score_breakdown = details["score_breakdown"]
        timeframe_averages = score_breakdown["timeframe_averages"]
        
        # 验证所有时间框架都被分析
        expected_timeframes = ["5m", "15m", "30m", "1h", "4h"]
        for tf in expected_timeframes:
            if tf in timeframe_averages:
                assert isinstance(timeframe_averages[tf], (int, float))
        
        # 验证时间框架权重的影响
        # 4h时间框架应该有更高的权重影响
        assert details["processed_signals_count"] >= 5  # 至少5个时间框架的信号
    
    def test_confidence_filtering(self):
        """测试置信度过滤"""
        # 创建包含低置信度信号的数据
        low_confidence_signals = {
            self.test_ticker: {
                "1h": {
                    "signal": "bullish",
                    "confidence": 30,  # 低于默认最小置信度(60)
                    "strategy_signals": {
                        "trend_following": {"signal": "bullish", "confidence": 25}
                    }
                },
                "4h": {
                    "signal": "bullish",
                    "confidence": 80,  # 高置信度
                    "strategy_signals": {
                        "momentum": {"signal": "bullish", "confidence": 75}
                    }
                }
            }
        }
        
        direction, score, details = self.analyzer.analyze_signals(
            low_confidence_signals, self.test_ticker
        )
        
        # 低置信度信号应该被过滤掉
        # 只有4h的信号应该被处理
        processed_count = details["processed_signals_count"]
        assert processed_count < 4  # 应该少于所有信号数量
    
    def test_signal_to_numeric_conversion(self):
        """测试信号转数值转换"""
        # 测试各种信号字符串的转换
        test_signals = [
            ("bullish", 1.0),
            ("bearish", -1.0),
            ("neutral", 0.0),
            ("buy", 1.0),
            ("sell", -1.0),
            ("hold", 0.0),
            ("long", 1.0),
            ("short", -1.0),
            ("bull market", 0.7),  # 部分匹配
            ("bear trend", -0.7),
            ("unknown", 0.0)  # 未知信号
        ]
        
        for signal_str, expected_value in test_signals:
            numeric_value = self.analyzer._signal_to_numeric(signal_str)
            assert numeric_value == expected_value, \
                f"信号 '{signal_str}' 转换错误，期望 {expected_value}，实际 {numeric_value}"
    
    def test_decision_confidence_calculation(self):
        """测试决策置信度计算"""
        # 测试不同分数对应的置信度
        test_scores = [
            (0.8, 85),   # 高分数 -> 高置信度
            (0.5, 60),   # 中等分数 -> 中等置信度
            (0.2, 50),   # 低分数 -> 低置信度
            (-0.7, 85),  # 强烈负分数 -> 高置信度
            (0.0, 30)    # 零分数 -> 低置信度
        ]
        
        for score, min_expected_confidence in test_scores:
            confidence = self.analyzer._calculate_decision_confidence(score)
            assert confidence >= min_expected_confidence, \
                f"分数 {score} 的置信度 {confidence} 低于期望的最小值 {min_expected_confidence}"
            assert 0 <= confidence <= 100, f"置信度 {confidence} 超出有效范围"
    
    def test_signal_summary_generation(self):
        """测试信号摘要生成"""
        # 创建分析结果
        technical_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker, signal_type="bullish"
        )
        analysis_result = self.analyzer.analyze_signals(
            technical_signals, self.test_ticker
        )
        
        # 生成摘要
        summary = self.analyzer.get_signal_summary(analysis_result)
        
        # 验证摘要内容
        assert "交易方向" in summary
        assert "信号强度" in summary
        assert "信号分数" in summary
        assert "决策置信度" in summary
        assert "信号数量" in summary
        
        # 验证摘要格式
        assert len(summary) > 50  # 摘要应该有足够的信息
        assert "|" in summary     # 使用分隔符格式
    
    def test_empty_signals_handling(self):
        """测试空信号处理"""
        # 测试完全空的信号
        empty_signals = {}
        direction, score, details = self.analyzer.analyze_signals(
            empty_signals, self.test_ticker
        )
        
        assert direction == TradingDirection.HOLD
        assert score == 0.0
        assert "reason" in details
        assert details["reason"] == "no_signals"
        
        # 测试ticker不存在的情况
        missing_ticker_signals = {"ETHUSDT": {"1h": {"signal": "bullish", "confidence": 80}}}
        direction2, score2, details2 = self.analyzer.analyze_signals(
            missing_ticker_signals, self.test_ticker
        )
        
        assert direction2 == TradingDirection.HOLD
        assert score2 == 0.0
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试格式错误的信号数据
        malformed_signals = {
            self.test_ticker: {
                "1h": "not_a_dict",  # 错误格式
                "4h": {
                    "signal": "bullish",
                    "confidence": "not_a_number"  # 错误类型
                }
            }
        }
        
        # 应该能够处理错误并返回默认结果
        direction, score, details = self.analyzer.analyze_signals(
            malformed_signals, self.test_ticker
        )
        
        assert direction in [TradingDirection.HOLD, TradingDirection.LONG, TradingDirection.SHORT]
        assert isinstance(score, (int, float))
        assert isinstance(details, dict)
    
    def test_weighted_score_calculation(self):
        """测试加权分数计算"""
        # 创建具有不同权重的信号
        test_signals = [
            {"timeframe": "5m", "signal_value": 1.0, "confidence": 80, "weight": 0.1},
            {"timeframe": "1h", "signal_value": 1.0, "confidence": 85, "weight": 0.25},
            {"timeframe": "4h", "signal_value": 1.0, "confidence": 90, "weight": 0.3},
            {"timeframe": "15m", "signal_value": -0.5, "confidence": 70, "weight": 0.15}
        ]
        
        score, breakdown = self.analyzer._calculate_weighted_score(test_signals)
        
        # 验证加权计算
        assert isinstance(score, float)
        assert "total_weight" in breakdown
        assert "signal_count" in breakdown
        assert breakdown["signal_count"] == len(test_signals)
        assert breakdown["total_weight"] > 0
        
        # 验证正负信号的平衡
        # 由于大部分是正信号，最终分数应该为正
        assert score > 0
    
    def test_direction_determination_thresholds(self):
        """测试方向决策阈值"""
        # 测试边界值
        test_cases = [
            (0.8, TradingDirection.LONG, SignalStrength.STRONG),    # 强烈看多
            (0.5, TradingDirection.LONG, SignalStrength.MODERATE),  # 适度看多
            (0.1, TradingDirection.HOLD, SignalStrength.WEAK),      # 弱信号
            (-0.1, TradingDirection.HOLD, SignalStrength.WEAK),     # 弱信号
            (-0.5, TradingDirection.SHORT, SignalStrength.MODERATE), # 适度看空
            (-0.8, TradingDirection.SHORT, SignalStrength.STRONG)   # 强烈看空
        ]
        
        for score, expected_direction, expected_strength in test_cases:
            direction, strength = self.analyzer._determine_direction(score)
            assert direction == expected_direction, \
                f"分数 {score} 的方向判断错误，期望 {expected_direction}，实际 {direction}"
            assert strength == expected_strength, \
                f"分数 {score} 的强度判断错误，期望 {expected_strength}，实际 {strength}"


class TestAnalyzeTradingSignalsFunction:
    """测试独立的analyze_trading_signals函数"""
    
    def test_function_integration(self):
        """测试函数集成"""
        technical_signals = TestDataFactory.create_technical_signals(
            ticker="BTCUSDT", signal_type="bullish"
        )
        
        result = analyze_trading_signals(technical_signals, "BTCUSDT")
        
        # 验证返回结果格式
        assert "direction" in result
        assert "signal_strength" in result
        assert "confidence" in result
        assert "analysis_details" in result
        
        # 验证结果类型
        assert result["direction"] in ["long", "short", "hold"]
        assert isinstance(result["signal_strength"], str)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["analysis_details"], dict)


class TestBidirectionalSignalAnalysis:
    """测试双向信号分析"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.analyzer = SignalAnalyzer()
        self.test_ticker = "BTCUSDT"
    
    def test_simultaneous_long_short_analysis(self):
        """测试同时分析多空信号"""
        # 创建看多信号
        bullish_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker, signal_type="bullish"
        )
        
        # 创建看空信号
        bearish_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker, signal_type="bearish"
        )
        
        # 分析两个方向
        long_direction, long_score, long_details = self.analyzer.analyze_signals(
            bullish_signals, self.test_ticker
        )
        
        short_direction, short_score, short_details = self.analyzer.analyze_signals(
            bearish_signals, self.test_ticker
        )
        
        # 验证方向一致性
        assert long_direction == TradingDirection.LONG
        assert short_direction == TradingDirection.SHORT
        
        # 验证分数符号
        assert long_score > 0
        assert short_score < 0
        
        # 验证置信度合理性
        assert long_details["decision_confidence"] > 0
        assert short_details["decision_confidence"] > 0
    
    def test_conflicting_signals_resolution(self):
        """测试冲突信号解决"""
        # 创建混合信号（既有多头也有空头倾向）
        mixed_signals = {
            self.test_ticker: {
                "5m": {"signal": "bearish", "confidence": 75, "strategy_signals": {}},
                "15m": {"signal": "bullish", "confidence": 80, "strategy_signals": {}},
                "1h": {"signal": "bearish", "confidence": 70, "strategy_signals": {}},
                "4h": {"signal": "bullish", "confidence": 85, "strategy_signals": {}},
                "cross_timeframe_analysis": {
                    "trend_alignment": "mixed",
                    "overall_signal_strength": "moderate"
                }
            }
        }
        
        direction, score, details = self.analyzer.analyze_signals(
            mixed_signals, self.test_ticker
        )
        
        # 冲突信号应该导致较低的置信度
        assert details["decision_confidence"] < 80
        
        # 分数应该不会太极端
        assert abs(score) < 0.7
        
        # 验证不同时间框架的权重影响
        timeframe_averages = details["score_breakdown"]["timeframe_averages"]
        assert len(timeframe_averages) > 0
    
    def test_position_aware_signal_analysis(self):
        """测试持仓感知的信号分析"""
        # 基础信号
        base_signals = TestDataFactory.create_technical_signals(
            ticker=self.test_ticker, signal_type="mixed"
        )
        
        # 测试不同持仓情况下的信号分析
        positions = [
            TestDataFactory.create_current_positions(position_type="empty"),
            TestDataFactory.create_current_positions(position_type="long"),
            TestDataFactory.create_current_positions(position_type="short")
        ]
        
        results = []
        for position in positions:
            direction, score, details = self.analyzer.analyze_signals(
                base_signals, self.test_ticker
            )
            results.append((direction, score, details))
        
        # 虽然SignalAnalyzer本身不考虑持仓，但信号应该保持一致
        directions = [r[0] for r in results]
        scores = [r[1] for r in results]
        
        # 相同信号应该产生相同的分析结果
        assert len(set(directions)) <= 2  # 允许一些随机性，但不应该差异太大
        
        # 分数差异不应该太大
        score_diff = max(scores) - min(scores)
        assert score_diff < 0.5  # 分数差异控制在合理范围


class TestPerformanceAndScalability:
    """测试性能和可扩展性"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.analyzer = SignalAnalyzer()
    
    def test_large_signal_dataset_performance(self):
        """测试大数据集性能"""
        import time
        
        # 创建大量信号数据
        large_signals = {}
        tickers = [f"SYMBOL{i}" for i in range(20)]
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        
        for ticker in tickers:
            large_signals.update(
                TestDataFactory.create_technical_signals(
                    ticker=ticker, 
                    timeframes=timeframes,
                    signal_type="mixed"
                )
            )
        
        # 测试处理时间
        start_time = time.time()
        
        for ticker in tickers[:5]:  # 测试前5个ticker
            direction, score, details = self.analyzer.analyze_signals(
                large_signals, ticker
            )
            assert direction in [TradingDirection.LONG, TradingDirection.SHORT, TradingDirection.HOLD]
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 性能应该在合理范围内（每个ticker处理时间 < 1秒）
        assert processing_time < 5.0, f"处理时间过长: {processing_time:.2f}秒"
    
    def test_memory_usage_efficiency(self):
        """测试内存使用效率"""
        import gc
        import sys
        
        # 获取初始内存使用
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # 处理多个分析任务
        for i in range(10):
            signals = TestDataFactory.create_technical_signals(
                ticker=f"TEST{i}",
                signal_type="mixed"
            )
            
            direction, score, details = self.analyzer.analyze_signals(
                signals, f"TEST{i}"
            )
            
            # 删除临时变量
            del signals, direction, score, details
        
        # 强制垃圾回收
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # 内存增长应该在合理范围内
        object_increase = final_objects - initial_objects
        assert object_increase < 1000, f"内存对象增长过多: {object_increase}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])