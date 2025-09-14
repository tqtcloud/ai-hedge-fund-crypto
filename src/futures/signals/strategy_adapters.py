"""
期货策略适配器

提供现有期货策略与新基类接口之间的适配，确保向后兼容性，
同时支持统一的策略管理和信号输出。
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import logging
from datetime import datetime

try:
    from .base_strategy import (
        FuturesBaseStrategy,
        StrategyConfig,
        StrategyOutput,
        SignalStrength,
        StrategyType
    )
    from ..models.data_models import FuturesSignal, TradingDirection, OperationType
except ImportError:
    # 处理相对导入问题
    from src.futures.signals.base_strategy import (
        FuturesBaseStrategy,
        StrategyConfig,
        StrategyOutput,
        SignalStrength,
        StrategyType
    )
    from src.futures.models.data_models import FuturesSignal, TradingDirection, OperationType

logger = logging.getLogger(__name__)


class FuturesMacdStrategyAdapter(FuturesBaseStrategy):
    """
    MACD策略适配器

    将现有的FuturesMacdStrategy适配到新的基类接口，
    保持原有功能的同时提供统一的接口。
    """

    def __init__(self, config: StrategyConfig):
        """初始化MACD策略适配器"""
        super().__init__(config)

        # 导入原始策略类
        try:
            from ...strategies.futures_macd_strategy import FuturesMacdStrategy
            self._original_strategy = FuturesMacdStrategy()
            logger.info("成功加载原始MACD策略")
        except ImportError as e:
            logger.error(f"无法导入原始MACD策略: {e}")
            raise

    def analyze(self,
                ticker: str,
                data: Dict[str, pd.DataFrame],
                current_price: float,
                **kwargs) -> Optional[StrategyOutput]:
        """
        执行MACD策略分析

        Args:
            ticker: 交易对符号
            data: 多时间框架数据
            current_price: 当前价格
            **kwargs: 其他参数

        Returns:
            策略输出结果
        """
        try:
            # 准备原始策略需要的数据格式
            agent_state = self._prepare_agent_state(ticker, data, **kwargs)

            # 调用原始策略
            original_result = self._original_strategy(agent_state)

            if not original_result or "data" not in original_result:
                return None

            # 提取分析结果
            analysis_data = original_result["data"].get("analyst_signals", {}).get(
                "futures_technical_analyst_agent", {}
            )

            if not analysis_data or ticker not in analysis_data:
                return None

            ticker_analysis = analysis_data[ticker]

            # 转换为新格式的信号
            futures_signal = self._convert_to_futures_signal(
                ticker, ticker_analysis, current_price
            )

            if not futures_signal:
                return None

            # 创建策略输出
            strategy_output = StrategyOutput(
                signal=futures_signal,
                metadata={
                    "strategy_type": "futures_macd",
                    "original_analysis": ticker_analysis,
                    "cross_timeframe_analysis": ticker_analysis.get("cross_timeframe_analysis", {}),
                    "processing_time": datetime.now().isoformat()
                },
                diagnostics=self._extract_diagnostics(ticker_analysis),
                performance_metrics=self._calculate_performance_metrics(ticker_analysis)
            )

            return strategy_output

        except Exception as e:
            logger.error(f"MACD策略分析失败: {e}")
            return None

    def get_signal_strength(self, data: Dict[str, Any]) -> SignalStrength:
        """计算MACD信号强度"""
        try:
            # 从分析数据中提取强度信息
            overall_strength = data.get("cross_timeframe_analysis", {}).get("overall_signal_strength", "moderate")

            strength_mapping = {
                "very_strong": SignalStrength.VERY_STRONG,
                "strong": SignalStrength.STRONG,
                "moderate": SignalStrength.MODERATE,
                "weak": SignalStrength.WEAK,
                "very_weak": SignalStrength.VERY_WEAK
            }

            return strength_mapping.get(overall_strength, SignalStrength.MODERATE)

        except Exception as e:
            logger.error(f"计算MACD信号强度失败: {e}")
            return SignalStrength.MODERATE

    def calculate_confidence(self, data: Dict[str, Any]) -> float:
        """计算MACD置信度"""
        try:
            # 从跨时间框架分析中获取整体置信度
            cross_tf_analysis = data.get("cross_timeframe_analysis", {})
            overall_confidence = cross_tf_analysis.get("overall_confidence", 50.0)

            # 如果有多个时间框架的数据，计算综合置信度
            timeframe_confidences = []
            for key, value in data.items():
                if key not in ["cross_timeframe_analysis"] and isinstance(value, dict):
                    conf = value.get("confidence", 0)
                    if conf > 0:
                        timeframe_confidences.append(conf)

            if timeframe_confidences:
                avg_tf_confidence = sum(timeframe_confidences) / len(timeframe_confidences)
                # 综合整体置信度和时间框架平均置信度
                final_confidence = (overall_confidence * 0.6 + avg_tf_confidence * 0.4)
            else:
                final_confidence = overall_confidence

            return min(100.0, max(0.0, final_confidence))

        except Exception as e:
            logger.error(f"计算MACD置信度失败: {e}")
            return 50.0

    def get_supported_timeframes(self) -> List[str]:
        """获取MACD策略支持的时间框架"""
        return ["1m", "5m", "15m", "1h", "4h", "1d"]

    def get_required_indicators(self) -> List[str]:
        """获取MACD策略需要的指标"""
        return ["macd", "macd_signal", "macd_histogram", "ema_12", "ema_26", "atr", "volume"]

    def _prepare_agent_state(self, ticker: str, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """准备原始策略需要的AgentState格式数据"""
        try:
            # 构建符合原始策略期望的数据结构
            agent_data = {
                "tickers": [ticker],
                "intervals": [],
                "metadata": {"show_reasoning": kwargs.get("show_reasoning", False)}
            }

            # 转换时间框架数据
            for timeframe, df in data.items():
                if not df.empty:
                    agent_data[f"{ticker}_{timeframe}"] = df
                    # 模拟interval对象
                    class MockInterval:
                        def __init__(self, value):
                            self.value = value

                    agent_data["intervals"].append(MockInterval(timeframe))

            return {"data": agent_data}

        except Exception as e:
            logger.error(f"准备AgentState数据失败: {e}")
            return {"data": {"tickers": [ticker], "intervals": []}}

    def _convert_to_futures_signal(self,
                                 ticker: str,
                                 analysis: Dict[str, Any],
                                 current_price: float) -> Optional[FuturesSignal]:
        """将原始策略结果转换为FuturesSignal"""
        try:
            # 优先使用跨时间框架分析的结果
            cross_tf = analysis.get("cross_timeframe_analysis", {})

            if cross_tf:
                signal_direction = cross_tf.get("dominant_signal", "neutral")
                confidence = cross_tf.get("overall_confidence", 50.0)

                # 获取期货交易建议
                futures_rec = cross_tf.get("futures_timeframe_recommendation", {})
                leverage = self._extract_leverage_from_recommendation(futures_rec)
            else:
                # 使用第一个可用时间框架的数据
                first_tf_data = None
                for key, value in analysis.items():
                    if key != "cross_timeframe_analysis" and isinstance(value, dict):
                        first_tf_data = value
                        break

                if not first_tf_data:
                    return None

                signal_direction = first_tf_data.get("signal", "neutral")
                confidence = first_tf_data.get("confidence", 50.0)

                futures_trading = first_tf_data.get("futures_trading_recommendation", {})
                leverage = futures_trading.get("leverage_recommendation", "conservative")

            # 转换方向
            direction = self._convert_direction(signal_direction)

            # 确定操作类型
            operation_type = OperationType.OPEN if direction != TradingDirection.NEUTRAL else OperationType.HOLD

            # 转换杠杆建议为数值
            suggested_leverage = self._convert_leverage_recommendation(leverage)

            # 计算仓位大小
            position_size = self.calculate_position_size(
                ticker, confidence, current_price, 10000.0, suggested_leverage
            )

            # 计算止损止盈
            sl_tp = self.calculate_stop_loss_take_profit(current_price, direction)

            # 创建期货信号
            signal = self.create_futures_signal(
                ticker=ticker,
                direction=direction,
                operation_type=operation_type,
                confidence=confidence,
                strength=self.get_signal_strength(analysis),
                current_price=current_price,
                leverage=suggested_leverage,
                metadata={
                    "original_signal": signal_direction,
                    "macd_analysis": analysis,
                    "adapter": "FuturesMacdStrategyAdapter"
                }
            )

            return signal

        except Exception as e:
            logger.error(f"转换期货信号失败: {e}")
            return None

    def _convert_direction(self, signal_direction: str) -> TradingDirection:
        """转换信号方向"""
        direction_mapping = {
            "long": TradingDirection.LONG,
            "short": TradingDirection.SHORT,
            "neutral": TradingDirection.NEUTRAL,
            "bullish": TradingDirection.LONG,
            "bearish": TradingDirection.SHORT
        }
        return direction_mapping.get(signal_direction.lower(), TradingDirection.NEUTRAL)

    def _convert_leverage_recommendation(self, leverage_rec: str) -> float:
        """转换杠杆建议为数值"""
        leverage_mapping = {
            "avoid": 1.0,
            "minimal": 2.0,
            "conservative": 5.0,
            "moderate": 10.0,
            "aggressive": 15.0
        }
        return leverage_mapping.get(leverage_rec, 5.0)

    def _extract_leverage_from_recommendation(self, futures_rec: Dict[str, Any]) -> str:
        """从期货建议中提取杠杆建议"""
        position_size = futures_rec.get("position_size", "conservative")

        size_to_leverage = {
            "minimal": "minimal",
            "conservative": "conservative",
            "moderate": "moderate",
            "aggressive": "aggressive"
        }

        return size_to_leverage.get(position_size, "conservative")

    def _extract_diagnostics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """提取诊断信息"""
        return {
            "cross_timeframe_consensus": analysis.get("cross_timeframe_analysis", {}).get("timeframe_consensus", 0),
            "dominant_timeframe": analysis.get("cross_timeframe_analysis", {}).get("dominant_timeframe", "unknown"),
            "conflict_areas": len(analysis.get("cross_timeframe_analysis", {}).get("conflict_areas", [])),
            "signal_count": len([k for k in analysis.keys() if k != "cross_timeframe_analysis"]),
            "data_quality": "good" if analysis else "poor"
        }

    def _calculate_performance_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算性能指标"""
        return {
            "signal_generation_time": datetime.now().timestamp(),
            "data_completeness": len(analysis) / 7,  # 假设7个时间框架为完整
            "cross_timeframe_alignment": analysis.get("cross_timeframe_analysis", {}).get("alignment_score", 0),
            "overall_strength_score": self._map_strength_to_score(
                analysis.get("cross_timeframe_analysis", {}).get("overall_signal_strength", "moderate")
            )
        }

    def _map_strength_to_score(self, strength: str) -> float:
        """映射强度字符串到数值分数"""
        strength_scores = {
            "very_weak": 0.2,
            "weak": 0.4,
            "moderate": 0.6,
            "strong": 0.8,
            "very_strong": 1.0
        }
        return strength_scores.get(strength, 0.6)


class FuturesRSIStrategyAdapter(FuturesBaseStrategy):
    """
    RSI策略适配器

    将现有的FuturesRSIStrategy适配到新的基类接口。
    """

    def __init__(self, config: StrategyConfig):
        """初始化RSI策略适配器"""
        super().__init__(config)

        # 导入原始策略类
        try:
            from ...strategies.futures_rsi_strategy import FuturesRSIStrategy
            self._original_strategy = FuturesRSIStrategy()
            logger.info("成功加载原始RSI策略")
        except ImportError as e:
            logger.error(f"无法导入原始RSI策略: {e}")
            raise

    def analyze(self,
                ticker: str,
                data: Dict[str, pd.DataFrame],
                current_price: float,
                **kwargs) -> Optional[StrategyOutput]:
        """执行RSI策略分析"""
        try:
            # 准备原始策略需要的数据格式
            agent_state = self._prepare_agent_state(ticker, data, **kwargs)

            # 调用原始策略
            original_result = self._original_strategy(agent_state)

            if not original_result or "data" not in original_result:
                return None

            # 提取策略输出
            strategy_output = original_result["data"].get("strategy_output", {})
            technical_analysis = strategy_output.get("technical_analysis", {})

            if not technical_analysis or ticker not in technical_analysis:
                return None

            ticker_analysis = technical_analysis[ticker]

            # 转换为新格式的信号
            futures_signal = self._convert_to_futures_signal(
                ticker, ticker_analysis, current_price
            )

            if not futures_signal:
                return None

            # 创建策略输出
            strategy_output_obj = StrategyOutput(
                signal=futures_signal,
                metadata={
                    "strategy_type": "futures_rsi",
                    "original_analysis": ticker_analysis,
                    "cross_timeframe_analysis": ticker_analysis.get("cross_timeframe_analysis", {}),
                    "processing_time": datetime.now().isoformat()
                },
                diagnostics=self._extract_diagnostics(ticker_analysis),
                performance_metrics=self._calculate_performance_metrics(ticker_analysis)
            )

            return strategy_output_obj

        except Exception as e:
            logger.error(f"RSI策略分析失败: {e}")
            return None

    def get_signal_strength(self, data: Dict[str, Any]) -> SignalStrength:
        """计算RSI信号强度"""
        try:
            # 从跨时间框架分析中获取整体强度
            cross_tf = data.get("cross_timeframe_analysis", {})
            overall_strength = cross_tf.get("overall_strength", "moderate")

            strength_mapping = {
                "weak": SignalStrength.WEAK,
                "moderate": SignalStrength.MODERATE,
                "strong": SignalStrength.STRONG
            }

            return strength_mapping.get(overall_strength, SignalStrength.MODERATE)

        except Exception as e:
            logger.error(f"计算RSI信号强度失败: {e}")
            return SignalStrength.MODERATE

    def calculate_confidence(self, data: Dict[str, Any]) -> float:
        """计算RSI置信度"""
        try:
            # 从跨时间框架分析中获取平均置信度
            cross_tf = data.get("cross_timeframe_analysis", {})
            avg_confidence = cross_tf.get("average_confidence", 50.0)

            return min(100.0, max(0.0, avg_confidence))

        except Exception as e:
            logger.error(f"计算RSI置信度失败: {e}")
            return 50.0

    def get_supported_timeframes(self) -> List[str]:
        """获取RSI策略支持的时间框架"""
        return ["1m", "5m", "15m", "1h", "4h", "1d"]

    def get_required_indicators(self) -> List[str]:
        """获取RSI策略需要的指标"""
        return ["rsi", "rsi_14", "overbought_threshold", "oversold_threshold", "atr", "volume"]

    def _prepare_agent_state(self, ticker: str, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """准备原始策略需要的AgentState格式数据"""
        try:
            agent_data = {
                "tickers": [ticker],
                "intervals": []
            }

            # 转换时间框架数据
            for timeframe, df in data.items():
                if not df.empty:
                    agent_data[f"{ticker}_{timeframe}"] = df

                    class MockInterval:
                        def __init__(self, value):
                            self.value = value

                    agent_data["intervals"].append(MockInterval(timeframe))

            return {"data": agent_data}

        except Exception as e:
            logger.error(f"准备RSI AgentState数据失败: {e}")
            return {"data": {"tickers": [ticker], "intervals": []}}

    def _convert_to_futures_signal(self,
                                 ticker: str,
                                 analysis: Dict[str, Any],
                                 current_price: float) -> Optional[FuturesSignal]:
        """将原始RSI策略结果转换为FuturesSignal"""
        try:
            # 优先使用跨时间框架分析
            cross_tf = analysis.get("cross_timeframe_analysis", {})

            if cross_tf:
                signal_direction = cross_tf.get("dominant_signal", "neutral")
                confidence = cross_tf.get("average_confidence", 50.0)
            else:
                # 使用第一个可用时间框架的数据
                first_tf_data = None
                for key, value in analysis.items():
                    if key != "cross_timeframe_analysis" and isinstance(value, dict):
                        first_tf_data = value
                        break

                if not first_tf_data:
                    return None

                signal_direction = first_tf_data.get("signal", "neutral")
                confidence = first_tf_data.get("confidence", 50.0)

            # 转换方向和操作类型
            direction = self._convert_direction(signal_direction)
            operation_type = OperationType.OPEN if direction != TradingDirection.NEUTRAL else OperationType.HOLD

            # RSI策略通常使用较低的杠杆
            suggested_leverage = min(self.config.default_leverage, 8.0)

            # 计算仓位大小
            position_size = self.calculate_position_size(
                ticker, confidence, current_price, 10000.0, suggested_leverage
            )

            # 创建期货信号
            signal = self.create_futures_signal(
                ticker=ticker,
                direction=direction,
                operation_type=operation_type,
                confidence=confidence,
                strength=self.get_signal_strength(analysis),
                current_price=current_price,
                leverage=suggested_leverage,
                metadata={
                    "original_signal": signal_direction,
                    "rsi_analysis": analysis,
                    "adapter": "FuturesRSIStrategyAdapter"
                }
            )

            return signal

        except Exception as e:
            logger.error(f"转换RSI期货信号失败: {e}")
            return None

    def _convert_direction(self, signal_direction: str) -> TradingDirection:
        """转换信号方向"""
        direction_mapping = {
            "long": TradingDirection.LONG,
            "short": TradingDirection.SHORT,
            "neutral": TradingDirection.NEUTRAL
        }
        return direction_mapping.get(signal_direction.lower(), TradingDirection.NEUTRAL)

    def _extract_diagnostics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """提取RSI诊断信息"""
        return {
            "signal_consensus": analysis.get("cross_timeframe_analysis", {}).get("signal_consensus", 0),
            "timeframe_count": analysis.get("cross_timeframe_analysis", {}).get("timeframe_count", 0),
            "overall_strength": analysis.get("cross_timeframe_analysis", {}).get("overall_strength", "weak"),
            "signal_distribution": analysis.get("cross_timeframe_analysis", {}).get("signal_distribution", {}),
            "data_quality": "good" if analysis else "poor"
        }

    def _calculate_performance_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算RSI性能指标"""
        return {
            "signal_generation_time": datetime.now().timestamp(),
            "timeframe_coverage": analysis.get("cross_timeframe_analysis", {}).get("timeframe_count", 0) / 6,
            "consensus_score": analysis.get("cross_timeframe_analysis", {}).get("signal_consensus", 0),
            "strength_score": self._map_strength_to_score(
                analysis.get("cross_timeframe_analysis", {}).get("overall_strength", "moderate")
            )
        }

    def _map_strength_to_score(self, strength: str) -> float:
        """映射强度字符串到数值分数"""
        strength_scores = {
            "weak": 0.4,
            "moderate": 0.6,
            "strong": 0.8
        }
        return strength_scores.get(strength, 0.6)