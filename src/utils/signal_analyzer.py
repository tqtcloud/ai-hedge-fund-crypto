"""
智能信号分析器

负责从多时间框架技术信号中智能决策交易方向，支持多空双向交易。
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TradingDirection(Enum):
    """交易方向枚举"""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"


class SignalStrength(Enum):
    """信号强度枚举"""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class SignalAnalyzer:
    """
    智能信号分析器
    
    从多时间框架和多策略信号中智能决策交易方向
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化信号分析器
        
        Args:
            config: 配置参数
        """
        self.config = config or self._get_default_config()
        logger.info("SignalAnalyzer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 时间框架权重（权重越高影响越大）
            "timeframe_weights": {
                "5m": 0.1,    # 短期噪音权重较低
                "15m": 0.15,
                "30m": 0.2,
                "1h": 0.25,   # 中期权重较高
                "4h": 0.3     # 长期趋势权重最高
            },
            
            # 策略类型权重
            "strategy_weights": {
                "trend_following": 0.3,      # 趋势跟踪
                "mean_reversion": 0.2,       # 均值回归
                "momentum": 0.25,            # 动量
                "volatility": 0.15,          # 波动率
                "statistical_arbitrage": 0.1 # 统计套利
            },
            
            # 信号强度阈值
            "signal_thresholds": {
                "strong_bullish": 0.7,    # 强烈看多阈值
                "moderate_bullish": 0.3,  # 适度看多阈值
                "neutral_zone": 0.1,      # 中性区间
                "moderate_bearish": -0.3, # 适度看空阈值
                "strong_bearish": -0.7    # 强烈看空阈值
            },
            
            # 置信度影响因子
            "confidence_factors": {
                "min_confidence": 60,     # 最小置信度要求
                "high_confidence": 80,    # 高置信度阈值
                "confidence_weight": 0.2  # 置信度权重
            },
            
            # 市场环境调整因子
            "market_adjustments": {
                "high_volatility_factor": 0.8,  # 高波动率时降低信号权重
                "low_liquidity_factor": 0.7,    # 低流动性时降低信号权重
                "trend_alignment_bonus": 0.2    # 趋势一致性奖励
            }
        }
    
    def analyze_signals(
        self, 
        technical_signals: Dict[str, Any], 
        ticker: str,
        market_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[TradingDirection, float, Dict[str, Any]]:
        """
        分析技术信号并决策交易方向
        
        Args:
            technical_signals: 技术分析信号数据
            ticker: 交易对标识
            market_context: 市场环境上下文
            
        Returns:
            (交易方向, 信号强度分数, 分析详情)
        """
        try:
            # 1. 提取和预处理信号
            processed_signals = self._extract_signals(technical_signals, ticker)
            if not processed_signals:
                logger.warning(f"No valid signals found for {ticker}")
                return TradingDirection.HOLD, 0.0, {"reason": "no_signals"}
            
            # 2. 计算加权信号分数
            signal_score, score_breakdown = self._calculate_weighted_score(processed_signals)
            
            # 3. 应用市场环境调整
            adjusted_score, adjustments = self._apply_market_adjustments(
                signal_score, market_context or {}
            )
            
            # 4. 决策交易方向
            direction, strength = self._determine_direction(adjusted_score)
            
            # 5. 生成分析详情
            analysis_details = {
                "signal_score": signal_score,
                "adjusted_score": adjusted_score,
                "score_breakdown": score_breakdown,
                "market_adjustments": adjustments,
                "signal_strength": strength.value,
                "processed_signals_count": len(processed_signals),
                "decision_confidence": self._calculate_decision_confidence(adjusted_score)
            }
            
            logger.info(f"Signal analysis for {ticker}: {direction.value} "
                       f"(score: {adjusted_score:.3f}, strength: {strength.value})")
            
            return direction, adjusted_score, analysis_details
            
        except Exception as e:
            logger.error(f"Error analyzing signals for {ticker}: {e}")
            return TradingDirection.HOLD, 0.0, {"error": str(e)}
    
    def _extract_signals(self, technical_signals: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
        """
        从技术分析数据中提取标准化信号
        
        Returns:
            标准化信号列表
        """
        signals = []
        
        # 获取ticker的信号数据
        ticker_data = technical_signals.get(ticker, {})
        
        for timeframe, timeframe_data in ticker_data.items():
            if not isinstance(timeframe_data, dict):
                continue
                
            # 提取主信号
            main_signal = timeframe_data.get("signal", "neutral")
            main_confidence = timeframe_data.get("confidence", 50)
            
            if main_confidence < self.config["confidence_factors"]["min_confidence"]:
                continue  # 跳过低置信度信号
            
            # 转换信号为数值
            signal_value = self._signal_to_numeric(main_signal)
            
            signals.append({
                "timeframe": timeframe,
                "signal_value": signal_value,
                "confidence": main_confidence,
                "original_signal": main_signal,
                "weight": self.config["timeframe_weights"].get(timeframe, 0.1)
            })
            
            # 提取策略子信号
            strategy_signals = timeframe_data.get("strategy_signals", {})
            for strategy_name, strategy_data in strategy_signals.items():
                if not isinstance(strategy_data, dict):
                    continue
                
                strategy_signal = strategy_data.get("signal", "neutral")
                strategy_confidence = strategy_data.get("confidence", 50)
                
                if strategy_confidence < self.config["confidence_factors"]["min_confidence"]:
                    continue
                
                strategy_value = self._signal_to_numeric(strategy_signal)
                strategy_weight = self.config["strategy_weights"].get(strategy_name, 0.1)
                
                signals.append({
                    "timeframe": timeframe,
                    "strategy": strategy_name,
                    "signal_value": strategy_value,
                    "confidence": strategy_confidence,
                    "original_signal": strategy_signal,
                    "weight": strategy_weight * 0.5  # 策略子信号权重降低
                })
        
        return signals
    
    def _signal_to_numeric(self, signal: str) -> float:
        """
        将信号字符串转换为数值
        
        Args:
            signal: 信号字符串 (bullish/bearish/neutral)
            
        Returns:
            数值化信号 (-1 to 1)
        """
        signal = signal.lower().strip()
        
        if signal in ["bullish", "buy", "long"]:
            return 1.0
        elif signal in ["bearish", "sell", "short"]:
            return -1.0
        elif signal in ["neutral", "hold"]:
            return 0.0
        else:
            # 处理其他可能的信号描述
            if "bull" in signal or "up" in signal:
                return 0.7
            elif "bear" in signal or "down" in signal:
                return -0.7
            else:
                return 0.0
    
    def _calculate_weighted_score(self, signals: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        计算加权信号分数
        
        Returns:
            (总分数, 分数明细)
        """
        if not signals:
            return 0.0, {}
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        # 按时间框架和策略分组统计
        timeframe_scores = {}
        strategy_scores = {}
        confidence_weighted_score = 0.0
        confidence_weight_sum = 0.0
        
        for signal in signals:
            # 基础权重 * 置信度调整
            confidence_factor = min(signal["confidence"] / 100.0, 1.0)
            adjusted_weight = signal["weight"] * confidence_factor
            
            # 计算加权分数
            weighted_score = signal["signal_value"] * adjusted_weight
            total_weighted_score += weighted_score
            total_weight += adjusted_weight
            
            # 置信度加权分数
            confidence_weighted_score += signal["signal_value"] * signal["confidence"] / 100.0
            confidence_weight_sum += signal["confidence"] / 100.0
            
            # 分组统计
            timeframe = signal["timeframe"]
            if timeframe not in timeframe_scores:
                timeframe_scores[timeframe] = []
            timeframe_scores[timeframe].append(signal["signal_value"])
            
            if "strategy" in signal:
                strategy = signal["strategy"]
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                strategy_scores[strategy].append(signal["signal_value"])
        
        # 计算最终分数
        final_score = total_weighted_score / max(total_weight, 0.001)
        
        # 应用置信度调整
        if confidence_weight_sum > 0:
            confidence_avg_score = confidence_weighted_score / confidence_weight_sum
            confidence_weight = self.config["confidence_factors"]["confidence_weight"]
            final_score = (final_score * (1 - confidence_weight) + 
                          confidence_avg_score * confidence_weight)
        
        # 生成分数明细
        score_breakdown = {
            "raw_weighted_score": total_weighted_score,
            "total_weight": total_weight,
            "confidence_adjusted_score": final_score,
            "timeframe_averages": {
                tf: sum(scores) / len(scores) 
                for tf, scores in timeframe_scores.items()
            },
            "strategy_averages": {
                st: sum(scores) / len(scores) 
                for st, scores in strategy_scores.items()
            },
            "signal_count": len(signals),
            "avg_confidence": sum(s["confidence"] for s in signals) / len(signals)
        }
        
        return final_score, score_breakdown
    
    def _apply_market_adjustments(
        self, 
        base_score: float, 
        market_context: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        应用市场环境调整
        
        Returns:
            (调整后分数, 调整明细)
        """
        adjustments = {}
        adjusted_score = base_score
        
        # 波动率调整
        volatility = market_context.get("volatility", 0.03)
        if volatility > 0.06:  # 高波动率环境
            volatility_factor = self.config["market_adjustments"]["high_volatility_factor"]
            adjusted_score *= volatility_factor
            adjustments["volatility"] = {
                "factor": volatility_factor,
                "reason": f"高波动率 ({volatility:.1%}) 降低信号权重"
            }
        
        # 流动性调整
        liquidity = market_context.get("liquidity_score", 1.0)
        if liquidity < 0.5:  # 低流动性
            liquidity_factor = self.config["market_adjustments"]["low_liquidity_factor"]
            adjusted_score *= liquidity_factor
            adjustments["liquidity"] = {
                "factor": liquidity_factor,
                "reason": f"低流动性 ({liquidity:.2f}) 降低信号权重"
            }
        
        # 趋势一致性奖励
        trend_alignment = market_context.get("trend_alignment", 0.0)
        if abs(trend_alignment) > 0.7:  # 强趋势一致性
            alignment_bonus = self.config["market_adjustments"]["trend_alignment_bonus"]
            if adjusted_score * trend_alignment > 0:  # 信号与趋势同向
                adjusted_score *= (1 + alignment_bonus)
                adjustments["trend_alignment"] = {
                    "bonus": alignment_bonus,
                    "reason": f"信号与趋势一致性高 ({trend_alignment:.2f})"
                }
        
        return adjusted_score, adjustments
    
    def _determine_direction(self, score: float) -> Tuple[TradingDirection, SignalStrength]:
        """
        根据分数决定交易方向和强度
        
        Returns:
            (交易方向, 信号强度)
        """
        thresholds = self.config["signal_thresholds"]
        
        if score >= thresholds["strong_bullish"]:
            return TradingDirection.LONG, SignalStrength.STRONG
        elif score >= thresholds["moderate_bullish"]:
            return TradingDirection.LONG, SignalStrength.MODERATE
        elif score <= thresholds["strong_bearish"]:
            return TradingDirection.SHORT, SignalStrength.STRONG
        elif score <= thresholds["moderate_bearish"]:
            return TradingDirection.SHORT, SignalStrength.MODERATE
        else:
            return TradingDirection.HOLD, SignalStrength.WEAK
    
    def _calculate_decision_confidence(self, score: float) -> float:
        """
        计算决策置信度
        
        Returns:
            置信度百分比 (0-100)
        """
        # 基于分数绝对值计算置信度
        abs_score = abs(score)
        
        if abs_score >= 0.7:
            return min(85 + (abs_score - 0.7) * 50, 95)  # 85-95%
        elif abs_score >= 0.3:
            return 60 + (abs_score - 0.3) * 62.5  # 60-85%
        else:
            return 30 + abs_score * 100  # 30-60%
    
    def get_signal_summary(self, analysis_result: Tuple[TradingDirection, float, Dict[str, Any]]) -> str:
        """
        生成信号分析摘要
        
        Returns:
            人类可读的分析摘要
        """
        direction, score, details = analysis_result
        
        summary_parts = [
            f"交易方向: {direction.value.upper()}",
            f"信号强度: {details.get('signal_strength', 'unknown')}",
            f"信号分数: {score:.3f}",
            f"决策置信度: {details.get('decision_confidence', 0):.1f}%",
            f"信号数量: {details.get('processed_signals_count', 0)}"
        ]
        
        # 添加主要调整因素
        adjustments = details.get('market_adjustments', {})
        if adjustments:
            adj_reasons = [adj.get('reason', '') for adj in adjustments.values()]
            summary_parts.append(f"市场调整: {'; '.join(adj_reasons)}")
        
        return " | ".join(summary_parts)


# 便捷函数
def analyze_trading_signals(
    technical_signals: Dict[str, Any], 
    ticker: str,
    market_context: Optional[Dict[str, Any]] = None
) -> Tuple[str, float, str]:
    """
    便捷的信号分析函数
    
    Returns:
        (交易方向字符串, 信号分数, 分析摘要)
    """
    analyzer = SignalAnalyzer()
    direction, score, details = analyzer.analyze_signals(
        technical_signals, ticker, market_context
    )
    
    summary = analyzer.get_signal_summary((direction, score, details))
    
    return direction.value, score, summary