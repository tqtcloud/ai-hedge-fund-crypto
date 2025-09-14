"""
期货风险管理节点

专门为期货交易设计的风险管理组件：
- 处理 long/short/neutral 信号语义
- 集成期货信号系统
- 保证金风险管理
- 杠杆控制
- 强平风险监控
"""

from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime
from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage

from src.graph.base_node import BaseNode
from src.graph.state import AgentState
from src.graph.utils import show_agent_reasoning

# 导入期货系统组件
from src.futures.signals.futures_signal_system import FuturesSignalSystem
from src.futures.models.data_models import TradingDirection, OperationType, RiskLevel
from src.futures.leverage.leverage_controller import LeverageStrategy
from src.futures.margin.margin_manager import MarginManager
from src.futures.market.market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)


class EnumJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理枚举类型的序列化"""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class FuturesRiskManagementNode(BaseNode):
    """
    期货风险管理节点

    核心功能：
    1. 处理期货信号语义 (long/short/neutral)
    2. 集成期货信号系统决策
    3. 保证金充足性检查
    4. 杠杆风险控制
    5. 仓位大小验证
    6. 强平风险评估
    """

    def __init__(self):
        super().__init__()
        self.node_name = "futures_risk_management"
        logger.info("期货风险管理节点初始化")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        执行期货风险管理决策

        Args:
            state: Agent状态，包含数据和组合信息

        Returns:
            更新的状态字典，包含风险管理决策
        """
        try:
            logger.info("🔍 开始期货风险管理分析...")

            data = state.get("data", {})
            metadata = state.get("metadata", {})

            # 获取期货交易组件
            portfolio = data.get("portfolio", {})
            futures_signal_system = portfolio.get("futures_signal_system")
            margin_manager = portfolio.get("margin_manager")
            market_analyzer = portfolio.get("market_analyzer")

            # 获取基础交易数据
            tickers = data.get("tickers", [])
            primary_interval = data.get("primary_interval")
            analyst_signals = data.get("analyst_signals", {})

            if not futures_signal_system:
                logger.warning("未找到期货信号系统，使用传统风险管理")
                return self._fallback_traditional_risk_management(state)

            # 初始化期货风险分析结果
            risk_analysis_results = {}

            # 为每个交易对进行期货风险分析
            for ticker in tickers:
                try:
                    # 期货风险分析（同步版本）
                    risk_result = self._analyze_futures_risk_for_ticker_sync(
                        ticker=ticker,
                        data=data,
                        futures_signal_system=futures_signal_system,
                        margin_manager=margin_manager,
                        market_analyzer=market_analyzer,
                        analyst_signals=analyst_signals
                    )

                    risk_analysis_results[ticker] = risk_result

                    logger.info(f"✅ {ticker} 期货风险分析完成: 风险等级={risk_result.get('risk_level', 'UNKNOWN')}")

                except Exception as e:
                    logger.error(f"❌ {ticker} 期货风险分析失败: {e}")
                    # 设置安全的默认风险结果
                    risk_analysis_results[ticker] = self._get_safe_default_risk_result(ticker)

            # 生成综合风险管理决策
            comprehensive_risk_decision = self._generate_comprehensive_risk_decision(
                risk_analysis_results, portfolio, margin_manager
            )

            # 构建风险管理消息
            risk_message_content = {
                "futures_risk_analysis": risk_analysis_results,
                "comprehensive_decision": comprehensive_risk_decision,
                "risk_management_summary": self._create_risk_summary(risk_analysis_results),
                "timestamp": datetime.now().isoformat(),
                "node_type": "futures_risk_management"
            }

            message = HumanMessage(
                content=json.dumps(risk_message_content, ensure_ascii=False, indent=2, cls=EnumJSONEncoder),
                name="futures_risk_management_agent"
            )

            # 显示推理过程（如果启用）
            if metadata.get("show_reasoning", False):
                show_agent_reasoning(risk_message_content, "期货风险管理分析师 (Futures Risk Management)")

            # 更新状态
            new_messages = state.get("messages", []) + [message]

            # 将风险决策添加到数据中供下游节点使用
            data["futures_risk_analysis"] = risk_analysis_results
            data["risk_management_decision"] = comprehensive_risk_decision

            return {
                "messages": new_messages,
                "data": data
            }

        except Exception as e:
            logger.error(f"💥 期货风险管理节点执行失败: {e}")
            return self._create_error_response(state, str(e))

    def _analyze_futures_risk_for_ticker_sync(
        self,
        ticker: str,
        data: Dict[str, Any],
        futures_signal_system: FuturesSignalSystem,
        margin_manager: Optional[MarginManager],
        market_analyzer: Optional[MarketAnalyzer],
        analyst_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为单个交易对执行期货风险分析（同步版本）
        """
        try:
            # 准备市场数据
            intervals = data.get("intervals", [])
            market_data = {}
            current_price = None

            for interval in intervals:
                df_key = f"{ticker}_{interval.value}"
                if df_key in data:
                    df = data[df_key]
                    if not df.empty:
                        market_data[interval.value] = df
                        if current_price is None:
                            current_price = float(df['close'].iloc[-1])

            if not market_data or current_price is None:
                logger.warning(f"{ticker} 缺少市场数据，跳过风险分析")
                return self._get_safe_default_risk_result(ticker)

            # 获取保证金状态（同步版本）
            margin_status = None
            if margin_manager:
                try:
                    # 使用同步方法获取保证金状态
                    margin_status = margin_manager.get_margin_status()
                except Exception as e:
                    logger.warning(f"获取保证金状态失败: {e}")

            # 生成期货信号（同步版本，不等待异步调用）
            futures_signal = None
            # 注意：这里暂时不调用异步信号生成，而是从analyst_signals中获取
            if analyst_signals:
                # 尝试从现有的分析师信号中提取期货信号
                for signal_source in analyst_signals.values():
                    if isinstance(signal_source, dict) and ticker in signal_source:
                        ticker_signals = signal_source[ticker]
                        if isinstance(ticker_signals, dict):
                            # 创建简化的期货信号对象
                            futures_signal = self._create_simplified_futures_signal(
                                ticker, ticker_signals, current_price
                            )
                            break

            # 风险因素评估
            risk_factors = self._assess_risk_factors(
                ticker, current_price, market_data, futures_signal,
                margin_status, analyst_signals
            )

            # 综合风险等级
            overall_risk_level = self._calculate_overall_risk_level(risk_factors)

            # 交易决策建议
            trading_decision = self._generate_trading_decision(
                ticker, futures_signal, risk_factors, overall_risk_level, margin_status
            )

            return {
                "ticker": ticker,
                "current_price": current_price,
                "futures_signal": futures_signal.__dict__ if hasattr(futures_signal, '__dict__') else futures_signal,
                "risk_factors": risk_factors,
                "risk_level": overall_risk_level.value,
                "trading_decision": trading_decision,
                "margin_status_summary": self._summarize_margin_status(margin_status),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"期货风险分析失败 {ticker}: {e}")
            return self._get_safe_default_risk_result(ticker)

    def _create_simplified_futures_signal(
        self,
        ticker: str,
        ticker_signals: Dict[str, Any],
        current_price: float
    ) -> Optional[Any]:
        """从分析师信号创建简化的期货信号"""
        try:
            # 创建简化的期货信号对象
            class SimplifiedFuturesSignal:
                def __init__(self, ticker, direction, confidence, leverage, position_size):
                    self.ticker = ticker
                    self.direction = self._convert_to_trading_direction(direction)
                    self.confidence = confidence
                    self.suggested_leverage = leverage
                    self.position_size = position_size

                def _convert_to_trading_direction(self, signal):
                    if isinstance(signal, str):
                        if signal.lower() in ['long', 'bullish', 'buy']:
                            return TradingDirection.LONG
                        elif signal.lower() in ['short', 'bearish', 'sell']:
                            return TradingDirection.SHORT
                        else:
                            return TradingDirection.NEUTRAL
                    return TradingDirection.NEUTRAL

            # 尝试从不同时间框架获取信号
            for timeframe_key, timeframe_data in ticker_signals.items():
                if isinstance(timeframe_data, dict) and 'signal' in timeframe_data:
                    signal = timeframe_data.get('signal', 'neutral')
                    confidence = timeframe_data.get('confidence', 50)

                    # 估算杠杆和仓位大小
                    estimated_leverage = 3.0  # 保守的默认杠杆
                    estimated_position_size = 100.0  # 默认仓位大小

                    return SimplifiedFuturesSignal(
                        ticker=ticker,
                        direction=signal,
                        confidence=confidence,
                        leverage=estimated_leverage,
                        position_size=estimated_position_size
                    )

            return None

        except Exception as e:
            logger.error(f"创建简化期货信号失败: {e}")
            return None

    async def _analyze_futures_risk_for_ticker(
        self,
        ticker: str,
        data: Dict[str, Any],
        futures_signal_system: FuturesSignalSystem,
        margin_manager: Optional[MarginManager],
        market_analyzer: Optional[MarketAnalyzer],
        analyst_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为单个交易对执行期货风险分析

        Args:
            ticker: 交易对符号
            data: 市场数据
            futures_signal_system: 期货信号系统
            margin_manager: 保证金管理器
            market_analyzer: 市场分析器
            analyst_signals: 分析师信号

        Returns:
            该交易对的风险分析结果
        """
        try:
            # 准备市场数据
            intervals = data.get("intervals", [])
            market_data = {}
            current_price = None

            for interval in intervals:
                df_key = f"{ticker}_{interval.value}"
                if df_key in data:
                    df = data[df_key]
                    if not df.empty:
                        market_data[interval.value] = df
                        if current_price is None:
                            current_price = float(df['close'].iloc[-1])

            if not market_data or current_price is None:
                logger.warning(f"{ticker} 缺少市场数据，跳过风险分析")
                return self._get_safe_default_risk_result(ticker)

            # 获取保证金状态
            margin_status = None
            if margin_manager:
                try:
                    await margin_manager.initialize()
                    margin_status = margin_manager.get_margin_status()
                except Exception as e:
                    logger.warning(f"获取保证金状态失败: {e}")

            # 生成期货信号（如果没有现成信号）
            futures_signal = None
            if futures_signal_system:
                try:
                    futures_signal = await futures_signal_system.generate_signal(
                        ticker=ticker,
                        market_data=market_data,
                        current_price=current_price,
                        current_positions=None,  # TODO: 获取实际持仓
                        margin_status=margin_status,
                        leverage_strategy=LeverageStrategy.MODERATE
                    )
                except Exception as e:
                    logger.warning(f"{ticker} 期货信号生成失败: {e}")

            # 风险因素评估
            risk_factors = self._assess_risk_factors(
                ticker, current_price, market_data, futures_signal,
                margin_status, analyst_signals
            )

            # 综合风险等级
            overall_risk_level = self._calculate_overall_risk_level(risk_factors)

            # 交易决策建议
            trading_decision = self._generate_trading_decision(
                ticker, futures_signal, risk_factors, overall_risk_level, margin_status
            )

            return {
                "ticker": ticker,
                "current_price": current_price,
                "futures_signal": futures_signal.__dict__ if futures_signal else None,
                "risk_factors": risk_factors,
                "risk_level": overall_risk_level.value,
                "trading_decision": trading_decision,
                "margin_status_summary": self._summarize_margin_status(margin_status),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"期货风险分析失败 {ticker}: {e}")
            return self._get_safe_default_risk_result(ticker)

    def _assess_risk_factors(
        self,
        ticker: str,
        current_price: float,
        market_data: Dict[str, Any],
        futures_signal: Optional[Any],
        margin_status: Optional[Any],
        analyst_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        """评估各项风险因素"""
        try:
            risk_factors = {
                "market_volatility": self._assess_market_volatility_risk(market_data),
                "leverage_risk": self._assess_leverage_risk(futures_signal, margin_status),
                "margin_risk": self._assess_margin_risk(margin_status),
                "liquidity_risk": self._assess_liquidity_risk(market_data),
                "signal_quality": self._assess_signal_quality_risk(futures_signal, analyst_signals),
                "position_sizing": self._assess_position_sizing_risk(futures_signal, margin_status)
            }

            # 计算综合风险评分
            risk_factors["composite_score"] = self._calculate_composite_risk_score(risk_factors)

            return risk_factors

        except Exception as e:
            logger.error(f"风险因素评估失败: {e}")
            return {"error": str(e), "composite_score": 80.0}  # 高风险作为安全默认值

    def _assess_market_volatility_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估市场波动率风险"""
        try:
            if not market_data:
                return {"score": 60.0, "level": "MEDIUM", "details": "缺少市场数据"}

            # 使用1小时数据分析波动率
            main_timeframe = '1h'
            if main_timeframe in market_data:
                df = market_data[main_timeframe]
                if 'close' in df.columns and len(df) > 20:
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * (24 ** 0.5)  # 日化波动率

                    if volatility > 0.05:  # > 5% 日波动率
                        return {"score": 80.0, "level": "HIGH", "details": f"高波动率: {volatility:.2%}"}
                    elif volatility > 0.03:  # > 3% 日波动率
                        return {"score": 60.0, "level": "MEDIUM", "details": f"中等波动率: {volatility:.2%}"}
                    else:
                        return {"score": 30.0, "level": "LOW", "details": f"低波动率: {volatility:.2%}"}

            return {"score": 50.0, "level": "MEDIUM", "details": "无法计算波动率"}

        except Exception as e:
            return {"score": 70.0, "level": "HIGH", "details": f"波动率分析失败: {e}"}

    def _assess_leverage_risk(self, futures_signal: Optional[Any], margin_status: Optional[Any]) -> Dict[str, Any]:
        """评估杠杆风险"""
        try:
            if not futures_signal:
                return {"score": 50.0, "level": "MEDIUM", "details": "无期货信号"}

            suggested_leverage = getattr(futures_signal, 'suggested_leverage', 1.0)

            if suggested_leverage > 20:
                return {"score": 95.0, "level": "CRITICAL", "details": f"杠杆过高: {suggested_leverage:.1f}x"}
            elif suggested_leverage > 10:
                return {"score": 75.0, "level": "HIGH", "details": f"高杠杆: {suggested_leverage:.1f}x"}
            elif suggested_leverage > 5:
                return {"score": 50.0, "level": "MEDIUM", "details": f"中等杠杆: {suggested_leverage:.1f}x"}
            elif suggested_leverage > 2:
                return {"score": 30.0, "level": "LOW", "details": f"低杠杆: {suggested_leverage:.1f}x"}
            else:
                return {"score": 15.0, "level": "VERY_LOW", "details": f"极低杠杆: {suggested_leverage:.1f}x"}

        except Exception as e:
            return {"score": 60.0, "level": "MEDIUM", "details": f"杠杆风险评估失败: {e}"}

    def _assess_margin_risk(self, margin_status: Optional[Any]) -> Dict[str, Any]:
        """评估保证金风险"""
        try:
            if not margin_status:
                return {"score": 40.0, "level": "MEDIUM", "details": "无保证金状态信息"}

            risk_level = getattr(margin_status, 'risk_level', None)
            margin_ratio = getattr(margin_status, 'margin_ratio', 0.0)

            if hasattr(risk_level, 'value'):
                risk_level_value = risk_level.value
            else:
                risk_level_value = str(risk_level)

            if risk_level_value == 'EMERGENCY':
                return {"score": 95.0, "level": "CRITICAL", "details": f"紧急保证金状态，比例: {margin_ratio:.2f}"}
            elif risk_level_value == 'CRITICAL':
                return {"score": 85.0, "level": "HIGH", "details": f"临界保证金状态，比例: {margin_ratio:.2f}"}
            elif risk_level_value == 'HIGH':
                return {"score": 70.0, "level": "HIGH", "details": f"高风险保证金状态，比例: {margin_ratio:.2f}"}
            elif risk_level_value == 'MEDIUM':
                return {"score": 45.0, "level": "MEDIUM", "details": f"中等保证金状态，比例: {margin_ratio:.2f}"}
            else:
                return {"score": 25.0, "level": "LOW", "details": f"良好保证金状态，比例: {margin_ratio:.2f}"}

        except Exception as e:
            return {"score": 50.0, "level": "MEDIUM", "details": f"保证金风险评估失败: {e}"}

    def _assess_liquidity_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估流动性风险"""
        try:
            if not market_data:
                return {"score": 50.0, "level": "MEDIUM", "details": "缺少市场数据"}

            # 使用成交量数据评估流动性
            for timeframe, df in market_data.items():
                if 'volume' in df.columns and len(df) > 10:
                    recent_volume = df['volume'].tail(5).mean()
                    avg_volume = df['volume'].tail(20).mean()

                    if avg_volume > 0:
                        volume_ratio = recent_volume / avg_volume

                        if volume_ratio > 1.5:
                            return {"score": 20.0, "level": "LOW", "details": f"高流动性，成交量比例: {volume_ratio:.2f}"}
                        elif volume_ratio > 0.8:
                            return {"score": 35.0, "level": "LOW", "details": f"正常流动性，成交量比例: {volume_ratio:.2f}"}
                        elif volume_ratio > 0.5:
                            return {"score": 60.0, "level": "MEDIUM", "details": f"流动性不足，成交量比例: {volume_ratio:.2f}"}
                        else:
                            return {"score": 80.0, "level": "HIGH", "details": f"流动性枯竭，成交量比例: {volume_ratio:.2f}"}

            return {"score": 45.0, "level": "MEDIUM", "details": "无成交量数据"}

        except Exception as e:
            return {"score": 55.0, "level": "MEDIUM", "details": f"流动性风险评估失败: {e}"}

    def _assess_signal_quality_risk(self, futures_signal: Optional[Any], analyst_signals: Dict[str, Any]) -> Dict[str, Any]:
        """评估信号质量风险"""
        try:
            if not futures_signal:
                return {"score": 60.0, "level": "MEDIUM", "details": "无期货信号"}

            confidence = getattr(futures_signal, 'confidence', 50.0)
            direction = getattr(futures_signal, 'direction', None)

            # 检查信号方向是否为期货语义
            if hasattr(direction, 'value'):
                direction_value = direction.value
            else:
                direction_value = str(direction)

            if direction_value not in ['long', 'short', 'neutral']:
                return {"score": 80.0, "level": "HIGH", "details": f"信号语义错误: {direction_value}"}

            # 基于置信度评估
            if confidence >= 80:
                return {"score": 20.0, "level": "LOW", "details": f"高质量信号，置信度: {confidence:.1f}%"}
            elif confidence >= 60:
                return {"score": 40.0, "level": "LOW", "details": f"良好信号，置信度: {confidence:.1f}%"}
            elif confidence >= 40:
                return {"score": 60.0, "level": "MEDIUM", "details": f"中等信号，置信度: {confidence:.1f}%"}
            else:
                return {"score": 80.0, "level": "HIGH", "details": f"低质量信号，置信度: {confidence:.1f}%"}

        except Exception as e:
            return {"score": 70.0, "level": "HIGH", "details": f"信号质量评估失败: {e}"}

    def _assess_position_sizing_risk(self, futures_signal: Optional[Any], margin_status: Optional[Any]) -> Dict[str, Any]:
        """评估仓位大小风险"""
        try:
            if not futures_signal:
                return {"score": 50.0, "level": "MEDIUM", "details": "无期货信号"}

            position_size = getattr(futures_signal, 'position_size', 0.0)

            if not margin_status:
                return {"score": 55.0, "level": "MEDIUM", "details": f"仓位大小: {position_size:.2f}，无保证金信息"}

            available_margin = getattr(margin_status, 'available_margin', 0.0)

            if available_margin > 0 and position_size > 0:
                position_margin_ratio = position_size / available_margin

                if position_margin_ratio > 0.5:  # 超过50%保证金
                    return {"score": 85.0, "level": "HIGH", "details": f"仓位过大，占用保证金: {position_margin_ratio:.1%}"}
                elif position_margin_ratio > 0.3:  # 超过30%保证金
                    return {"score": 60.0, "level": "MEDIUM", "details": f"仓位适中，占用保证金: {position_margin_ratio:.1%}"}
                else:
                    return {"score": 30.0, "level": "LOW", "details": f"仓位合理，占用保证金: {position_margin_ratio:.1%}"}

            return {"score": 40.0, "level": "MEDIUM", "details": "无法计算仓位保证金比例"}

        except Exception as e:
            return {"score": 65.0, "level": "MEDIUM", "details": f"仓位大小风险评估失败: {e}"}

    def _calculate_composite_risk_score(self, risk_factors: Dict[str, Any]) -> float:
        """计算综合风险评分"""
        try:
            # 定义各风险因素权重
            weights = {
                "market_volatility": 0.20,
                "leverage_risk": 0.25,
                "margin_risk": 0.25,
                "liquidity_risk": 0.15,
                "signal_quality": 0.10,
                "position_sizing": 0.05
            }

            total_score = 0.0
            total_weight = 0.0

            for factor, weight in weights.items():
                if factor in risk_factors and isinstance(risk_factors[factor], dict):
                    score = risk_factors[factor].get('score', 50.0)
                    total_score += score * weight
                    total_weight += weight

            if total_weight > 0:
                return total_score / total_weight
            else:
                return 50.0

        except Exception as e:
            logger.error(f"综合风险评分计算失败: {e}")
            return 75.0  # 高风险默认值

    def _calculate_overall_risk_level(self, risk_factors: Dict[str, Any]) -> RiskLevel:
        """计算整体风险等级"""
        try:
            composite_score = risk_factors.get("composite_score", 50.0)

            if composite_score >= 80:
                return RiskLevel.EMERGENCY
            elif composite_score >= 65:
                return RiskLevel.CRITICAL
            elif composite_score >= 50:
                return RiskLevel.HIGH
            elif composite_score >= 30:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except Exception as e:
            logger.error(f"风险等级计算失败: {e}")
            return RiskLevel.HIGH

    def _generate_trading_decision(
        self,
        ticker: str,
        futures_signal: Optional[Any],
        risk_factors: Dict[str, Any],
        overall_risk_level: RiskLevel,
        margin_status: Optional[Any]
    ) -> Dict[str, Any]:
        """生成交易决策建议"""
        try:
            decision = {
                "action": "HOLD",  # OPEN_LONG, OPEN_SHORT, CLOSE, HOLD, REJECT
                "confidence": 50.0,
                "risk_adjusted_size": 0.0,
                "max_leverage": 1.0,
                "reasons": [],
                "warnings": []
            }

            # 如果整体风险过高，拒绝交易
            if overall_risk_level in [RiskLevel.EMERGENCY, RiskLevel.CRITICAL]:
                decision["action"] = "REJECT"
                decision["reasons"].append(f"整体风险等级过高: {overall_risk_level.value}")
                return decision

            # 如果没有期货信号，保持持仓
            if not futures_signal:
                decision["reasons"].append("缺少期货交易信号")
                return decision

            # 解析期货信号
            signal_direction = getattr(futures_signal, 'direction', None)
            signal_confidence = getattr(futures_signal, 'confidence', 50.0)
            suggested_leverage = getattr(futures_signal, 'suggested_leverage', 1.0)
            position_size = getattr(futures_signal, 'position_size', 0.0)

            if hasattr(signal_direction, 'value'):
                direction_value = signal_direction.value
            else:
                direction_value = str(signal_direction)

            # 基于信号方向决定行动
            if direction_value == 'long' and signal_confidence > 60:
                decision["action"] = "OPEN_LONG"
                decision["confidence"] = signal_confidence
            elif direction_value == 'short' and signal_confidence > 60:
                decision["action"] = "OPEN_SHORT"
                decision["confidence"] = signal_confidence
            elif direction_value == 'neutral':
                decision["action"] = "CLOSE"
                decision["confidence"] = signal_confidence
                decision["reasons"].append("信号为中性，建议平仓")

            # 风险调整
            composite_risk = risk_factors.get("composite_score", 50.0)

            # 调整仓位大小
            risk_adjustment_factor = max(0.1, min(1.0, (100 - composite_risk) / 100))
            decision["risk_adjusted_size"] = position_size * risk_adjustment_factor

            # 调整杠杆
            max_safe_leverage = min(suggested_leverage, 20.0)  # 硬性限制20x
            if overall_risk_level == RiskLevel.HIGH:
                max_safe_leverage = min(max_safe_leverage, 5.0)
            elif overall_risk_level == RiskLevel.MEDIUM:
                max_safe_leverage = min(max_safe_leverage, 10.0)

            decision["max_leverage"] = max_safe_leverage

            # 添加具体原因
            decision["reasons"].append(f"期货信号方向: {direction_value}, 置信度: {signal_confidence:.1f}%")
            decision["reasons"].append(f"综合风险评分: {composite_risk:.1f}")
            decision["reasons"].append(f"风险调整系数: {risk_adjustment_factor:.2f}")

            # 添加警告
            if composite_risk > 70:
                decision["warnings"].append("高风险环境，建议谨慎操作")
            if suggested_leverage > max_safe_leverage:
                decision["warnings"].append(f"建议杠杆已从{suggested_leverage:.1f}x调整为{max_safe_leverage:.1f}x")

            return decision

        except Exception as e:
            logger.error(f"交易决策生成失败: {e}")
            return {
                "action": "REJECT",
                "confidence": 0.0,
                "risk_adjusted_size": 0.0,
                "max_leverage": 1.0,
                "reasons": [f"决策生成失败: {e}"],
                "warnings": ["系统错误，建议人工检查"]
            }

    def _summarize_margin_status(self, margin_status: Optional[Any]) -> Dict[str, Any]:
        """汇总保证金状态"""
        if not margin_status:
            return {"status": "unknown", "details": "无保证金信息"}

        try:
            return {
                "available_margin": getattr(margin_status, 'available_margin', 0.0),
                "used_margin": getattr(margin_status, 'used_margin', 0.0),
                "margin_ratio": getattr(margin_status, 'margin_ratio', 0.0),
                "risk_level": getattr(margin_status, 'risk_level', RiskLevel.MEDIUM).value,
                "can_trade": getattr(margin_status, 'can_trade', False),
                "can_open_position": getattr(margin_status, 'can_open_position', False)
            }
        except Exception as e:
            return {"status": "error", "details": f"保证金状态解析失败: {e}"}

    def _generate_comprehensive_risk_decision(
        self,
        risk_analysis_results: Dict[str, Any],
        portfolio: Dict[str, Any],
        margin_manager: Optional[MarginManager]
    ) -> Dict[str, Any]:
        """生成综合风险管理决策"""
        try:
            # 汇总所有交易对的风险情况
            total_tickers = len(risk_analysis_results)
            high_risk_count = 0
            trading_decisions = {}
            overall_warnings = []

            for ticker, result in risk_analysis_results.items():
                risk_level = result.get("risk_level", "HIGH")
                if risk_level in ["CRITICAL", "EMERGENCY", "HIGH"]:
                    high_risk_count += 1

                trading_decision = result.get("trading_decision", {})
                trading_decisions[ticker] = trading_decision

            # 计算整体风险比例
            high_risk_ratio = high_risk_count / total_tickers if total_tickers > 0 else 0

            # 整体决策
            if high_risk_ratio > 0.7:
                overall_action = "SUSPEND_TRADING"
                overall_warnings.append(f"超过70%的交易对处于高风险状态 ({high_risk_count}/{total_tickers})")
            elif high_risk_ratio > 0.4:
                overall_action = "REDUCE_EXPOSURE"
                overall_warnings.append(f"40%以上交易对处于高风险状态 ({high_risk_count}/{total_tickers})")
            else:
                overall_action = "NORMAL_TRADING"

            return {
                "overall_action": overall_action,
                "high_risk_ratio": high_risk_ratio,
                "high_risk_count": high_risk_count,
                "total_tickers": total_tickers,
                "trading_decisions": trading_decisions,
                "overall_warnings": overall_warnings,
                "portfolio_summary": {
                    "total_cash": portfolio.get("cash", 0.0),
                    "margin_used": portfolio.get("margin_used", 0.0),
                    "futures_mode": True
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"综合风险决策生成失败: {e}")
            return {
                "overall_action": "SUSPEND_TRADING",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _create_risk_summary(self, risk_analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """创建风险分析摘要"""
        try:
            summary = {
                "total_analyzed": len(risk_analysis_results),
                "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0, "EMERGENCY": 0},
                "average_composite_score": 0.0,
                "key_risks": [],
                "recommendations": []
            }

            total_score = 0.0
            for ticker, result in risk_analysis_results.items():
                risk_level = result.get("risk_level", "HIGH")
                summary["risk_distribution"][risk_level] = summary["risk_distribution"].get(risk_level, 0) + 1

                risk_factors = result.get("risk_factors", {})
                composite_score = risk_factors.get("composite_score", 50.0)
                total_score += composite_score

            # 计算平均风险评分
            if summary["total_analyzed"] > 0:
                summary["average_composite_score"] = total_score / summary["total_analyzed"]

            # 生成关键风险和建议
            if summary["risk_distribution"]["EMERGENCY"] > 0:
                summary["key_risks"].append("存在紧急风险状况")
                summary["recommendations"].append("立即停止交易并检查系统")

            if summary["risk_distribution"]["CRITICAL"] > 0:
                summary["key_risks"].append("存在临界风险状况")
                summary["recommendations"].append("暂停开新仓，考虑减仓")

            if summary["average_composite_score"] > 70:
                summary["key_risks"].append("整体风险评分偏高")
                summary["recommendations"].append("降低杠杆和仓位规模")

            return summary

        except Exception as e:
            logger.error(f"风险摘要创建失败: {e}")
            return {"error": str(e)}

    def _get_safe_default_risk_result(self, ticker: str) -> Dict[str, Any]:
        """获取安全的默认风险结果"""
        return {
            "ticker": ticker,
            "current_price": 0.0,
            "futures_signal": None,
            "risk_factors": {"composite_score": 80.0},  # 高风险默认值
            "risk_level": "HIGH",
            "trading_decision": {
                "action": "REJECT",
                "confidence": 0.0,
                "risk_adjusted_size": 0.0,
                "max_leverage": 1.0,
                "reasons": ["数据不足，安全起见拒绝交易"],
                "warnings": ["缺少必要的风险分析数据"]
            },
            "margin_status_summary": {"status": "unknown"},
            "timestamp": datetime.now().isoformat(),
            "error": "使用安全默认值"
        }

    def _fallback_traditional_risk_management(self, state: AgentState) -> Dict[str, Any]:
        """回退到传统风险管理"""
        logger.warning("回退到传统风险管理模式")

        # 创建基本的风险管理消息
        risk_message = HumanMessage(
            content=json.dumps({
                "risk_management_type": "traditional_fallback",
                "action": "HOLD",
                "reason": "期货风险管理系统不可用，回退到传统模式",
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False, cls=EnumJSONEncoder),
            name="traditional_risk_management_agent"
        )

        return {
            "messages": state.get("messages", []) + [risk_message],
            "data": state.get("data", {})
        }

    def _create_error_response(self, state: AgentState, error_message: str) -> Dict[str, Any]:
        """创建错误响应"""
        error_message_obj = HumanMessage(
            content=json.dumps({
                "error": error_message,
                "risk_management_status": "failed",
                "action": "SUSPEND_TRADING",
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False, cls=EnumJSONEncoder),
            name="futures_risk_management_error"
        )

        return {
            "messages": state.get("messages", []) + [error_message_obj],
            "data": state.get("data", {})
        }