"""
期货信号格式转换适配器

解决期货信号系统与投资组合管理节点之间的格式兼容性问题
将期货信号系统的输出格式转换为投资组合管理节点期望的格式
"""

from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging

from ..models.data_models import (
    FuturesSignal, TradingDirection, OperationType, RiskLevel
)
from .futures_signal_system import PositionOperation

logger = logging.getLogger(__name__)


class FuturesSignalFormatAdapter:
    """
    期货信号格式转换适配器

    负责将期货信号系统的各种输出格式统一转换为
    投资组合管理节点能识别的标准格式
    """

    def __init__(self):
        """初始化适配器"""
        self.adapter_name = "futures_signal_format_adapter"

        # 定义信号映射规则
        self._position_operation_mapping = {
            # 期货信号系统PositionOperation到投资组合action的映射
            PositionOperation.OPEN_LONG: "OPEN_LONG",
            PositionOperation.OPEN_SHORT: "OPEN_SHORT",
            PositionOperation.ADD_LONG: "OPEN_LONG",      # 加仓视为开仓
            PositionOperation.ADD_SHORT: "OPEN_SHORT",    # 加仓视为开仓
            PositionOperation.REDUCE_LONG: "CLOSE",       # 减仓视为平仓
            PositionOperation.REDUCE_SHORT: "CLOSE",      # 减仓视为平仓
            PositionOperation.CLOSE_LONG: "CLOSE",
            PositionOperation.CLOSE_SHORT: "CLOSE",
            PositionOperation.REVERSE_TO_LONG: "OPEN_LONG",
            PositionOperation.REVERSE_TO_SHORT: "OPEN_SHORT",
            PositionOperation.HOLD: "HOLD"
        }

        # 定义描述性文本映射规则
        self._text_pattern_mapping = {
            # 处理描述性文本格式（如"short + open"）
            "short + open": "OPEN_SHORT",
            "long + open": "OPEN_LONG",
            "short + close": "CLOSE",
            "long + close": "CLOSE",
            "open short": "OPEN_SHORT",
            "open long": "OPEN_LONG",
            "close short": "CLOSE",
            "close long": "CLOSE",
            "short": "OPEN_SHORT",  # 默认短头为开空
            "long": "OPEN_LONG",    # 默认多头为开多
            "hold": "HOLD",
            "neutral": "HOLD"
        }

        logger.info(f"🔧 {self.adapter_name} 初始化完成")

    def adapt_futures_signal_to_portfolio_action(
        self,
        signal_data: Union[FuturesSignal, Dict[str, Any], str],
        ticker: str = None
    ) -> Dict[str, Any]:
        """
        将期货信号转换为投资组合管理节点期望的action格式

        Args:
            signal_data: 期货信号数据（可以是FuturesSignal对象、字典或字符串）
            ticker: 交易对符号

        Returns:
            投资组合管理节点期望的格式化信号
        """
        try:
            logger.info(f"🔄 开始转换期货信号格式: {type(signal_data)}")

            # 处理不同类型的输入
            if isinstance(signal_data, FuturesSignal):
                return self._adapt_futures_signal_object(signal_data)
            elif isinstance(signal_data, dict):
                return self._adapt_signal_dict(signal_data, ticker)
            elif isinstance(signal_data, str):
                return self._adapt_signal_string(signal_data, ticker)
            elif hasattr(signal_data, 'direction') and hasattr(signal_data, 'ticker'):
                # 处理SimplifiedFuturesSignal或其他类似对象
                return self._adapt_simplified_futures_signal(signal_data)
            else:
                logger.warning(f"⚠️ 未知的信号格式类型: {type(signal_data)}")
                return self._create_default_action(ticker)

        except Exception as e:
            logger.error(f"❌ 信号格式转换失败: {e}")
            return self._create_error_action(ticker, str(e))

    def _adapt_futures_signal_object(self, signal: FuturesSignal) -> Dict[str, Any]:
        """转换FuturesSignal对象"""
        try:
            # 根据方向和操作类型确定action
            if signal.direction == TradingDirection.LONG:
                if signal.operation_type == OperationType.OPEN:
                    action = "OPEN_LONG"
                elif signal.operation_type == OperationType.CLOSE:
                    action = "CLOSE"
                else:
                    action = "HOLD"
            elif signal.direction == TradingDirection.SHORT:
                if signal.operation_type == OperationType.OPEN:
                    action = "OPEN_SHORT"
                elif signal.operation_type == OperationType.CLOSE:
                    action = "CLOSE"
                else:
                    action = "HOLD"
            else:  # NEUTRAL
                action = "HOLD"

            return {
                "action": action,
                "ticker": signal.ticker,
                "confidence": signal.confidence,
                "risk_adjusted_size": signal.position_size or 0.0,
                "max_leverage": signal.suggested_leverage,
                "entry_price": signal.entry_price,
                "take_profit_price": signal.take_profit_price,
                "stop_loss_price": signal.stop_loss_price,
                "signal_strength": signal.strength,
                "risk_level": signal.risk_level.value,
                "timestamp": signal.timestamp.isoformat(),
                "reasons": [f"期货信号: {signal.direction.value} {signal.operation_type.value}"],
                "metadata": {
                    "signal_source": "futures_signal_system",
                    "strategy_source": signal.strategy_source,
                    "signal_id": signal.signal_id,
                    "original_signal": signal.to_dict()
                }
            }

        except Exception as e:
            logger.error(f"❌ FuturesSignal对象转换失败: {e}")
            return self._create_error_action(signal.ticker, str(e))

    def _adapt_simplified_futures_signal(self, signal) -> Dict[str, Any]:
        """转换SimplifiedFuturesSignal或类似对象"""
        try:
            # 从direction确定action（由于SimplifiedFuturesSignal缺少operation_type，默认为开仓操作）
            if hasattr(signal, 'direction'):
                if hasattr(signal.direction, 'value'):
                    # 如果direction是枚举类型
                    direction_value = signal.direction.value.lower()
                else:
                    # 如果direction是字符串
                    direction_value = str(signal.direction).lower()

                if direction_value in ['short', 'bearish', 'sell']:
                    action = "OPEN_SHORT"
                elif direction_value in ['long', 'bullish', 'buy']:
                    action = "OPEN_LONG"
                else:
                    action = "HOLD"
            else:
                action = "HOLD"

            return {
                "action": action,
                "ticker": getattr(signal, 'ticker', 'UNKNOWN'),
                "confidence": getattr(signal, 'confidence', 50.0),
                "risk_adjusted_size": getattr(signal, 'position_size', 0.0),
                "max_leverage": getattr(signal, 'suggested_leverage', 1.0),
                "entry_price": None,
                "take_profit_price": None,
                "stop_loss_price": None,
                "signal_strength": 0.6,  # 默认强度
                "risk_level": "MEDIUM",
                "timestamp": datetime.now().isoformat(),
                "reasons": [f"简化期货信号: {direction_value} -> {action}"],
                "metadata": {
                    "signal_source": "simplified_futures_signal",
                    "signal_type": str(type(signal)),
                    "original_direction": direction_value
                }
            }

        except Exception as e:
            logger.error(f"❌ SimplifiedFuturesSignal转换失败: {e}")
            return self._create_error_action(getattr(signal, 'ticker', 'UNKNOWN'), str(e))

    def _adapt_signal_dict(self, signal_dict: Dict[str, Any], ticker: str) -> Dict[str, Any]:
        """转换信号字典"""
        try:
            # 尝试从字典中提取action信息
            action = "HOLD"

            # 检查是否包含PositionOperation
            if "operation" in signal_dict:
                operation = signal_dict["operation"]
                if isinstance(operation, PositionOperation):
                    action = self._position_operation_mapping.get(operation, "HOLD")
                elif isinstance(operation, str):
                    action = self._map_text_to_action(operation)

            # 检查是否包含方向和操作信息
            elif "direction" in signal_dict and "operation_type" in signal_dict:
                direction = signal_dict["direction"]
                operation_type = signal_dict["operation_type"]

                if direction in ["long", "LONG"] and operation_type in ["open", "OPEN"]:
                    action = "OPEN_LONG"
                elif direction in ["short", "SHORT"] and operation_type in ["open", "OPEN"]:
                    action = "OPEN_SHORT"
                elif operation_type in ["close", "CLOSE"]:
                    action = "CLOSE"

            # 检查是否包含简单的action字段
            elif "action" in signal_dict:
                raw_action = signal_dict["action"]
                action = self._map_text_to_action(str(raw_action))

            return {
                "action": action,
                "ticker": ticker or signal_dict.get("ticker", "UNKNOWN"),
                "confidence": signal_dict.get("confidence", 50.0),
                "risk_adjusted_size": signal_dict.get("position_size", 0.0),
                "max_leverage": signal_dict.get("suggested_leverage", 1.0),
                "entry_price": signal_dict.get("entry_price"),
                "take_profit_price": signal_dict.get("take_profit_price"),
                "stop_loss_price": signal_dict.get("stop_loss_price"),
                "signal_strength": signal_dict.get("strength", 0.5),
                "risk_level": signal_dict.get("risk_level", "MEDIUM"),
                "timestamp": signal_dict.get("timestamp", datetime.now().isoformat()),
                "reasons": signal_dict.get("reasons", [f"信号转换: {action}"]),
                "metadata": {
                    "signal_source": "signal_dict_adapter",
                    "original_signal": signal_dict
                }
            }

        except Exception as e:
            logger.error(f"❌ 信号字典转换失败: {e}")
            return self._create_error_action(ticker, str(e))

    def _adapt_signal_string(self, signal_str: str, ticker: str) -> Dict[str, Any]:
        """转换信号字符串（如'short + open'）"""
        try:
            # 规范化字符串
            normalized_signal = signal_str.lower().strip()

            # 映射到action
            action = self._map_text_to_action(normalized_signal)

            return {
                "action": action,
                "ticker": ticker or "UNKNOWN",
                "confidence": 50.0,  # 字符串信号默认置信度
                "risk_adjusted_size": 0.0,
                "max_leverage": 1.0,
                "signal_strength": 0.5,
                "risk_level": "MEDIUM",
                "timestamp": datetime.now().isoformat(),
                "reasons": [f"文本信号转换: '{signal_str}' -> {action}"],
                "metadata": {
                    "signal_source": "text_signal_adapter",
                    "original_signal": signal_str
                }
            }

        except Exception as e:
            logger.error(f"❌ 信号字符串转换失败: {e}")
            return self._create_error_action(ticker, str(e))

    def _map_text_to_action(self, text: str) -> str:
        """将文本映射到action"""
        text = text.lower().strip()

        # 直接匹配
        if text in self._text_pattern_mapping:
            return self._text_pattern_mapping[text]

        # 模糊匹配
        for pattern, action in self._text_pattern_mapping.items():
            if pattern in text:
                return action

        # 特殊处理"short + open"格式
        if "short" in text and "open" in text:
            return "OPEN_SHORT"
        elif "long" in text and "open" in text:
            return "OPEN_LONG"
        elif "close" in text:
            return "CLOSE"
        elif "short" in text:
            return "OPEN_SHORT"
        elif "long" in text:
            return "OPEN_LONG"

        logger.warning(f"⚠️ 无法映射文本信号: {text}，默认为HOLD")
        return "HOLD"

    def _create_default_action(self, ticker: str) -> Dict[str, Any]:
        """创建默认action"""
        return {
            "action": "HOLD",
            "ticker": ticker or "UNKNOWN",
            "confidence": 0.0,
            "risk_adjusted_size": 0.0,
            "max_leverage": 1.0,
            "signal_strength": 0.0,
            "risk_level": "LOW",
            "timestamp": datetime.now().isoformat(),
            "reasons": ["默认持仓信号"],
            "metadata": {
                "signal_source": "default_adapter",
                "adapter_reason": "unknown_signal_format"
            }
        }

    def _create_error_action(self, ticker: str, error_msg: str) -> Dict[str, Any]:
        """创建错误action"""
        return {
            "action": "REJECT",
            "ticker": ticker or "UNKNOWN",
            "confidence": 0.0,
            "risk_adjusted_size": 0.0,
            "max_leverage": 1.0,
            "signal_strength": 0.0,
            "risk_level": "HIGH",
            "timestamp": datetime.now().isoformat(),
            "reasons": [f"信号转换错误: {error_msg}"],
            "metadata": {
                "signal_source": "error_adapter",
                "error_message": error_msg
            }
        }

    def validate_converted_signal(self, converted_signal: Dict[str, Any]) -> bool:
        """验证转换后的信号格式"""
        try:
            required_fields = ["action", "ticker", "confidence"]
            for field in required_fields:
                if field not in converted_signal:
                    logger.error(f"❌ 转换后信号缺少必需字段: {field}")
                    return False

            valid_actions = ["OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD", "REJECT"]
            if converted_signal["action"] not in valid_actions:
                logger.error(f"❌ 无效的action值: {converted_signal['action']}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ 信号验证失败: {e}")
            return False


# 工厂函数
def create_signal_adapter() -> FuturesSignalFormatAdapter:
    """创建信号格式适配器实例"""
    return FuturesSignalFormatAdapter()


# 便捷函数
def adapt_signal(
    signal_data: Union[FuturesSignal, Dict[str, Any], str],
    ticker: str = None
) -> Dict[str, Any]:
    """
    快速转换信号格式的便捷函数

    Args:
        signal_data: 需要转换的信号数据
        ticker: 交易对符号

    Returns:
        转换后的信号格式
    """
    adapter = create_signal_adapter()
    return adapter.adapt_futures_signal_to_portfolio_action(signal_data, ticker)