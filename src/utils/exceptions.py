"""
合约交易异常处理模块

定义合约交易过程中可能出现的各种异常类型，
提供详细的错误信息和恢复建议。
"""
from typing import Optional, Dict, Any


class ContractTradingError(Exception):
    """
    合约交易基础异常类
    
    所有合约交易相关异常的基类，提供统一的异常处理接口
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN",
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None
    ):
        """
        初始化合约交易异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            context: 错误上下文信息
            recovery_suggestion: 恢复建议
        """
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        
    def get_context(self) -> Dict[str, Any]:
        """获取错误上下文信息"""
        return self.context
        
    def get_recovery_suggestion(self) -> Optional[str]:
        """获取恢复建议"""
        return self.recovery_suggestion


class MarginInsufficientError(ContractTradingError):
    """
    保证金不足异常
    
    当账户保证金不足以支持当前交易时抛出
    """
    
    def __init__(
        self,
        required_margin: float,
        available_margin: float,
        ticker: str,
        position_size: float,
        leverage: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        初始化保证金不足异常
        
        Args:
            required_margin: 需要的保证金
            available_margin: 可用保证金
            ticker: 交易对符号
            position_size: 仓位大小
            leverage: 杠杆倍数
            context: 额外上下文信息
        """
        shortage = required_margin - available_margin
        message = (
            f"保证金不足: 需要 {required_margin:.2f} USDT，"
            f"可用 {available_margin:.2f} USDT，"
            f"缺口 {shortage:.2f} USDT"
        )
        
        error_context = {
            "ticker": ticker,
            "required_margin": required_margin,
            "available_margin": available_margin,
            "shortage": shortage,
            "position_size": position_size,
            "leverage": leverage
        }
        if context:
            error_context.update(context)
            
        recovery_suggestion = (
            f"建议: 1) 降低仓位大小至 {(available_margin / required_margin * position_size):.2f}，"
            f"2) 降低杠杆倍数至 {(available_margin / required_margin * leverage):.1f}x，"
            f"3) 追加保证金 {shortage:.2f} USDT"
        )
        
        super().__init__(
            message=message,
            error_code="MARGIN_INSUFFICIENT",
            context=error_context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.required_margin = required_margin
        self.available_margin = available_margin
        self.shortage = shortage
        self.ticker = ticker
        self.position_size = position_size
        self.leverage = leverage


class LeverageExceedsLimitError(ContractTradingError):
    """
    杠杆倍数超限异常
    
    当杠杆倍数超过交易所或风险管理限制时抛出
    """
    
    def __init__(
        self,
        requested_leverage: float,
        max_allowed_leverage: float,
        ticker: str,
        reason: str = "exchange_limit",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        初始化杠杆超限异常
        
        Args:
            requested_leverage: 请求的杠杆倍数
            max_allowed_leverage: 最大允许杠杆倍数
            ticker: 交易对符号
            reason: 超限原因 (exchange_limit, risk_limit, volatility_limit)
            context: 额外上下文信息
        """
        message = (
            f"杠杆倍数超限: 请求 {requested_leverage}x，"
            f"最大允许 {max_allowed_leverage}x ({reason})"
        )
        
        error_context = {
            "ticker": ticker,
            "requested_leverage": requested_leverage,
            "max_allowed_leverage": max_allowed_leverage,
            "reason": reason
        }
        if context:
            error_context.update(context)
            
        recovery_suggestion = (
            f"建议: 将杠杆倍数调整至 {max_allowed_leverage}x 或更低"
        )
        
        super().__init__(
            message=message,
            error_code="LEVERAGE_EXCEEDS_LIMIT",
            context=error_context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.requested_leverage = requested_leverage
        self.max_allowed_leverage = max_allowed_leverage
        self.ticker = ticker
        self.reason = reason


class LiquidationRiskError(ContractTradingError):
    """
    强制平仓风险异常
    
    当持仓面临强制平仓风险时抛出
    """
    
    def __init__(
        self,
        current_price: float,
        liquidation_price: float,
        distance_to_liquidation: float,
        risk_level: str,
        ticker: str,
        position_info: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ):
        """
        初始化强平风险异常
        
        Args:
            current_price: 当前价格
            liquidation_price: 强平价格
            distance_to_liquidation: 距离强平的距离 (百分比)
            risk_level: 风险等级 (high, critical, emergency)
            ticker: 交易对符号
            position_info: 持仓信息
            context: 额外上下文信息
        """
        message = (
            f"强平风险 ({risk_level}): 当前价格 {current_price:.4f}，"
            f"强平价格 {liquidation_price:.4f}，"
            f"距离强平 {distance_to_liquidation:.2f}%"
        )
        
        error_context = {
            "ticker": ticker,
            "current_price": current_price,
            "liquidation_price": liquidation_price,
            "distance_to_liquidation": distance_to_liquidation,
            "risk_level": risk_level,
            "position_info": position_info
        }
        if context:
            error_context.update(context)
            
        # 根据风险等级提供不同的恢复建议
        if risk_level == "emergency":
            recovery_suggestion = "紧急操作: 立即平仓或大幅降低杠杆"
        elif risk_level == "critical":
            recovery_suggestion = "关键操作: 设置紧急止损或降低仓位50%"
        else:  # high
            recovery_suggestion = "高风险警告: 考虑降低仓位或增加保证金"
            
        super().__init__(
            message=message,
            error_code="LIQUIDATION_RISK",
            context=error_context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.current_price = current_price
        self.liquidation_price = liquidation_price
        self.distance_to_liquidation = distance_to_liquidation
        self.risk_level = risk_level
        self.ticker = ticker
        self.position_info = position_info


class PositionSizeError(ContractTradingError):
    """
    仓位大小异常
    
    当计算的仓位大小不合理时抛出
    """
    
    def __init__(
        self,
        position_size: float,
        issue_type: str,
        ticker: str,
        min_size: Optional[float] = None,
        max_size: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        初始化仓位大小异常
        
        Args:
            position_size: 计算的仓位大小
            issue_type: 问题类型 (too_small, too_large, invalid)
            ticker: 交易对符号
            min_size: 最小仓位大小
            max_size: 最大仓位大小
            context: 额外上下文信息
        """
        if issue_type == "too_small":
            message = f"仓位过小: {position_size:.6f}，最小要求 {min_size:.6f}"
        elif issue_type == "too_large":
            message = f"仓位过大: {position_size:.2f}，最大允许 {max_size:.2f}"
        else:  # invalid
            message = f"仓位大小无效: {position_size}"
            
        error_context = {
            "ticker": ticker,
            "position_size": position_size,
            "issue_type": issue_type,
            "min_size": min_size,
            "max_size": max_size
        }
        if context:
            error_context.update(context)
            
        if issue_type == "too_small":
            recovery_suggestion = f"建议: 增加仓位至最小值 {min_size:.6f} 或取消交易"
        elif issue_type == "too_large":
            recovery_suggestion = f"建议: 降低仓位至最大值 {max_size:.2f}"
        else:
            recovery_suggestion = "建议: 重新计算仓位大小"
            
        super().__init__(
            message=message,
            error_code="POSITION_SIZE_ERROR",
            context=error_context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.position_size = position_size
        self.issue_type = issue_type
        self.ticker = ticker
        self.min_size = min_size
        self.max_size = max_size


class OrderExecutionError(ContractTradingError):
    """
    订单执行异常
    
    当订单执行失败或参数无效时抛出
    """
    
    def __init__(
        self,
        order_params: Dict[str, Any],
        failure_reason: str,
        ticker: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        初始化订单执行异常
        
        Args:
            order_params: 订单参数
            failure_reason: 失败原因
            ticker: 交易对符号
            context: 额外上下文信息
        """
        message = f"订单执行失败: {failure_reason}"
        
        error_context = {
            "ticker": ticker,
            "order_params": order_params,
            "failure_reason": failure_reason
        }
        if context:
            error_context.update(context)
            
        recovery_suggestion = "建议: 检查订单参数并重新提交，或联系技术支持"
        
        super().__init__(
            message=message,
            error_code="ORDER_EXECUTION_ERROR",
            context=error_context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.order_params = order_params
        self.failure_reason = failure_reason
        self.ticker = ticker


class RiskLimitExceededError(ContractTradingError):
    """
    风险限制超出异常
    
    当交易违反风险管理规则时抛出
    """
    
    def __init__(
        self,
        risk_type: str,
        current_value: float,
        limit_value: float,
        ticker: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        初始化风险限制异常
        
        Args:
            risk_type: 风险类型 (drawdown, var, exposure, correlation)
            current_value: 当前值
            limit_value: 限制值
            ticker: 交易对符号
            context: 额外上下文信息
        """
        message = (
            f"风险限制超出 ({risk_type}): 当前值 {current_value:.4f}，"
            f"限制值 {limit_value:.4f}"
        )
        
        error_context = {
            "ticker": ticker,
            "risk_type": risk_type,
            "current_value": current_value,
            "limit_value": limit_value
        }
        if context:
            error_context.update(context)
            
        recovery_suggestion = f"建议: 降低{risk_type}风险至限制值以下"
        
        super().__init__(
            message=message,
            error_code="RISK_LIMIT_EXCEEDED",
            context=error_context,
            recovery_suggestion=recovery_suggestion
        )
        
        self.risk_type = risk_type
        self.current_value = current_value
        self.limit_value = limit_value
        self.ticker = ticker