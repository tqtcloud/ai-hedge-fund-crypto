"""
合约交易错误恢复机制

提供自动错误恢复和参数调整功能，确保在遇到异常时
能够自动降级并继续执行交易策略。
"""
import logging
from typing import Dict, Any, Tuple, Optional
from .exceptions import (
    ContractTradingError,
    MarginInsufficientError,
    LeverageExceedsLimitError,
    LiquidationRiskError,
    PositionSizeError,
    RiskLimitExceededError
)

logger = logging.getLogger(__name__)


class ErrorRecoveryManager:
    """
    错误恢复管理器
    
    负责处理各种合约交易异常，并提供自动恢复机制
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化错误恢复管理器
        
        Args:
            config: 恢复策略配置
        """
        self.config = config or self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认恢复策略配置"""
        return {
            "margin_insufficient": {
                "max_retry_attempts": 3,
                "position_reduction_factor": 0.8,
                "leverage_reduction_factor": 0.8,
                "min_position_size": 10.0
            },
            "leverage_exceeds_limit": {
                "max_retry_attempts": 2,
                "leverage_reduction_step": 1,
                "min_leverage": 1
            },
            "liquidation_risk": {
                "emergency_close_threshold": 5.0,  # 5%距离强平
                "critical_stop_loss_factor": 0.02,  # 2%紧急止损
                "position_reduction_factor": 0.5
            },
            "position_size_error": {
                "max_retry_attempts": 2,
                "size_adjustment_factor": 0.9
            }
        }
    
    def handle_margin_insufficient_error(
        self,
        error: MarginInsufficientError,
        basic_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        处理保证金不足异常
        
        Args:
            error: 保证金不足异常
            basic_params: 基础交易参数
            
        Returns:
            (是否成功恢复, 调整后的参数)
        """
        logger.warning(f"处理保证金不足异常: {error}")
        
        config = self.config["margin_insufficient"]
        max_attempts = config["max_retry_attempts"]
        
        for attempt in range(max_attempts):
            try:
                # 策略1: 降低仓位大小
                if attempt == 0:
                    adjusted_params = self._reduce_position_size(
                        basic_params, 
                        error.available_margin / error.required_margin
                    )
                    logger.info(f"尝试降低仓位大小: {adjusted_params['position_size']}")
                    
                # 策略2: 降低杠杆倍数
                elif attempt == 1:
                    adjusted_params = self._reduce_leverage(
                        basic_params,
                        config["leverage_reduction_factor"]
                    )
                    logger.info(f"尝试降低杠杆倍数: {adjusted_params['leverage']}")
                    
                # 策略3: 同时降低仓位和杠杆
                else:
                    adjusted_params = self._reduce_position_and_leverage(
                        basic_params,
                        config["position_reduction_factor"],
                        config["leverage_reduction_factor"]
                    )
                    logger.info("尝试同时降低仓位和杠杆")
                
                # 验证调整后的参数
                if self._validate_margin_requirement(adjusted_params, error.available_margin):
                    logger.info("保证金不足问题已解决")
                    return True, adjusted_params
                    
            except Exception as e:
                logger.error(f"恢复尝试 {attempt + 1} 失败: {e}")
                continue
        
        logger.error("保证金不足问题无法自动恢复")
        return False, basic_params
    
    def handle_leverage_exceeds_limit_error(
        self,
        error: LeverageExceedsLimitError,
        basic_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        处理杠杆超限异常
        
        Args:
            error: 杠杆超限异常
            basic_params: 基础交易参数
            
        Returns:
            (是否成功恢复, 调整后的参数)
        """
        logger.warning(f"处理杠杆超限异常: {error}")
        
        config = self.config["leverage_exceeds_limit"]
        adjusted_params = basic_params.copy()
        
        # 直接调整到最大允许杠杆
        adjusted_params["leverage"] = error.max_allowed_leverage
        
        # 重新计算相关参数
        adjusted_params = self._recalculate_dependent_params(adjusted_params)
        
        logger.info(f"杠杆已调整至: {adjusted_params['leverage']}")
        return True, adjusted_params
    
    def handle_liquidation_risk_error(
        self,
        error: LiquidationRiskError,
        basic_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        处理强平风险异常
        
        Args:
            error: 强平风险异常
            basic_params: 基础交易参数
            
        Returns:
            (是否成功恢复, 调整后的参数)
        """
        logger.warning(f"处理强平风险异常: {error}")
        
        config = self.config["liquidation_risk"]
        adjusted_params = basic_params.copy()
        
        if error.risk_level == "emergency":
            # 紧急情况：建议平仓
            logger.critical("强平风险极高，建议立即平仓")
            adjusted_params["operation"] = "close"
            adjusted_params["urgency"] = "emergency"
            
        elif error.risk_level == "critical":
            # 关键情况：设置紧急止损
            stop_loss_price = self._calculate_emergency_stop_loss(
                error.current_price,
                error.liquidation_price,
                config["critical_stop_loss_factor"]
            )
            adjusted_params["stop_loss"] = stop_loss_price
            adjusted_params["urgency"] = "high"
            logger.warning(f"设置紧急止损价格: {stop_loss_price}")
            
        else:  # high risk
            # 高风险：降低仓位
            adjusted_params = self._reduce_position_size(
                adjusted_params,
                config["position_reduction_factor"]
            )
            logger.info(f"降低仓位至: {adjusted_params['position_size']}")
        
        return True, adjusted_params
    
    def handle_position_size_error(
        self,
        error: PositionSizeError,
        basic_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        处理仓位大小异常
        
        Args:
            error: 仓位大小异常
            basic_params: 基础交易参数
            
        Returns:
            (是否成功恢复, 调整后的参数)
        """
        logger.warning(f"处理仓位大小异常: {error}")
        
        adjusted_params = basic_params.copy()
        
        if error.issue_type == "too_small":
            if error.min_size:
                adjusted_params["position_size"] = error.min_size
                logger.info(f"仓位调整至最小值: {error.min_size}")
            else:
                # 取消交易
                logger.info("仓位过小，建议取消交易")
                return False, basic_params
                
        elif error.issue_type == "too_large":
            if error.max_size:
                adjusted_params["position_size"] = error.max_size
                logger.info(f"仓位调整至最大值: {error.max_size}")
            else:
                # 使用默认调整因子
                config = self.config["position_size_error"]
                adjusted_params["position_size"] *= config["size_adjustment_factor"]
                logger.info(f"仓位按因子调整: {adjusted_params['position_size']}")
        
        # 重新计算相关参数
        adjusted_params = self._recalculate_dependent_params(adjusted_params)
        
        return True, adjusted_params
    
    def handle_risk_limit_exceeded_error(
        self,
        error: RiskLimitExceededError,
        basic_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        处理风险限制超出异常
        
        Args:
            error: 风险限制异常
            basic_params: 基础交易参数
            
        Returns:
            (是否成功恢复, 调整后的参数)
        """
        logger.warning(f"处理风险限制异常: {error}")
        
        adjusted_params = basic_params.copy()
        reduction_factor = error.limit_value / error.current_value
        
        if error.risk_type in ["drawdown", "exposure"]:
            # 降低仓位大小
            adjusted_params = self._reduce_position_size(adjusted_params, reduction_factor)
        elif error.risk_type == "var":
            # 降低杠杆和仓位
            adjusted_params = self._reduce_position_and_leverage(
                adjusted_params, reduction_factor, 0.8
            )
        
        logger.info(f"风险调整完成，{error.risk_type}降低至限制值以下")
        return True, adjusted_params
    
    def _reduce_position_size(
        self, 
        params: Dict[str, Any], 
        reduction_factor: float
    ) -> Dict[str, Any]:
        """降低仓位大小"""
        adjusted_params = params.copy()
        adjusted_params["position_size"] *= reduction_factor
        adjusted_params["contract_quantity"] *= reduction_factor
        adjusted_params["contract_value"] *= reduction_factor
        return adjusted_params
    
    def _reduce_leverage(
        self, 
        params: Dict[str, Any], 
        reduction_factor: float
    ) -> Dict[str, Any]:
        """降低杠杆倍数"""
        adjusted_params = params.copy()
        new_leverage = max(1, int(params["leverage"] * reduction_factor))
        adjusted_params["leverage"] = new_leverage
        return self._recalculate_dependent_params(adjusted_params)
    
    def _reduce_position_and_leverage(
        self,
        params: Dict[str, Any],
        position_factor: float,
        leverage_factor: float
    ) -> Dict[str, Any]:
        """同时降低仓位和杠杆"""
        adjusted_params = self._reduce_position_size(params, position_factor)
        return self._reduce_leverage(adjusted_params, leverage_factor)
    
    def _recalculate_dependent_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """重新计算依赖参数"""
        adjusted_params = params.copy()
        
        # 重新计算合约价值和数量
        position_size = adjusted_params["position_size"]
        leverage = adjusted_params.get("leverage", 1)
        current_price = adjusted_params.get("current_price", 1.0)
        
        adjusted_params["contract_value"] = position_size / leverage
        adjusted_params["contract_quantity"] = position_size / current_price if current_price > 0 else 0
        
        # 安全地计算position_ratio
        original_position_size = params.get("position_size", position_size)
        if original_position_size > 0:
            ratio_adjustment = position_size / original_position_size
            adjusted_params["position_ratio"] = min(
                adjusted_params.get("position_ratio", 0.1) * ratio_adjustment,
                0.1
            )
        else:
            adjusted_params["position_ratio"] = 0.001
        
        return adjusted_params
    
    def _validate_margin_requirement(
        self, 
        params: Dict[str, Any], 
        available_margin: float
    ) -> bool:
        """验证保证金要求"""
        required_margin = params["position_size"] / params["leverage"]
        return required_margin <= available_margin * 0.95  # 留5%缓冲
    
    def _calculate_emergency_stop_loss(
        self,
        current_price: float,
        liquidation_price: float,
        stop_loss_factor: float
    ) -> float:
        """计算紧急止损价格"""
        if current_price > liquidation_price:  # 多头仓位
            return current_price * (1 - stop_loss_factor)
        else:  # 空头仓位
            return current_price * (1 + stop_loss_factor)
    
    def apply_recovery_strategy(
        self,
        error: ContractTradingError,
        basic_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        根据异常类型应用相应的恢复策略
        
        Args:
            error: 合约交易异常
            basic_params: 基础交易参数
            
        Returns:
            (是否成功恢复, 调整后的参数)
        """
        try:
            if isinstance(error, MarginInsufficientError):
                return self.handle_margin_insufficient_error(error, basic_params)
            elif isinstance(error, LeverageExceedsLimitError):
                return self.handle_leverage_exceeds_limit_error(error, basic_params)
            elif isinstance(error, LiquidationRiskError):
                return self.handle_liquidation_risk_error(error, basic_params)
            elif isinstance(error, PositionSizeError):
                return self.handle_position_size_error(error, basic_params)
            elif isinstance(error, RiskLimitExceededError):
                return self.handle_risk_limit_exceeded_error(error, basic_params)
            else:
                logger.warning(f"未知的异常类型: {type(error)}")
                return False, basic_params
                
        except Exception as e:
            logger.error(f"恢复策略执行失败: {e}")
            return False, basic_params