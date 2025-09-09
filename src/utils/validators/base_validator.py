"""
数据验证器基类

定义验证器的基础接口和通用功能，所有具体验证器都继承自此基类。
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """验证问题严重程度枚举"""
    INFO = "info"           # 信息性提示
    WARNING = "warning"     # 警告
    ERROR = "error"         # 错误
    CRITICAL = "critical"   # 严重错误


@dataclass
class ValidationResult:
    """验证结果数据类"""
    is_valid: bool                          # 是否验证通过
    severity: ValidationSeverity            # 问题严重程度
    field_name: str                         # 字段名称
    message: str                            # 验证消息
    current_value: Any = None               # 当前值
    expected_range: Optional[Dict[str, Any]] = None  # 期望范围
    suggestion: Optional[str] = None        # 修正建议
    corrected_value: Any = None             # 建议修正值
    context: Optional[Dict[str, Any]] = None # 上下文信息
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"[{self.severity.value.upper()}] {self.field_name}: {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "is_valid": self.is_valid,
            "severity": self.severity.value,
            "field_name": self.field_name,
            "message": self.message,
            "current_value": self.current_value,
            "expected_range": self.expected_range,
            "suggestion": self.suggestion,
            "corrected_value": self.corrected_value,
            "context": self.context
        }


class BaseValidator(ABC):
    """
    验证器基类
    
    定义所有验证器的通用接口和基础功能
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化验证器
        
        Args:
            name: 验证器名称
            config: 验证器配置
        """
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
        self.warning_only = config.get("warning_only", False)
        self.auto_correction = config.get("auto_correction", {})
        
    @abstractmethod
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> List[ValidationResult]:
        """
        执行验证
        
        Args:
            data: 待验证数据
            context: 验证上下文
            
        Returns:
            验证结果列表
        """
        pass
    
    def is_enabled(self) -> bool:
        """检查验证器是否启用"""
        return self.enabled
    
    def should_auto_correct(self) -> bool:
        """检查是否应该自动修正"""
        return self.auto_correction.get("enabled", False)
    
    def _create_result(
        self, 
        is_valid: bool, 
        severity: ValidationSeverity,
        field_name: str,
        message: str,
        current_value: Any = None,
        expected_range: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        corrected_value: Any = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        创建验证结果
        
        Args:
            is_valid: 是否验证通过
            severity: 严重程度
            field_name: 字段名称
            message: 验证消息
            current_value: 当前值
            expected_range: 期望范围
            suggestion: 修正建议
            corrected_value: 建议修正值
            context: 上下文信息
            
        Returns:
            验证结果对象
        """
        # 如果设置为仅警告模式，将错误降级为警告
        if self.warning_only and severity == ValidationSeverity.ERROR:
            severity = ValidationSeverity.WARNING
            
        return ValidationResult(
            is_valid=is_valid,
            severity=severity,
            field_name=field_name,
            message=message,
            current_value=current_value,
            expected_range=expected_range,
            suggestion=suggestion,
            corrected_value=corrected_value,
            context=context
        )
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        获取嵌套字典中的值
        
        Args:
            data: 数据字典
            key_path: 键路径，使用点分隔（如 "basic_params.leverage"）
            default: 默认值
            
        Returns:
            字段值
        """
        try:
            value = data
            for key in key_path.split('.'):
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (KeyError, TypeError, AttributeError):
            return default
    
    def _set_nested_value(self, data: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        设置嵌套字典中的值
        
        Args:
            data: 数据字典
            key_path: 键路径
            value: 要设置的值
        """
        try:
            keys = key_path.split('.')
            current = data
            
            # 导航到倒数第二层
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # 设置最后一层的值
            current[keys[-1]] = value
        except (KeyError, TypeError, AttributeError) as e:
            logger.error(f"设置嵌套值失败: {key_path}, error: {e}")
    
    def _safe_float_conversion(self, value: Any, default: float = 0.0) -> float:
        """
        安全的浮点数转换
        
        Args:
            value: 待转换值
            default: 转换失败时的默认值
            
        Returns:
            转换后的浮点数
        """
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_int_conversion(self, value: Any, default: int = 0) -> int:
        """
        安全的整数转换
        
        Args:
            value: 待转换值
            default: 转换失败时的默认值
            
        Returns:
            转换后的整数
        """
        try:
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _is_number(self, value: Any) -> bool:
        """
        检查值是否为数字
        
        Args:
            value: 待检查值
            
        Returns:
            是否为数字
        """
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def _clamp_value(self, value: float, min_val: float, max_val: float) -> float:
        """
        将值限制在指定范围内
        
        Args:
            value: 原始值
            min_val: 最小值
            max_val: 最大值
            
        Returns:
            限制后的值
        """
        return max(min_val, min(value, max_val))
    
    def log_validation_result(self, result: ValidationResult) -> None:
        """
        记录验证结果日志
        
        Args:
            result: 验证结果
        """
        log_level = {
            ValidationSeverity.INFO: logging.INFO,
            ValidationSeverity.WARNING: logging.WARNING,
            ValidationSeverity.ERROR: logging.ERROR,
            ValidationSeverity.CRITICAL: logging.CRITICAL
        }.get(result.severity, logging.INFO)
        
        logger.log(log_level, f"{self.name}: {result}")
        
        if result.suggestion:
            logger.log(log_level, f"建议: {result.suggestion}")


class ValidationMixin:
    """验证器混入类，提供通用验证方法"""
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: List[str]) -> List[ValidationResult]:
        """
        验证必需字段是否存在
        
        Args:
            data: 数据字典
            required_fields: 必需字段列表
            
        Returns:
            验证结果列表
        """
        results = []
        for field in required_fields:
            value = self._get_nested_value(data, field)
            if value is None:
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name=field,
                        message=f"必需字段 {field} 缺失",
                        suggestion=f"请提供 {field} 字段的值"
                    )
                )
        return results
    
    def validate_data_types(self, data: Dict[str, Any], type_mapping: Dict[str, type]) -> List[ValidationResult]:
        """
        验证字段数据类型
        
        Args:
            data: 数据字典
            type_mapping: 字段类型映射
            
        Returns:
            验证结果列表
        """
        results = []
        for field, expected_type in type_mapping.items():
            value = self._get_nested_value(data, field)
            if value is not None and not isinstance(value, expected_type):
                results.append(
                    self._create_result(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name=field,
                        message=f"字段 {field} 类型错误，期望 {expected_type.__name__}，实际 {type(value).__name__}",
                        current_value=value,
                        suggestion=f"请将 {field} 转换为 {expected_type.__name__} 类型"
                    )
                )
        return results