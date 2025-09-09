"""
数据验证器模块

提供全面的合约交易数据验证功能，包括：
- 数值范围验证
- 逻辑一致性验证  
- 风险约束验证
- 成本合理性验证
"""

from .base_validator import BaseValidator, ValidationResult, ValidationSeverity
from .numerical_validator import NumericalRangeValidator
from .logical_validator import LogicalConsistencyValidator
from .risk_validator import RiskConstraintValidator
from .cost_validator import CostReasonabilityValidator
from .composite_validator import CompositeValidator
from .validator_factory import ValidatorFactory, create_validator, get_validator_config

__all__ = [
    "BaseValidator",
    "ValidationResult", 
    "ValidationSeverity",
    "NumericalRangeValidator",
    "LogicalConsistencyValidator",
    "RiskConstraintValidator",
    "CostReasonabilityValidator",
    "CompositeValidator",
    "ValidatorFactory",
    "create_validator",
    "get_validator_config"
]