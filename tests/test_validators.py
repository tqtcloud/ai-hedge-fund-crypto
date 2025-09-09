"""
数据验证器单元测试

测试各种验证器的功能和行为
"""

import pytest
import yaml
from pathlib import Path
from src.utils.validators import (
    create_validator, 
    NumericalRangeValidator, 
    LogicalConsistencyValidator,
    RiskConstraintValidator,
    CostReasonabilityValidator,
    CompositeValidator,
    ValidationSeverity
)


class TestNumericalRangeValidator:
    """测试数值范围验证器"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = {
            "numerical_constraints": {
                "leverage": {"min": 1, "max": 125, "warning_threshold": 50},
                "confidence": {"min": 0, "max": 100, "reliable_threshold": 65},
                "price": {"min": 0.000001, "max": 1000000},
                "position_size": {"min_value": 10.0, "max_value": 1000000.0}
            }
        }
        self.validator = NumericalRangeValidator(self.config)
    
    def test_valid_leverage(self):
        """测试有效杠杆验证"""
        data = {"basic_params": {"leverage": 10}}
        results = self.validator.validate(data)
        
        # 应该没有错误或警告
        errors = [r for r in results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        assert len(errors) == 0
    
    def test_invalid_leverage_too_high(self):
        """测试过高杠杆验证"""
        data = {"basic_params": {"leverage": 150}}
        results = self.validator.validate(data)
        
        # 应该有错误
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "杠杆倍数超限" in errors[0].message
        assert errors[0].corrected_value == 125  # 应该建议修正为最大值
    
    def test_invalid_leverage_too_low(self):
        """测试过低杠杆验证"""
        data = {"basic_params": {"leverage": 0.5}}
        results = self.validator.validate(data)
        
        # 应该有错误
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "杠杆倍数过低" in errors[0].message
    
    def test_high_leverage_warning(self):
        """测试高杠杆警告"""
        data = {"basic_params": {"leverage": 60}}
        results = self.validator.validate(data)
        
        # 应该有警告
        warnings = [r for r in results if r.severity == ValidationSeverity.WARNING]
        assert len(warnings) >= 1
        assert "杠杆倍数偏高" in warnings[0].message
    
    def test_confidence_validation(self):
        """测试置信度验证"""
        # 测试正常置信度
        data = {"decision_metadata": {"confidence": 75}}
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0
        
        # 测试超出范围的置信度
        data = {"decision_metadata": {"confidence": 150}}
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "置信度超限" in errors[0].message
    
    def test_price_validation(self):
        """测试价格验证"""
        # 测试负价格
        data = {"basic_params": {"current_price": -10}}
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "价格必须为正数" in errors[0].message
        
        # 测试正常价格
        data = {"basic_params": {"current_price": 50000}}
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0


class TestLogicalConsistencyValidator:
    """测试逻辑一致性验证器"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = {
            "logical_constraints": {
                "direction_stop_logic": {},
                "leverage_volatility": {"high_volatility_max_leverage": 10}
            }
        }
        self.validator = LogicalConsistencyValidator(self.config)
    
    def test_long_position_logic(self):
        """测试多头仓位逻辑"""
        data = {
            "basic_params": {
                "direction": "long",
                "current_price": 50000,
                "entry_price_target": 50000
            },
            "risk_management": {
                "stop_loss": 48000,  # 正确：低于入场价
                "take_profit": 52000  # 正确：高于入场价
            }
        }
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0
    
    def test_long_position_wrong_stop_loss(self):
        """测试多头仓位错误止损"""
        data = {
            "basic_params": {
                "direction": "long",
                "current_price": 50000,
                "entry_price_target": 50000
            },
            "risk_management": {
                "stop_loss": 52000,  # 错误：高于入场价
                "take_profit": 53000
            }
        }
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "多头止损价格错误" in errors[0].message
    
    def test_short_position_logic(self):
        """测试空头仓位逻辑"""
        data = {
            "basic_params": {
                "direction": "short",
                "current_price": 50000,
                "entry_price_target": 50000
            },
            "risk_management": {
                "stop_loss": 52000,  # 正确：高于入场价
                "take_profit": 48000  # 正确：低于入场价
            }
        }
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0
    
    def test_high_volatility_leverage(self):
        """测试高波动率环境下的杠杆"""
        data = {
            "basic_params": {"leverage": 20}
        }
        context = {"volatility": 0.08}  # 8%波动率
        
        results = self.validator.validate(data, context)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "高波动率环境下杠杆过高" in errors[0].message


class TestRiskConstraintValidator:
    """测试风险约束验证器"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = {
            "risk_constraints": {
                "margin_utilization": {
                    "warning_level": 0.7,
                    "critical_level": 0.85,
                    "emergency_level": 0.95
                },
                "liquidation_distance": {
                    "min_safe_distance": 15.0,
                    "warning_distance": 10.0,
                    "critical_distance": 5.0
                }
            }
        }
        self.validator = RiskConstraintValidator(self.config)
    
    def test_safe_margin_utilization(self):
        """测试安全保证金使用率"""
        data = {"margin_management": {"margin_utilization": 0.5}}
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0
    
    def test_high_margin_utilization(self):
        """测试高保证金使用率"""
        data = {"margin_management": {"margin_utilization": 0.9}}
        results = self.validator.validate(data)
        critical_errors = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        assert len(critical_errors) >= 1
        assert "保证金使用率极高" in critical_errors[0].message
    
    def test_liquidation_distance(self):
        """测试强平距离验证"""
        data = {
            "basic_params": {
                "current_price": 50000,
                "direction": "long"
            },
            "risk_management": {
                "liquidation_price": 45000  # 10%距离
            }
        }
        results = self.validator.validate(data)
        warnings = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(warnings) >= 1
        assert "强平距离过近" in warnings[0].message


class TestCostReasonabilityValidator:
    """测试成本合理性验证器"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = {
            "cost_constraints": {
                "trading_fees": {"max_fee_rate": 0.001, "high_fee_warning": 0.0008},
                "funding_rate": {"high_rate_threshold": 0.01}
            }
        }
        self.validator = CostReasonabilityValidator(self.config)
    
    def test_reasonable_trading_fees(self):
        """测试合理的交易手续费"""
        data = {
            "basic_params": {"position_size": 10000},
            "cost_benefit_analysis": {"estimated_trading_fee": 5}  # 0.05%费率
        }
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0
    
    def test_high_trading_fees(self):
        """测试过高的交易手续费"""
        data = {
            "basic_params": {"position_size": 10000},
            "cost_benefit_analysis": {"estimated_trading_fee": 15}  # 0.15%费率
        }
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "手续费率过高" in errors[0].message
    
    def test_negative_expected_value(self):
        """测试负期望值"""
        data = {
            "cost_benefit_analysis": {"expected_value": -100}
        }
        results = self.validator.validate(data)
        errors = [r for r in results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) >= 1
        assert "期望值为负" in errors[0].message


class TestCompositeValidator:
    """测试复合验证器"""
    
    def setup_method(self):
        """测试前的设置"""
        self.config = {
            "validation_level": "moderate",
            "numerical_constraints": {
                "leverage": {"min": 1, "max": 125},
                "confidence": {"min": 0, "max": 100}
            },
            "logical_constraints": {
                "direction_stop_logic": {}
            },
            "risk_constraints": {
                "margin_utilization": {"critical_level": 0.85}
            },
            "cost_constraints": {
                "trading_fees": {"max_fee_rate": 0.001}
            },
            "validation_levels": {
                "moderate": {
                    "enabled_validators": ["numerical", "logical", "risk", "cost"],
                    "warning_only": []
                }
            },
            "auto_correction": {
                "enabled": True,
                "max_adjustments": 3
            }
        }
        self.validator = CompositeValidator(self.config)
    
    def test_comprehensive_validation(self):
        """测试综合验证"""
        data = {
            "basic_params": {
                "leverage": 10,
                "direction": "long",
                "current_price": 50000,
                "entry_price_target": 50000,
                "position_size": 10000
            },
            "risk_management": {
                "stop_loss": 48000,
                "take_profit": 52000
            },
            "decision_metadata": {
                "confidence": 75
            },
            "cost_benefit_analysis": {
                "estimated_trading_fee": 5,
                "expected_value": 200
            }
        }
        
        is_valid, results, corrected_data = self.validator.validate(data)
        
        # 这个数据应该通过验证
        assert is_valid == True
        
        # 不应该有严重错误
        critical_errors = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        assert len(critical_errors) == 0
    
    def test_validation_with_corrections(self):
        """测试带修正的验证"""
        data = {
            "basic_params": {
                "leverage": 150,  # 超出限制，应该被修正
                "position_size": 10000
            },
            "decision_metadata": {
                "confidence": 75
            }
        }
        
        is_valid, results, corrected_data = self.validator.validate(data)
        
        # 检查是否有修正
        corrections = [r for r in results if r.corrected_value is not None]
        assert len(corrections) > 0
    
    def test_validation_report_generation(self):
        """测试验证报告生成"""
        data = {
            "basic_params": {"leverage": 200},  # 无效数据
            "decision_metadata": {"confidence": 150}  # 无效数据
        }
        
        is_valid, results, corrected_data = self.validator.validate(data)
        report = self.validator.format_validation_report(results)
        
        assert "交易数据验证报告" in report
        assert "错误" in report
        assert len(report) > 100  # 报告应该有一定长度


class TestValidatorFactory:
    """测试验证器工厂"""
    
    def test_create_validator(self):
        """测试创建验证器"""
        validator = create_validator(validation_level="moderate")
        assert isinstance(validator, CompositeValidator)
        assert validator.validation_level == "moderate"
    
    def test_create_validator_with_custom_config(self):
        """测试使用自定义配置创建验证器"""
        custom_config = {
            "numerical_constraints": {
                "leverage": {"min": 1, "max": 50}  # 自定义最大杠杆
            }
        }
        
        validator = create_validator(
            validation_level="strict",
            custom_config=custom_config
        )
        
        assert isinstance(validator, CompositeValidator)
        assert validator.validation_level == "strict"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])