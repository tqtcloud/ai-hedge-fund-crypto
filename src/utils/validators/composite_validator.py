"""
复合验证器

整合所有验证器，提供统一的验证入口点和结果汇总功能。
支持不同的验证级别和自动修正功能。
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from .base_validator import BaseValidator, ValidationResult, ValidationSeverity
from .numerical_validator import NumericalRangeValidator
from .logical_validator import LogicalConsistencyValidator
from .risk_validator import RiskConstraintValidator
from .cost_validator import CostReasonabilityValidator

logger = logging.getLogger(__name__)


class CompositeValidator:
    """
    复合验证器
    
    整合多个验证器，提供统一的验证接口和结果管理
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化复合验证器
        
        Args:
            config: 验证器配置，包含所有验证器的设置
        """
        self.config = config
        self.validation_level = config.get("validation_level", "moderate")
        self.auto_correction = config.get("auto_correction", {})
        
        # 获取验证级别配置
        level_config = config.get("validation_levels", {}).get(self.validation_level, {})
        self.enabled_validators = level_config.get("enabled_validators", ["numerical", "logical", "risk", "cost"])
        self.warning_only_validators = level_config.get("warning_only", [])
        
        # 初始化各个验证器
        self.validators = {}
        self._initialize_validators(config)
        
        # 验证结果统计
        self.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "warnings": 0,
            "errors": 0,
            "critical_errors": 0,
            "auto_corrections": 0
        }
    
    def _initialize_validators(self, config: Dict[str, Any]) -> None:
        """初始化所有验证器"""
        
        # 数值范围验证器
        if "numerical" in self.enabled_validators:
            validator_config = config.copy()
            validator_config["warning_only"] = "numerical" in self.warning_only_validators
            self.validators["numerical"] = NumericalRangeValidator(validator_config)
        
        # 逻辑一致性验证器
        if "logical" in self.enabled_validators:
            validator_config = config.copy()
            validator_config["warning_only"] = "logical" in self.warning_only_validators
            self.validators["logical"] = LogicalConsistencyValidator(validator_config)
        
        # 风险约束验证器
        if "risk" in self.enabled_validators:
            validator_config = config.copy()
            validator_config["warning_only"] = "risk" in self.warning_only_validators
            self.validators["risk"] = RiskConstraintValidator(validator_config)
        
        # 成本合理性验证器
        if "cost" in self.enabled_validators:
            validator_config = config.copy()
            validator_config["warning_only"] = "cost" in self.warning_only_validators
            self.validators["cost"] = CostReasonabilityValidator(validator_config)
    
    def validate(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[ValidationResult], Dict[str, Any]]:
        """
        执行完整的数据验证
        
        Args:
            data: 待验证的交易数据
            context: 验证上下文信息
            
        Returns:
            (是否通过验证, 验证结果列表, 修正后的数据)
        """
        all_results = []
        corrected_data = data.copy()
        has_critical_errors = False
        
        logger.info(f"开始执行 {self.validation_level} 级别验证，启用验证器: {list(self.validators.keys())}")
        
        # 逐个执行验证器
        for validator_name, validator in self.validators.items():
            try:
                logger.debug(f"执行 {validator_name} 验证器")
                results = validator.validate(corrected_data, context)
                
                # 处理验证结果和自动修正
                corrected_results, data_corrections = self._process_validation_results(
                    results, corrected_data, validator_name
                )
                
                all_results.extend(corrected_results)
                
                # 应用数据修正
                if data_corrections:
                    for field_path, corrected_value in data_corrections.items():
                        self._set_nested_value(corrected_data, field_path, corrected_value)
                        logger.info(f"自动修正字段 {field_path}: {corrected_value}")
                        self.validation_stats["auto_corrections"] += 1
                
            except Exception as e:
                logger.error(f"{validator_name} 验证器执行失败: {str(e)}")
                all_results.append(
                    ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        field_name=f"{validator_name}_validator",
                        message=f"验证器执行异常: {str(e)}",
                        suggestion="请检查数据格式和验证器配置"
                    )
                )
        
        # 统计验证结果
        self._update_validation_stats(all_results)
        
        # 判断整体是否通过验证
        has_critical_errors = any(r.severity == ValidationSeverity.CRITICAL for r in all_results)
        has_errors = any(r.severity == ValidationSeverity.ERROR for r in all_results)
        
        # 根据验证级别决定是否通过
        is_valid = self._determine_overall_validity(all_results, has_critical_errors, has_errors)
        
        # 生成验证摘要
        summary = self._generate_validation_summary(all_results, is_valid)
        
        logger.info(f"验证完成: {'通过' if is_valid else '失败'}，"
                   f"问题数量 - 严重: {summary['critical_count']}，"
                   f"错误: {summary['error_count']}，"
                   f"警告: {summary['warning_count']}")
        
        return is_valid, all_results, corrected_data
    
    def _process_validation_results(
        self, 
        results: List[ValidationResult], 
        data: Dict[str, Any], 
        validator_name: str
    ) -> Tuple[List[ValidationResult], Dict[str, str]]:
        """
        处理验证结果并执行自动修正
        
        Args:
            results: 验证结果列表
            data: 当前数据
            validator_name: 验证器名称
            
        Returns:
            (处理后的验证结果, 需要修正的数据字典)
        """
        processed_results = []
        data_corrections = {}
        
        for result in results:
            # 检查是否需要自动修正
            if (not result.is_valid and 
                result.corrected_value is not None and 
                self._should_auto_correct(validator_name, result)):
                
                # 执行自动修正
                data_corrections[result.field_name] = result.corrected_value
                
                # 更新验证结果为已修正
                corrected_result = ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    field_name=result.field_name,
                    message=f"已自动修正: {result.message}",
                    current_value=result.current_value,
                    expected_range=result.expected_range,
                    suggestion=f"值已从 {result.current_value} 修正为 {result.corrected_value}",
                    corrected_value=result.corrected_value,
                    context=result.context
                )
                processed_results.append(corrected_result)
            else:
                processed_results.append(result)
        
        return processed_results, data_corrections
    
    def _should_auto_correct(self, validator_name: str, result: ValidationResult) -> bool:
        """
        判断是否应该自动修正
        
        Args:
            validator_name: 验证器名称
            result: 验证结果
            
        Returns:
            是否应该自动修正
        """
        if not self.auto_correction.get("enabled", False):
            return False
        
        # 只对特定严重程度的问题进行自动修正
        auto_correct_severities = [ValidationSeverity.ERROR, ValidationSeverity.WARNING]
        if result.severity not in auto_correct_severities:
            return False
        
        # 检查是否超过最大修正次数
        max_adjustments = self.auto_correction.get("max_adjustments", 3)
        if self.validation_stats["auto_corrections"] >= max_adjustments:
            return False
        
        # 检查是否保留用户意图
        preserve_intent = self.auto_correction.get("preserve_intent", True)
        if preserve_intent and result.severity == ValidationSeverity.WARNING:
            return False  # 警告级别的问题不自动修正，保留用户意图
        
        return True
    
    def _determine_overall_validity(
        self, 
        results: List[ValidationResult], 
        has_critical: bool, 
        has_errors: bool
    ) -> bool:
        """
        根据验证级别和结果确定整体是否有效
        
        Args:
            results: 验证结果列表
            has_critical: 是否有严重错误
            has_errors: 是否有错误
            
        Returns:
            整体是否有效
        """
        if has_critical:
            return False
        
        if self.validation_level == "strict":
            # 严格模式：任何错误或警告都不通过
            return not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING] 
                          for r in results if not r.is_valid)
        elif self.validation_level == "moderate":
            # 适中模式：只有错误不通过，警告可以通过
            return not has_errors
        elif self.validation_level == "lenient":
            # 宽松模式：只有严重错误不通过
            return not has_critical
        else:
            # 默认为适中模式
            return not has_errors
    
    def _update_validation_stats(self, results: List[ValidationResult]) -> None:
        """更新验证统计信息"""
        self.validation_stats["total_validations"] = len(results)
        self.validation_stats["passed_validations"] = sum(1 for r in results if r.is_valid)
        
        for result in results:
            if result.severity == ValidationSeverity.WARNING:
                self.validation_stats["warnings"] += 1
            elif result.severity == ValidationSeverity.ERROR:
                self.validation_stats["errors"] += 1
            elif result.severity == ValidationSeverity.CRITICAL:
                self.validation_stats["critical_errors"] += 1
    
    def _generate_validation_summary(
        self, 
        results: List[ValidationResult], 
        is_valid: bool
    ) -> Dict[str, Any]:
        """
        生成验证摘要
        
        Args:
            results: 验证结果列表
            is_valid: 整体是否有效
            
        Returns:
            验证摘要字典
        """
        summary = {
            "is_valid": is_valid,
            "validation_level": self.validation_level,
            "total_checks": len(results),
            "critical_count": sum(1 for r in results if r.severity == ValidationSeverity.CRITICAL),
            "error_count": sum(1 for r in results if r.severity == ValidationSeverity.ERROR),
            "warning_count": sum(1 for r in results if r.severity == ValidationSeverity.WARNING),
            "info_count": sum(1 for r in results if r.severity == ValidationSeverity.INFO),
            "auto_corrections": self.validation_stats["auto_corrections"],
            "enabled_validators": list(self.validators.keys()),
            "validation_stats": self.validation_stats.copy()
        }
        
        # 按严重程度分组结果
        summary["results_by_severity"] = {
            "critical": [r for r in results if r.severity == ValidationSeverity.CRITICAL],
            "error": [r for r in results if r.severity == ValidationSeverity.ERROR],
            "warning": [r for r in results if r.severity == ValidationSeverity.WARNING],
            "info": [r for r in results if r.severity == ValidationSeverity.INFO]
        }
        
        # 按验证器分组结果
        summary["results_by_validator"] = {}
        for result in results:
            validator_name = result.field_name.split('.')[0] if '.' in result.field_name else "unknown"
            if validator_name not in summary["results_by_validator"]:
                summary["results_by_validator"][validator_name] = []
            summary["results_by_validator"][validator_name].append(result)
        
        return summary
    
    def get_critical_issues(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """获取严重问题列表"""
        return [r for r in results if r.severity == ValidationSeverity.CRITICAL]
    
    def get_error_issues(self, results: List[ValidationResult]) -> List[ValidationResult]:
        """获取错误问题列表"""
        return [r for r in results if r.severity == ValidationSeverity.ERROR]
    
    def get_suggestions(self, results: List[ValidationResult]) -> List[str]:
        """获取所有修正建议"""
        suggestions = []
        for result in results:
            if result.suggestion and not result.is_valid:
                suggestions.append(f"{result.field_name}: {result.suggestion}")
        return suggestions
    
    def format_validation_report(self, results: List[ValidationResult]) -> str:
        """
        格式化验证报告
        
        Args:
            results: 验证结果列表
            
        Returns:
            格式化的验证报告字符串
        """
        summary = self._generate_validation_summary(results, 
                                                   self._determine_overall_validity(results, 
                                                   any(r.severity == ValidationSeverity.CRITICAL for r in results),
                                                   any(r.severity == ValidationSeverity.ERROR for r in results)))
        
        report = []
        report.append("=" * 60)
        report.append("交易数据验证报告")
        report.append("=" * 60)
        report.append(f"验证级别: {summary['validation_level'].upper()}")
        report.append(f"整体结果: {'✓ 通过' if summary['is_valid'] else '✗ 失败'}")
        report.append(f"启用验证器: {', '.join(summary['enabled_validators'])}")
        report.append("")
        
        # 统计信息
        report.append("统计信息:")
        report.append(f"  总检查项: {summary['total_checks']}")
        report.append(f"  严重错误: {summary['critical_count']}")
        report.append(f"  错误: {summary['error_count']}")
        report.append(f"  警告: {summary['warning_count']}")
        report.append(f"  信息: {summary['info_count']}")
        report.append(f"  自动修正: {summary['auto_corrections']}")
        report.append("")
        
        # 详细问题列表
        if summary['critical_count'] > 0:
            report.append("🚨 严重错误:")
            for result in summary['results_by_severity']['critical']:
                report.append(f"  • {result.field_name}: {result.message}")
                if result.suggestion:
                    report.append(f"    建议: {result.suggestion}")
            report.append("")
        
        if summary['error_count'] > 0:
            report.append("❌ 错误:")
            for result in summary['results_by_severity']['error']:
                report.append(f"  • {result.field_name}: {result.message}")
                if result.suggestion:
                    report.append(f"    建议: {result.suggestion}")
            report.append("")
        
        if summary['warning_count'] > 0:
            report.append("⚠️ 警告:")
            for result in summary['results_by_severity']['warning']:
                report.append(f"  • {result.field_name}: {result.message}")
                if result.suggestion:
                    report.append(f"    建议: {result.suggestion}")
            report.append("")
        
        if summary['info_count'] > 0:
            report.append("ℹ️ 信息:")
            for result in summary['results_by_severity']['info']:
                report.append(f"  • {result.field_name}: {result.message}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _get_nested_value(self, data: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """获取嵌套字典中的值"""
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
        """设置嵌套字典中的值"""
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
    
    def reset_stats(self) -> None:
        """重置验证统计信息"""
        self.validation_stats = {
            "total_validations": 0,
            "passed_validations": 0,
            "warnings": 0,
            "errors": 0,
            "critical_errors": 0,
            "auto_corrections": 0
        }
    
    def get_validation_stats(self) -> Dict[str, int]:
        """获取验证统计信息"""
        return self.validation_stats.copy()