# 合约交易数据验证系统

## 概述

本系统为AI量化交易提供全面的数据验证功能，确保所有交易参数都符合业务规则和风险要求。验证系统包含四个核心验证器，支持多级验证模式和自动修正功能。

## 核心特性

### 🔍 四大验证器类型

1. **数值范围验证器** (`NumericalRangeValidator`)
   - 杠杆倍数：1-125倍，警告阈值50倍
   - 置信度：0-100%，可信阈值65%
   - 价格：必须为正数，合理范围检查
   - 仓位规模：最小10 USDT，最大100万 USDT
   - 风险收益比：最小0.5，建议>1.5

2. **逻辑一致性验证器** (`LogicalConsistencyValidator`)
   - 交易方向与止损止盈关系验证
   - 杠杆与波动率匹配性检查
   - 仓位规模与账户平衡验证
   - 时间框架信号一致性分析
   - 风险收益比计算一致性

3. **风险约束验证器** (`RiskConstraintValidator`)
   - 保证金使用率：警告70%，危险85%，紧急95%
   - 强平距离：最小安全距离15%，警告10%，危险5%
   - VaR限制：日VaR<2%，周VaR<5%
   - 最大回撤：警告5%，限制15%
   - 集中度风险：单一资产<40%，前三大<70%

4. **成本合理性验证器** (`CostReasonabilityValidator`)
   - 交易手续费：最大0.1%，警告0.08%
   - 资金费率：高费率1%/8小时，极端2%/8小时
   - 滑点成本：最大预期0.1%，警告0.05%
   - 持仓成本：最大日成本0.5%，盈亏平衡时间<7天
   - 收益率期望：最小年化10%，夏普比率>1.0

### 📊 三种验证级别

- **严格模式** (`strict`)：所有约束都必须满足
- **适中模式** (`moderate`)：核心约束必须满足，其他可警告
- **宽松模式** (`lenient`)：只检查基本安全约束

### 🔧 自动修正功能

- 智能参数调整：超限值自动修正为合理范围
- 保留用户意图：在可能情况下尽量保持原始策略
- 修正限制：最多3次调整，避免过度修改
- 透明度：所有修正都有详细记录和说明

## 快速开始

### 基础使用

```python
from src.utils.validators import create_validator

# 创建验证器
validator = create_validator(validation_level="moderate")

# 准备交易数据
trading_data = {
    "basic_params": {
        "leverage": 10,
        "direction": "long", 
        "current_price": 50000.0,
        "position_size": 5000.0
    },
    "risk_management": {
        "stop_loss": 47500.0,
        "take_profit": 52500.0,
        "risk_reward_ratio": 1.0
    },
    "decision_metadata": {
        "confidence": 75
    }
}

# 执行验证
is_valid, results, corrected_data = validator.validate(trading_data)

# 检查结果
if is_valid:
    print("✓ 验证通过")
else:
    print("✗ 验证失败")
    for result in results:
        if not result.is_valid:
            print(f"问题: {result.message}")
            if result.suggestion:
                print(f"建议: {result.suggestion}")
```

### 在PortfolioManagementNode中的集成

验证系统已自动集成到投资组合管理节点中：

```python
from src.graph.portfolio_management_node import PortfolioManagementNode

# 创建节点（可指定验证级别）
node = PortfolioManagementNode(validation_level="moderate")

# 正常调用，验证会自动执行
result = node(state)

# 验证结果包含在决策输出中
validation_info = result["messages"][0].content["BTCUSDT"]["validation"]
print(f"验证通过: {validation_info['is_valid']}")
print(f"错误数量: {validation_info['error_count']}")
print(f"建议: {validation_info['suggestions']}")
```

## 配置文件

### 默认配置位置
```
config/validation_constraints.yaml
```

### 自定义配置示例

```python
# 使用自定义配置
custom_config = {
    "numerical_constraints": {
        "leverage": {
            "min": 1,
            "max": 50,  # 更严格的杠杆限制
            "warning_threshold": 20
        }
    },
    "auto_correction": {
        "enabled": True,
        "max_adjustments": 5,
        "preserve_intent": False
    }
}

validator = create_validator(
    validation_level="strict",
    custom_config=custom_config
)
```

## 验证结果结构

### ValidationResult对象

```python
@dataclass
class ValidationResult:
    is_valid: bool                    # 是否通过验证
    severity: ValidationSeverity      # 问题严重程度
    field_name: str                   # 字段名称
    message: str                      # 验证消息
    current_value: Any                # 当前值
    expected_range: Dict              # 期望范围
    suggestion: str                   # 修正建议
    corrected_value: Any              # 建议修正值
    context: Dict                     # 上下文信息
```

### 严重程度级别

- `ValidationSeverity.INFO`：信息性提示
- `ValidationSeverity.WARNING`：警告
- `ValidationSeverity.ERROR`：错误
- `ValidationSeverity.CRITICAL`：严重错误

## 高级功能

### 验证报告生成

```python
# 生成详细验证报告
is_valid, results, _ = validator.validate(data)
report = validator.format_validation_report(results)
print(report)
```

输出示例：
```
============================================================
交易数据验证报告
============================================================
验证级别: MODERATE
整体结果: ✗ 失败
启用验证器: numerical, logical, risk, cost

统计信息:
  总检查项: 15
  严重错误: 1
  错误: 3
  警告: 2
  信息: 0
  自动修正: 2

🚨 严重错误:
  • margin_management.margin_utilization: 保证金使用率极高: 98.0%
    建议: 立即降低仓位或增加保证金，避免强制平仓

❌ 错误:
  • basic_params.leverage: 杠杆倍数超限: 150，最大值为 125
    建议: 建议将杠杆倍数调整为 125
============================================================
```

### 性能监控

```python
import time

# 测量验证性能
start_time = time.time()
validator.validate(large_dataset)
end_time = time.time()

print(f"验证耗时: {end_time - start_time:.3f}秒")
```

### 缓存和优化

验证器工厂使用缓存机制提高性能：

```python
from src.utils.validators import validator_factory

# 清除缓存
validator_factory.clear_cache()

# 重新加载配置
validator_factory.reload_config()
```

## 最佳实践

### 1. 选择合适的验证级别

- **开发阶段**：使用 `strict` 模式确保数据质量
- **生产环境**：使用 `moderate` 模式平衡安全和灵活性
- **紧急情况**：使用 `lenient` 模式允许更多交易通过

### 2. 合理使用自动修正

```python
# 保守的自动修正设置
auto_correction_config = {
    "enabled": True,
    "max_adjustments": 2,      # 限制修正次数
    "preserve_intent": True    # 保留用户意图
}
```

### 3. 监控验证统计

```python
# 定期检查验证统计
stats = validator.get_validation_stats()
if stats["error_count"] / stats["total_validations"] > 0.1:
    logger.warning("验证错误率过高，需要检查数据质量")
```

### 4. 处理验证失败

```python
is_valid, results, corrected_data = validator.validate(data)

if not is_valid:
    # 检查是否有严重错误
    critical_issues = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
    if critical_issues:
        # 停止交易
        return {"action": "hold", "reason": "严重验证错误"}
    else:
        # 使用修正后的数据或降低仓位
        return process_with_corrections(corrected_data)
```

## 错误诊断

### 常见问题

1. **配置文件找不到**
   ```
   ERROR: 验证器配置文件未找到
   ```
   解决：检查 `config/validation_constraints.yaml` 是否存在

2. **验证器创建失败**
   ```
   ERROR: 创建验证器实例失败
   ```
   解决：检查配置文件格式和必需字段

3. **数据类型错误**
   ```
   WARNING: 字段类型错误，期望 float，实际 str
   ```
   解决：确保数值字段使用正确的数据类型

### 调试技巧

1. **启用详细日志**
   ```python
   import logging
   logging.getLogger('src.utils.validators').setLevel(logging.DEBUG)
   ```

2. **查看验证上下文**
   ```python
   # 检查传递给验证器的上下文
   context = {
       "account_balance": portfolio_value,
       "volatility": current_volatility,
       "debug": True  # 启用调试模式
   }
   validator.validate(data, context)
   ```

3. **单独测试验证器**
   ```python
   # 测试单个验证器
   numerical_validator = NumericalRangeValidator(config)
   results = numerical_validator.validate(test_data)
   ```

## 扩展开发

### 添加新的验证规则

1. **继承BaseValidator**
   ```python
   from src.utils.validators.base_validator import BaseValidator
   
   class CustomValidator(BaseValidator):
       def validate(self, data, context=None):
           results = []
           # 实现自定义验证逻辑
           return results
   ```

2. **集成到复合验证器**
   ```python
   # 在CompositeValidator中添加新验证器
   self.validators["custom"] = CustomValidator(config)
   ```

### 修改约束配置

编辑 `config/validation_constraints.yaml` 文件：

```yaml
# 添加新的数值约束
numerical_constraints:
  custom_field:
    min: 0
    max: 100
    warning_threshold: 80
```

## 性能指标

- **验证速度**：单次验证 < 100ms
- **内存使用**：< 50MB per validator instance
- **缓存命中率**：> 90% for repeated validations
- **错误检测率**：> 95% for known issue patterns

## 更新日志

### v1.0.0 (2025-01-09)
- 初始版本发布
- 四大验证器完整实现
- 三级验证模式支持
- 自动修正功能
- PortfolioManagementNode集成

## 支持和反馈

如有问题或建议，请查看：
- 单元测试：`tests/test_validators.py`
- 集成测试：`tests/test_portfolio_validation_integration.py`
- 演示示例：`examples/validation_demo.py`