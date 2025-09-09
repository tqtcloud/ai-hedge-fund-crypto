# 合约交易异常处理机制

## 概述

本项目实现了一套完整的合约交易异常处理机制，能够在各种错误情况下自动检测、诊断和恢复，确保交易系统的稳定性和可靠性。

## 异常类型体系

### 基础异常类 - ContractTradingError

所有合约交易相关异常的基类，提供统一的异常处理接口：

```python
class ContractTradingError(Exception):
    def __init__(self, message, error_code, context, recovery_suggestion):
        # 提供详细的错误信息和恢复建议
```

### 具体异常类型

#### 1. MarginInsufficientError - 保证金不足异常

**触发条件**：
- 账户保证金不足以支持当前交易
- 计算的所需保证金超过可用保证金

**自动恢复策略**：
1. 降低仓位大小至可承受范围
2. 降低杠杆倍数减少保证金需求
3. 同时调整仓位和杠杆

**使用示例**：
```python
try:
    position_size, position_ratio = calculator.calculate_position_size(...)
except MarginInsufficientError as e:
    # 自动恢复处理
    recovered, adjusted_params = recovery_manager.handle_margin_insufficient_error(e, basic_params)
```

#### 2. LeverageExceedsLimitError - 杠杆超限异常

**触发条件**：
- 请求的杠杆倍数超过交易所限制
- 杠杆倍数超过风险管理规则
- 基于波动率的杠杆限制

**自动恢复策略**：
- 自动调整至最大允许杠杆
- 重新计算相关的仓位参数

#### 3. LiquidationRiskError - 强平风险异常

**触发条件**：
- 距离强平价格过近（< 15%）
- 风险等级：emergency（< 5%）、critical（< 10%）、high（< 15%）

**自动恢复策略**：
- Emergency：建议立即平仓
- Critical：设置紧急止损
- High：降低仓位大小

#### 4. PositionSizeError - 仓位大小异常

**触发条件**：
- 计算的仓位过小（低于最小交易单位）
- 计算的仓位过大（超过风险限制）
- 仓位大小无效或异常

**自动恢复策略**：
- 调整至合规的仓位大小
- 或建议取消交易

#### 5. RiskLimitExceededError - 风险限制超出异常

**触发条件**：
- 风险暴露超过设定阈值
- VaR（风险价值）超限
- 最大回撤风险超限

**自动恢复策略**：
- 降低仓位大小
- 调整杠杆倍数
- 重新平衡风险参数

## 错误恢复管理器

### ErrorRecoveryManager 类

负责统一管理所有异常的恢复处理：

```python
class ErrorRecoveryManager:
    def apply_recovery_strategy(self, error, basic_params):
        # 根据异常类型自动选择合适的恢复策略
        # 返回 (是否恢复成功, 调整后的参数)
```

### 恢复策略配置

```python
config = {
    "margin_insufficient": {
        "max_retry_attempts": 3,
        "position_reduction_factor": 0.8,
        "leverage_reduction_factor": 0.8,
        "min_position_size": 10.0
    },
    "liquidation_risk": {
        "emergency_close_threshold": 5.0,
        "critical_stop_loss_factor": 0.02,
        "position_reduction_factor": 0.5
    }
}
```

## 集成到现有系统

### PortfolioCalculator 增强

```python
class PortfolioCalculator:
    @staticmethod
    def calculate_leverage(risk_data, technical_data, ticker):
        # 检查杠杆限制
        if leverage > exchange_max_leverage:
            raise LeverageExceedsLimitError(...)
        
    @staticmethod  
    def calculate_position_size(...):
        # 检查保证金要求
        if required_margin > available_margin:
            raise MarginInsufficientError(...)
```

### PortfolioManagementNode 增强

主要方法都已集成异常处理：

1. **calculate_basic_params()** - 带完整异常处理和恢复
2. **design_risk_management()** - 强平风险检测和处理
3. **assess_technical_risk()** - 包装器异常处理
4. **design_execution_strategy()** - 包装器异常处理

### 使用流程

```python
def enhanced_trading_flow():
    try:
        # 计算基础参数（带自动恢复）
        basic_params = portfolio_node.calculate_basic_params(...)
        
        # 设计风险管理（带强平检测）
        risk_management = portfolio_node.design_risk_management(...)
        
        # 检查恢复记录
        if "_calculation_errors" in basic_params:
            logger.info("检测到异常，已自动恢复")
            
    except ContractTradingError as e:
        # 最后的异常处理
        logger.error(f"无法恢复的错误: {e}")
```

## 监控和日志

### 异常监控

所有异常都包含丰富的上下文信息：

```python
{
    "error_type": "MarginInsufficientError",
    "error_code": "MARGIN_INSUFFICIENT", 
    "ticker": "BTCUSDT",
    "required_margin": 1000.0,
    "available_margin": 800.0,
    "recovery_applied": True
}
```

### 风险监控信息

```python
{
    "_risk_monitoring": {
        "distance_to_liquidation": 12.5,
        "liquidation_risk_level": "high",
        "risk_percentage_status": "normal",
        "volatility_level": "high"
    }
}
```

## 保守参数后备机制

当所有恢复尝试都失败时，系统会自动使用保守的默认参数：

```python
conservative_params = {
    "leverage": 1,  # 最小杠杆
    "position_size": min(portfolio_cash * 0.01, 50.0),  # 1%资金
    "risk_percentage": 1.0,  # 1%固定风险
    "_is_conservative_fallback": True
}
```

## 测试验证

### 单元测试

- `test_exceptions.py` - 完整的异常处理测试
- `test_simple_exceptions.py` - 简化的功能验证

### 集成测试

- 正常交易流程测试
- 保证金不足场景测试  
- 高杠杆限制测试
- 强平风险场景测试

## 最佳实践

### 1. 错误处理原则

- **快速失败**：及时发现和处理异常
- **自动恢复**：优先尝试自动修复
- **降级处理**：无法修复时使用保守参数
- **详细日志**：记录所有异常和恢复过程

### 2. 配置建议

- 根据不同交易对调整限制参数
- 定期审查和更新恢复策略配置
- 监控恢复成功率，优化恢复算法

### 3. 运维监控

- 建立异常告警机制
- 监控异常频率和类型分布
- 定期分析异常原因，改进系统

## 总结

该异常处理机制提供了：

✅ **完整的异常类型覆盖** - 涵盖所有常见交易风险  
✅ **智能自动恢复** - 多策略尝试，最大化成功率  
✅ **保守降级方案** - 确保系统在任何情况下都能继续运行  
✅ **丰富的监控信息** - 便于问题诊断和系统优化  
✅ **无缝系统集成** - 不影响现有业务逻辑  

这套机制大大提高了合约交易系统的稳定性和可靠性，确保在各种异常情况下都能安全、稳定地运行。