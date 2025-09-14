# 期货保证金管理器 (Margin Manager)

期货保证金管理器是一个完整的保证金管理系统，提供实时保证金计算、风险评估、强平价格计算和监控功能。

## 功能特性

### 核心功能
- ✅ **保证金计算**: 初始保证金和维持保证金计算
- ✅ **强平价格计算**: 多头/空头强制平仓价格计算
- ✅ **实时数据集成**: WebSocket数据流集成（含Mock实现）
- ✅ **保证金充足性检查**: 开仓前保证金验证
- ✅ **风险评估**: 多级风险等级评估
- ✅ **实时监控**: 保证金健康状态持续监控
- ✅ **完整类型注解**: 全面的类型提示支持
- ✅ **错误处理**: 完善的异常处理和日志记录

### 架构设计

```
src/futures/margin/
├── __init__.py              # 模块导出
├── margin_manager.py        # 核心保证金管理器
├── test_margin_manager.py   # 完整测试套件
├── usage_example.py         # 使用示例
└── README.md               # 本文档
```

## 快速开始

### 基本使用

```python
import asyncio
from src.futures.margin import MarginManager, MarginConfig

async def main():
    # 1. 创建配置
    config = MarginConfig(
        initial_cash=10000.0,        # 初始资金
        initial_margin_rate=0.1,     # 10%初始保证金率
        maintenance_margin_rate=0.05, # 5%维持保证金率
        margin_call_threshold=0.3,   # 30%追保阈值
        liquidation_threshold=0.8,   # 80%强平阈值
        max_leverage=20              # 最大20倍杠杆
    )

    # 2. 创建管理器
    margin_manager = MarginManager(config)

    try:
        # 3. 初始化
        await margin_manager.initialize()

        # 4. 计算保证金
        initial_margin = margin_manager.calculate_initial_margin(
            position_value=5000.0,  # 5000 USDT仓位价值
            leverage=10             # 10倍杠杆
        )
        print(f"所需初始保证金: {initial_margin} USDT")

        # 5. 计算强平价格
        liquidation_price = margin_manager.calculate_liquidation_price(
            entry_price=50000.0,    # 入场价格
            position_size=0.1,      # 仓位大小
            margin=initial_margin,  # 保证金
            side="long"            # 多头
        )
        print(f"强平价格: {liquidation_price} USDT")

        # 6. 检查保证金充足性
        status = margin_manager.check_margin_sufficiency(initial_margin)
        print(f"可以开仓: {status.can_open_position}")

    finally:
        await margin_manager.cleanup()

# 运行示例
asyncio.run(main())
```

### 保证金监控

```python
async def monitoring_example():
    margin_manager = MarginManager(config)

    try:
        await margin_manager.initialize()

        # 启动实时监控
        await margin_manager.start_monitoring()

        # 监控将自动检测风险等级并发出警告
        await asyncio.sleep(10)  # 运行10秒监控

        # 停止监控
        await margin_manager.stop_monitoring()

    finally:
        await margin_manager.cleanup()
```

## API 文档

### MarginConfig 配置类

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_cash` | float | 1000.0 | 初始资金 (USDT) |
| `initial_margin_rate` | float | 0.1 | 初始保证金率 (10%) |
| `maintenance_margin_rate` | float | 0.05 | 维持保证金率 (5%) |
| `margin_call_threshold` | float | 0.3 | 追保阈值 (30%) |
| `liquidation_threshold` | float | 0.8 | 强平阈值 (80%) |
| `emergency_threshold` | float | 0.9 | 紧急阈值 (90%) |
| `max_leverage` | int | 20 | 最大杠杆倍数 |
| `currency` | str | "USDT" | 保证金货币 |

### MarginManager 主要方法

#### 初始化方法

```python
async def initialize() -> None
```
初始化管理器，建立WebSocket连接并获取账户数据。

#### 保证金计算方法

```python
def calculate_initial_margin(
    position_value: float,
    leverage: int
) -> float
```
计算开仓所需的初始保证金。

```python
def calculate_maintenance_margin(
    position_value: float,
    leverage: int
) -> float
```
计算维持仓位所需的保证金。

#### 强平价格计算

```python
def calculate_liquidation_price(
    entry_price: float,
    position_size: float,
    margin: float,
    side: str  # "long" | "short"
) -> float
```
计算强制平仓价格。

#### 保证金状态检查

```python
def check_margin_sufficiency(
    required_margin: float
) -> MarginStatus
```
检查保证金是否充足，返回详细的保证金状态。

#### 风险评估

```python
async def assess_risk_level() -> RiskLevel
```
评估当前风险等级：
- `LOW`: 低风险 (< 20%使用率)
- `MEDIUM`: 中等风险 (20%-30%使用率)
- `HIGH`: 高风险 (30%-80%使用率)
- `CRITICAL`: 关键风险 (80%-90%使用率)
- `EMERGENCY`: 紧急风险 (> 90%使用率)

#### 监控方法

```python
async def start_monitoring() -> None
async def stop_monitoring() -> None
```
启动/停止保证金健康状态监控。

#### 实用方法

```python
def get_available_margin() -> float
def get_margin_ratio() -> float
def get_status_summary() -> Dict[str, Any]
```

## 测试

运行完整测试套件：

```bash
uv run python -m src.futures.margin.test_margin_manager
```

运行使用示例：

```bash
uv run python -m src.futures.margin.usage_example
```

## 配置集成

### 与config.yaml集成

保证金管理器会自动读取项目配置文件中的相关参数：

```yaml
# config.yaml
initial_cash: 10000
futures:
  risk:
    max_leverage: 20
    margin_requirement: 0.1
    stop_loss_percentage: 0.02
    daily_loss_limit: 500.0
```

### WebSocket集成

当前提供Mock WebSocket接口用于测试，在实际部署时会连接到真实的币安期货WebSocket流：

```python
# 自定义WebSocket接口
custom_ws_interface = YourWebSocketInterface()
margin_manager = MarginManager(config, custom_ws_interface)
```

## 错误处理

管理器使用项目统一的异常体系：

```python
from src.utils.exceptions import MarginInsufficientError

try:
    status = margin_manager.check_margin_sufficiency(required_margin)
    if not status.can_open_position:
        # 处理保证金不足情况
        pass
except MarginInsufficientError as e:
    print(f"保证金不足: {e}")
    print(f"建议: {e.recovery_suggestion}")
```

## 性能特性

- **异步设计**: 全面支持异步操作
- **实时计算**: 基于最新WebSocket数据
- **内存效率**: 最小化内存占用
- **快速响应**: 毫秒级计算响应
- **可扩展**: 支持多个交易对并发管理

## 安全特性

- **参数验证**: 严格的输入参数验证
- **范围检查**: 杠杆、保证金率等参数范围验证
- **异常处理**: 完善的错误恢复机制
- **日志记录**: 详细的操作日志记录
- **资源清理**: 自动资源清理和连接管理

## 注意事项

1. **测试环境**: 当前使用Mock WebSocket接口，生产环境需要连接真实数据流
2. **配置调优**: 根据实际交易策略调整保证金率和风险阈值
3. **监控频率**: 默认每5秒检查一次，可根据需要调整
4. **资源管理**: 使用完毕后请调用`cleanup()`方法清理资源

## 未来扩展

- [ ] 支持多种保证金模式（全仓/逐仓）
- [ ] 集成实时价格预警
- [ ] 支持自动风险控制操作
- [ ] 添加历史保证金数据分析
- [ ] 支持多交易所适配

## 相关文档

- [期货交易配置文档](../config/README.md)
- [WebSocket接口文档](../interfaces/README.md)
- [数据模型文档](../models/README.md)
- [设计文档](.kiro/specs/futures-trading-refactor/design.md)