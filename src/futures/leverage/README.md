# 智能杠杆控制系统

智能杠杆控制系统是期货交易中的核心风险管理组件，提供多维度动态杠杆计算功能，确保交易安全性和收益性的最佳平衡。

## 🎯 核心特性

### 多维度动态杠杆计算
- **市场波动率分析**: 基于实时波动率自动调整杠杆倍数
- **流动性评估**: 考虑交易对流动性差异，优化杠杆配置
- **风险等级管控**: 根据账户风险状态分层限制杠杆
- **仓位集中度管理**: 防止过度集中带来的系统性风险
- **市场状态适应**: 针对不同市场环境（趋势、震荡、高波动）智能调整

### 安全机制
- **多重验证**: 多层次安全检查，防止异常杠杆
- **紧急降杠杆**: 极端情况下自动降低杠杆保护资金
- **实时监控**: 持续监控市场条件和风险指标
- **限制管理**: 灵活的交易对特定限制配置

## 🏗️ 系统架构

```
杠杆控制系统
├── LeverageController (核心控制器)
│   ├── 杠杆计算引擎
│   ├── 风险评估模块
│   ├── 市场分析集成
│   └── 缓存管理
├── 数据模型层
│   ├── LeverageConfig (配置模型)
│   ├── LeverageCalculationResult (计算结果)
│   ├── LeverageLimits (限制管理)
│   └── MarketCondition (市场条件)
├── 配置管理
│   └── LeverageConfigManager
└── 集成接口
    ├── MarketAnalyzer 集成
    ├── RiskManagement 集成
    └── Position 管理集成
```

## 🔄 核心算法

### 杠杆计算公式

```python
optimal_leverage = base_leverage × volatility_multiplier × liquidity_factor ×
                   market_adjustment × position_adjustment × risk_adjustment ×
                   time_adjustment × correlation_factor × momentum_factor
```

### 调整因子说明

| 因子 | 范围 | 说明 |
|------|------|------|
| volatility_multiplier | 0.6-1.2 | 基于市场波动率的调整 |
| liquidity_multiplier | 0.6-1.1 | 基于流动性的调整 |
| market_adjustment | 0.5-1.1 | 基于市场状态的调整 |
| position_adjustment | 0.7-1.0 | 基于仓位集中度的调整 |
| risk_adjustment | 0.3-1.0 | 基于账户风险等级的调整 |
| time_adjustment | 0.9-1.0 | 基于交易时段的调整 |

## 🚀 快速开始

### 基本使用

```python
from src.futures.leverage import LeverageController, LeverageStrategy

# 创建杠杆控制器
controller = LeverageController()

# 计算最优杠杆
result = await controller.calculate_optimal_leverage(
    ticker='BTCUSDT',
    strategy=LeverageStrategy.MODERATE
)

print(f"建议杠杆: {result.applied_leverage}")
print(f"风险等级: {result.risk_level.value}")
print(f"调整建议: {result.recommendation}")
```

### 高级配置

```python
from src.futures.leverage import LeverageConfig, LeverageLimits

# 创建自定义配置
config = LeverageConfig()
config.default_leverage = 8.0
config.emergency_leverage_cap = 3.0

# 添加交易对特定限制
btc_limits = LeverageLimits(ticker='BTCUSDT', max_leverage=125.0)
config.add_symbol_limits('BTCUSDT', btc_limits)

# 使用自定义配置
controller = LeverageController(config=config)
```

### 集成风险管理

```python
# 传入当前仓位和保证金状态
result = await controller.calculate_optimal_leverage(
    ticker='BTCUSDT',
    strategy=LeverageStrategy.MODERATE,
    current_positions=positions_list,
    margin_status=margin_info
)

# 检查计算结果
if result.has_warnings():
    print(f"警告: {result.warnings}")

if result.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
    print("账户风险过高，建议立即减仓")
```

## 📊 风险等级体系

### 账户风险等级

| 等级 | 保证金率 | 杠杆限制 | 交易权限 |
|------|----------|----------|----------|
| LOW | < 50% | 100% | 全部功能 |
| MEDIUM | 50%-70% | 80% | 全部功能 |
| HIGH | 70%-80% | 60% | 限制开仓 |
| CRITICAL | 80%-90% | 40% | 禁止交易 |
| EMERGENCY | > 90% | 20% | 强制平仓 |

### 杠杆策略配置

| 策略 | 基础杠杆 | 适用场景 | 风险特征 |
|------|----------|----------|----------|
| CONSERVATIVE | 3x | 新手/稳健投资 | 低风险低收益 |
| MODERATE | 5x | 一般交易者 | 平衡风险收益 |
| AGGRESSIVE | 8x | 经验丰富者 | 高风险高收益 |
| EXPERT | 12x | 专业交易员 | 极高风险 |

## 🔧 配置管理

### 配置文件示例

```yaml
# leverage_config.yaml
default_leverage: 5.0

base_leverage_by_strategy:
  conservative: 3.0
  moderate: 5.0
  aggressive: 8.0
  expert: 12.0

# 风险控制参数
max_adjustment_percentage: 50.0
min_adjustment_percentage: 10.0
emergency_leverage_cap: 2.0
high_volatility_threshold: 0.4
low_liquidity_threshold: 0.3

# 时间窗口设置
volatility_window_minutes: 60
liquidity_window_minutes: 30
position_check_interval_seconds: 30

# 安全开关
enable_emergency_reduction: true
enable_position_size_limits: true
enable_correlation_checks: true

# 交易对特定限制
symbol_specific_limits:
  BTCUSDT:
    max_leverage: 125.0
    min_leverage: 1.0
    volatility_threshold: 0.3
    risk_based_limits:
      low: 125.0
      medium: 100.0
      high: 75.0
      critical: 50.0
      emergency: 25.0
```

### 配置管理

```python
from src.futures.leverage import LeverageConfigManager
from pathlib import Path

# 创建配置管理器
config_manager = LeverageConfigManager()

# 加载配置
config = config_manager.load_config(Path('leverage_config.yaml'))

# 验证配置
errors = config_manager.validate_config(config)
if errors:
    print(f"配置错误: {errors}")

# 保存配置
config_manager.save_config(config, Path('new_config.yaml'))
```

## 🔌 系统集成

### 与MarketAnalyzer集成

```python
from src.futures.market import MarketAnalyzer

# 创建市场分析器
market_analyzer = MarketAnalyzer(api_client)

# 集成到杠杆控制器
controller = LeverageController(
    config=config,
    market_analyzer=market_analyzer
)

# 自动获取市场数据进行杠杆计算
result = await controller.calculate_optimal_leverage(
    ticker='BTCUSDT',
    strategy=LeverageStrategy.MODERATE
)
```

### 与RiskManagement集成

```python
# 在风险管理节点中使用杠杆控制器
class RiskManagementNode:
    def __init__(self):
        self.leverage_controller = LeverageController(config)

    async def assess_position_risk(self, signal):
        # 计算建议杠杆
        leverage_result = await self.leverage_controller.calculate_optimal_leverage(
            ticker=signal.ticker,
            strategy=self.get_user_strategy(),
            current_positions=self.get_current_positions(),
            margin_status=self.get_margin_status()
        )

        # 应用杠杆建议
        signal.suggested_leverage = leverage_result.applied_leverage
        return leverage_result
```

## 📈 性能监控

### 统计信息

```python
# 获取控制器统计
stats = controller.get_leverage_statistics()
print(f"计算次数: {stats['calculation_count']}")
print(f"调整统计: {stats['adjustment_stats']}")
print(f"缓存使用: {stats['cache_size']}")
```

### 实时监控

```python
async def monitor_leverage_adjustments():
    controller = LeverageController()

    while True:
        for ticker in ['BTCUSDT', 'ETHUSDT']:
            result = await controller.calculate_optimal_leverage(
                ticker=ticker,
                strategy=LeverageStrategy.MODERATE
            )

            if result.is_adjustment_significant(threshold=0.2):
                print(f"显著调整: {ticker} 杠杆调整至 {result.applied_leverage:.2f}")

            if result.has_warnings():
                print(f"警告: {ticker} - {result.warnings}")

        await asyncio.sleep(30)  # 30秒检查一次
```

## 🧪 测试

### 运行单元测试

```bash
cd src/futures/leverage
python -m pytest test_leverage_controller.py -v
```

### 测试覆盖率

```bash
cd src/futures/leverage
python -m pytest test_leverage_controller.py --cov=leverage_controller --cov-report=html
```

### 运行使用示例

```bash
cd src/futures/leverage
python usage_examples.py
```

## 🐛 故障排除

### 常见问题

1. **异步方法调用问题**
   ```python
   # ❌ 错误 - 在同步环境中调用异步方法
   result = controller.calculate_optimal_leverage(...)

   # ✅ 正确 - 使用同步包装器
   result = controller.calculate_leverage_sync(...)

   # ✅ 正确 - 在异步环境中调用
   result = await controller.calculate_optimal_leverage(...)
   ```

2. **配置验证失败**
   ```python
   errors = config_manager.validate_config(config)
   if errors:
       for error in errors:
           print(f"配置错误: {error}")
   ```

3. **市场数据获取失败**
   ```python
   # 手动提供市场数据作为备份
   market_data = {
       'volatility': 0.2,
       'liquidity_score': 0.7,
       'market_regime': 'calm'
   }

   result = await controller.calculate_optimal_leverage(
       ticker='BTCUSDT',
       market_data=market_data
   )
   ```

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 控制器将输出详细的调试信息
controller = LeverageController()
```

## 📝 更新日志

### v1.0.0 (当前版本)
- ✅ 多维度动态杠杆计算
- ✅ 完整的风险管理体系
- ✅ MarketAnalyzer集成
- ✅ 灵活的配置管理
- ✅ 全面的单元测试
- ✅ 详细的使用示例

### 未来计划
- 🔄 机器学习优化杠杆算法
- 🔄 更多市场指标集成
- 🔄 实时新闻情感分析
- 🔄 历史回测功能
- 🔄 Web界面管理工具

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 支持

如有问题或建议，请：
- 创建 [Issue](https://github.com/your-repo/issues)
- 查看 [文档](https://your-docs-url)
- 联系维护团队

---

*智能杠杆控制系统 - 让期货交易更安全、更智能* 🚀