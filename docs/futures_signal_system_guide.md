# 期货信号系统使用指南

## 概述

`FuturesSignalSystem` 是一个完整的期货交易信号生成系统，提供多时间框架信号聚合、智能杠杆控制、风险评估等功能。

## 核心功能

### 1. 多时间框架信号聚合
- 支持 5m、15m、30m、1h、4h 等时间框架
- 使用加权算法聚合不同时间框架的信号
- 时间框架权重配置：5m(0.1), 15m(0.15), 30m(0.2), 1h(0.3), 4h(0.25)

### 2. 信号强度计算
- 计算综合信号强度（-1 到 1 范围）
- 基于信号阈值确定交易方向：
  - Strong Long: ≥ 0.7
  - Long: ≥ 0.5
  - Neutral: -0.2 到 0.2
  - Short: ≤ -0.5
  - Strong Short: ≤ -0.7

### 3. 置信度计算
提供多维度置信度评估：
- **信号一致性**: 同方向信号占比
- **时间框架一致性**: 重要时间框架对齐度
- **成交量确认**: 成交量放大验证
- **趋势对齐度**: 基于移动平均线的趋势分析
- **动量强度**: RSI和价格动量评估
- **支撑阻力确认**: 关键价位确认

### 4. 操作类型决策
根据当前持仓和信号强度决定操作：
- `OPEN`: 开仓
- `ADD`: 加仓
- `REDUCE`: 减仓
- `CLOSE`: 平仓
- `HOLD`: 持仓不变

### 5. 杠杆和风险管理
- 集成智能杠杆控制器
- 基于账户风险等级动态调整
- 考虑市场条件和波动率
- 计算最优仓位大小

### 6. 止盈止损计算
- 动态风险收益比（1.5-2.5）
- 基于杠杆调整止损距离
- 考虑信号置信度和强度
- 精确的价格精度处理

## 快速开始

### 基本使用

```python
import asyncio
from src.futures.signals import FuturesSignalSystem
from src.futures.leverage.leverage_controller import LeverageController

# 创建系统实例
leverage_controller = LeverageController()
signal_system = FuturesSignalSystem(leverage_controller=leverage_controller)

# 初始化系统
signal_system.initialize()

# 准备市场数据 (多时间框架)
market_data = {
    '5m': pd.DataFrame(...),   # 5分钟数据
    '15m': pd.DataFrame(...),  # 15分钟数据
    '30m': pd.DataFrame(...),  # 30分钟数据
    '1h': pd.DataFrame(...),   # 1小时数据
    '4h': pd.DataFrame(...)    # 4小时数据
}

# 生成交易信号
async def generate_signal():
    signal = await signal_system.generate_signal(
        ticker="BTCUSDT",
        market_data=market_data,
        current_price=45000.0,
        current_positions=positions,  # 可选
        margin_status=margin_status,  # 可选
        leverage_strategy=LeverageStrategy.MODERATE
    )

    if signal:
        print(f"交易方向: {signal.direction.value}")
        print(f"操作类型: {signal.operation_type.value}")
        print(f"建议杠杆: {signal.suggested_leverage}x")
        print(f"置信度: {signal.confidence:.1f}%")
        print(f"止盈价格: ${signal.take_profit_price:,.2f}")
        print(f"止损价格: ${signal.stop_loss_price:,.2f}")

# 运行
asyncio.run(generate_signal())
```

### 完整配置示例

```python
from src.futures.signals import FuturesSignalSystem, FuturesStrategyManager
from src.futures.leverage.leverage_controller import LeverageController
from src.futures.market.market_analyzer import MarketAnalyzer

# 创建组件
strategy_manager = FuturesStrategyManager()
leverage_controller = LeverageController()
market_analyzer = MarketAnalyzer()  # 可选

# 添加策略到管理器
strategy_manager.add_strategy("futures_macd", weight=0.4)
strategy_manager.add_strategy("futures_rsi", weight=0.35)

# 创建完整配置的信号系统
signal_system = FuturesSignalSystem(
    strategy_manager=strategy_manager,
    leverage_controller=leverage_controller,
    market_analyzer=market_analyzer,
    risk_params={
        'max_position_ratio': 0.2,
        'min_position_size': 10.0,
        'max_position_size': 10000.0
    }
)
```

## 数据结构

### 输入数据格式

#### 市场数据 (market_data)
```python
market_data = {
    '1h': pd.DataFrame({
        'timestamp': [...],
        'open': [...],
        'high': [...],
        'low': [...],
        'close': [...],
        'volume': [...]
    })
}
```

#### 保证金状态 (margin_status)
```python
from src.futures.models.data_models import MarginStatus, RiskLevel

margin_status = MarginStatus(
    total_balance=10000.0,
    available_margin=5000.0,
    used_margin=3000.0,
    margin_ratio=0.3,
    risk_level=RiskLevel.MEDIUM,
    can_trade=True,
    can_open_position=True
)
```

#### 当前持仓 (current_positions)
```python
from src.futures.models.data_models import Position, PositionSide

positions = [
    Position(
        ticker="BTCUSDT",
        side=PositionSide.LONG,
        size=0.1,
        entry_price=44000.0,
        current_price=45000.0,
        leverage=5.0,
        initial_margin=900.0,
        maintenance_margin=180.0,
        liquidation_price=40000.0
    )
]
```

### 输出信号格式

```python
# FuturesSignal 对象包含以下字段:
signal = FuturesSignal(
    ticker="BTCUSDT",                    # 交易对
    direction=TradingDirection.LONG,     # 交易方向
    operation_type=OperationType.OPEN,   # 操作类型
    confidence=85.5,                     # 置信度 (0-100)
    strength=0.742,                      # 信号强度 (0-1)

    # 杠杆和仓位
    suggested_leverage=5.0,              # 建议杠杆
    position_size=4200.0,                # 仓位大小 (USDT)

    # 价格信息
    entry_price=45000.0,                 # 入场价格
    current_price=45000.0,               # 当前价格
    take_profit_price=47250.0,           # 止盈价格
    stop_loss_price=43200.0,             # 止损价格

    # 风险和时间
    risk_level=RiskLevel.MEDIUM,         # 风险等级
    expiry_time=datetime(...),           # 信号过期时间

    # 元数据
    metadata={
        'expected_return': 2.5,          # 预期收益 (%)
        'expected_duration_hours': 12.0, # 预期持仓时间
        'risk_score': 35.0,              # 风险评分 (0-100)
        'processing_time_ms': 45.2       # 处理时间 (毫秒)
    }
)
```

## 置信度指标详解

### SignalConfidenceMetrics

```python
from src.futures.signals import SignalConfidenceMetrics

metrics = SignalConfidenceMetrics(
    signal_consistency=0.8,      # 信号一致性 (0-1)
    timeframe_agreement=0.7,     # 时间框架一致性 (0-1)
    volume_confirmation=0.6,     # 成交量确认度 (0-1)
    trend_alignment=0.75,        # 趋势对齐度 (0-1)
    momentum_strength=0.65,      # 动量强度 (0-1)
    support_resistance=0.7       # 支撑阻力确认 (0-1)
)

# 计算总体置信度
overall_confidence = metrics.calculate_overall_confidence()
print(f"总体置信度: {overall_confidence:.1%}")
```

### 置信度权重配置
- 信号一致性: 25%
- 时间框架一致性: 20%
- 成交量确认: 15%
- 趋势对齐度: 15%
- 动量强度: 15%
- 支撑阻力确认: 10%

## 系统监控

### 获取系统状态

```python
status = signal_system.get_system_status()
print(f"系统状态: {status}")

# 输出示例:
{
    "is_initialized": True,
    "last_signal_time": "2023-01-01T12:00:00",
    "performance_stats": {
        "total_signals": 150,
        "successful_signals": 142,
        "avg_processing_time": 0.085,
        "direction_distribution": {
            "long": 65,
            "short": 45,
            "neutral": 40
        }
    },
    "cached_signals": 5,
    "active_strategies": 2
}
```

### 重置统计数据

```python
# 重置性能统计
signal_system.reset_performance_stats()

# 清理缓存
signal_system.clear_cache()
```

## 最佳实践

### 1. 数据质量检查
确保市场数据完整且时间对齐：
```python
def validate_market_data(market_data):
    required_timeframes = ['5m', '15m', '30m', '1h', '4h']
    required_columns = ['open', 'high', 'low', 'close', 'volume']

    for tf in required_timeframes:
        if tf not in market_data:
            raise ValueError(f"缺少时间框架: {tf}")

        df = market_data[tf]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"时间框架 {tf} 缺少必要列")

        if len(df) < 50:
            raise ValueError(f"时间框架 {tf} 数据量不足")
```

### 2. 错误处理
```python
try:
    signal = await signal_system.generate_signal(...)
    if signal:
        # 处理信号
        process_signal(signal)
    else:
        logger.warning("未生成有效信号")

except Exception as e:
    logger.error(f"信号生成失败: {e}")
    # 执行降级逻辑
    handle_signal_failure()
```

### 3. 性能优化
```python
# 批量处理多个交易对
async def process_multiple_tickers(tickers, market_data_dict):
    tasks = []
    for ticker in tickers:
        task = signal_system.generate_signal(
            ticker=ticker,
            market_data=market_data_dict[ticker],
            current_price=get_current_price(ticker)
        )
        tasks.append(task)

    # 并发执行
    signals = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果
    valid_signals = [s for s in signals if isinstance(s, FuturesSignal)]
    return valid_signals
```

### 4. 风险控制
```python
def validate_signal_risk(signal, account_balance):
    """验证信号风险是否可接受"""

    # 检查杠杆限制
    if signal.suggested_leverage > 10:
        logger.warning(f"杠杆过高: {signal.suggested_leverage}")
        return False

    # 检查仓位大小
    max_position = account_balance * 0.2  # 最大20%仓位
    if signal.position_size > max_position:
        logger.warning(f"仓位过大: {signal.position_size} > {max_position}")
        return False

    # 检查风险等级
    if signal.risk_level == RiskLevel.EMERGENCY:
        logger.warning("风险等级过高")
        return False

    return True
```

## 测试和调试

### 运行测试
```bash
# 基础功能测试
python simple_test_futures_signal_system.py

# 完整功能测试（需要策略支持）
python test_futures_signal_system.py
```

### 调试模式
```python
import logging

# 启用详细日志
logging.getLogger('src.futures.signals').setLevel(logging.DEBUG)

# 或者设置特定组件的日志级别
logging.getLogger('src.futures.signals.futures_signal_system').setLevel(logging.DEBUG)
```

### 开发环境配置
确保使用testnet环境进行开发测试：
```python
import os
os.environ['BINANCE_ENVIRONMENT'] = 'testnet'
```

## 故障排除

### 常见问题

1. **导入错误**
   - 确保项目路径正确
   - 检查依赖模块是否已安装

2. **策略管理器无策略**
   - 系统会发出警告但仍可运行
   - 添加真实策略或使用模拟策略进行测试

3. **杠杆计算默认值**
   - 如果市场分析器不可用，会使用默认市场条件
   - 这是正常行为，不影响基本功能

4. **数据质量问题**
   - 确保市场数据包含所有必要时间框架
   - 检查数据的时间对齐和完整性

### 性能调优

1. **减少计算复杂度**
   - 根据需要选择时间框架
   - 调整置信度计算的精度

2. **缓存优化**
   - 定期清理过期缓存
   - 监控内存使用情况

3. **并发控制**
   - 合理设置线程池大小
   - 避免过度并发导致的资源竞争

## 更新日志

### v1.0.0 (当前版本)
- ✅ 完整的多时间框架信号聚合
- ✅ 智能杠杆控制集成
- ✅ 多维度置信度计算
- ✅ 操作类型决策矩阵
- ✅ 动态止盈止损计算
- ✅ 预期收益和风险评估
- ✅ 基础功能测试验证

### 下一版本计划
- 🔄 真实策略集成
- 🔄 完整的端到端测试
- 🔄 性能基准测试
- 🔄 止盈止损专用模块

## 贡献指南

1. 遵循现有代码风格
2. 添加适当的类型注解
3. 编写单元测试
4. 更新相关文档
5. 确保向后兼容性

---

*更新时间: 2025-09-14*