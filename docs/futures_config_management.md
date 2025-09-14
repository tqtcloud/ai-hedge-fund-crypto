# 期货配置管理系统

本系统提供期货交易的配置验证、API管理和环境检查功能，确保期货交易的安全性和可靠性。

## 功能概述

### 1. 期货配置验证器 (`FuturesConfigValidator`)
- 验证期货交易相关配置的完整性和正确性
- 支持testnet/mainnet环境验证
- 验证保证金、杠杆等期货特有参数
- 集成到现有配置系统

### 2. API安全管理器 (`FuturesAPIManager`)
- 安全管理Binance期货API密钥
- 支持testnet/mainnet不同的API配置
- API权限和有效性检查
- 环境变量安全加载

### 3. 环境配置检查器 (`EnvironmentChecker`)
- 自动检测和切换testnet/mainnet环境
- 验证不同环境的配置完整性
- 提供环境状态报告

## 配置文件结构

### config.yaml 新增期货配置部分

```yaml
# 期货交易配置
futures:
  # 环境配置 (默认使用testnet确保安全)
  environment: testnet  # testnet | mainnet
  
  # 保证金类型
  margin_type: CROSSED  # CROSSED | ISOLATED
  
  # 交易对配置
  symbols: ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
  
  # 默认杠杆配置
  default_leverage:
    BTCUSDT: 10
    ETHUSDT: 10
    BNBUSDT: 5
  
  # 允许的订单类型
  allowed_order_types: ["MARKET", "LIMIT", "STOP", "TAKE_PROFIT"]
  
  # 风险控制配置
  risk:
    max_leverage: 20                # 最大杠杆倍数
    max_position_size: 1000.0       # 最大仓位大小 (USDT)
    stop_loss_percentage: 0.02      # 止损百分比 (2%)
    take_profit_percentage: 0.05    # 止盈百分比 (5%)
    daily_loss_limit: 500.0         # 日损失限额 (USDT)
    max_open_positions: 5           # 最大开仓数量
    margin_requirement: 0.1         # 保证金要求比例 (10%)
  
  # WebSocket配置
  websocket:
    enable_user_data_stream: true   # 启用用户数据流
    enable_market_data_stream: true # 启用市场数据流
    reconnect_interval: 5           # 重连间隔 (秒)
    max_reconnect_attempts: 10      # 最大重连尝试次数
    heartbeat_interval: 30          # 心跳间隔 (秒)
    buffer_size: 1024              # 缓冲区大小
```

### 环境变量配置

在 `.env` 文件中配置API密钥：

```bash
# Binance API keys (required for data access)
BINANCE_API_KEY=your-binance-api-key
BINANCE_API_SECRET=your-binance-api-secret

# Binance Futures API keys (optional, separate keys for futures trading)
BINANCE_FUTURES_API_KEY=your-binance-futures-api-key
BINANCE_FUTURES_API_SECRET=your-binance-futures-api-secret

# Binance Futures Testnet API keys (recommended for testing)
BINANCE_FUTURES_TESTNET_API_KEY=your-binance-futures-testnet-api-key
BINANCE_FUTURES_TESTNET_API_SECRET=your-binance-futures-testnet-api-secret

# Optional: Encryption key for securing API credentials in files
ENCRYPTION_KEY=your-encryption-key-for-api-credentials
```

## 使用方法

### 1. 命令行工具

使用 `scripts/futures_config_cli.py` 进行各种配置管理操作：

```bash
# 验证配置文件
python scripts/futures_config_cli.py validate --config config.yaml --verbose

# 检查testnet API连接
python scripts/futures_config_cli.py check-api --testnet --verbose

# 检查环境状态并保存报告
python scripts/futures_config_cli.py check-env --testnet --output testnet_report.yaml

# 检查所有环境
python scripts/futures_config_cli.py check-all

# 生成默认配置文件
python scripts/futures_config_cli.py generate --output new_config.yaml

# 生成mainnet配置文件（更保守的设置）
python scripts/futures_config_cli.py generate --output mainnet_config.yaml --mainnet
```

### 2. Python API

在代码中使用期货配置管理功能：

```python
import asyncio
from src.futures.config import FuturesConfigValidator, FuturesAPIManager, EnvironmentChecker
from src.futures.config.api_manager import APIEnvironment

async def setup_futures_trading():
    # 1. 验证配置
    validator = FuturesConfigValidator()
    config = validator.validate_from_file("config.yaml")
    
    # 2. 检查环境并选择最佳环境
    checker = EnvironmentChecker()
    optimal_env = await checker.auto_switch_environment()
    
    # 3. 设置API管理器
    api_manager = FuturesAPIManager(optimal_env)
    credentials = api_manager.load_credentials_from_env()
    
    # 4. 验证API连接
    api_result = await api_manager.validate_api_credentials()
    
    if api_result['valid']:
        print("✅ 期货交易系统准备就绪")
        return api_manager, config
    else:
        raise Exception("❌ API验证失败")

# 运行设置
api_manager, config = asyncio.run(setup_futures_trading())
```

### 3. 集成到现有系统

更新现有的设置加载方式以支持期货配置验证：

```python
from src.utils.settings import load_settings_with_futures_validation

# 加载设置并进行期货配置验证
settings = load_settings_with_futures_validation("config.yaml")

# 检查是否包含期货配置
if settings.futures:
    print(f"期货环境: {settings.futures.environment}")
    print(f"交易对: {settings.futures.symbols}")
    print(f"最大杠杆: {settings.futures.risk.max_leverage}")
```

## 安全建议

### 1. 环境选择
- **开发和测试**: 始终使用 `testnet` 环境
- **生产环境**: 充分测试后才使用 `mainnet` 环境
- 系统默认推荐 `testnet` 环境确保安全

### 2. API密钥管理
- 使用独立的期货API密钥
- 定期轮换API密钥
- 限制API权限仅包含必要的期货交易权限
- 考虑使用加密存储敏感凭证

### 3. 风险控制
- 设置合理的杠杆限制
- 配置止损和止盈比例
- 设置日损失限额
- 限制同时开仓数量

### 4. 配置验证
- 定期验证配置文件
- 监控环境健康状态
- 检查API连接和权限
- 记录所有配置变更

## 故障排除

### 常见问题

1. **配置验证失败**
   - 检查config.yaml格式是否正确
   - 验证期货配置部分的参数值
   - 确认所有必需字段都已配置

2. **API连接失败**
   - 检查环境变量是否正确设置
   - 验证API密钥是否有效
   - 确认网络连接和防火墙设置
   - 检查API权限是否包含期货交易

3. **环境检查异常**
   - 检查网络连通性
   - 验证DNS解析
   - 确认防火墙设置允许WebSocket连接

### 调试模式

启用详细日志记录：

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 在CLI工具中使用 --verbose 选项获取详细输出
python scripts/futures_config_cli.py validate --config config.yaml --verbose
```

## 文件结构

```
src/futures/config/
├── __init__.py                 # 模块初始化
├── config_validator.py         # 期货配置验证器
├── api_manager.py              # API安全管理器
├── environment_checker.py      # 环境配置检查器
└── usage_example.py            # 使用示例

scripts/
└── futures_config_cli.py       # 命令行工具

tests/
└── test_futures_config.py      # 测试文件

docs/
└── futures_config_management.md # 本文档
```

## 扩展功能

系统设计支持以下扩展：

1. **多交易所支持**: 轻松扩展到其他期货交易所
2. **配置模板**: 为不同交易策略提供预设配置
3. **监控集成**: 集成到监控系统进行实时状态检查
4. **自动化部署**: 支持CI/CD流程中的配置验证

## 许可证

本项目遵循项目主许可证。使用期货交易功能需要自行承担风险，建议在充分测试后使用。