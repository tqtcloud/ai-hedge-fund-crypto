"""
Binance SDK集成使用示例

演示如何使用重构后的期货交易数据类和SDK管理器。
包含常见使用场景和最佳实践。
"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

from .sdk_manager import (
    get_global_sdk_manager,
    BinanceSDKManager,
    SDKCredentials,
    create_sdk_manager_from_config
)
from ..models.data_models import (
    FuturesSignal,
    Position,
    MarginStatus,
    TradingDirection,
    OperationType,
    PositionSide,
    BinanceSDKCompatibility,
    create_position_from_binance_data,
    create_margin_status_from_binance_data,
    convert_signal_to_binance_order
)
from ..constants import (
    get_current_environment,
    is_testnet_environment,
    get_sdk_environment_config
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FuturesTradingExample:
    """期货交易SDK使用示例类"""
    
    def __init__(self):
        """初始化示例类"""
        self.sdk_manager = get_global_sdk_manager()
        logger.info(f"示例类初始化完成 - 环境: {get_current_environment()}")
    
    async def example_1_basic_setup(self):
        """示例1: 基本设置和连接测试"""
        print("\n=== 示例1: 基本设置和连接测试 ===")
        
        # 检查SDK可用性
        is_available = self.sdk_manager.is_sdk_available()
        print(f"SDK可用状态: {is_available}")
        
        # 获取环境信息
        env_info = self.sdk_manager.get_environment_info()
        print(f"当前环境: {env_info['environment']}")
        print(f"是否测试网: {env_info['is_testnet']}")
        print(f"REST API URL: {env_info['rest_api_url']}")
        print(f"WebSocket URL: {env_info['ws_streams_url']}")
        
        # 测试连接
        if is_available:
            connection_test = self.sdk_manager.test_connection()
            print(f"连接测试结果: {connection_test}")
        else:
            print("SDK不可用，跳过连接测试")
    
    async def example_2_create_trading_signal(self):
        """示例2: 创建和使用交易信号"""
        print("\n=== 示例2: 创建和使用交易信号 ===")
        
        # 创建交易信号
        signal = FuturesSignal(
            ticker="BTCUSDT",
            direction=TradingDirection.LONG,
            operation_type=OperationType.OPEN,
            confidence=85.0,
            strength=0.75,
            suggested_leverage=10.0,
            position_size=1000.0,  # USDT
            entry_price=50000.0,
            take_profit_price=55000.0,
            stop_loss_price=48000.0,
            strategy_source="example_strategy",
            signal_id="example_signal_001"
        )
        
        print(f"交易信号创建成功:")
        print(f"- 交易对: {signal.ticker}")
        print(f"- 方向: {signal.direction.value}")
        print(f"- 操作: {signal.operation_type.value}")
        print(f"- 置信度: {signal.confidence}%")
        print(f"- 建议杠杆: {signal.suggested_leverage}x")
        
        # 计算风险收益比
        risk_reward = signal.calculate_risk_reward_ratio()
        if risk_reward:
            print(f"- 风险收益比: {risk_reward:.2f}")
        
        # 检查信号有效性
        is_valid = signal.is_valid()
        print(f"- 信号有效性: {is_valid}")
        
        # 转换为字典格式（包含Binance格式）
        signal_dict = signal.to_dict()
        print(f"- 包含Binance格式: {'binance_order_format' in signal_dict}")
        
        # 转换为Binance订单格式
        if self.sdk_manager.is_sdk_available():
            try:
                order_params = convert_signal_to_binance_order(signal, quantity=0.02)
                print(f"Binance订单参数: {order_params}")
            except Exception as e:
                print(f"转换订单格式时出错: {e}")
        
        return signal
    
    async def example_3_position_management(self):
        """示例3: 仓位管理示例"""
        print("\n=== 示例3: 仓位管理示例 ===")
        
        if not self.sdk_manager.is_sdk_available():
            print("SDK不可用，使用模拟数据演示")
            # 创建模拟仓位数据
            mock_position = Position(
                ticker="BTCUSDT",
                side=PositionSide.LONG,
                size=0.02,
                entry_price=50000.0,
                current_price=51000.0,
                leverage=10.0,
                initial_margin=100.0,
                maintenance_margin=5.0
            )
            
            print(f"模拟仓位:")
            print(f"- 交易对: {mock_position.ticker}")
            print(f"- 方向: {mock_position.side.value}")
            print(f"- 大小: {mock_position.size}")
            print(f"- 未实现盈亏: {mock_position.unrealized_pnl:.2f} USDT")
            print(f"- 投资回报率: {mock_position.roe_percentage:.2f}%")
            print(f"- 风险等级: {mock_position.get_risk_level().value}")
            
            return [mock_position]
        
        try:
            # 获取真实持仓信息
            rest_client = self.sdk_manager.get_rest_client()
            response = rest_client.rest_api.position_information()
            position_data = response.data()
            
            positions = []
            for pos_data in position_data:
                if float(getattr(pos_data, 'positionAmt', 0)) != 0:  # 只处理有持仓的
                    position = Position.from_binance_position_response(pos_data)
                    positions.append(position)
            
            print(f"当前持仓数量: {len(positions)}")
            for i, pos in enumerate(positions, 1):
                print(f"\n持仓 {i}:")
                print(f"- 交易对: {pos.ticker}")
                print(f"- 方向: {pos.side.value}")
                print(f"- 大小: {pos.size}")
                print(f"- 入场价格: {pos.entry_price}")
                print(f"- 当前价格: {pos.current_price}")
                print(f"- 未实现盈亏: {pos.unrealized_pnl:.2f} USDT")
                print(f"- 投资回报率: {pos.roe_percentage:.2f}%")
                print(f"- 风险等级: {pos.get_risk_level().value}")
                
                # 检查强平风险
                if pos.is_at_risk():
                    print(f"⚠️ 警告: 距离强平较近!")
            
            return positions
            
        except Exception as e:
            logger.error(f"获取持仓信息失败: {e}")
            return []
    
    async def example_4_margin_status(self):
        """示例4: 保证金状态管理"""
        print("\n=== 示例4: 保证金状态管理 ===")
        
        if not self.sdk_manager.is_sdk_available():
            print("SDK不可用，使用模拟数据演示")
            # 创建模拟保证金状态
            mock_margin = MarginStatus(
                total_balance=10000.0,
                available_margin=8000.0,
                used_margin=2000.0,
                margin_ratio=0.2
            )
            
            print(f"模拟保证金状态:")
            print(f"- 总余额: {mock_margin.total_balance:.2f} USDT")
            print(f"- 可用保证金: {mock_margin.available_margin:.2f} USDT")
            print(f"- 已用保证金: {mock_margin.used_margin:.2f} USDT")
            print(f"- 保证金率: {mock_margin.margin_ratio:.1%}")
            print(f"- 风险等级: {mock_margin.risk_level.value}")
            print(f"- 可以交易: {mock_margin.can_trade}")
            print(f"- 可以开仓: {mock_margin.can_open_position}")
            
            return mock_margin
        
        try:
            # 获取账户信息
            rest_client = self.sdk_manager.get_rest_client()
            response = rest_client.rest_api.account_information()
            account_data = response.data()
            
            # 转换为项目的MarginStatus对象
            margin_status = MarginStatus.from_binance_account_response(account_data)
            
            print(f"账户保证金状态:")
            print(f"- 总余额: {margin_status.total_balance:.2f} USDT")
            print(f"- 可用保证金: {margin_status.available_margin:.2f} USDT")
            print(f"- 已用保证金: {margin_status.used_margin:.2f} USDT")
            print(f"- 保证金率: {margin_status.margin_ratio:.1%}")
            print(f"- 风险等级: {margin_status.risk_level.value}")
            print(f"- 账户状态: {margin_status.account_status}")
            print(f"- 可以交易: {margin_status.can_trade}")
            print(f"- 可以开仓: {margin_status.can_open_position}")
            
            # 计算最大仓位大小
            max_position = margin_status.get_max_position_size(
                leverage=10.0,
                price=50000.0
            )
            print(f"- 最大仓位大小(10x杠杆): {max_position:.6f} BTC")
            
            return margin_status
            
        except Exception as e:
            logger.error(f"获取保证金状态失败: {e}")
            return None
    
    async def example_5_websocket_streams(self):
        """示例5: WebSocket数据流订阅"""
        print("\n=== 示例5: WebSocket数据流订阅 ===")
        
        if not self.sdk_manager.is_sdk_available():
            print("SDK不可用，跳过WebSocket示例")
            return
        
        try:
            ws_client = self.sdk_manager.get_websocket_streams_client()
            connection = None
            
            try:
                connection = await ws_client.websocket_streams.create_connection()
                
                # 订阅K线数据
                stream = await connection.klineStream(
                    symbol="btcusdt",
                    interval="1m"
                )
                
                # 数据计数器
                message_count = 0
                max_messages = 5  # 只接收5条消息作为示例
                
                def handle_kline_data(data):
                    nonlocal message_count
                    message_count += 1
                    
                    print(f"接收到K线数据 #{message_count}:")
                    kline = data.get('k', {})
                    print(f"- 交易对: {kline.get('s', 'N/A')}")
                    print(f"- 开盘价: {kline.get('o', 'N/A')}")
                    print(f"- 最高价: {kline.get('h', 'N/A')}")
                    print(f"- 最低价: {kline.get('l', 'N/A')}")
                    print(f"- 收盘价: {kline.get('c', 'N/A')}")
                    print(f"- 成交量: {kline.get('v', 'N/A')}")
                    print("-" * 40)
                
                # 设置消息处理器
                stream.on("message", handle_kline_data)
                
                print(f"开始接收BTCUSDT 1分钟K线数据（最多{max_messages}条）...")
                
                # 等待接收指定数量的消息
                while message_count < max_messages:
                    await asyncio.sleep(0.1)
                
                # 取消订阅
                await stream.unsubscribe()
                print("已取消订阅K线数据流")
                
            finally:
                if connection:
                    await connection.close_connection(close_session=True)
                    print("WebSocket连接已关闭")
                    
        except Exception as e:
            logger.error(f"WebSocket流订阅失败: {e}")
    
    async def example_6_websocket_api(self):
        """示例6: WebSocket API调用"""
        print("\n=== 示例6: WebSocket API调用 ===")
        
        if not self.sdk_manager.is_sdk_available():
            print("SDK不可用，跳过WebSocket API示例")
            return
        
        try:
            ws_api_client = self.sdk_manager.get_websocket_api_client()
            connection = None
            
            try:
                connection = await ws_api_client.websocket_api.create_connection()
                
                # 通过WebSocket API获取持仓信息
                response = await ws_api_client.websocket_api.position_information()
                position_data = response.data()
                
                print("通过WebSocket API获取的持仓信息:")
                active_positions = 0
                
                for pos_data in position_data:
                    position_amt = float(getattr(pos_data, 'positionAmt', 0))
                    if position_amt != 0:
                        active_positions += 1
                        symbol = getattr(pos_data, 'symbol', 'N/A')
                        side = getattr(pos_data, 'positionSide', 'N/A')
                        entry_price = getattr(pos_data, 'entryPrice', 'N/A')
                        unrealized_pnl = getattr(pos_data, 'unRealizedProfit', 'N/A')
                        
                        print(f"- {symbol} {side}: 数量={position_amt}, 入场价={entry_price}, 盈亏={unrealized_pnl}")
                
                if active_positions == 0:
                    print("- 当前无活跃持仓")
                else:
                    print(f"总计: {active_positions} 个活跃持仓")
                
                # 获取账户信息
                account_response = await ws_api_client.websocket_api.account_information()
                account_data = account_response.data()
                
                total_balance = getattr(account_data, 'totalWalletBalance', 'N/A')
                available_balance = getattr(account_data, 'availableBalance', 'N/A')
                
                print(f"\n账户余额信息:")
                print(f"- 总余额: {total_balance} USDT")
                print(f"- 可用余额: {available_balance} USDT")
                
            finally:
                if connection:
                    await connection.close_connection(close_session=True)
                    print("WebSocket API连接已关闭")
                    
        except Exception as e:
            logger.error(f"WebSocket API调用失败: {e}")
    
    async def example_7_order_management(self):
        """示例7: 订单管理示例"""
        print("\n=== 示例7: 订单管理示例 ===")
        
        # 创建示例交易信号
        signal = FuturesSignal(
            ticker="BTCUSDT",
            direction=TradingDirection.LONG,
            operation_type=OperationType.OPEN,
            confidence=80.0,
            strength=0.6,
            suggested_leverage=5.0,
            position_size=100.0,  # 100 USDT
            entry_price=50000.0,
            strategy_source="example_strategy"
        )
        
        print("基于交易信号的订单管理:")
        print(f"- 原始信号: {signal.ticker} {signal.direction.value} {signal.operation_type.value}")
        
        # 转换为订单格式
        if self.sdk_manager.is_sdk_available():
            try:
                # 计算订单数量（基于USDT金额）
                quantity = signal.position_size / signal.entry_price if signal.entry_price else 0.001
                
                order_params = convert_signal_to_binance_order(signal, quantity=quantity)
                
                print(f"转换后的订单参数:")
                for key, value in order_params.items():
                    print(f"  {key}: {value}")
                
                # 注意: 这里只是演示订单参数生成，不会实际提交订单
                print("\n注意: 这是演示模式，未实际提交订单到交易所")
                print("在生产环境中，可以使用以下代码提交订单:")
                print("rest_client = sdk_manager.get_rest_client()")
                print("response = rest_client.rest_api.new_order(**order_params)")
                
            except Exception as e:
                logger.error(f"订单参数生成失败: {e}")
        else:
            print("SDK不可用，无法生成订单参数")
    
    async def example_8_error_handling(self):
        """示例8: 错误处理示例"""
        print("\n=== 示例8: 错误处理示例 ===")
        
        # 演示各种错误处理场景
        scenarios = [
            "网络连接错误",
            "认证失败",
            "参数无效",
            "SDK不可用"
        ]
        
        for scenario in scenarios:
            print(f"\n处理场景: {scenario}")
            
            try:
                if scenario == "SDK不可用":
                    # 模拟SDK不可用的情况
                    if not self.sdk_manager.is_sdk_available():
                        raise ImportError("Binance SDK不可用")
                    print("- SDK可用，跳过此错误场景")
                
                elif scenario == "认证失败":
                    # 演示认证相关的处理
                    if not self.sdk_manager.credentials:
                        raise ValueError("未设置认证凭据")
                    print("- 认证凭据已设置")
                
                elif scenario == "参数无效":
                    # 演示参数验证
                    invalid_signal = FuturesSignal(
                        ticker="INVALID",
                        direction=TradingDirection.LONG,
                        operation_type=OperationType.OPEN,
                        confidence=150.0,  # 无效值 > 100
                        strength=0.5
                    )
                    print("- 这行代码不应该执行到，因为上面应该抛出异常")
                
                elif scenario == "网络连接错误":
                    # 这里只是演示，不会真正产生网络错误
                    print("- 网络连接正常（演示模式）")
                
            except ValueError as e:
                print(f"- 捕获到参数错误: {e}")
            except ImportError as e:
                print(f"- 捕获到导入错误: {e}")
            except Exception as e:
                print(f"- 捕获到其他错误: {type(e).__name__}: {e}")
        
        print("\n错误处理最佳实践:")
        print("1. 始终检查SDK可用性")
        print("2. 验证输入参数")
        print("3. 使用适当的异常类型")
        print("4. 记录详细错误信息")
        print("5. 提供降级方案")

    async def run_all_examples(self):
        """运行所有示例"""
        print("🚀 开始运行Binance SDK集成示例")
        print("=" * 60)
        
        examples = [
            self.example_1_basic_setup,
            self.example_2_create_trading_signal,
            self.example_3_position_management,
            self.example_4_margin_status,
            self.example_5_websocket_streams,
            self.example_6_websocket_api,
            self.example_7_order_management,
            self.example_8_error_handling
        ]
        
        for i, example in enumerate(examples, 1):
            try:
                await example()
            except Exception as e:
                logger.error(f"示例{i}执行失败: {e}")
            
            # 在示例之间添加分隔
            if i < len(examples):
                print("\n" + "=" * 60)
                await asyncio.sleep(1)  # 短暂暂停
        
        print("\n✅ 所有示例执行完成")


# 独立运行的示例函数
async def quick_start_example():
    """快速开始示例"""
    print("🚀 Binance SDK集成 - 快速开始示例")
    
    # 1. 获取SDK管理器
    sdk_manager = get_global_sdk_manager()
    
    # 2. 检查状态
    print(f"SDK可用: {sdk_manager.is_sdk_available()}")
    print(f"当前环境: {get_current_environment()}")
    
    # 3. 创建交易信号
    signal = FuturesSignal(
        ticker="BTCUSDT",
        direction=TradingDirection.LONG,
        operation_type=OperationType.OPEN,
        confidence=75.0,
        strength=0.6,
        suggested_leverage=10.0
    )
    
    print(f"交易信号: {signal.ticker} {signal.direction.value}")
    
    # 4. 转换为订单格式（如果SDK可用）
    if sdk_manager.is_sdk_available():
        try:
            order_params = convert_signal_to_binance_order(signal)
            print(f"订单参数已生成: {order_params.get('symbol', 'N/A')}")
        except Exception as e:
            print(f"订单转换失败: {e}")
    
    print("✅ 快速示例完成")


if __name__ == "__main__":
    """主程序入口"""
    
    # 可以选择运行完整示例或快速示例
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # 快速示例
        asyncio.run(quick_start_example())
    else:
        # 完整示例
        example = FuturesTradingExample()
        asyncio.run(example.run_all_examples())