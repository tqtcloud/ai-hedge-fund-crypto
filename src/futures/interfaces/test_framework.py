"""
期货WebSocket框架测试脚本

用于验证WebSocket框架的基本功能，包括：
- 数据格式解析测试
- Mock数据生成测试
- WebSocket接口功能测试
- 错误处理测试
"""

import asyncio
import json
import time
from typing import List, Dict, Any
from decimal import Decimal

from .data_formats import (
    parse_websocket_message, EventType, KlineData, DepthUpdate,
    MarkPriceUpdate, AggTradeData, ErrorMessage
)
from .mock_data_generator import (
    FuturesMockDataGenerator, MockMarketConfig, MockAccountConfig
)
from .mock_websocket_client import create_mock_client
from .websocket_interface import create_kline_stream, create_depth_stream


class FrameworkTester:
    """框架测试器"""
    
    def __init__(self):
        self.test_results: Dict[str, bool] = {}
        self.error_messages: List[str] = []
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        self.test_results[test_name] = success
        status = "✓" if success else "✗"
        print(f"{status} {test_name}: {message}")
        if not success and message:
            self.error_messages.append(f"{test_name}: {message}")
    
    def test_data_format_parsing(self):
        """测试数据格式解析"""
        print("\n=== 测试数据格式解析 ===")
        
        # 测试K线数据解析
        kline_raw = {
            "e": "kline",
            "E": 1640995200000,
            "s": "BTCUSDT",
            "k": {
                "t": 1640995140000,
                "T": 1640995199999,
                "s": "BTCUSDT",
                "i": "1m",
                "f": 100,
                "L": 200,
                "o": "46929.10000000",
                "c": "47000.10000000",
                "h": "47100.10000000",
                "l": "46900.10000000",
                "v": "10.12345678",
                "n": 100,
                "x": True,
                "q": "475123.45678900",
                "V": "5.12345678",
                "Q": "240123.45678900",
                "B": "0"
            }
        }
        
        try:
            parsed = parse_websocket_message(kline_raw)
            if isinstance(parsed, KlineData):
                self.log_test("K线数据解析", True, f"解析成功，价格: {parsed.close_price}")
            else:
                self.log_test("K线数据解析", False, f"解析类型错误: {type(parsed)}")
        except Exception as e:
            self.log_test("K线数据解析", False, f"解析异常: {e}")
        
        # 测试深度数据解析
        depth_raw = {
            "e": "depthUpdate",
            "E": 1640995200000,
            "T": 1640995200000,
            "s": "BTCUSDT",
            "U": 1000001,
            "u": 1000010,
            "pu": 1000000,
            "b": [["46950.00", "1.23"], ["46949.00", "2.45"]],
            "a": [["47000.00", "1.67"], ["47001.00", "3.21"]]
        }
        
        try:
            parsed = parse_websocket_message(depth_raw)
            if isinstance(parsed, DepthUpdate):
                self.log_test("深度数据解析", True, f"解析成功，{len(parsed.bids)}档买单")
            else:
                self.log_test("深度数据解析", False, f"解析类型错误: {type(parsed)}")
        except Exception as e:
            self.log_test("深度数据解析", False, f"解析异常: {e}")
        
        # 测试错误数据处理
        invalid_data = {"invalid": "data"}
        try:
            parsed = parse_websocket_message(invalid_data)
            self.log_test("无效数据处理", True, f"处理成功，类型: {type(parsed).__name__}")
        except Exception as e:
            self.log_test("无效数据处理", False, f"处理异常: {e}")
    
    async def test_mock_data_generator(self):
        """测试Mock数据生成器"""
        print("\n=== 测试Mock数据生成器 ===")
        
        try:
            # 创建生成器
            generator = FuturesMockDataGenerator(
                symbols=["BTCUSDT", "ETHUSDT"],
                market_config=MockMarketConfig(base_price=Decimal("50000")),
                account_config=MockAccountConfig()
            )
            
            # 测试K线数据生成
            kline_data = await generator.generate_kline_data("BTCUSDT", "1m")
            if isinstance(kline_data, KlineData) and kline_data.symbol == "BTCUSDT":
                self.log_test("Mock K线生成", True, f"价格: {kline_data.close_price}")
            else:
                self.log_test("Mock K线生成", False, "生成数据格式错误")
            
            # 测试深度数据生成
            depth_data = await generator.generate_depth_update("BTCUSDT")
            if isinstance(depth_data, DepthUpdate) and len(depth_data.bids) > 0:
                self.log_test("Mock深度生成", True, f"{len(depth_data.bids)}档深度")
            else:
                self.log_test("Mock深度生成", False, "生成数据格式错误")
            
            # 测试标记价格生成
            mark_price = await generator.generate_mark_price_update("BTCUSDT")
            if isinstance(mark_price, MarkPriceUpdate) and mark_price.mark_price > 0:
                self.log_test("Mock标记价格生成", True, f"标记价格: {mark_price.mark_price}")
            else:
                self.log_test("Mock标记价格生成", False, "生成数据格式错误")
            
            # 测试交易数据生成
            trade_data = await generator.generate_agg_trade_data("BTCUSDT")
            if isinstance(trade_data, AggTradeData) and trade_data.price > 0:
                self.log_test("Mock交易数据生成", True, f"交易价格: {trade_data.price}")
            else:
                self.log_test("Mock交易数据生成", False, "生成数据格式错误")
            
            # 测试账户更新生成
            account_update = await generator.generate_account_update()
            if account_update and len(account_update.balances) > 0:
                self.log_test("Mock账户数据生成", True, f"{len(account_update.balances)}个余额")
            else:
                self.log_test("Mock账户数据生成", False, "生成数据格式错误")
        
        except Exception as e:
            self.log_test("Mock数据生成器", False, f"测试异常: {e}")
    
    async def test_mock_websocket_client(self):
        """测试Mock WebSocket客户端"""
        print("\n=== 测试Mock WebSocket客户端 ===")
        
        received_messages = []
        
        def message_callback(message):
            received_messages.append(message)
        
        try:
            # 创建客户端
            client = create_mock_client(
                symbols=["BTCUSDT"],
                use_realistic_data=True,
                log_level="ERROR"  # 减少日志输出
            )
            
            # 设置回调
            client.set_global_callback(message_callback)
            
            # 启动客户端
            await client.start()
            self.log_test("Mock客户端启动", True, "启动成功")
            
            # 订阅数据流
            success1 = await client.subscribe(create_kline_stream("BTCUSDT", "1m"))
            success2 = await client.subscribe(create_depth_stream("BTCUSDT", 10))
            
            if success1 and success2:
                self.log_test("数据流订阅", True, "K线和深度订阅成功")
            else:
                self.log_test("数据流订阅", False, f"订阅失败: K线={success1}, 深度={success2}")
            
            # 等待接收消息（给更多时间让异步任务启动）
            await asyncio.sleep(5)
            
            # 检查接收到的消息
            if len(received_messages) > 0:
                self.log_test("消息接收", True, f"接收到 {len(received_messages)} 条消息")
                
                # 分析消息类型
                kline_count = sum(1 for msg in received_messages if isinstance(msg, KlineData))
                depth_count = sum(1 for msg in received_messages if isinstance(msg, DepthUpdate))
                
                self.log_test("消息类型统计", True, f"K线: {kline_count}, 深度: {depth_count}")
            else:
                self.log_test("消息接收", False, "未收到任何消息")
            
            # 测试连接状态
            status = client.get_connection_status()
            if status.is_connected:
                self.log_test("连接状态检查", True, "连接正常")
            else:
                self.log_test("连接状态检查", False, "连接异常")
            
            # 测试订阅管理
            subscriptions = client.get_subscriptions()
            if len(subscriptions) >= 2:
                self.log_test("订阅管理", True, f"{len(subscriptions)} 个活跃订阅")
            else:
                self.log_test("订阅管理", False, f"订阅数量异常: {len(subscriptions)}")
            
            # 测试取消订阅
            unsubscribe_success = await client.unsubscribe(create_kline_stream("BTCUSDT", "1m"))
            if unsubscribe_success:
                self.log_test("取消订阅", True, "K线订阅取消成功")
            else:
                self.log_test("取消订阅", False, "取消订阅失败")
            
            # 停止客户端
            await client.stop()
            self.log_test("Mock客户端停止", True, "停止成功")
        
        except Exception as e:
            self.log_test("Mock WebSocket客户端", False, f"测试异常: {e}")
    
    def test_utility_functions(self):
        """测试工具函数"""
        print("\n=== 测试工具函数 ===")
        
        try:
            # 测试流名称创建函数
            from .websocket_interface import (
                create_kline_stream, create_depth_stream, create_trade_stream,
                create_ticker_stream, create_mark_price_stream, create_multiplex_stream
            )
            
            kline_stream = create_kline_stream("BTCUSDT", "1m")
            expected_kline = "btcusdt@kline_1m"
            if kline_stream == expected_kline:
                self.log_test("K线流名称创建", True, f"创建: {kline_stream}")
            else:
                self.log_test("K线流名称创建", False, f"期望: {expected_kline}, 实际: {kline_stream}")
            
            depth_stream = create_depth_stream("ETHUSDT", 20, "100ms")
            expected_depth = "ethusdt@depth20@100ms"
            if depth_stream == expected_depth:
                self.log_test("深度流名称创建", True, f"创建: {depth_stream}")
            else:
                self.log_test("深度流名称创建", False, f"期望: {expected_depth}, 实际: {depth_stream}")
            
            trade_stream = create_trade_stream("BNBUSDT")
            expected_trade = "bnbusdt@aggTrade"
            if trade_stream == expected_trade:
                self.log_test("交易流名称创建", True, f"创建: {trade_stream}")
            else:
                self.log_test("交易流名称创建", False, f"期望: {expected_trade}, 实际: {trade_stream}")
            
            # 测试多路复用流
            streams = ["btcusdt@kline_1m", "ethusdt@depth20"]
            multiplex_stream = create_multiplex_stream(streams)
            expected_multiplex = "btcusdt@kline_1m/ethusdt@depth20"
            if multiplex_stream == expected_multiplex:
                self.log_test("多路复用流创建", True, f"创建: {multiplex_stream}")
            else:
                self.log_test("多路复用流创建", False, f"期望: {expected_multiplex}, 实际: {multiplex_stream}")
        
        except Exception as e:
            self.log_test("工具函数测试", False, f"测试异常: {e}")
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*50)
        print("测试总结")
        print("="*50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过数: {passed_tests}")
        print(f"失败数: {failed_tests}")
        print(f"成功率: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%")
        
        if failed_tests > 0:
            print("\n失败的测试:")
            for error in self.error_messages:
                print(f"  - {error}")
        else:
            print("\n🎉 所有测试都通过了！")
        
        print("="*50)


async def run_all_tests():
    """运行所有测试"""
    print("期货WebSocket框架测试")
    print("="*50)
    
    tester = FrameworkTester()
    
    # 运行各项测试
    tester.test_data_format_parsing()
    await tester.test_mock_data_generator()
    await tester.test_mock_websocket_client()
    tester.test_utility_functions()
    
    # 打印总结
    tester.print_summary()
    
    return tester.test_results


if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_all_tests())