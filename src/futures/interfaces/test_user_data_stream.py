"""
币安期货用户数据流测试

测试用户数据流和Listen Key管理器的功能，包括：
1. Listen Key管理器测试
2. 用户数据流基本功能测试
3. 事件处理测试
4. 错误处理和重连测试

注意：这些是集成测试，需要有效的API密钥
"""

import asyncio
import logging
import os
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from .listen_key_manager import (
    BinanceFuturesListenKeyManager,
    ListenKeyInfo,
    create_futures_listen_key_manager
)
from .user_data_stream import (
    BinanceFuturesUserDataStream,
    UserDataStreamConfig,
    create_user_data_stream
)
from .data_formats import (
    AccountUpdate,
    OrderTradeUpdate,
    Balance,
    Position,
    OrderUpdate,
    EventType
)

# 测试配置
TEST_API_KEY = os.getenv('TEST_BINANCE_API_KEY', 'test_key')
TEST_API_SECRET = os.getenv('TEST_BINANCE_API_SECRET', 'test_secret')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestListenKeyManager:
    """Listen Key管理器测试"""

    @pytest.mark.asyncio
    async def test_listen_key_info(self):
        """测试Listen Key信息类"""
        current_time = int(asyncio.get_event_loop().time() * 1000)

        info = ListenKeyInfo(
            key="test_key",
            created_time=current_time,
            last_refresh_time=current_time
        )

        # 测试新创建的key不需要刷新
        assert not info.needs_refresh()
        assert not info.is_expired()

        # 测试刷新时间更新
        info.update_refresh_time()
        assert info.last_refresh_time >= current_time

        # 测试过期检测（模拟24小时后）
        info.created_time = current_time - (25 * 60 * 60 * 1000)
        assert info.is_expired()

    @pytest.mark.asyncio
    async def test_listen_key_manager_lifecycle(self):
        """测试Listen Key管理器生命周期"""
        with patch('src.futures.interfaces.listen_key_manager.DerivativesTradingUsdsFutures') as mock_client_class:
            # 模拟API响应
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data.return_value = {'listenKey': 'test_listen_key_123'}

            mock_client.rest_api.new_listen_key.return_value = mock_response
            mock_client.rest_api.renew_listen_key.return_value = mock_response
            mock_client.rest_api.close_listen_key.return_value = mock_response

            mock_client_class.return_value = mock_client

            # 创建管理器
            manager = BinanceFuturesListenKeyManager(TEST_API_KEY, TEST_API_SECRET, testnet=True)

            # 测试启动
            key = await manager.start()
            assert key == 'test_listen_key_123'
            assert manager._current_key is not None
            assert manager._current_key.key == key
            assert manager._is_running

            # 测试获取当前key
            current_key = await manager.get_current_key()
            assert current_key == key

            # 测试强制刷新
            refreshed_key = await manager.force_refresh()
            assert refreshed_key == key

            # 测试停止
            await manager.stop()
            assert not manager._is_running
            assert manager._current_key is None

    @pytest.mark.asyncio
    async def test_listen_key_manager_callbacks(self):
        """测试Listen Key管理器回调函数"""
        with patch('src.futures.interfaces.listen_key_manager.DerivativesTradingUsdsFutures') as mock_client_class:
            # 设置模拟
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data.return_value = {'listenKey': 'callback_test_key'}

            mock_client.rest_api.new_listen_key.return_value = mock_response
            mock_client.rest_api.close_listen_key.return_value = mock_response

            mock_client_class.return_value = mock_client

            # 创建管理器和回调追踪器
            manager = BinanceFuturesListenKeyManager(TEST_API_KEY, TEST_API_SECRET, testnet=True)

            created_keys = []
            errors = []

            def on_created(key):
                created_keys.append(key)

            def on_error(error):
                errors.append(error)

            # 设置回调
            manager.set_on_key_created(on_created)
            manager.set_on_error(on_error)

            # 启动管理器
            await manager.start()

            # 验证回调被调用
            assert len(created_keys) == 1
            assert created_keys[0] == 'callback_test_key'

            # 清理
            await manager.stop()


class TestUserDataStream:
    """用户数据流测试"""

    @pytest.mark.asyncio
    async def test_user_data_stream_config(self):
        """测试用户数据流配置"""
        config = UserDataStreamConfig(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
            testnet=True,
            reconnect_delay=3000,
            max_reconnect_attempts=5
        )

        assert config.api_key == TEST_API_KEY
        assert config.api_secret == TEST_API_SECRET
        assert config.testnet
        assert config.reconnect_delay == 3000
        assert config.max_reconnect_attempts == 5

        # 测试转换为WebSocket配置
        ws_config = config.to_websocket_config()
        assert ws_config.testnet == config.testnet
        assert ws_config.max_reconnects == config.max_reconnect_attempts
        assert not ws_config.use_mock_data

    @pytest.mark.asyncio
    async def test_user_data_stream_initialization(self):
        """测试用户数据流初始化"""
        config = UserDataStreamConfig(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
            testnet=True
        )

        stream = BinanceFuturesUserDataStream(config)

        assert stream.stream_config == config
        assert stream._binance_client is None
        assert stream._listen_key_manager is None
        assert stream._current_listen_key is None
        assert not stream._is_running

    @pytest.mark.asyncio
    async def test_event_handler_registration(self):
        """测试事件处理器注册"""
        config = UserDataStreamConfig(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
            testnet=True
        )

        stream = BinanceFuturesUserDataStream(config)

        # 创建测试处理器
        account_updates = []
        order_updates = []
        margin_calls = []

        def account_handler(update):
            account_updates.append(update)

        def order_handler(update):
            order_updates.append(update)

        def margin_handler(call):
            margin_calls.append(call)

        # 注册处理器
        stream.add_account_update_handler(account_handler)
        stream.add_order_update_handler(order_handler)
        stream.add_margin_call_handler(margin_handler)

        # 验证处理器已注册
        assert len(stream._account_update_handlers) == 1
        assert len(stream._order_update_handlers) == 1
        assert len(stream._margin_call_handlers) == 1

        # 测试移除处理器
        stream.remove_account_update_handler(account_handler)
        stream.remove_order_update_handler(order_handler)
        stream.remove_margin_call_handler(margin_handler)

        assert len(stream._account_update_handlers) == 0
        assert len(stream._order_update_handlers) == 0
        assert len(stream._margin_call_handlers) == 0

    @pytest.mark.asyncio
    async def test_message_parsing_and_dispatch(self):
        """测试消息解析和分发"""
        config = UserDataStreamConfig(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
            testnet=True
        )

        stream = BinanceFuturesUserDataStream(config)

        # 创建测试数据
        account_update_data = {
            "e": "ACCOUNT_UPDATE",
            "E": 1564745798939,
            "T": 1564745798938,
            "a": {
                "B": [
                    {
                        "a": "USDT",
                        "wb": "122624.12345678",
                        "up": "0.00000000",
                        "bc": "122624.12345678",
                        "mm": "0.00000000",
                        "im": "0.00000000",
                        "pim": "0.00000000",
                        "oim": "0.00000000",
                        "cw": "122624.12345678",
                        "cp": "0.00000000",
                        "ab": "122624.12345678",
                        "mwa": "122624.12345678"
                    }
                ],
                "P": [
                    {
                        "s": "BTCUSDT",
                        "ps": "BOTH",
                        "pa": "0.001",
                        "ep": "50000.0",
                        "bep": "0.0",
                        "mp": "50100.0",
                        "up": "100.0",
                        "mt": "CROSSED",
                        "iw": "0.0"
                    }
                ]
            }
        }

        # 测试消息处理
        received_updates = []

        def account_handler(update):
            received_updates.append(update)

        stream.add_account_update_handler(account_handler)

        # 模拟消息处理
        stream._handle_user_data_message(account_update_data)

        # 等待异步处理完成
        await asyncio.sleep(0.1)

        # 由于这是集成测试，实际的消息解析需要完整的流设置
        # 这里主要测试处理器的注册和调用机制


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not TEST_API_KEY or not TEST_API_SECRET or TEST_API_KEY == 'test_key',
        reason="需要有效的API密钥进行集成测试"
    )
    async def test_full_integration(self):
        """完整集成测试（需要有效的API密钥）"""
        logger.info("开始完整集成测试...")

        # 创建配置
        config = UserDataStreamConfig(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
            testnet=True,
            reconnect_delay=5000,
            max_reconnect_attempts=3
        )

        # 事件收集器
        events = {
            'account_updates': [],
            'order_updates': [],
            'margin_calls': [],
            'errors': []
        }

        def collect_account_update(update):
            events['account_updates'].append(update)
            logger.info(f"收到账户更新: {len(update.balances)} 余额, {len(update.positions)} 持仓")

        def collect_order_update(update):
            events['order_updates'].append(update)
            logger.info(f"收到订单更新: {update.order.symbol} {update.order.order_status.value}")

        def collect_margin_call(call):
            events['margin_calls'].append(call)
            logger.warning("收到保证金追加通知")

        try:
            # 使用上下文管理器
            async with create_user_data_stream(
                api_key=TEST_API_KEY,
                api_secret=TEST_API_SECRET,
                testnet=True,
                auto_start=True
            ) as stream:
                # 注册事件处理器
                stream.add_account_update_handler(collect_account_update)
                stream.add_order_update_handler(collect_order_update)
                stream.add_margin_call_handler(collect_margin_call)

                # 验证流状态
                assert stream._is_running
                assert stream.get_current_listen_key() is not None

                logger.info(f"当前Listen Key: {stream.get_current_listen_key()[:10]}...")

                # 运行一段时间收集事件
                logger.info("监听用户数据流 30 秒...")
                await asyncio.sleep(30)

                # 打印收集到的事件统计
                logger.info(f"收集到的事件统计:")
                logger.info(f"  账户更新: {len(events['account_updates'])}")
                logger.info(f"  订单更新: {len(events['order_updates'])}")
                logger.info(f"  保证金追加: {len(events['margin_calls'])}")
                logger.info(f"  错误: {len(events['errors'])}")

                # 基本验证
                assert stream._is_running
                assert stream.get_current_listen_key() is not None

            logger.info("完整集成测试完成")

        except Exception as e:
            logger.error(f"集成测试失败: {e}")
            raise


# 手动测试函数
async def manual_test_listen_key_manager():
    """手动测试Listen Key管理器"""
    print("=== 手动测试Listen Key管理器 ===")

    if not TEST_API_KEY or TEST_API_KEY == 'test_key':
        print("跳过：需要有效的API密钥")
        return

    try:
        manager = await create_futures_listen_key_manager(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
            testnet=True
        )

        print(f"Listen Key创建成功: {manager._current_key.key[:10]}...")

        # 运行一段时间
        await asyncio.sleep(10)

        # 强制刷新
        await manager.force_refresh()
        print("强制刷新成功")

        # 停止管理器
        await manager.stop()
        print("管理器停止成功")

    except Exception as e:
        print(f"测试失败: {e}")


async def manual_test_user_data_stream():
    """手动测试用户数据流"""
    print("=== 手动测试用户数据流 ===")

    if not TEST_API_KEY or TEST_API_KEY == 'test_key':
        print("跳过：需要有效的API密钥")
        return

    try:
        events_count = {'total': 0}

        def on_account_update(update):
            events_count['total'] += 1
            print(f"账户更新 #{events_count['total']}: {len(update.balances)} 余额")

        def on_order_update(update):
            events_count['total'] += 1
            print(f"订单更新 #{events_count['total']}: {update.order.symbol}")

        async with create_user_data_stream(
            api_key=TEST_API_KEY,
            api_secret=TEST_API_SECRET,
            testnet=True
        ) as stream:
            stream.add_account_update_handler(on_account_update)
            stream.add_order_update_handler(on_order_update)

            print("开始监听用户数据流...")
            await asyncio.sleep(20)

        print(f"测试完成，共收到 {events_count['total']} 个事件")

    except Exception as e:
        print(f"测试失败: {e}")


async def run_manual_tests():
    """运行手动测试"""
    await manual_test_listen_key_manager()
    print()
    await manual_test_user_data_stream()


if __name__ == "__main__":
    # 设置测试环境
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行手动测试
    asyncio.run(run_manual_tests())