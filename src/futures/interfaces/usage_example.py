"""
币安期货用户数据流使用示例

展示如何使用BinanceFuturesUserDataStream和ListenKeyManager：
1. 基本使用方法
2. 事件处理器注册
3. 错误处理和重连
4. 与现有架构集成

注意：运行前需要设置正确的API Key和Secret
"""

import asyncio
import logging
import os
from typing import Dict, Any
from decimal import Decimal

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
    OrderUpdate
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FuturesAccountMonitor:
    """期货账户监控器示例"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # 用户数据流
        self.user_stream: Optional[BinanceFuturesUserDataStream] = None

        # 账户状态
        self.current_balances: Dict[str, Balance] = {}
        self.current_positions: Dict[str, Position] = {}
        self.recent_orders: list = []

        # 统计信息
        self.total_events = 0
        self.account_updates = 0
        self.order_updates = 0
        self.errors = 0

    async def start(self) -> None:
        """启动账户监控"""
        try:
            logger.info("启动期货账户监控...")

            # 创建用户数据流配置
            config = UserDataStreamConfig(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                reconnect_delay=5000,
                max_reconnect_attempts=10,
                listen_key_auto_refresh=True,
                enable_compression=True
            )

            # 创建用户数据流
            self.user_stream = BinanceFuturesUserDataStream(config)

            # 注册事件处理器
            self._register_event_handlers()

            # 启动用户数据流
            await self.user_stream.start()

            logger.info("期货账户监控启动成功")

        except Exception as e:
            logger.error(f"启动期货账户监控失败: {e}")
            raise

    async def stop(self) -> None:
        """停止账户监控"""
        try:
            logger.info("停止期货账户监控...")

            if self.user_stream:
                await self.user_stream.stop()
                self.user_stream = None

            logger.info("期货账户监控已停止")

        except Exception as e:
            logger.error(f"停止期货账户监控失败: {e}")

    def _register_event_handlers(self) -> None:
        """注册事件处理器"""
        if not self.user_stream:
            return

        # 注册账户更新处理器
        self.user_stream.add_account_update_handler(self._handle_account_update)

        # 注册订单更新处理器
        self.user_stream.add_order_update_handler(self._handle_order_update)

        # 注册保证金追加通知处理器
        self.user_stream.add_margin_call_handler(self._handle_margin_call)

        # 注册Listen Key过期处理器
        self.user_stream.add_listen_key_expired_handler(self._handle_listen_key_expired)

        # 注册全局消息回调
        self.user_stream.set_global_callback(self._handle_global_message)

    def _handle_account_update(self, update: AccountUpdate) -> None:
        """处理账户更新事件"""
        try:
            self.total_events += 1
            self.account_updates += 1

            logger.info(f"收到账户更新事件 - 时间: {update.event_time}")

            # 更新余额信息
            for balance in update.balances:
                self.current_balances[balance.asset] = balance
                logger.info(f"余额更新 - {balance.asset}: "
                           f"钱包余额={balance.wallet_balance}, "
                           f"未实现盈亏={balance.unrealized_pnl}, "
                           f"保证金余额={balance.margin_balance}")

            # 更新持仓信息
            for position in update.positions:
                position_key = f"{position.symbol}_{position.position_side.value}"
                self.current_positions[position_key] = position

                if position.position_amount != 0:
                    logger.info(f"持仓更新 - {position.symbol} {position.position_side.value}: "
                               f"数量={position.position_amount}, "
                               f"入场价格={position.entry_price}, "
                               f"标记价格={position.mark_price}, "
                               f"未实现盈亏={position.unrealized_pnl}")

            # 账户风险监控
            self._check_account_risk(update)

        except Exception as e:
            logger.error(f"处理账户更新事件失败: {e}")
            self.errors += 1

    def _handle_order_update(self, update: OrderTradeUpdate) -> None:
        """处理订单更新事件"""
        try:
            self.total_events += 1
            self.order_updates += 1

            order = update.order
            logger.info(f"收到订单更新事件 - 时间: {update.event_time}")
            logger.info(f"订单信息 - {order.symbol}: "
                       f"ID={order.order_id}, "
                       f"状态={order.order_status.value}, "
                       f"类型={order.order_type.value}, "
                       f"方向={order.order_side}, "
                       f"数量={order.original_quantity}, "
                       f"价格={order.original_price}, "
                       f"已成交数量={order.cumulative_filled_quantity}")

            # 保存最近的订单
            self.recent_orders.append({
                'timestamp': update.event_time,
                'symbol': order.symbol,
                'order_id': order.order_id,
                'status': order.order_status.value,
                'side': order.order_side,
                'type': order.order_type.value,
                'quantity': float(order.original_quantity),
                'price': float(order.original_price),
                'filled_quantity': float(order.cumulative_filled_quantity)
            })

            # 只保留最近100个订单
            if len(self.recent_orders) > 100:
                self.recent_orders = self.recent_orders[-100:]

            # 订单执行分析
            if order.order_status.value == 'FILLED':
                logger.info(f"订单完全成交 - {order.symbol}: "
                           f"平均价格={order.average_price}, "
                           f"手续费={order.commission_amount} {order.commission_asset}")

            elif order.order_status.value == 'PARTIALLY_FILLED':
                remaining = order.original_quantity - order.cumulative_filled_quantity
                logger.info(f"订单部分成交 - {order.symbol}: "
                           f"剩余数量={remaining}")

            elif order.order_status.value in ['CANCELED', 'REJECTED', 'EXPIRED']:
                logger.warning(f"订单未成交 - {order.symbol}: 状态={order.order_status.value}")

        except Exception as e:
            logger.error(f"处理订单更新事件失败: {e}")
            self.errors += 1

    def _handle_margin_call(self, data: Dict[str, Any]) -> None:
        """处理保证金追加通知"""
        try:
            self.total_events += 1

            logger.warning("收到保证金追加通知!")
            logger.warning(f"保证金追加详情: {data}")

            # 这里可以实现自动风险管理措施
            # 例如：自动平仓、发送警报等
            asyncio.create_task(self._handle_margin_call_emergency(data))

        except Exception as e:
            logger.error(f"处理保证金追加通知失败: {e}")
            self.errors += 1

    def _handle_listen_key_expired(self, listen_key: str) -> None:
        """处理Listen Key过期"""
        try:
            logger.warning(f"Listen Key过期: {listen_key[:10]}...")
            # Listen Key管理器会自动处理重连

        except Exception as e:
            logger.error(f"处理Listen Key过期失败: {e}")
            self.errors += 1

    def _handle_global_message(self, message: Any) -> None:
        """处理全局消息"""
        try:
            # 记录所有消息用于调试
            if hasattr(message, 'event_type'):
                logger.debug(f"收到消息: {message.event_type}")
            else:
                logger.debug(f"收到未知消息: {type(message)}")

        except Exception as e:
            logger.error(f"处理全局消息失败: {e}")

    def _check_account_risk(self, update: AccountUpdate) -> None:
        """检查账户风险"""
        try:
            for balance in update.balances:
                # 检查保证金比例
                if balance.margin_balance > 0:
                    margin_ratio = balance.maint_margin / balance.margin_balance
                    if margin_ratio > 0.8:  # 维持保证金比例超过80%
                        logger.warning(f"高风险警告 - {balance.asset}: "
                                     f"维持保证金比例={margin_ratio:.2%}")

                # 检查可用余额
                if balance.available_balance < balance.margin_balance * Decimal('0.1'):
                    logger.warning(f"低可用余额警告 - {balance.asset}: "
                                 f"可用余额={balance.available_balance}")

        except Exception as e:
            logger.error(f"检查账户风险失败: {e}")

    async def _handle_margin_call_emergency(self, data: Dict[str, Any]) -> None:
        """处理保证金追加紧急情况"""
        try:
            logger.warning("执行保证金追加紧急处理...")

            # 这里可以实现紧急风险管理措施：
            # 1. 发送紧急通知
            # 2. 自动平仓部分持仓
            # 3. 调整杠杆倍数
            # 4. 暂停新订单

            # 示例：记录紧急事件
            emergency_info = {
                'timestamp': data.get('E', 0),
                'event_type': 'MARGIN_CALL',
                'data': data,
                'action': 'logged'
            }

            logger.critical(f"保证金追加紧急事件: {emergency_info}")

        except Exception as e:
            logger.error(f"处理保证金追加紧急情况失败: {e}")

    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        return {
            'stream_status': 'running' if self.user_stream else 'stopped',
            'current_listen_key': self.user_stream.get_current_listen_key() if self.user_stream else None,
            'last_message_time': self.user_stream.get_last_message_time() if self.user_stream else None,
            'reconnect_count': self.user_stream.get_reconnect_count() if self.user_stream else 0,
            'statistics': {
                'total_events': self.total_events,
                'account_updates': self.account_updates,
                'order_updates': self.order_updates,
                'errors': self.errors
            },
            'balances_count': len(self.current_balances),
            'positions_count': len(self.current_positions),
            'recent_orders_count': len(self.recent_orders)
        }

    async def print_status_loop(self, interval: int = 30) -> None:
        """定期打印状态报告"""
        while True:
            try:
                report = self.get_status_report()
                logger.info("=== 账户监控状态报告 ===")
                logger.info(f"流状态: {report['stream_status']}")
                logger.info(f"总事件数: {report['statistics']['total_events']}")
                logger.info(f"账户更新: {report['statistics']['account_updates']}")
                logger.info(f"订单更新: {report['statistics']['order_updates']}")
                logger.info(f"错误次数: {report['statistics']['errors']}")
                logger.info(f"当前余额种类: {report['balances_count']}")
                logger.info(f"当前持仓数: {report['positions_count']}")
                logger.info(f"最近订单数: {report['recent_orders_count']}")
                logger.info("========================")

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"打印状态报告失败: {e}")
                await asyncio.sleep(interval)


async def basic_usage_example():
    """基本使用示例"""
    logger.info("=== 基本使用示例 ===")

    # 从环境变量获取API密钥（出于安全考虑）
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logger.error("请设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET")
        return

    try:
        # 方法1: 直接使用便捷函数
        async with create_user_data_stream(
            api_key=api_key,
            api_secret=api_secret,
            testnet=True,
            auto_start=True
        ) as stream:
            # 添加简单的事件处理器
            def on_account_update(update: AccountUpdate):
                logger.info(f"账户更新: {len(update.balances)} 个余额, {len(update.positions)} 个持仓")

            def on_order_update(update: OrderTradeUpdate):
                order = update.order
                logger.info(f"订单更新: {order.symbol} {order.order_status.value}")

            stream.add_account_update_handler(on_account_update)
            stream.add_order_update_handler(on_order_update)

            # 运行10秒
            logger.info("监听用户数据流 10 秒...")
            await asyncio.sleep(10)

        logger.info("基本使用示例完成")

    except Exception as e:
        logger.error(f"基本使用示例失败: {e}")


async def advanced_monitor_example():
    """高级监控示例"""
    logger.info("=== 高级监控示例 ===")

    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')

    if not api_key or not api_secret:
        logger.error("请设置环境变量 BINANCE_API_KEY 和 BINANCE_API_SECRET")
        return

    try:
        # 创建高级账户监控器
        monitor = FuturesAccountMonitor(api_key, api_secret, testnet=True)

        # 启动监控
        await monitor.start()

        # 启动状态报告任务
        status_task = asyncio.create_task(monitor.print_status_loop(30))

        try:
            # 运行60秒
            logger.info("运行高级监控 60 秒...")
            await asyncio.sleep(60)

        finally:
            # 清理资源
            status_task.cancel()
            try:
                await status_task
            except asyncio.CancelledError:
                pass

            await monitor.stop()

        logger.info("高级监控示例完成")

    except Exception as e:
        logger.error(f"高级监控示例失败: {e}")


async def main():
    """主函数"""
    logger.info("开始运行币安期货用户数据流示例")

    try:
        # 运行基本使用示例
        await basic_usage_example()

        # 等待一下
        await asyncio.sleep(2)

        # 运行高级监控示例
        await advanced_monitor_example()

        logger.info("所有示例运行完成")

    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序运行失败: {e}")


if __name__ == "__main__":
    # 设置更详细的日志级别用于演示
    logging.getLogger().setLevel(logging.DEBUG)

    # 运行示例
    asyncio.run(main())