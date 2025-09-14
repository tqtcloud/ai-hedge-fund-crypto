"""
保证金管理器核心功能实现

该模块提供完整的期货保证金管理功能，包括：
- 基础保证金计算和强平价格计算
- WebSocket数据集成接口
- 保证金充足性检查和风险评估
- 实时保证金监控和告警
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from ..models.data_models import MarginStatus, RiskLevel
from ..interfaces.websocket_interface import FuturesWebSocketInterface
from ...utils.exceptions import MarginInsufficientError


logger = logging.getLogger(__name__)


@dataclass
class MarginConfig:
    """保证金配置类"""
    initial_cash: float = 1000.0                    # 初始资金
    initial_margin_rate: float = 0.1                # 初始保证金率 (10%)
    maintenance_margin_rate: float = 0.05            # 维持保证金率 (5%)
    margin_call_threshold: float = 0.3               # 追保阈值 (30%)
    liquidation_threshold: float = 0.8               # 强平阈值 (80%)
    emergency_threshold: float = 0.9                 # 紧急阈值 (90%)
    max_leverage: int = 20                           # 最大杠杆倍数
    currency: str = "USDT"                           # 保证金货币


class WebSocketTradingInterface:
    """WebSocket交易接口的Mock实现

    在实际实现中，这将连接到真实的WebSocket接口
    目前提供模拟数据用于测试和开发
    """

    def __init__(self):
        self.connected = False
        self.mock_account_data = {
            'totalWalletBalance': 1000.0,
            'totalUnrealizedProfit': 0.0,
            'totalMarginBalance': 1000.0,
            'totalPositionInitialMargin': 0.0,
            'totalOpenOrderInitialMargin': 0.0,
            'availableBalance': 1000.0,
            'totalCrossWalletBalance': 1000.0,
            'totalCrossUnPnl': 0.0,
            'maxWithdrawAmount': 1000.0
        }

    async def get_account_status(self) -> Dict[str, Any]:
        """获取账户状态信息（Mock实现）"""
        await asyncio.sleep(0.01)  # 模拟网络延迟
        return {
            'e': 'ACCOUNT_UPDATE',
            'a': {
                'B': [{'a': 'USDT', 'wb': str(self.mock_account_data['totalWalletBalance']),
                       'cw': str(self.mock_account_data['availableBalance'])}],
                'P': []
            }
        }

    async def connect(self):
        """连接WebSocket"""
        self.connected = True
        logger.info("WebSocket连接已建立")

    async def disconnect(self):
        """断开WebSocket连接"""
        self.connected = False
        logger.info("WebSocket连接已断开")


class MarginManager:
    """保证金管理核心组件

    功能包括：
    - 保证金计算和验证
    - 强平价格计算
    - 实时账户数据监控
    - 风险评估和预警
    - 保证金健康状态监控
    """

    def __init__(self, config: MarginConfig, ws_interface: Optional[WebSocketTradingInterface] = None):
        """初始化保证金管理器

        Args:
            config: 保证金配置
            ws_interface: WebSocket接口，如果为None则创建默认Mock接口
        """
        self.config = config
        self.ws_interface = ws_interface or WebSocketTradingInterface()

        # 基础配置参数
        self.initial_cash = config.initial_cash
        self.initial_margin_rate = config.initial_margin_rate
        self.maintenance_margin_rate = config.maintenance_margin_rate
        self.margin_call_threshold = config.margin_call_threshold
        self.liquidation_threshold = config.liquidation_threshold
        self.emergency_threshold = config.emergency_threshold
        self.max_leverage = config.max_leverage

        # 实时账户信息（从WebSocket更新）
        self.account_data = {
            'totalWalletBalance': 0.0,               # 钱包总余额
            'totalUnrealizedProfit': 0.0,            # 未实现盈亏
            'totalMarginBalance': 0.0,               # 保证金余额
            'totalPositionInitialMargin': 0.0,       # 持仓保证金
            'totalOpenOrderInitialMargin': 0.0,      # 挂单保证金
            'availableBalance': 0.0,                 # 可用余额
            'totalCrossWalletBalance': 0.0,          # 全仓钱包余额
            'totalCrossUnPnl': 0.0,                  # 全仓未实现盈亏
            'maxWithdrawAmount': 0.0                 # 最大可转出
        }

        # 监控状态
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        logger.info(f"保证金管理器初始化完成，初始资金: {self.initial_cash} {config.currency}")

    async def initialize(self) -> None:
        """初始化：从WebSocket获取最新账户信息"""
        try:
            # 建立WebSocket连接
            await self.ws_interface.connect()

            # 获取实时账户数据
            account_status = await self.ws_interface.get_account_status()
            self.update_account_data(account_status)

            # 如果WebSocket数据为空，使用配置文件的初始值
            if self.account_data['totalWalletBalance'] == 0:
                logger.warning("无法获取WebSocket账户数据，使用配置文件初始值")
                self.account_data['totalWalletBalance'] = self.initial_cash
                self.account_data['availableBalance'] = self.initial_cash
                self.account_data['totalMarginBalance'] = self.initial_cash

            logger.info(f"保证金管理器初始化成功，账户余额: {self.account_data['totalWalletBalance']}")

        except Exception as e:
            logger.error(f"保证金管理器初始化失败: {e}")
            # 降级使用配置文件数据
            self.account_data['totalWalletBalance'] = self.initial_cash
            self.account_data['availableBalance'] = self.initial_cash
            self.account_data['totalMarginBalance'] = self.initial_cash

    def update_account_data(self, ws_message: Dict[str, Any]) -> None:
        """更新账户数据（从WebSocket推送）

        Args:
            ws_message: WebSocket消息数据
        """
        try:
            if ws_message.get('e') == 'ACCOUNT_UPDATE':
                # 处理账户更新事件
                account = ws_message.get('a', {})

                # 更新余额信息
                if 'B' in account:  # 余额数组
                    for balance in account['B']:
                        if balance['a'] == 'USDT':  # USDT余额
                            self.account_data['totalWalletBalance'] = float(balance['wb'])
                            self.account_data['availableBalance'] = float(balance['cw'])

                # 更新仓位信息
                if 'P' in account:  # 仓位数组
                    total_margin = 0.0
                    for position in account['P']:
                        total_margin += float(position.get('iw', 0))  # 仓位保证金
                    self.account_data['totalPositionInitialMargin'] = total_margin

                # 重新计算保证金余额
                self.account_data['totalMarginBalance'] = (
                    self.account_data['totalWalletBalance'] +
                    self.account_data.get('totalUnrealizedProfit', 0.0)
                )

                logger.debug(f"账户数据已更新: {self.account_data}")

        except Exception as e:
            logger.error(f"更新账户数据失败: {e}")

    def calculate_initial_margin(self, position_value: float, leverage: int) -> float:
        """计算开仓所需初始保证金

        Args:
            position_value: 仓位价值（价格 * 数量）
            leverage: 杠杆倍数

        Returns:
            所需初始保证金

        Raises:
            ValueError: 当参数无效时
        """
        if position_value <= 0:
            raise ValueError(f"仓位价值必须大于0: {position_value}")
        if leverage <= 0 or leverage > self.max_leverage:
            raise ValueError(f"杠杆倍数无效，必须在1-{self.max_leverage}之间: {leverage}")

        # 基础保证金 = 仓位价值 / 杠杆 * 初始保证金率
        base_margin = position_value / leverage
        required_margin = base_margin * self.initial_margin_rate

        logger.debug(f"计算初始保证金 - 仓位价值: {position_value}, 杠杆: {leverage}, 所需保证金: {required_margin}")
        return required_margin

    def calculate_maintenance_margin(self, position_value: float, leverage: int) -> float:
        """计算维持仓位所需保证金

        Args:
            position_value: 仓位价值（价格 * 数量）
            leverage: 杠杆倍数

        Returns:
            维持保证金
        """
        if position_value <= 0:
            raise ValueError(f"仓位价值必须大于0: {position_value}")
        if leverage <= 0:
            raise ValueError(f"杠杆倍数必须大于0: {leverage}")

        # 维持保证金 = 仓位价值 / 杠杆 * 维持保证金率
        base_margin = position_value / leverage
        maintenance_margin = base_margin * self.maintenance_margin_rate

        return maintenance_margin

    def get_margin_ratio(self) -> float:
        """计算当前保证金使用率（基于实时WebSocket数据）

        Returns:
            保证金使用率 (0.0 - 1.0)
        """
        total_margin = self.account_data.get('totalMarginBalance', 0.0)
        used_margin = self.account_data.get('totalPositionInitialMargin', 0.0)

        if total_margin <= 0:
            return 0.0

        ratio = used_margin / total_margin
        return min(ratio, 1.0)  # 确保不超过1.0

    def get_available_margin(self) -> float:
        """获取可用保证金（实时）

        Returns:
            可用保证金数量
        """
        # 优先使用WebSocket实时数据
        available = self.account_data.get('availableBalance', 0.0)
        if available > 0:
            return available

        # 降级使用计算值
        total_margin = self.account_data.get('totalMarginBalance', self.initial_cash)
        used_margin = self.account_data.get('totalPositionInitialMargin', 0.0)
        return max(0.0, total_margin - used_margin)

    def calculate_liquidation_price(self,
                                   entry_price: float,
                                   position_size: float,
                                   margin: float,
                                   side: str) -> float:
        """计算强平价格

        Args:
            entry_price: 入场价格
            position_size: 仓位大小（正数表示多头，负数表示空头）
            margin: 保证金数量
            side: 仓位方向 ("long" | "short")

        Returns:
            强平价格

        Raises:
            ValueError: 当参数无效时
        """
        if entry_price <= 0:
            raise ValueError(f"入场价格必须大于0: {entry_price}")
        if abs(position_size) == 0:
            raise ValueError(f"仓位大小不能为0: {position_size}")
        if margin <= 0:
            raise ValueError(f"保证金必须大于0: {margin}")
        if side not in ["long", "short"]:
            raise ValueError(f"仓位方向必须是'long'或'short': {side}")

        # 计算有效杠杆
        position_value = entry_price * abs(position_size)
        leverage = position_value / margin

        # 强平价格计算
        # 多头强平价格 = 入场价格 * (1 - 维持保证金率 / 杠杆)
        # 空头强平价格 = 入场价格 * (1 + 维持保证金率 / 杠杆)
        maintenance_factor = self.maintenance_margin_rate / leverage

        if side == "long":
            liquidation_price = entry_price * (1 - maintenance_factor)
        else:  # short
            liquidation_price = entry_price * (1 + maintenance_factor)

        # 确保强平价格为正数
        liquidation_price = max(0.01, liquidation_price)

        logger.debug(f"强平价格计算 - 入场价格: {entry_price}, 方向: {side}, "
                    f"杠杆: {leverage:.2f}, 强平价格: {liquidation_price}")

        return liquidation_price

    def check_margin_sufficiency(self, required_margin: float) -> MarginStatus:
        """检查保证金是否充足（基于实时数据）

        Args:
            required_margin: 需要的保证金数量

        Returns:
            保证金状态对象
        """
        available = self.get_available_margin()
        total_balance = self.account_data.get('totalWalletBalance', 0.0)
        used_margin = self.account_data.get('totalPositionInitialMargin', 0.0)
        margin_ratio = self.get_margin_ratio()

        # 计算维持保证金要求
        maintenance_requirement = required_margin * (self.maintenance_margin_rate / self.initial_margin_rate)

        # 计算最大可开仓位价值（基于可用保证金和最大杠杆）
        max_position_value = available * self.max_leverage / self.initial_margin_rate

        # 判断是否可以开仓 - 修复逻辑：确保可用保证金确实大于等于需求
        can_open_position = available >= required_margin and required_margin > 0

        # 创建MarginStatus对象
        margin_status = MarginStatus(
            total_balance=total_balance,
            available_margin=available,
            used_margin=used_margin,
            margin_ratio=margin_ratio,
            initial_margin_requirement=required_margin,
            maintenance_margin_requirement=maintenance_requirement,
            can_open_position=True,  # 临时设置，稍后会基于实际逻辑修改
            currency=self.config.currency,
            update_time=datetime.now(),
            # 设置保证金阈值
            margin_call_threshold=self.margin_call_threshold,
            liquidation_threshold=self.liquidation_threshold,
            # 额外信息
            metadata={
                'max_position_value': max_position_value,
                'leverage_limit': self.max_leverage,
                'initial_margin_rate': self.initial_margin_rate,
                'maintenance_margin_rate': self.maintenance_margin_rate
            }
        )

        # 在__post_init__完成后，基于保证金充足性重新设置开仓权限
        # 如果保证金不足，即使风险等级允许开仓也不能开仓
        if not can_open_position:
            margin_status.can_open_position = False
            margin_status.account_status = "insufficient_margin"

        return margin_status

    async def assess_risk_level(self) -> RiskLevel:
        """评估当前风险等级

        Returns:
            风险等级枚举
        """
        margin_ratio = self.get_margin_ratio()

        # 根据保证金使用率判断风险等级
        # 注意：这里的阈值应该和margin_ratio的计算保持一致
        # margin_ratio = used_margin / total_margin，所以阈值应该相应调整
        if margin_ratio >= self.emergency_threshold:
            return RiskLevel.EMERGENCY
        elif margin_ratio >= self.liquidation_threshold:
            return RiskLevel.CRITICAL
        elif margin_ratio >= self.margin_call_threshold:
            return RiskLevel.HIGH
        elif margin_ratio >= 0.2:  # 20%使用率
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def start_monitoring(self) -> None:
        """启动保证金健康状态监控"""
        if self.is_monitoring:
            logger.warning("保证金监控已在运行")
            return

        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_margin_health())
        logger.info("保证金健康状态监控已启动")

    async def stop_monitoring(self) -> None:
        """停止保证金健康状态监控"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        logger.info("保证金健康状态监控已停止")

    async def _monitor_margin_health(self) -> None:
        """持续监控保证金健康状态"""
        logger.info("开始监控保证金健康状态")

        while self.is_monitoring:
            try:
                margin_ratio = self.get_margin_ratio()
                risk_level = await self.assess_risk_level()

                # 根据风险等级执行相应动作
                if risk_level == RiskLevel.EMERGENCY:
                    logger.critical(f"保证金使用率极高: {margin_ratio:.2%} - 立即执行紧急操作!")
                    await self._trigger_emergency_action()
                elif risk_level == RiskLevel.CRITICAL:
                    logger.error(f"保证金使用率过高: {margin_ratio:.2%} - 强制平仓风险!")
                    await self._suggest_risk_reduction()
                elif risk_level == RiskLevel.HIGH:
                    logger.warning(f"保证金使用率较高: {margin_ratio:.2%} - 建议降低风险")
                    await self._suggest_risk_reduction()
                elif risk_level == RiskLevel.MEDIUM:
                    logger.info(f"保证金使用率中等: {margin_ratio:.2%} - 保持关注")
                else:
                    logger.debug(f"保证金使用率正常: {margin_ratio:.2%}")

                await asyncio.sleep(5.0)  # 每5秒检查一次

            except Exception as e:
                logger.error(f"保证金健康监控出错: {e}")
                await asyncio.sleep(10.0)  # 出错时等待更长时间

    async def _trigger_emergency_action(self) -> None:
        """触发紧急操作"""
        logger.critical("触发紧急保证金操作 - 建议立即减少仓位或追加保证金")
        # 在实际实现中，这里可能会:
        # 1. 发送紧急通知
        # 2. 自动减少仓位
        # 3. 限制新开仓

    async def _suggest_risk_reduction(self) -> None:
        """建议风险降低措施"""
        available = self.get_available_margin()
        used = self.account_data.get('totalPositionInitialMargin', 0.0)

        logger.warning(f"建议降低风险 - 当前可用保证金: {available:.2f}, "
                      f"已用保证金: {used:.2f}")
        # 在实际实现中，这里可能会:
        # 1. 计算建议减仓数量
        # 2. 发送风险警告
        # 3. 限制高风险操作

    async def cleanup(self) -> None:
        """清理资源"""
        await self.stop_monitoring()
        if self.ws_interface:
            await self.ws_interface.disconnect()
        logger.info("保证金管理器资源清理完成")

    def get_margin_status(self) -> MarginStatus:
        """
        获取保证金状态对象

        Returns:
            MarginStatus: 保证金状态对象
        """
        try:
            # 获取当前余额信息
            total_wallet_balance = self.account_data.get('totalWalletBalance', 0.0)
            unrealized_pnl = self.account_data.get('totalUnrealizedProfit', 0.0)
            total_balance = total_wallet_balance + unrealized_pnl

            used_margin = self.account_data.get('totalPositionInitialMargin', 0.0) + \
                         self.account_data.get('totalOpenOrderInitialMargin', 0.0)
            available_margin = max(0, self.account_data.get('availableBalance', 0.0))

            # 计算保证金率
            margin_ratio = used_margin / total_balance if total_balance > 0 else 0.0

            # 评估风险等级
            if margin_ratio >= self.config.emergency_threshold:
                risk_level = RiskLevel.EMERGENCY
                can_trade = False
                can_open_position = False
            elif margin_ratio >= self.config.liquidation_threshold:
                risk_level = RiskLevel.CRITICAL
                can_trade = True
                can_open_position = False
            elif margin_ratio >= self.config.margin_call_threshold:
                risk_level = RiskLevel.HIGH
                can_trade = True
                can_open_position = True
            elif margin_ratio >= 0.15:  # 15%以上使用率
                risk_level = RiskLevel.MEDIUM
                can_trade = True
                can_open_position = True
            else:
                risk_level = RiskLevel.LOW
                can_trade = True
                can_open_position = True

            return MarginStatus(
                total_balance=total_balance,
                available_margin=available_margin,
                used_margin=used_margin,
                margin_ratio=margin_ratio,
                risk_level=risk_level,
                can_trade=can_trade,
                can_open_position=can_open_position,
                update_time=datetime.now()
            )

        except Exception as e:
            logger.error(f"获取保证金状态失败: {e}")
            # 返回安全默认状态
            return MarginStatus(
                total_balance=0.0,
                available_margin=0.0,
                used_margin=0.0,
                margin_ratio=1.0,
                risk_level=RiskLevel.EMERGENCY,
                can_trade=False,
                can_open_position=False,
                update_time=datetime.now()
            )

    def get_status_summary(self) -> Dict[str, Any]:
        """获取保证金状态摘要

        Returns:
            状态摘要字典
        """
        return {
            "total_balance": self.account_data.get('totalWalletBalance', 0.0),
            "available_margin": self.get_available_margin(),
            "used_margin": self.account_data.get('totalPositionInitialMargin', 0.0),
            "margin_ratio": self.get_margin_ratio(),
            "risk_level": asyncio.create_task(self.assess_risk_level()),
            "is_monitoring": self.is_monitoring,
            "currency": self.config.currency,
            "last_update": datetime.now().isoformat()
        }