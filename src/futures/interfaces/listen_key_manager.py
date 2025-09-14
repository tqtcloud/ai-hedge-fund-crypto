"""
币安期货Listen Key管理器

实现Listen Key的生命周期管理，包括：
- 获取新的Listen Key
- 自动刷新（每30分钟）
- 删除Listen Key
- 错误处理和重试机制

基于币安官方SDK实现，支持testnet和mainnet环境。
"""

import asyncio
import logging
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from binance_common.configuration import ConfigurationRestAPI
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import DerivativesTradingUsdsFutures


@dataclass
class ListenKeyInfo:
    """Listen Key信息"""
    key: str                                    # Listen Key字符串
    created_time: int                          # 创建时间（毫秒）
    last_refresh_time: int                     # 最后刷新时间（毫秒）
    refresh_interval: int = 30 * 60 * 1000    # 刷新间隔（毫秒，默认30分钟）
    max_lifetime: int = 24 * 60 * 60 * 1000   # 最大生命周期（毫秒，默认24小时）

    def is_expired(self) -> bool:
        """检查Listen Key是否已过期"""
        current_time = int(time.time() * 1000)
        return (current_time - self.created_time) >= self.max_lifetime

    def needs_refresh(self) -> bool:
        """检查Listen Key是否需要刷新"""
        current_time = int(time.time() * 1000)
        return (current_time - self.last_refresh_time) >= self.refresh_interval

    def update_refresh_time(self) -> None:
        """更新最后刷新时间"""
        self.last_refresh_time = int(time.time() * 1000)


class ListenKeyManager(ABC):
    """Listen Key管理器抽象基类"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        """
        初始化Listen Key管理器

        Args:
            api_key: Binance API Key
            api_secret: Binance API Secret
            testnet: 是否使用testnet环境
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.logger = self._setup_logger()

        # Listen Key管理
        self._current_key: Optional[ListenKeyInfo] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._is_running = False

        # 回调函数
        self._on_key_created: Optional[Callable[[str], None]] = None
        self._on_key_refreshed: Optional[Callable[[str], None]] = None
        self._on_key_expired: Optional[Callable[[str], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @abstractmethod
    async def _create_listen_key(self) -> str:
        """创建新的Listen Key（抽象方法）"""
        pass

    @abstractmethod
    async def _refresh_listen_key(self, listen_key: str) -> bool:
        """刷新Listen Key（抽象方法）"""
        pass

    @abstractmethod
    async def _delete_listen_key(self, listen_key: str) -> bool:
        """删除Listen Key（抽象方法）"""
        pass

    async def start(self) -> str:
        """
        启动Listen Key管理器并获取初始Key

        Returns:
            初始的Listen Key字符串

        Raises:
            Exception: 如果创建Listen Key失败
        """
        if self._is_running:
            self.logger.warning("Listen Key管理器已在运行")
            return self._current_key.key if self._current_key else ""

        self.logger.info("启动Listen Key管理器")

        try:
            # 创建初始Listen Key
            key = await self._create_listen_key()
            current_time = int(time.time() * 1000)

            self._current_key = ListenKeyInfo(
                key=key,
                created_time=current_time,
                last_refresh_time=current_time
            )

            # 启动自动刷新任务
            self._is_running = True
            self._refresh_task = asyncio.create_task(self._auto_refresh_loop())

            # 调用创建回调
            if self._on_key_created:
                try:
                    self._on_key_created(key)
                except Exception as e:
                    self.logger.error(f"创建回调执行失败: {e}")

            self.logger.info(f"Listen Key管理器启动成功, Key: {key[:10]}...")
            return key

        except Exception as e:
            self.logger.error(f"启动Listen Key管理器失败: {e}")
            self._is_running = False
            raise

    async def stop(self) -> None:
        """停止Listen Key管理器"""
        if not self._is_running:
            return

        self.logger.info("停止Listen Key管理器")
        self._is_running = False

        # 取消刷新任务
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass

        # 删除当前Listen Key
        if self._current_key:
            try:
                await self._delete_listen_key(self._current_key.key)
                self.logger.info(f"删除Listen Key: {self._current_key.key[:10]}...")
            except Exception as e:
                self.logger.error(f"删除Listen Key失败: {e}")

        self._current_key = None
        self._refresh_task = None

        self.logger.info("Listen Key管理器已停止")

    async def get_current_key(self) -> Optional[str]:
        """
        获取当前有效的Listen Key

        Returns:
            当前Listen Key字符串，如果没有或已过期则返回None
        """
        if not self._current_key or self._current_key.is_expired():
            return None
        return self._current_key.key

    async def force_refresh(self) -> str:
        """
        强制刷新Listen Key

        Returns:
            刷新后的Listen Key字符串

        Raises:
            Exception: 如果刷新失败
        """
        if not self._current_key:
            raise Exception("没有当前的Listen Key需要刷新")

        try:
            success = await self._refresh_listen_key(self._current_key.key)
            if success:
                self._current_key.update_refresh_time()

                # 调用刷新回调
                if self._on_key_refreshed:
                    try:
                        self._on_key_refreshed(self._current_key.key)
                    except Exception as e:
                        self.logger.error(f"刷新回调执行失败: {e}")

                self.logger.info(f"强制刷新Listen Key成功: {self._current_key.key[:10]}...")
                return self._current_key.key
            else:
                raise Exception("刷新Listen Key失败")

        except Exception as e:
            self.logger.error(f"强制刷新Listen Key失败: {e}")
            raise

    async def _auto_refresh_loop(self) -> None:
        """自动刷新循环"""
        self.logger.debug("启动自动刷新循环")

        while self._is_running:
            try:
                await asyncio.sleep(60)  # 每分钟检查一次

                if not self._current_key:
                    continue

                # 检查是否已过期
                if self._current_key.is_expired():
                    self.logger.warning("Listen Key已过期，重新创建")

                    # 调用过期回调
                    if self._on_key_expired:
                        try:
                            self._on_key_expired(self._current_key.key)
                        except Exception as e:
                            self.logger.error(f"过期回调执行失败: {e}")

                    # 创建新的Listen Key
                    try:
                        new_key = await self._create_listen_key()
                        current_time = int(time.time() * 1000)

                        old_key = self._current_key.key
                        self._current_key = ListenKeyInfo(
                            key=new_key,
                            created_time=current_time,
                            last_refresh_time=current_time
                        )

                        # 删除旧Key
                        try:
                            await self._delete_listen_key(old_key)
                        except Exception as e:
                            self.logger.error(f"删除过期Listen Key失败: {e}")

                        # 调用创建回调
                        if self._on_key_created:
                            try:
                                self._on_key_created(new_key)
                            except Exception as e:
                                self.logger.error(f"创建回调执行失败: {e}")

                        self.logger.info(f"重新创建Listen Key成功: {new_key[:10]}...")

                    except Exception as e:
                        self.logger.error(f"重新创建Listen Key失败: {e}")
                        if self._on_error:
                            try:
                                self._on_error(e)
                            except Exception as cb_e:
                                self.logger.error(f"错误回调执行失败: {cb_e}")

                # 检查是否需要刷新
                elif self._current_key.needs_refresh():
                    self.logger.debug("Listen Key需要刷新")

                    try:
                        success = await self._refresh_listen_key(self._current_key.key)
                        if success:
                            self._current_key.update_refresh_time()

                            # 调用刷新回调
                            if self._on_key_refreshed:
                                try:
                                    self._on_key_refreshed(self._current_key.key)
                                except Exception as e:
                                    self.logger.error(f"刷新回调执行失败: {e}")

                            self.logger.debug(f"自动刷新Listen Key成功: {self._current_key.key[:10]}...")
                        else:
                            self.logger.warning("自动刷新Listen Key失败")

                    except Exception as e:
                        self.logger.error(f"自动刷新Listen Key异常: {e}")
                        if self._on_error:
                            try:
                                self._on_error(e)
                            except Exception as cb_e:
                                self.logger.error(f"错误回调执行失败: {cb_e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自动刷新循环异常: {e}")
                if self._on_error:
                    try:
                        self._on_error(e)
                    except Exception as cb_e:
                        self.logger.error(f"错误回调执行失败: {cb_e}")
                await asyncio.sleep(10)  # 异常后等待10秒再继续

        self.logger.debug("自动刷新循环结束")

    # 回调函数设置方法
    def set_on_key_created(self, callback: Callable[[str], None]) -> None:
        """设置Listen Key创建回调函数"""
        self._on_key_created = callback

    def set_on_key_refreshed(self, callback: Callable[[str], None]) -> None:
        """设置Listen Key刷新回调函数"""
        self._on_key_refreshed = callback

    def set_on_key_expired(self, callback: Callable[[str], None]) -> None:
        """设置Listen Key过期回调函数"""
        self._on_key_expired = callback

    def set_on_error(self, callback: Callable[[Exception], None]) -> None:
        """设置错误回调函数"""
        self._on_error = callback


class BinanceFuturesListenKeyManager(ListenKeyManager):
    """币安期货Listen Key管理器实现"""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        super().__init__(api_key, api_secret, testnet)

        # 初始化币安客户端
        self._rest_client = self._create_rest_client()

    def _create_rest_client(self) -> DerivativesTradingUsdsFutures:
        """创建REST API客户端"""
        try:
            # 根据testnet设置选择URL
            if self.testnet:
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
                base_url = DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
            else:
                from binance_common.constants import DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
                base_url = DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL

            config = ConfigurationRestAPI(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_url=base_url
            )

            return DerivativesTradingUsdsFutures(config_rest_api=config)

        except Exception as e:
            self.logger.error(f"创建REST客户端失败: {e}")
            raise

    async def _create_listen_key(self) -> str:
        """创建新的Listen Key"""
        try:
            self.logger.debug("正在创建新的Listen Key...")

            # 调用币安API创建Listen Key
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self._rest_client.rest_api.new_listen_key
            )

            if response and hasattr(response, 'data') and response.data:
                data = response.data()
                if isinstance(data, dict) and 'listenKey' in data:
                    listen_key = data['listenKey']
                    self.logger.info(f"成功创建Listen Key: {listen_key[:10]}...")
                    return listen_key
                else:
                    raise Exception(f"无效的API响应格式: {data}")
            else:
                raise Exception("API返回空响应")

        except Exception as e:
            self.logger.error(f"创建Listen Key失败: {e}")
            raise Exception(f"创建Listen Key失败: {str(e)}")

    async def _refresh_listen_key(self, listen_key: str) -> bool:
        """刷新Listen Key"""
        try:
            self.logger.debug(f"正在刷新Listen Key: {listen_key[:10]}...")

            # 调用币安API刷新Listen Key
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._rest_client.rest_api.renew_listen_key(listen_key=listen_key)
            )

            # 币安API刷新成功通常返回空响应或状态码200
            if response is not None:
                self.logger.debug(f"成功刷新Listen Key: {listen_key[:10]}...")
                return True
            else:
                self.logger.warning(f"刷新Listen Key可能失败: {listen_key[:10]}...")
                return False

        except Exception as e:
            self.logger.error(f"刷新Listen Key失败: {e}")
            return False

    async def _delete_listen_key(self, listen_key: str) -> bool:
        """删除Listen Key"""
        try:
            self.logger.debug(f"正在删除Listen Key: {listen_key[:10]}...")

            # 调用币安API删除Listen Key
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._rest_client.rest_api.close_listen_key(listen_key=listen_key)
            )

            # 币安API删除成功通常返回空响应或状态码200
            if response is not None:
                self.logger.debug(f"成功删除Listen Key: {listen_key[:10]}...")
                return True
            else:
                self.logger.warning(f"删除Listen Key可能失败: {listen_key[:10]}...")
                return False

        except Exception as e:
            self.logger.error(f"删除Listen Key失败: {e}")
            return False


# 便捷函数
async def create_futures_listen_key_manager(
    api_key: str,
    api_secret: str,
    testnet: bool = True,
    auto_start: bool = True
) -> BinanceFuturesListenKeyManager:
    """
    创建并启动币安期货Listen Key管理器

    Args:
        api_key: Binance API Key
        api_secret: Binance API Secret
        testnet: 是否使用testnet环境
        auto_start: 是否自动启动

    Returns:
        配置好的Listen Key管理器实例
    """
    manager = BinanceFuturesListenKeyManager(api_key, api_secret, testnet)

    if auto_start:
        await manager.start()

    return manager