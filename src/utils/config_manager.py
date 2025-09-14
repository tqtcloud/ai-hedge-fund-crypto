"""
配置管理器

支持配置的动态更新、热重载和环境切换功能。
提供配置变更监听、缓存管理和配置同步等功能。
"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml

from .settings import Settings, load_settings_with_futures_validation, get_environment_config, BinanceSDKSettings


logger = logging.getLogger(__name__)


class ConfigChangeHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.last_modified = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = event.src_path
        if not file_path.endswith('.yaml'):
            return
            
        # 防止重复触发
        current_time = time.time()
        if file_path in self.last_modified:
            if current_time - self.last_modified[file_path] < 1:  # 1秒内不重复处理
                return
        
        self.last_modified[file_path] = current_time
        logger.info(f"检测到配置文件变更: {file_path}")
        
        # 延迟一点时间确保文件写入完成
        threading.Timer(0.5, lambda: self.callback(file_path)).start()


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.settings: Optional[Settings] = None
        self.observers: List[Observer] = []
        self.change_callbacks: List[Callable[[Settings], None]] = []
        self._lock = threading.RLock()
        self._config_cache: Dict[str, Any] = {}
        self._last_loaded = 0
        
        # 初始化配置
        self._load_initial_config()
        
    def _load_initial_config(self) -> None:
        """加载初始配置"""
        try:
            self.settings = load_settings_with_futures_validation(str(self.config_path))
            self._last_loaded = time.time()
            logger.info(f"配置加载成功: {self.config_path}")
            
            # 缓存环境信息
            self._config_cache["environment_info"] = get_environment_config(self.settings)
            
            # 缓存SDK配置信息
            self._config_cache["sdk_info"] = self._get_sdk_config_info()
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            raise
    
    def get_settings(self) -> Optional[Settings]:
        """获取当前配置"""
        with self._lock:
            return self.settings
    
    def reload_config(self, force: bool = False) -> bool:
        """
        重新加载配置
        
        Args:
            force: 是否强制重载，忽略时间检查
            
        Returns:
            bool: 重载是否成功
        """
        try:
            with self._lock:
                # 检查文件修改时间
                if not force:
                    file_mtime = self.config_path.stat().st_mtime
                    if file_mtime <= self._last_loaded:
                        return True
                
                logger.info("开始重新加载配置...")
                old_settings = self.settings
                
                # 加载新配置
                new_settings = load_settings_with_futures_validation(str(self.config_path))
                
                # 验证配置变更
                if self._validate_config_change(old_settings, new_settings):
                    self.settings = new_settings
                    self._last_loaded = time.time()
                    
                    # 更新缓存
                    self._config_cache["environment_info"] = get_environment_config(self.settings)
                    self._config_cache["sdk_info"] = self._get_sdk_config_info()
                    
                    logger.info("配置重载成功")
                    
                    # 通知所有回调
                    self._notify_config_change()
                    
                    return True
                else:
                    logger.error("配置变更验证失败，保持原配置")
                    return False
                    
        except Exception as e:
            logger.error(f"配置重载失败: {e}")
            return False
    
    def _validate_config_change(self, old_config: Optional[Settings], new_config: Settings) -> bool:
        """
        验证配置变更的有效性
        
        Args:
            old_config: 旧配置
            new_config: 新配置
            
        Returns:
            bool: 配置变更是否有效
        """
        if not old_config:
            return True
        
        # 检查关键配置项是否发生不兼容的变更
        critical_changes = []
        
        # 检查期货环境变更
        if old_config.futures and new_config.futures:
            old_env = old_config.get_effective_futures_environment()
            new_env = new_config.get_effective_futures_environment()
            
            if old_env != new_env:
                critical_changes.append(f"期货环境: {old_env} -> {new_env}")
        
        # 检查交易模式变更
        if old_config.mode != new_config.mode:
            critical_changes.append(f"交易模式: {old_config.mode} -> {new_config.mode}")
        
        if critical_changes:
            logger.warning(f"检测到关键配置变更: {critical_changes}")
            
            # 如果从非live模式切换到live模式，需要额外验证
            if old_config.mode != 'live' and new_config.mode == 'live':
                if not self._validate_live_mode_requirements(new_config):
                    return False
        
        return True
    
    def _validate_live_mode_requirements(self, config: Settings) -> bool:
        """
        验证live模式的必要条件
        
        Args:
            config: 配置对象
            
        Returns:
            bool: 是否满足live模式要求
        """
        if not config.futures:
            logger.error("Live模式需要期货配置")
            return False
        
        # 验证API凭证
        if not config.futures.validate_api_credentials():
            logger.error("Live模式需要有效的API凭证")
            return False
        
        # 验证风险控制参数
        if config.futures.risk.daily_loss_limit <= 0:
            logger.error("Live模式需要设置有效的日损失限额")
            return False
        
        return True
    
    def switch_environment(self, env_name: str) -> bool:
        """
        切换环境配置
        
        Args:
            env_name: 环境名称
            
        Returns:
            bool: 切换是否成功
        """
        try:
            with self._lock:
                if not self.settings:
                    logger.error("配置未初始化")
                    return False
                
                if not self.settings.environment:
                    logger.error("未配置环境设置")
                    return False
                
                if env_name not in self.settings.environment.profiles:
                    logger.error(f"环境 {env_name} 不存在")
                    return False
                
                old_env = self.settings.environment.active
                self.settings.switch_environment(env_name)
                
                # 更新缓存
                self._config_cache["environment_info"] = get_environment_config(self.settings)
                self._config_cache["sdk_info"] = self._get_sdk_config_info()
                
                logger.info(f"环境切换成功: {old_env} -> {env_name}")
                
                # 通知配置变更
                self._notify_config_change()
                
                return True
                
        except Exception as e:
            logger.error(f"环境切换失败: {e}")
            return False
    
    def get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        with self._lock:
            return self._config_cache.get("environment_info", {})
    
    def get_sdk_info(self) -> Dict[str, Any]:
        """获取SDK配置信息"""
        with self._lock:
            return self._config_cache.get("sdk_info", {})
    
    def _get_sdk_config_info(self) -> Dict[str, Any]:
        """内部方法：获取SDK配置信息"""
        if not self.settings or not self.settings.binance_sdk:
            return {"configured": False}
        
        sdk_config = self.settings.binance_sdk
        return {
            "configured": True,
            "environment": sdk_config.get_effective_environment(),
            "credentials_valid": sdk_config.validate_credentials(),
            "api_key_configured": bool(sdk_config.api.get_resolved_credentials().get("api_key")),
            "private_key_configured": bool(sdk_config.api.private_key_path),
            "proxy_enabled": {
                "rest_api": sdk_config.rest_api.proxy.enabled,
                "websocket_api": sdk_config.websocket_api.proxy.enabled,
                "websocket_streams": sdk_config.websocket_streams.proxy.enabled
            },
            "timeout_configs": {
                "rest_api": sdk_config.rest_api.timeout,
                "websocket_api": sdk_config.websocket_api.timeout,
                "websocket_streams": sdk_config.websocket_streams.timeout
            }
        }
    
    def update_sdk_environment(self, env_type: str) -> bool:
        """
        更新SDK环境配置
        
        Args:
            env_type: 环境类型 (testnet/mainnet)
            
        Returns:
            bool: 更新是否成功
        """
        try:
            with self._lock:
                if not self.settings or not self.settings.binance_sdk:
                    logger.error("SDK配置未初始化")
                    return False
                
                if env_type not in ['testnet', 'mainnet']:
                    logger.error(f"不支持的环境类型: {env_type}")
                    return False
                
                old_env = self.settings.binance_sdk.get_effective_environment()
                self.settings.binance_sdk.environment.type = env_type
                
                # 更新缓存
                self._config_cache["sdk_info"] = self._get_sdk_config_info()
                
                logger.info(f"SDK环境更新成功: {old_env} -> {env_type}")
                
                # 通知配置变更
                self._notify_config_change()
                
                return True
                
        except Exception as e:
            logger.error(f"SDK环境更新失败: {e}")
            return False
    
    def validate_sdk_configuration(self) -> Dict[str, Any]:
        """
        验证SDK配置的完整性
        
        Returns:
            Dict[str, Any]: 验证结果
        """
        with self._lock:
            if not self.settings or not self.settings.binance_sdk:
                return {
                    "valid": False,
                    "error": "SDK配置未初始化"
                }
            
            try:
                sdk_config = self.settings.binance_sdk
                validation_result = {
                    "valid": True,
                    "environment": sdk_config.get_effective_environment(),
                    "credentials_valid": sdk_config.validate_credentials(),
                    "warnings": [],
                    "errors": []
                }
                
                # 检查凭证配置
                credentials = sdk_config.api.get_resolved_credentials()
                if not credentials.get("api_key"):
                    validation_result["warnings"].append("API Key未配置")
                if not credentials.get("api_secret"):
                    validation_result["warnings"].append("API Secret未配置")
                
                # 检查代理配置
                if sdk_config.rest_api.proxy.enabled and not sdk_config.rest_api.proxy.host:
                    validation_result["errors"].append("REST API代理已启用但未配置主机")
                
                if sdk_config.websocket_api.proxy.enabled and not sdk_config.websocket_api.proxy.host:
                    validation_result["errors"].append("WebSocket API代理已启用但未配置主机")
                
                if sdk_config.websocket_streams.proxy.enabled and not sdk_config.websocket_streams.proxy.host:
                    validation_result["errors"].append("WebSocket流代理已启用但未配置主机")
                
                # 检查超时配置合理性
                if sdk_config.rest_api.timeout > 60000:  # 1分钟
                    validation_result["warnings"].append(f"REST API超时时间过长: {sdk_config.rest_api.timeout}ms")
                
                # 检查私钥文件
                if sdk_config.api.private_key_path and not os.path.exists(sdk_config.api.private_key_path):
                    validation_result["errors"].append(f"私钥文件不存在: {sdk_config.api.private_key_path}")
                
                # 如果有错误，标记为无效
                if validation_result["errors"]:
                    validation_result["valid"] = False
                
                return validation_result
                
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"SDK配置验证失败: {e}"
                }
    
    def add_change_callback(self, callback: Callable[[Settings], None]) -> None:
        """
        添加配置变更回调
        
        Args:
            callback: 配置变更回调函数
        """
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[Settings], None]) -> None:
        """
        移除配置变更回调
        
        Args:
            callback: 要移除的回调函数
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def _notify_config_change(self) -> None:
        """通知所有配置变更回调"""
        if not self.settings:
            return
            
        for callback in self.change_callbacks:
            try:
                callback(self.settings)
            except Exception as e:
                logger.error(f"配置变更回调执行失败: {e}")
    
    def start_watching(self) -> None:
        """开始监控配置文件变更"""
        if not self.config_path.exists():
            logger.warning(f"配置文件不存在，无法开启监控: {self.config_path}")
            return
        
        # 监控配置文件所在目录
        config_dir = self.config_path.parent
        
        event_handler = ConfigChangeHandler(self._on_config_file_changed)
        observer = Observer()
        observer.schedule(event_handler, str(config_dir), recursive=False)
        observer.start()
        
        self.observers.append(observer)
        logger.info(f"开始监控配置文件变更: {config_dir}")
    
    def stop_watching(self) -> None:
        """停止监控配置文件变更"""
        for observer in self.observers:
            observer.stop()
            observer.join()
        
        self.observers.clear()
        logger.info("已停止配置文件变更监控")
    
    def _on_config_file_changed(self, file_path: str) -> None:
        """配置文件变更回调"""
        if Path(file_path).name == self.config_path.name:
            logger.info(f"配置文件发生变更: {file_path}")
            self.reload_config()
    
    def export_config(self, output_path: str, exclude_sensitive: bool = True) -> bool:
        """
        导出当前配置到文件
        
        Args:
            output_path: 输出文件路径
            exclude_sensitive: 是否排除敏感信息
            
        Returns:
            bool: 导出是否成功
        """
        try:
            with self._lock:
                if not self.settings:
                    logger.error("没有可导出的配置")
                    return False
                
                # 读取原始配置数据
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # 如果需要排除敏感信息
                if exclude_sensitive and 'futures' in config_data:
                    futures_config = config_data['futures']
                    if 'security' in futures_config:
                        # 移除敏感的环境变量配置
                        security_config = futures_config['security'].copy()
                        security_config['api_key_env'] = "YOUR_API_KEY_ENV_VAR"
                        security_config['secret_key_env'] = "YOUR_SECRET_KEY_ENV_VAR"
                        futures_config['security'] = security_config
                
                # 添加导出时间戳
                config_data['exported_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                config_data['exported_from'] = str(self.config_path)
                
                # 写入文件
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
                
                logger.info(f"配置导出成功: {output_path}")
                return True
                
        except Exception as e:
            logger.error(f"配置导出失败: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        获取配置摘要信息
        
        Returns:
            Dict[str, Any]: 配置摘要
        """
        with self._lock:
            if not self.settings:
                return {"error": "配置未初始化"}
            
            summary = {
                "config_file": str(self.config_path),
                "last_loaded": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self._last_loaded)),
                "mode": self.settings.mode,
                "environment_info": self.get_environment_info(),
                "futures_enabled": self.settings.futures is not None,
                "binance_sdk_enabled": self.settings.binance_sdk is not None
            }
            
            if self.settings.futures:
                summary.update({
                    "futures_environment": self.settings.get_effective_futures_environment(),
                    "api_credentials_configured": self.settings.futures.validate_api_credentials(),
                    "symbols_count": len(self.settings.futures.symbols),
                    "risk_management_enabled": True
                })
            
            if self.settings.binance_sdk:
                sdk_info = self.get_sdk_info()
                summary.update({
                    "sdk_environment": sdk_info.get("environment"),
                    "sdk_credentials_valid": sdk_info.get("credentials_valid"),
                    "sdk_proxy_enabled": any(sdk_info.get("proxy_enabled", {}).values()),
                    "sdk_validation": self.validate_sdk_configuration()
                })
            
            return summary
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_watching()


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: str = "config.yaml") -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    
    if _config_manager is None or str(_config_manager.config_path) != config_path:
        if _config_manager:
            _config_manager.stop_watching()
        _config_manager = ConfigManager(config_path)
    
    return _config_manager


def initialize_config_manager(config_path: str = "config.yaml", auto_watch: bool = True) -> ConfigManager:
    """
    初始化配置管理器
    
    Args:
        config_path: 配置文件路径
        auto_watch: 是否自动监控文件变更
        
    Returns:
        ConfigManager: 初始化的配置管理器
    """
    manager = get_config_manager(config_path)
    
    if auto_watch:
        manager.start_watching()
    
    return manager