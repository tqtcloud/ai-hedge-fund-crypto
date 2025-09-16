"""
性能监控系统 - 实时监控和分析

主要功能：
1. 实时性能指标收集
2. 性能阈值告警
3. 详细性能报告生成
4. 可视化仪表板数据
"""

import time
import psutil
import asyncio
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import json
import logging
from pathlib import Path
from collections import deque, defaultdict
import threading
from enum import Enum
import tracemalloc
import numpy as np

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    CACHE_HIT_RATE = "cache_hit_rate"


@dataclass
class PerformanceMetric:
    """性能指标"""
    timestamp: datetime
    component: str
    operation: str
    metric_type: MetricType
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """性能告警"""
    timestamp: datetime
    severity: str  # info, warning, critical
    component: str
    message: str
    metric: PerformanceMetric
    threshold: float


class PerformanceMonitor:
    """
    性能监控器

    提供全面的性能监控和分析能力
    """

    def __init__(
        self,
        log_dir: str = "logs/performance",
        enable_memory_profiling: bool = True,
        alert_callback: Optional[Callable[[Alert], None]] = None
    ):
        """
        初始化性能监控器

        Args:
            log_dir: 日志目录
            enable_memory_profiling: 是否启用内存分析
            alert_callback: 告警回调函数
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.enable_memory_profiling = enable_memory_profiling
        self.alert_callback = alert_callback

        # 性能指标存储
        self.metrics: deque = deque(maxlen=10000)  # 最多保存10000条记录
        self.operations_in_progress: Dict[str, Dict] = {}

        # 性能阈值
        self.thresholds = {
            'data_fetch': {
                'warning': 2000,    # 2秒
                'critical': 5000    # 5秒
            },
            'strategy_calc': {
                'warning': 1000,    # 1秒
                'critical': 3000    # 3秒
            },
            'signal_process': {
                'warning': 500,     # 500ms
                'critical': 1500    # 1.5秒
            },
            'api_call': {
                'warning': 1000,    # 1秒
                'critical': 3000    # 3秒
            },
            'memory_usage_mb': {
                'warning': 2048,    # 2GB
                'critical': 4096    # 4GB
            },
            'cpu_usage_percent': {
                'warning': 70,
                'critical': 90
            }
        }

        # 统计信息
        self.stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'max_time': 0,
            'min_time': float('inf'),
            'errors': 0
        })

        # 启动后台监控线程
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._background_monitor)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        # 启用内存分析
        if enable_memory_profiling:
            tracemalloc.start()

        logger.info("性能监控器已启动")

    def start_operation(
        self,
        operation_id: str,
        component: str,
        operation_type: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        开始监控操作

        Args:
            operation_id: 操作ID
            component: 组件名称
            operation_type: 操作类型
            metadata: 元数据

        Returns:
            操作ID
        """
        self.operations_in_progress[operation_id] = {
            'start_time': time.perf_counter(),
            'component': component,
            'operation_type': operation_type,
            'metadata': metadata or {},
            'cpu_start': psutil.cpu_percent(interval=0),
            'memory_start': psutil.Process().memory_info().rss / 1024 / 1024
        }

        return operation_id

    def end_operation(
        self,
        operation_id: str,
        success: bool = True,
        result_metadata: Optional[Dict] = None
    ) -> Optional[PerformanceMetric]:
        """
        结束监控操作

        Args:
            operation_id: 操作ID
            success: 是否成功
            result_metadata: 结果元数据

        Returns:
            性能指标
        """
        if operation_id not in self.operations_in_progress:
            logger.warning(f"操作ID不存在: {operation_id}")
            return None

        op_data = self.operations_in_progress.pop(operation_id)
        duration_ms = (time.perf_counter() - op_data['start_time']) * 1000

        # 创建性能指标
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            component=op_data['component'],
            operation=op_data['operation_type'],
            metric_type=MetricType.LATENCY,
            value=duration_ms,
            unit='ms',
            metadata={
                **op_data['metadata'],
                **(result_metadata or {}),
                'success': success,
                'cpu_usage': psutil.cpu_percent(interval=0) - op_data['cpu_start'],
                'memory_delta_mb': (
                    psutil.Process().memory_info().rss / 1024 / 1024 -
                    op_data['memory_start']
                )
            }
        )

        # 存储指标
        self.metrics.append(metric)

        # 更新统计
        self._update_statistics(op_data['operation_type'], duration_ms, success)

        # 检查阈值
        self._check_thresholds(metric)

        return metric

    def record_metric(
        self,
        component: str,
        operation: str,
        metric_type: MetricType,
        value: float,
        unit: str,
        metadata: Optional[Dict] = None
    ) -> PerformanceMetric:
        """
        记录自定义指标

        Args:
            component: 组件名称
            operation: 操作名称
            metric_type: 指标类型
            value: 指标值
            unit: 单位
            metadata: 元数据

        Returns:
            性能指标
        """
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            metric_type=metric_type,
            value=value,
            unit=unit,
            metadata=metadata or {}
        )

        self.metrics.append(metric)
        return metric

    def _update_statistics(self, operation_type: str, duration_ms: float, success: bool):
        """更新统计信息"""
        stats = self.stats[operation_type]
        stats['count'] += 1
        stats['total_time'] += duration_ms
        stats['max_time'] = max(stats['max_time'], duration_ms)
        stats['min_time'] = min(stats['min_time'], duration_ms)

        if not success:
            stats['errors'] += 1

    def _check_thresholds(self, metric: PerformanceMetric):
        """检查性能阈值"""
        operation_thresholds = self.thresholds.get(metric.operation, {})

        if not operation_thresholds:
            return

        warning_threshold = operation_thresholds.get('warning')
        critical_threshold = operation_thresholds.get('critical')

        alert = None

        if critical_threshold and metric.value > critical_threshold:
            alert = Alert(
                timestamp=datetime.now(),
                severity='critical',
                component=metric.component,
                message=f"{metric.operation} 严重超时: {metric.value:.2f}{metric.unit} > {critical_threshold}{metric.unit}",
                metric=metric,
                threshold=critical_threshold
            )
        elif warning_threshold and metric.value > warning_threshold:
            alert = Alert(
                timestamp=datetime.now(),
                severity='warning',
                component=metric.component,
                message=f"{metric.operation} 轻微超时: {metric.value:.2f}{metric.unit} > {warning_threshold}{metric.unit}",
                metric=metric,
                threshold=warning_threshold
            )

        if alert:
            self._handle_alert(alert)

    def _handle_alert(self, alert: Alert):
        """处理告警"""
        # 记录日志
        if alert.severity == 'critical':
            logger.error(f"性能告警: {alert.message}")
        elif alert.severity == 'warning':
            logger.warning(f"性能告警: {alert.message}")
        else:
            logger.info(f"性能告警: {alert.message}")

        # 调用回调函数
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"告警回调失败: {e}")

    def _background_monitor(self):
        """后台监控线程"""
        while self.monitoring:
            try:
                # 监控系统资源
                self._monitor_system_resources()

                # 每30秒生成一次报告
                time.sleep(30)

            except Exception as e:
                logger.error(f"后台监控失败: {e}")

    def _monitor_system_resources(self):
        """监控系统资源"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        self.record_metric(
            component='system',
            operation='cpu_usage',
            metric_type=MetricType.RESOURCE_USAGE,
            value=cpu_percent,
            unit='%'
        )

        # 内存使用
        memory = psutil.Process().memory_info()
        memory_mb = memory.rss / 1024 / 1024
        self.record_metric(
            component='system',
            operation='memory_usage',
            metric_type=MetricType.RESOURCE_USAGE,
            value=memory_mb,
            unit='MB',
            metadata={'memory_percent': psutil.virtual_memory().percent}
        )

        # 检查资源阈值
        if cpu_percent > self.thresholds['cpu_usage_percent']['critical']:
            alert = Alert(
                timestamp=datetime.now(),
                severity='critical',
                component='system',
                message=f"CPU使用率过高: {cpu_percent:.1f}%",
                metric=PerformanceMetric(
                    timestamp=datetime.now(),
                    component='system',
                    operation='cpu_usage',
                    metric_type=MetricType.RESOURCE_USAGE,
                    value=cpu_percent,
                    unit='%'
                ),
                threshold=self.thresholds['cpu_usage_percent']['critical']
            )
            self._handle_alert(alert)

        if memory_mb > self.thresholds['memory_usage_mb']['critical']:
            alert = Alert(
                timestamp=datetime.now(),
                severity='critical',
                component='system',
                message=f"内存使用过高: {memory_mb:.0f}MB",
                metric=PerformanceMetric(
                    timestamp=datetime.now(),
                    component='system',
                    operation='memory_usage',
                    metric_type=MetricType.RESOURCE_USAGE,
                    value=memory_mb,
                    unit='MB'
                ),
                threshold=self.thresholds['memory_usage_mb']['critical']
            )
            self._handle_alert(alert)

    def get_statistics(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计信息

        Args:
            operation_type: 操作类型（可选）

        Returns:
            统计信息
        """
        if operation_type:
            stats = self.stats.get(operation_type, {})
            if stats['count'] > 0:
                return {
                    'operation': operation_type,
                    'count': stats['count'],
                    'avg_time_ms': stats['total_time'] / stats['count'],
                    'max_time_ms': stats['max_time'],
                    'min_time_ms': stats['min_time'] if stats['min_time'] != float('inf') else 0,
                    'error_rate': stats['errors'] / stats['count'] * 100,
                    'total_errors': stats['errors']
                }
            return {}

        # 返回所有统计
        all_stats = {}
        for op_type, stats in self.stats.items():
            if stats['count'] > 0:
                all_stats[op_type] = {
                    'count': stats['count'],
                    'avg_time_ms': stats['total_time'] / stats['count'],
                    'max_time_ms': stats['max_time'],
                    'min_time_ms': stats['min_time'] if stats['min_time'] != float('inf') else 0,
                    'error_rate': stats['errors'] / stats['count'] * 100,
                    'total_errors': stats['errors']
                }
        return all_stats

    def get_recent_metrics(
        self,
        minutes: int = 5,
        component: Optional[str] = None,
        operation: Optional[str] = None
    ) -> List[PerformanceMetric]:
        """
        获取最近的性能指标

        Args:
            minutes: 时间范围（分钟）
            component: 组件过滤
            operation: 操作过滤

        Returns:
            性能指标列表
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = []

        for metric in self.metrics:
            if metric.timestamp < cutoff_time:
                continue

            if component and metric.component != component:
                continue

            if operation and metric.operation != operation:
                continue

            recent_metrics.append(metric)

        return recent_metrics

    def generate_report(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        生成性能报告

        Args:
            filepath: 保存路径（可选）

        Returns:
            报告数据
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_operations': sum(s['count'] for s in self.stats.values()),
                'total_errors': sum(s['errors'] for s in self.stats.values()),
                'avg_latency_ms': self._calculate_overall_avg_latency(),
                'error_rate': self._calculate_overall_error_rate()
            },
            'statistics': self.get_statistics(),
            'recent_metrics': [
                asdict(m) for m in self.get_recent_metrics(minutes=60)
            ][:100],  # 最近100条
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'memory_percent': psutil.virtual_memory().percent
            }
        }

        # 添加内存分析（如果启用）
        if self.enable_memory_profiling:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            report['memory_profile'] = [
                {
                    'file': stat.traceback.format()[0],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                }
                for stat in top_stats
            ]

        # 保存报告
        if filepath is None:
            filepath = self.log_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filepath, 'w') as f:
            # 处理datetime序列化
            def default(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, MetricType):
                    return obj.value
                return str(obj)

            json.dump(report, f, indent=2, default=default)

        logger.info(f"性能报告已保存到: {filepath}")
        return report

    def _calculate_overall_avg_latency(self) -> float:
        """计算总体平均延迟"""
        total_time = sum(s['total_time'] for s in self.stats.values())
        total_count = sum(s['count'] for s in self.stats.values())

        if total_count == 0:
            return 0

        return total_time / total_count

    def _calculate_overall_error_rate(self) -> float:
        """计算总体错误率"""
        total_errors = sum(s['errors'] for s in self.stats.values())
        total_count = sum(s['count'] for s in self.stats.values())

        if total_count == 0:
            return 0

        return (total_errors / total_count) * 100

    def shutdown(self):
        """关闭监控器"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        # 生成最终报告
        self.generate_report()

        # 停止内存分析
        if self.enable_memory_profiling:
            tracemalloc.stop()

        logger.info("性能监控器已关闭")


class PerformanceContext:
    """
    性能监控上下文管理器

    用于自动监控代码块的性能
    """

    def __init__(
        self,
        monitor: PerformanceMonitor,
        operation_id: str,
        component: str,
        operation_type: str,
        metadata: Optional[Dict] = None
    ):
        self.monitor = monitor
        self.operation_id = operation_id
        self.component = component
        self.operation_type = operation_type
        self.metadata = metadata
        self.success = True
        self.result_metadata = {}

    def __enter__(self):
        self.monitor.start_operation(
            self.operation_id,
            self.component,
            self.operation_type,
            self.metadata
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.success = False
            self.result_metadata['error'] = str(exc_val)

        self.monitor.end_operation(
            self.operation_id,
            self.success,
            self.result_metadata
        )

        return False  # 不抑制异常

    def add_metadata(self, key: str, value: Any):
        """添加结果元数据"""
        self.result_metadata[key] = value


# 使用示例
async def example_usage():
    """性能监控使用示例"""

    def alert_handler(alert: Alert):
        """告警处理函数"""
        print(f"⚠️ 性能告警 [{alert.severity}]: {alert.message}")

    # 创建监控器
    monitor = PerformanceMonitor(alert_callback=alert_handler)

    # 示例1: 手动监控
    op_id = monitor.start_operation(
        operation_id="fetch_data_001",
        component="data_fetcher",
        operation_type="data_fetch"
    )

    # 模拟数据获取
    await asyncio.sleep(0.5)

    monitor.end_operation(op_id, success=True, result_metadata={'records': 1000})

    # 示例2: 使用上下文管理器
    with PerformanceContext(
        monitor,
        "calc_strategy_001",
        "strategy_engine",
        "strategy_calc"
    ) as ctx:
        # 模拟策略计算
        await asyncio.sleep(0.3)
        ctx.add_metadata('signals_generated', 5)

    # 示例3: 记录自定义指标
    monitor.record_metric(
        component='cache',
        operation='cache_hit_rate',
        metric_type=MetricType.CACHE_HIT_RATE,
        value=85.5,
        unit='%'
    )

    # 等待一段时间让后台监控运行
    await asyncio.sleep(5)

    # 获取统计信息
    stats = monitor.get_statistics()
    print(f"统计信息: {json.dumps(stats, indent=2)}")

    # 生成报告
    report = monitor.generate_report()
    print(f"报告已生成: {report['summary']}")

    # 关闭监控器
    monitor.shutdown()


if __name__ == "__main__":
    asyncio.run(example_usage())