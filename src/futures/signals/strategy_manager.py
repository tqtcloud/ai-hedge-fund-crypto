"""
期货策略管理器

管理多个策略实例的运行、信号输出聚合、权重分配和验证，
确保策略输出的正确性和一致性。
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
import pandas as pd

try:
    from .base_strategy import FuturesBaseStrategy, StrategyOutput, SignalStrength
    from .strategy_factory import FuturesStrategyFactory, strategy_factory
    from ..models.data_models import FuturesSignal, TradingDirection, OperationType, ValidationResult, ValidationSeverity
    from ..constants import RiskParameters
except ImportError:
    # 处理相对导入问题
    from src.futures.signals.base_strategy import FuturesBaseStrategy, StrategyOutput, SignalStrength
    from src.futures.signals.strategy_factory import FuturesStrategyFactory, strategy_factory
    from src.futures.models.data_models import FuturesSignal, TradingDirection, OperationType, ValidationResult, ValidationSeverity
    from src.futures.constants import RiskParameters

logger = logging.getLogger(__name__)


class AggregationMethod(Enum):
    """信号聚合方法"""
    WEIGHTED_AVERAGE = "weighted_average"     # 加权平均
    MAJORITY_VOTE = "majority_vote"           # 多数投票
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # 置信度加权
    HIGHEST_CONFIDENCE = "highest_confidence"    # 最高置信度
    ENSEMBLE = "ensemble"                     # 集成方法


class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"           # 基础验证
    STANDARD = "standard"     # 标准验证
    STRICT = "strict"         # 严格验证


@dataclass
class StrategyWeight:
    """策略权重配置"""
    strategy_name: str
    weight: float = 1.0
    confidence_multiplier: float = 1.0
    enabled: bool = True
    max_influence: float = 0.5  # 单策略最大影响力

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"权重必须在0-1范围内: {self.weight}")
        if not 0 <= self.max_influence <= 1:
            raise ValueError(f"最大影响力必须在0-1范围内: {self.max_influence}")


@dataclass
class AggregatedSignal:
    """聚合信号结果"""
    final_signal: Optional[FuturesSignal]
    contributing_signals: List[FuturesSignal]
    aggregation_metadata: Dict[str, Any]
    validation_results: List[ValidationResult]
    confidence_scores: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_signal": self.final_signal.to_dict() if self.final_signal else None,
            "contributing_signals": [s.to_dict() for s in self.contributing_signals],
            "aggregation_metadata": self.aggregation_metadata,
            "validation_results": [v.to_dict() for v in self.validation_results],
            "confidence_scores": self.confidence_scores,
            "timestamp": self.timestamp.isoformat()
        }


class FuturesStrategyManager:
    """
    期货策略管理器

    负责管理多个策略实例的运行，提供信号聚合、权重分配、
    输出验证等功能，确保系统的稳定性和一致性。
    """

    def __init__(self,
                 factory: Optional[FuturesStrategyFactory] = None,
                 max_workers: int = 4,
                 default_aggregation_method: AggregationMethod = AggregationMethod.CONFIDENCE_WEIGHTED):
        """
        初始化策略管理器

        Args:
            factory: 策略工厂实例
            max_workers: 最大并发工作线程数
            default_aggregation_method: 默认聚合方法
        """
        self.factory = factory or strategy_factory
        self.max_workers = max_workers
        self.default_aggregation_method = default_aggregation_method

        # 管理的策略实例
        self.strategies: Dict[str, FuturesBaseStrategy] = {}

        # 策略权重配置
        self.strategy_weights: Dict[str, StrategyWeight] = {}

        # 运行状态
        self.is_running = False
        self.last_analysis_time: Optional[datetime] = None

        # 性能统计
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_processing_time": 0.0,
            "signal_distribution": {}
        }

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        logger.info(f"初始化期货策略管理器，最大工作线程: {self.max_workers}")

    def add_strategy(self,
                    strategy_name: str,
                    config: Optional[Dict[str, Any]] = None,
                    weight: float = 1.0,
                    confidence_multiplier: float = 1.0) -> bool:
        """
        添加策略到管理器

        Args:
            strategy_name: 策略名称
            config: 策略配置
            weight: 策略权重
            confidence_multiplier: 置信度倍数

        Returns:
            是否成功添加
        """
        try:
            # 创建策略实例
            strategy = self.factory.create_strategy(strategy_name, config)

            # 添加到管理器
            self.strategies[strategy_name] = strategy

            # 设置权重配置
            self.strategy_weights[strategy_name] = StrategyWeight(
                strategy_name=strategy_name,
                weight=weight,
                confidence_multiplier=confidence_multiplier
            )

            logger.info(f"成功添加策略到管理器: {strategy_name} (权重: {weight})")
            return True

        except Exception as e:
            logger.error(f"添加策略失败 '{strategy_name}': {e}")
            return False

    def remove_strategy(self, strategy_name: str) -> bool:
        """
        从管理器移除策略

        Args:
            strategy_name: 策略名称

        Returns:
            是否成功移除
        """
        try:
            if strategy_name in self.strategies:
                del self.strategies[strategy_name]

            if strategy_name in self.strategy_weights:
                del self.strategy_weights[strategy_name]

            logger.info(f"成功移除策略: {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"移除策略失败 '{strategy_name}': {e}")
            return False

    def update_strategy_weight(self,
                             strategy_name: str,
                             weight: Optional[float] = None,
                             confidence_multiplier: Optional[float] = None,
                             enabled: Optional[bool] = None) -> bool:
        """
        更新策略权重配置

        Args:
            strategy_name: 策略名称
            weight: 新权重
            confidence_multiplier: 置信度倍数
            enabled: 是否启用

        Returns:
            是否成功更新
        """
        try:
            if strategy_name not in self.strategy_weights:
                logger.error(f"策略权重配置不存在: {strategy_name}")
                return False

            weight_config = self.strategy_weights[strategy_name]

            if weight is not None:
                weight_config.weight = weight

            if confidence_multiplier is not None:
                weight_config.confidence_multiplier = confidence_multiplier

            if enabled is not None:
                weight_config.enabled = enabled

            logger.info(f"更新策略权重配置: {strategy_name}")
            return True

        except Exception as e:
            logger.error(f"更新策略权重失败 '{strategy_name}': {e}")
            return False

    def get_active_strategies(self) -> List[str]:
        """
        获取活跃策略列表

        Returns:
            活跃策略名称列表
        """
        active_strategies = []
        for name, strategy in self.strategies.items():
            weight_config = self.strategy_weights.get(name)
            if strategy.is_enabled and (not weight_config or weight_config.enabled):
                active_strategies.append(name)
        return active_strategies

    def analyze_ticker(self,
                      ticker: str,
                      data: Dict[str, pd.DataFrame],
                      current_price: float,
                      aggregation_method: Optional[AggregationMethod] = None,
                      validation_level: ValidationLevel = ValidationLevel.STANDARD,
                      **kwargs) -> Optional[AggregatedSignal]:
        """
        对指定交易对进行多策略分析

        Args:
            ticker: 交易对符号
            data: 多时间框架数据
            current_price: 当前价格
            aggregation_method: 聚合方法
            validation_level: 验证级别
            **kwargs: 其他参数

        Returns:
            聚合信号结果
        """
        try:
            start_time = datetime.now()
            active_strategies = self.get_active_strategies()

            if not active_strategies:
                logger.warning("没有活跃的策略可用于分析")
                return None

            logger.info(f"开始多策略分析: {ticker}, 活跃策略: {len(active_strategies)}")

            # 并行执行策略分析
            strategy_outputs = self._run_strategies_parallel(
                active_strategies, ticker, data, current_price, **kwargs
            )

            if not strategy_outputs:
                logger.warning(f"所有策略分析都失败: {ticker}")
                return None

            # 提取有效信号
            valid_signals = []
            for output in strategy_outputs.values():
                if output and output.signal and output.signal.is_valid():
                    valid_signals.append(output.signal)

            if not valid_signals:
                logger.warning(f"没有有效信号产生: {ticker}")
                return None

            # 聚合信号
            aggregated_signal = self._aggregate_signals(
                valid_signals,
                ticker,
                current_price,
                aggregation_method or self.default_aggregation_method
            )

            # 验证聚合结果
            validation_results = self._validate_aggregated_signal(
                aggregated_signal, validation_level
            )

            # 创建最终结果
            result = AggregatedSignal(
                final_signal=aggregated_signal,
                contributing_signals=valid_signals,
                aggregation_metadata={
                    "method": (aggregation_method or self.default_aggregation_method).value,
                    "active_strategies": active_strategies,
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "ticker": ticker,
                    "current_price": current_price
                },
                validation_results=validation_results,
                confidence_scores={s.strategy_source: s.confidence for s in valid_signals}
            )

            # 更新统计
            self._update_performance_stats(True, datetime.now() - start_time)

            logger.info(f"多策略分析完成: {ticker}, 信号: {aggregated_signal.direction.value if aggregated_signal else 'None'}")

            return result

        except Exception as e:
            logger.error(f"多策略分析失败 '{ticker}': {e}")
            self._update_performance_stats(False, datetime.now() - start_time if 'start_time' in locals() else timedelta(0))
            return None

    def _run_strategies_parallel(self,
                               strategy_names: List[str],
                               ticker: str,
                               data: Dict[str, pd.DataFrame],
                               current_price: float,
                               **kwargs) -> Dict[str, Optional[StrategyOutput]]:
        """
        并行运行多个策略

        Args:
            strategy_names: 策略名称列表
            ticker: 交易对符号
            data: 数据
            current_price: 当前价格
            **kwargs: 其他参数

        Returns:
            策略输出字典
        """
        strategy_outputs = {}

        # 提交并行任务
        future_to_strategy = {}
        for strategy_name in strategy_names:
            if strategy_name in self.strategies:
                future = self.executor.submit(
                    self._run_single_strategy,
                    strategy_name,
                    ticker,
                    data,
                    current_price,
                    **kwargs
                )
                future_to_strategy[future] = strategy_name

        # 收集结果
        for future in as_completed(future_to_strategy, timeout=30):
            strategy_name = future_to_strategy[future]
            try:
                result = future.result()
                strategy_outputs[strategy_name] = result
            except Exception as e:
                logger.error(f"策略 '{strategy_name}' 执行失败: {e}")
                strategy_outputs[strategy_name] = None

        return strategy_outputs

    def _run_single_strategy(self,
                           strategy_name: str,
                           ticker: str,
                           data: Dict[str, pd.DataFrame],
                           current_price: float,
                           **kwargs) -> Optional[StrategyOutput]:
        """
        运行单个策略

        Args:
            strategy_name: 策略名称
            ticker: 交易对符号
            data: 数据
            current_price: 当前价格
            **kwargs: 其他参数

        Returns:
            策略输出
        """
        try:
            strategy = self.strategies[strategy_name]

            # 验证数据
            if not strategy.validate_data(data):
                logger.warning(f"策略 '{strategy_name}' 数据验证失败")
                return None

            # 执行分析
            result = strategy.analyze(ticker, data, current_price, **kwargs)

            return result

        except Exception as e:
            logger.error(f"单策略执行失败 '{strategy_name}': {e}")
            return None

    def _aggregate_signals(self,
                         signals: List[FuturesSignal],
                         ticker: str,
                         current_price: float,
                         method: AggregationMethod) -> Optional[FuturesSignal]:
        """
        聚合多个信号

        Args:
            signals: 信号列表
            ticker: 交易对符号
            current_price: 当前价格
            method: 聚合方法

        Returns:
            聚合后的信号
        """
        try:
            if not signals:
                return None

            if method == AggregationMethod.HIGHEST_CONFIDENCE:
                return self._aggregate_by_highest_confidence(signals)
            elif method == AggregationMethod.MAJORITY_VOTE:
                return self._aggregate_by_majority_vote(signals, ticker, current_price)
            elif method == AggregationMethod.WEIGHTED_AVERAGE:
                return self._aggregate_by_weighted_average(signals, ticker, current_price)
            elif method == AggregationMethod.CONFIDENCE_WEIGHTED:
                return self._aggregate_by_confidence_weighted(signals, ticker, current_price)
            elif method == AggregationMethod.ENSEMBLE:
                return self._aggregate_by_ensemble(signals, ticker, current_price)
            else:
                logger.warning(f"不支持的聚合方法: {method}, 使用置信度加权")
                return self._aggregate_by_confidence_weighted(signals, ticker, current_price)

        except Exception as e:
            logger.error(f"信号聚合失败: {e}")
            return None

    def _aggregate_by_highest_confidence(self, signals: List[FuturesSignal]) -> FuturesSignal:
        """按最高置信度聚合"""
        return max(signals, key=lambda s: s.confidence)

    def _aggregate_by_majority_vote(self,
                                  signals: List[FuturesSignal],
                                  ticker: str,
                                  current_price: float) -> Optional[FuturesSignal]:
        """按多数投票聚合"""
        # 统计方向投票
        direction_votes = {TradingDirection.LONG: 0, TradingDirection.SHORT: 0, TradingDirection.NEUTRAL: 0}

        for signal in signals:
            direction_votes[signal.direction] += 1

        # 找到获票最多的方向
        winning_direction = max(direction_votes, key=direction_votes.get)

        if direction_votes[winning_direction] <= len(signals) // 2:
            # 没有明确多数，返回中性信号
            winning_direction = TradingDirection.NEUTRAL

        # 计算平均置信度
        avg_confidence = sum(s.confidence for s in signals) / len(signals)

        # 找到该方向的信号作为模板
        template_signal = next((s for s in signals if s.direction == winning_direction), signals[0])

        # 创建聚合信号
        return self._create_aggregated_signal(
            template_signal, winning_direction, avg_confidence, ticker, current_price, "majority_vote"
        )

    def _aggregate_by_confidence_weighted(self,
                                        signals: List[FuturesSignal],
                                        ticker: str,
                                        current_price: float) -> Optional[FuturesSignal]:
        """按置信度加权聚合"""
        total_weight = 0
        weighted_scores = {TradingDirection.LONG: 0, TradingDirection.SHORT: 0, TradingDirection.NEUTRAL: 0}

        for signal in signals:
            # 获取策略权重
            strategy_weight = self.strategy_weights.get(signal.strategy_source)
            if strategy_weight and strategy_weight.enabled:
                weight = strategy_weight.weight * strategy_weight.confidence_multiplier
            else:
                weight = 1.0

            # 置信度加权
            confidence_weight = signal.confidence / 100.0 * weight
            weighted_scores[signal.direction] += confidence_weight
            total_weight += weight

            # 详细日志记录每个信号的贡献
            logger.debug(
                f"  信号贡献: 策略={signal.strategy_source} 方向={signal.direction.value} "
                f"置信度={signal.confidence:.1f}% 策略权重={weight:.2f} "
                f"有效权重={confidence_weight:.3f}"
            )

        if total_weight == 0:
            return None

        # 找到得分最高的方向
        winning_direction = max(weighted_scores, key=weighted_scores.get)
        winning_score = weighted_scores[winning_direction] / total_weight

        # 输出各方向得分情况
        logger.debug(
            f"方向得分分布: LONG={weighted_scores[TradingDirection.LONG]/total_weight:.3f} "
            f"SHORT={weighted_scores[TradingDirection.SHORT]/total_weight:.3f} "
            f"NEUTRAL={weighted_scores[TradingDirection.NEUTRAL]/total_weight:.3f}"
        )

        # 计算最终置信度
        final_confidence = min(100.0, winning_score * 100)

        # 如果得分太低，设为中性（降低阈值提高敏感度）
        if final_confidence < 20:  # 从30降低到20，提高信号敏感度
            winning_direction = TradingDirection.NEUTRAL
            final_confidence = 50.0

        # 添加详细调试日志
        logger.debug(
            f"置信度加权聚合结果: 胜出方向={winning_direction.value} "
            f"胜出得分={winning_score:.3f} 最终置信度={final_confidence:.1f}% "
            f"总权重={total_weight:.3f} 信号数={len(signals)}"
        )

        # 选择最佳模板信号
        template_signal = max(signals, key=lambda s: s.confidence)

        return self._create_aggregated_signal(
            template_signal, winning_direction, final_confidence, ticker, current_price, "confidence_weighted"
        )

    def _aggregate_by_weighted_average(self,
                                     signals: List[FuturesSignal],
                                     ticker: str,
                                     current_price: float) -> Optional[FuturesSignal]:
        """按权重平均聚合"""
        total_weight = 0
        weighted_confidence = 0
        direction_weights = {TradingDirection.LONG: 0, TradingDirection.SHORT: 0, TradingDirection.NEUTRAL: 0}

        for signal in signals:
            # 获取策略权重
            strategy_weight = self.strategy_weights.get(signal.strategy_source)
            weight = strategy_weight.weight if strategy_weight and strategy_weight.enabled else 1.0

            direction_weights[signal.direction] += weight
            weighted_confidence += signal.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return None

        # 计算加权置信度
        final_confidence = weighted_confidence / total_weight

        # 选择权重最大的方向
        winning_direction = max(direction_weights, key=direction_weights.get)

        # 选择模板信号
        template_signal = max(signals, key=lambda s: s.confidence)

        return self._create_aggregated_signal(
            template_signal, winning_direction, final_confidence, ticker, current_price, "weighted_average"
        )

    def _aggregate_by_ensemble(self,
                             signals: List[FuturesSignal],
                             ticker: str,
                             current_price: float) -> Optional[FuturesSignal]:
        """集成方法聚合"""
        # 组合多种方法的结果
        methods = [
            AggregationMethod.CONFIDENCE_WEIGHTED,
            AggregationMethod.MAJORITY_VOTE,
            AggregationMethod.HIGHEST_CONFIDENCE
        ]

        ensemble_signals = []
        for method in methods:
            try:
                result = self._aggregate_signals(signals, ticker, current_price, method)
                if result:
                    ensemble_signals.append(result)
            except:
                continue

        if not ensemble_signals:
            return None

        # 对集成结果再次进行置信度加权聚合
        return self._aggregate_by_confidence_weighted(ensemble_signals, ticker, current_price)

    def _create_aggregated_signal(self,
                                template_signal: FuturesSignal,
                                direction: TradingDirection,
                                confidence: float,
                                ticker: str,
                                current_price: float,
                                method: str) -> FuturesSignal:
        """创建聚合信号"""
        return FuturesSignal(
            ticker=ticker,
            direction=direction,
            operation_type=OperationType.OPEN if direction != TradingDirection.NEUTRAL else OperationType.HOLD,
            confidence=confidence,
            strength=template_signal.strength,
            suggested_leverage=template_signal.suggested_leverage,
            position_size=template_signal.position_size,
            entry_price=current_price,
            current_price=current_price,
            take_profit_price=template_signal.take_profit_price,
            stop_loss_price=template_signal.stop_loss_price,
            take_profit_ratio=template_signal.take_profit_ratio,
            stop_loss_ratio=template_signal.stop_loss_ratio,
            expiry_time=template_signal.expiry_time,
            risk_level=template_signal.risk_level,
            strategy_source=f"aggregated_{method}",
            signal_id=f"agg_{ticker}_{int(datetime.now().timestamp())}",
            metadata={
                "aggregation_method": method,
                "contributing_strategies": len([s for s in self.strategies.values() if s.is_enabled]),
                "original_template": template_signal.strategy_source
            }
        )

    def _validate_aggregated_signal(self,
                                  signal: Optional[FuturesSignal],
                                  level: ValidationLevel) -> List[ValidationResult]:
        """验证聚合信号"""
        validation_results = []

        if not signal:
            validation_results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field_name="signal",
                message="聚合信号为空"
            ))
            return validation_results

        # 基础验证
        if level in [ValidationLevel.BASIC, ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            validation_results.extend(self._basic_signal_validation(signal))

        # 标准验证
        if level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            validation_results.extend(self._standard_signal_validation(signal))

        # 严格验证
        if level == ValidationLevel.STRICT:
            validation_results.extend(self._strict_signal_validation(signal))

        return validation_results

    def _basic_signal_validation(self, signal: FuturesSignal) -> List[ValidationResult]:
        """基础信号验证"""
        results = []

        # 检查信号基本属性
        if not signal.ticker:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field_name="ticker",
                message="交易对不能为空"
            ))

        if not 0 <= signal.confidence <= 100:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field_name="confidence",
                message=f"置信度必须在0-100范围内: {signal.confidence}"
            ))

        if signal.suggested_leverage <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field_name="leverage",
                message=f"杠杆必须大于0: {signal.suggested_leverage}"
            ))

        return results

    def _standard_signal_validation(self, signal: FuturesSignal) -> List[ValidationResult]:
        """标准信号验证"""
        results = []

        # 检查价格合理性
        if signal.entry_price and signal.current_price:
            price_diff = abs(signal.entry_price - signal.current_price) / signal.current_price
            if price_diff > 0.05:  # 价格差异超过5%
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="price_consistency",
                    message=f"入场价格与当前价格差异过大: {price_diff:.2%}"
                ))

        # 检查止损止盈合理性
        if signal.take_profit_price and signal.stop_loss_price and signal.entry_price:
            risk_reward = signal.calculate_risk_reward_ratio()
            if risk_reward and risk_reward < 1.0:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    field_name="risk_reward_ratio",
                    message=f"风险收益比过低: {risk_reward:.2f}"
                ))

        return results

    def _strict_signal_validation(self, signal: FuturesSignal) -> List[ValidationResult]:
        """严格信号验证"""
        results = []

        # 检查杠杆限制
        if signal.suggested_leverage > 20:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                field_name="leverage_limit",
                message=f"杠杆过高，建议降低: {signal.suggested_leverage}"
            ))

        # 检查仓位大小合理性
        if signal.position_size and signal.position_size > 10000:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                field_name="position_size",
                message=f"仓位大小可能过大: {signal.position_size}"
            ))

        return results

    def _update_performance_stats(self, success: bool, processing_time: timedelta):
        """更新性能统计"""
        try:
            self.performance_stats["total_analyses"] += 1

            if success:
                self.performance_stats["successful_analyses"] += 1
            else:
                self.performance_stats["failed_analyses"] += 1

            # 更新平均处理时间
            current_avg = self.performance_stats["avg_processing_time"]
            total = self.performance_stats["total_analyses"]
            new_time = processing_time.total_seconds()

            self.performance_stats["avg_processing_time"] = (current_avg * (total - 1) + new_time) / total

            self.last_analysis_time = datetime.now()

        except Exception as e:
            logger.error(f"更新性能统计失败: {e}")

    def get_manager_status(self) -> Dict[str, Any]:
        """
        获取管理器状态

        Returns:
            状态信息字典
        """
        active_strategies = self.get_active_strategies()

        return {
            "total_strategies": len(self.strategies),
            "active_strategies": len(active_strategies),
            "active_strategy_names": active_strategies,
            "is_running": self.is_running,
            "last_analysis_time": self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            "performance_stats": self.performance_stats,
            "strategy_weights": {name: weight.__dict__ for name, weight in self.strategy_weights.items()},
            "default_aggregation_method": self.default_aggregation_method.value,
            "max_workers": self.max_workers
        }

    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "avg_processing_time": 0.0,
            "signal_distribution": {}
        }
        logger.info("重置策略管理器性能统计")

    def shutdown(self):
        """关闭管理器"""
        try:
            self.is_running = False
            self.executor.shutdown(wait=True)
            logger.info("期货策略管理器已关闭")
        except Exception as e:
            logger.error(f"关闭策略管理器失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        self.is_running = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()

    def __str__(self) -> str:
        """字符串表示"""
        return f"FuturesStrategyManager(strategies={len(self.strategies)}, active={len(self.get_active_strategies())})"

    def __repr__(self) -> str:
        """详细字符串表示"""
        return (f"FuturesStrategyManager(strategies={list(self.strategies.keys())}, "
                f"active={self.get_active_strategies()}, method={self.default_aggregation_method.value})")


# 创建全局策略管理器实例
strategy_manager = FuturesStrategyManager()