"""
智能杠杆控制器

实现多维度动态杠杆计算，包括：
- 基于市场波动率的自适应调整
- 考虑账户风险等级的分层限制
- 交易对流动性的差异化管理
- 实时市场条件的动态响应
"""
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import asyncio
from dataclasses import dataclass
import math

# 导入项目模块
from ..models.data_models import RiskLevel, MarginStatus, Position
from ..market.market_analyzer import MarketAnalyzer
from .leverage_models import (
    LeverageConfig,
    LeverageCalculationResult,
    LeverageAdjustmentFactor,
    LeverageLimits,
    MarketCondition,
    LeverageStrategy,
    MarketRegime,
    LiquidityLevel
)
from ...utils.exceptions import (
    LeverageExceedsLimitError,
    MarginInsufficientError,
    ContractTradingError
)

logger = logging.getLogger(__name__)


@dataclass
class PortfolioRiskMetrics:
    """投资组合风险指标"""
    total_exposure: float = 0.0                    # 总敞口
    correlation_risk: float = 0.0                  # 相关性风险
    concentration_risk: float = 0.0                # 集中度风险
    liquidity_risk: float = 0.0                    # 流动性风险
    var_estimate: float = 0.0                      # VaR估计
    sharpe_ratio: Optional[float] = None           # 夏普比率


class LeverageController:
    """
    智能杠杆控制器

    提供多维度动态杠杆计算功能，包括波动率、风险等级、
    流动性等因子的综合分析，实现安全智能的杠杆控制。

    核心计算公式：
    optimal_leverage = base_leverage * volatility_multiplier *
                      liquidity_factor * market_adjustment *
                      position_adjustment * risk_adjustment
    """

    def __init__(
        self,
        config: Optional[LeverageConfig] = None,
        market_analyzer: Optional[MarketAnalyzer] = None
    ):
        """
        初始化杠杆控制器

        Args:
            config: 杠杆配置（可选，使用默认配置）
            market_analyzer: 市场分析器（可选）
        """
        self.config = config or LeverageConfig()
        self.market_analyzer = market_analyzer

        # 缓存数据
        self._market_conditions_cache: Dict[str, MarketCondition] = {}
        self._limits_cache: Dict[str, LeverageLimits] = {}
        self._calculation_cache: Dict[str, LeverageCalculationResult] = {}

        # 统计信息
        self._calculation_count = 0
        self._adjustment_stats = {
            "total_adjustments": 0,
            "emergency_reductions": 0,
            "volatility_adjustments": 0,
            "liquidity_adjustments": 0
        }

        logger.info("杠杆控制器已初始化")

    async def calculate_optimal_leverage(
        self,
        ticker: str,
        strategy: LeverageStrategy = LeverageStrategy.MODERATE,
        requested_leverage: Optional[float] = None,
        current_positions: Optional[List[Position]] = None,
        margin_status: Optional[MarginStatus] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> LeverageCalculationResult:
        """
        计算最优杠杆倍数

        Args:
            ticker: 交易对符号
            strategy: 杠杆策略
            requested_leverage: 请求的杠杆倍数（可选）
            current_positions: 当前持仓列表（可选）
            margin_status: 保证金状态（可选）
            market_data: 市场数据（可选）

        Returns:
            杠杆计算结果
        """
        self._calculation_count += 1

        try:
            # 初始化计算结果
            result = LeverageCalculationResult(
                ticker=ticker,
                calculated_leverage=0.0,
                applied_leverage=0.0,
                base_leverage=self.config.get_base_leverage(strategy),
                requested_leverage=requested_leverage
            )

            result.add_calculation_step(f"开始计算{ticker}的最优杠杆，策略: {strategy.value}")

            # 1. 获取基础杠杆
            base_leverage = self.config.get_base_leverage(strategy)
            result.add_calculation_step(f"基础杠杆: {base_leverage}")

            # 2. 获取市场条件
            market_condition = await self._get_market_condition(ticker, market_data)
            result.market_condition = market_condition

            # 3. 计算调整因子
            adjustment_factors = await self._calculate_adjustment_factors(
                ticker=ticker,
                market_condition=market_condition,
                current_positions=current_positions,
                margin_status=margin_status
            )
            result.adjustment_factors = adjustment_factors

            # 4. 计算初步杠杆
            calculated_leverage = base_leverage * adjustment_factors.get_combined_multiplier()
            result.calculated_leverage = calculated_leverage
            result.add_calculation_step(
                f"计算杠杆: {base_leverage} * {adjustment_factors.get_combined_multiplier():.4f} = {calculated_leverage:.2f}"
            )

            # 5. 应用限制
            limits = self._get_leverage_limits(ticker)
            result.limits = limits

            # 获取有效限制
            risk_level = self._assess_risk_level(margin_status, current_positions, market_condition)
            result.risk_level = risk_level

            effective_limit = limits.get_effective_limit(
                risk_level=risk_level,
                strategy=strategy,
                market_regime=market_condition.market_regime if market_condition else MarketRegime.CALM
            )
            result.effective_limit = effective_limit

            # 6. 应用最终杠杆
            final_leverage = min(calculated_leverage, effective_limit)
            final_leverage = max(final_leverage, limits.min_leverage)

            # 如果请求了特定杠杆，比较并决定
            if requested_leverage is not None:
                if requested_leverage <= effective_limit:
                    final_leverage = requested_leverage
                    result.add_calculation_step(f"应用请求杠杆: {requested_leverage}")
                else:
                    result.add_warning(
                        f"请求杠杆{requested_leverage}超过限制{effective_limit}，已调整"
                    )

            result.applied_leverage = final_leverage
            result.add_calculation_step(f"最终杠杆: {final_leverage}")

            # 7. 生成建议
            self._generate_recommendations(result)

            # 8. 验证结果
            self._validate_leverage_result(result)

            # 9. 设置有效期
            result.valid_until = datetime.now() + timedelta(minutes=5)

            # 缓存结果
            cache_key = f"{ticker}_{strategy.value}_{datetime.now().strftime('%Y%m%d%H%M')}"
            self._calculation_cache[cache_key] = result

            logger.info(
                f"杠杆计算完成: {ticker} 策略={strategy.value} "
                f"基础={base_leverage} 计算={calculated_leverage:.2f} "
                f"最终={final_leverage} 调整倍数={adjustment_factors.get_combined_multiplier():.4f}"
            )

            return result

        except Exception as e:
            error_msg = f"杠杆计算失败: {str(e)}"
            logger.error(error_msg)

            # 创建错误结果
            error_result = LeverageCalculationResult(
                ticker=ticker,
                calculated_leverage=1.0,  # 默认最小杠杆
                applied_leverage=1.0,
                base_leverage=self.config.get_base_leverage(strategy),
                requested_leverage=requested_leverage
            )
            error_result.add_error(error_msg)

            return error_result

    def calculate_leverage_sync(
        self,
        ticker: str,
        strategy: LeverageStrategy = LeverageStrategy.MODERATE,
        requested_leverage: Optional[float] = None,
        current_positions: Optional[List[Position]] = None,
        margin_status: Optional[MarginStatus] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> LeverageCalculationResult:
        """
        同步计算最优杠杆倍数（包装异步方法）

        Args:
            ticker: 交易对符号
            strategy: 杠杆策略
            requested_leverage: 请求的杠杆倍数（可选）
            current_positions: 当前持仓列表（可选）
            margin_status: 保证金状态（可选）
            market_data: 市场数据（可选）

        Returns:
            杠杆计算结果
        """
        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果循环正在运行，创建新任务
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self.calculate_optimal_leverage(
                                ticker, strategy, requested_leverage,
                                current_positions, margin_status, market_data
                            )
                        )
                        return future.result()
                else:
                    return loop.run_until_complete(
                        self.calculate_optimal_leverage(
                            ticker, strategy, requested_leverage,
                            current_positions, margin_status, market_data
                        )
                    )
            except RuntimeError:
                # 没有事件循环，创建新的
                return asyncio.run(
                    self.calculate_optimal_leverage(
                        ticker, strategy, requested_leverage,
                        current_positions, margin_status, market_data
                    )
                )
        except Exception as e:
            logger.error(f"同步杠杆计算失败: {e}")
            # 返回错误结果
            error_result = LeverageCalculationResult(
                ticker=ticker,
                calculated_leverage=1.0,
                applied_leverage=1.0,
                base_leverage=self.config.get_base_leverage(strategy)
            )
            error_result.add_error(f"同步计算失败: {str(e)}")
            return error_result

    async def _get_market_condition(
        self,
        ticker: str,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Optional[MarketCondition]:
        """
        获取市场条件

        Args:
            ticker: 交易对
            market_data: 市场数据

        Returns:
            市场条件对象
        """
        # 检查缓存
        cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self._market_conditions_cache:
            cached = self._market_conditions_cache[cache_key]
            if cached.is_data_fresh(max_age_seconds=300):  # 5分钟有效
                return cached

        try:
            # 如果有市场分析器，使用专门的杠杆支持数据接口
            if self.market_analyzer:
                try:
                    # 使用专门为杠杆控制器设计的增强数据接口
                    leverage_data = await self.market_analyzer.get_leverage_support_data(ticker)
                    
                    if leverage_data:
                        # 从增强数据中提取市场条件
                        enhanced_volatility = leverage_data.get('enhanced_volatility', {})
                        volume_characteristics = leverage_data.get('volume_characteristics', {})
                        market_sentiment = leverage_data.get('market_sentiment', {})
                        
                        # 使用更丰富的数据构建市场条件
                        volatility = enhanced_volatility.get('current', 0.2)
                        volume_strength = volume_characteristics.get('strength', 0.5)
                        sentiment_score = market_sentiment.get('fear_greed_index', 50.0) / 100.0
                        
                        # 根据多维数据计算流动性评分
                        liquidity_score = min(max(volume_strength * sentiment_score * 1.2, 0.1), 1.0)
                        
                        # 确定市场状态
                        market_regime = self._determine_market_regime_from_enhanced_data(leverage_data)
                        liquidity_level = self._determine_liquidity_level_from_enhanced_data(leverage_data)
                        
                        market_condition = MarketCondition(
                            ticker=ticker,
                            volatility=volatility,
                            liquidity_score=liquidity_score,
                            market_regime=market_regime,
                            liquidity_level=liquidity_level,
                            atr_percentage=leverage_data.get('risk_adjustments', {}).get('max_drawdown', 0.1) * 100,
                            volume_ratio=volume_characteristics.get('strength', 1.0),
                            spread_bps=None  # 可以后续从其他数据源获取
                        )
                        
                        # 缓存结果
                        self._market_conditions_cache[cache_key] = market_condition
                        return market_condition
                        
                except Exception as analyzer_error:
                    logger.warning(f"使用增强市场分析器失败，回退到基础分析: {analyzer_error}")
                    
                    # 回退：尝试使用基础的市场条件分析
                    try:
                        analysis = await self.market_analyzer.analyze_market_conditions(ticker)
                        if analysis:
                            market_condition = MarketCondition(
                                ticker=ticker,
                                volatility=analysis.volatility_metrics.historical_volatility if analysis.volatility_metrics else 0.2,
                                liquidity_score=0.7,  # 默认值，可以从其他数据改进
                                market_regime=self._determine_market_regime_from_analysis(analysis),
                                liquidity_level=self._determine_liquidity_level_from_analysis(analysis)
                            )

                            # 缓存结果
                            self._market_conditions_cache[cache_key] = market_condition
                            return market_condition
                    except Exception as fallback_error:
                        logger.error(f"市场分析器回退方案也失败: {fallback_error}")

            # 如果提供了市场数据，使用它
            if market_data:
                market_condition = MarketCondition(
                    ticker=ticker,
                    volatility=market_data.get('volatility', 0.2),
                    liquidity_score=market_data.get('liquidity_score', 0.7),
                    market_regime=MarketRegime(market_data.get('market_regime', 'calm')),
                    liquidity_level=LiquidityLevel(market_data.get('liquidity_level', 'medium'))
                )

                self._market_conditions_cache[cache_key] = market_condition
                return market_condition

            # 默认市场条件
            default_condition = MarketCondition(
                ticker=ticker,
                volatility=0.2,  # 默认波动率
                liquidity_score=0.7,  # 默认流动性
                market_regime=MarketRegime.CALM,
                liquidity_level=LiquidityLevel.MEDIUM
            )

            logger.warning(f"使用默认市场条件: {ticker}")
            return default_condition

        except Exception as e:
            logger.error(f"获取市场条件失败: {e}")
            return None

    async def _calculate_adjustment_factors(
        self,
        ticker: str,
        market_condition: Optional[MarketCondition],
        current_positions: Optional[List[Position]] = None,
        margin_status: Optional[MarginStatus] = None
    ) -> LeverageAdjustmentFactor:
        """
        计算调整因子

        Args:
            ticker: 交易对
            market_condition: 市场条件
            current_positions: 当前持仓
            margin_status: 保证金状态

        Returns:
            杠杆调整因子
        """
        factors = LeverageAdjustmentFactor()

        try:
            # 1. 波动率调整
            if market_condition:
                factors.volatility_multiplier = market_condition.get_volatility_multiplier()

                # 流动性调整
                factors.liquidity_multiplier = market_condition.get_liquidity_multiplier()

                # 市场状态调整
                factors.market_adjustment = self._get_market_regime_adjustment(
                    market_condition.market_regime
                )

            # 2. 仓位调整
            if current_positions:
                factors.position_adjustment = self._calculate_position_adjustment(
                    ticker, current_positions
                )

            # 3. 风险调整
            if margin_status:
                factors.risk_adjustment = self._calculate_risk_adjustment(margin_status)

            # 4. 时间调整（基于交易时段）
            factors.time_adjustment = self._calculate_time_adjustment()

            # 5. 特殊调整因子
            factors.news_impact_factor = await self._calculate_news_impact(ticker)
            factors.correlation_factor = self._calculate_correlation_factor(
                ticker, current_positions
            )
            factors.momentum_factor = self._calculate_momentum_factor(market_condition)

            # 更新统计
            combined = factors.get_combined_multiplier()
            if combined < 0.8:
                self._adjustment_stats["volatility_adjustments"] += 1
            if factors.liquidity_multiplier < 0.9:
                self._adjustment_stats["liquidity_adjustments"] += 1
            if combined < 0.5:
                self._adjustment_stats["emergency_reductions"] += 1

            self._adjustment_stats["total_adjustments"] += 1

        except Exception as e:
            logger.error(f"计算调整因子失败: {e}")
            # 返回保守的调整因子
            factors = LeverageAdjustmentFactor(
                volatility_multiplier=0.8,
                liquidity_multiplier=0.8,
                market_adjustment=0.8,
                position_adjustment=0.9,
                risk_adjustment=0.9
            )

        return factors

    def _get_market_regime_adjustment(self, regime: MarketRegime) -> float:
        """获取市场状态调整因子"""
        regime_adjustments = {
            MarketRegime.CALM: 1.0,
            MarketRegime.VOLATILE: 0.8,
            MarketRegime.TRENDING: 1.1,
            MarketRegime.RANGING: 0.9,
            MarketRegime.HIGH_VOLATILITY: 0.6
        }
        return regime_adjustments.get(regime, 0.8)

    def _calculate_position_adjustment(
        self,
        ticker: str,
        positions: List[Position]
    ) -> float:
        """计算仓位调整因子"""
        try:
            # 计算同一交易对的仓位集中度
            same_symbol_exposure = sum(
                pos.notional_value or 0 for pos in positions
                if pos.ticker == ticker
            )

            # 计算总敞口
            total_exposure = sum(pos.notional_value or 0 for pos in positions)

            if total_exposure == 0:
                return 1.0

            concentration = same_symbol_exposure / total_exposure

            # 基于集中度调整杠杆
            if concentration > 0.5:  # 超过50%集中度
                return 0.7
            elif concentration > 0.3:  # 超过30%集中度
                return 0.85
            else:
                return 1.0

        except Exception as e:
            logger.error(f"计算仓位调整失败: {e}")
            return 0.9

    def _calculate_risk_adjustment(self, margin_status: MarginStatus) -> float:
        """计算风险调整因子"""
        try:
            # 基于保证金率调整
            if margin_status.risk_level == RiskLevel.EMERGENCY:
                return 0.3
            elif margin_status.risk_level == RiskLevel.CRITICAL:
                return 0.5
            elif margin_status.risk_level == RiskLevel.HIGH:
                return 0.7
            elif margin_status.risk_level == RiskLevel.MEDIUM:
                return 0.9
            else:
                return 1.0

        except Exception as e:
            logger.error(f"计算风险调整失败: {e}")
            return 0.8

    def _calculate_time_adjustment(self) -> float:
        """计算时间调整因子"""
        try:
            current_hour = datetime.now().hour

            # 基于UTC时间的主要交易时段调整
            if 0 <= current_hour < 6:  # 亚洲时段
                return 0.95
            elif 6 <= current_hour < 14:  # 欧洲时段
                return 1.0
            elif 14 <= current_hour < 22:  # 美国时段
                return 1.0
            else:  # 低流动性时段
                return 0.9

        except Exception:
            return 1.0

    async def _calculate_news_impact(self, ticker: str) -> float:
        """计算新闻影响因子"""
        # TODO: 实现新闻情感分析
        # 当前返回中性影响
        return 1.0

    def _calculate_correlation_factor(
        self,
        ticker: str,
        positions: Optional[List[Position]]
    ) -> float:
        """计算相关性调整因子"""
        if not positions or not self.config.enable_correlation_checks:
            return 1.0

        try:
            # 简化的相关性检查
            # TODO: 实现更复杂的相关性计算
            same_base_currency_count = sum(
                1 for pos in positions
                if pos.ticker.startswith(ticker.split('USDT')[0])
            )

            if same_base_currency_count > 3:
                return 0.8  # 降低杠杆
            elif same_base_currency_count > 1:
                return 0.9
            else:
                return 1.0

        except Exception as e:
            logger.error(f"计算相关性因子失败: {e}")
            return 0.95

    def _calculate_momentum_factor(
        self,
        market_condition: Optional[MarketCondition]
    ) -> float:
        """计算动量调整因子"""
        if not market_condition:
            return 1.0

        try:
            # 基于市场状态调整动量因子
            if market_condition.market_regime == MarketRegime.TRENDING:
                return 1.1  # 趋势市场增加杠杆
            elif market_condition.market_regime == MarketRegime.RANGING:
                return 0.9  # 震荡市场降低杠杆
            else:
                return 1.0

        except Exception:
            return 1.0

    def _get_leverage_limits(self, ticker: str) -> LeverageLimits:
        """获取交易对杠杆限制"""
        # 检查缓存
        if ticker in self._limits_cache:
            return self._limits_cache[ticker]

        # 检查配置中的特定限制
        symbol_limits = self.config.get_symbol_limits(ticker)
        if symbol_limits:
            self._limits_cache[ticker] = symbol_limits
            return symbol_limits

        # 创建默认限制
        default_limits = self._create_default_limits(ticker)
        self._limits_cache[ticker] = default_limits

        return default_limits

    def _create_default_limits(self, ticker: str) -> LeverageLimits:
        """创建默认杠杆限制"""
        # 基于交易对类型设置不同的默认限制
        if ticker.endswith('USDT'):
            if ticker in ['BTCUSDT', 'ETHUSDT']:
                max_leverage = 125.0  # 主流币种
            elif ticker in ['ADAUSDT', 'DOTUSDT', 'LINKUSDT']:
                max_leverage = 75.0   # 二线币种
            else:
                max_leverage = 50.0   # 其他币种
        else:
            max_leverage = 20.0  # 非USDT交易对

        return LeverageLimits(ticker=ticker, max_leverage=max_leverage)

    def _assess_risk_level(
        self,
        margin_status: Optional[MarginStatus],
        positions: Optional[List[Position]],
        market_condition: Optional[MarketCondition]
    ) -> RiskLevel:
        """评估当前风险等级"""
        try:
            risk_scores = []

            # 保证金风险
            if margin_status:
                risk_scores.append(self._get_margin_risk_score(margin_status))

            # 仓位风险
            if positions:
                risk_scores.append(self._get_position_risk_score(positions))

            # 市场风险
            if market_condition:
                risk_scores.append(self._get_market_risk_score(market_condition))

            if not risk_scores:
                return RiskLevel.MEDIUM

            # 取最高风险等级
            max_risk_score = max(risk_scores)

            if max_risk_score >= 0.9:
                return RiskLevel.EMERGENCY
            elif max_risk_score >= 0.8:
                return RiskLevel.CRITICAL
            elif max_risk_score >= 0.6:
                return RiskLevel.HIGH
            elif max_risk_score >= 0.4:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW

        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return RiskLevel.MEDIUM

    def _get_margin_risk_score(self, margin_status: MarginStatus) -> float:
        """获取保证金风险评分"""
        return margin_status.margin_ratio

    def _get_position_risk_score(self, positions: List[Position]) -> float:
        """获取仓位风险评分"""
        try:
            # 计算平均距离强平距离
            liquidation_distances = [
                abs(pos.distance_to_liquidation or 100.0)
                for pos in positions
                if pos.distance_to_liquidation is not None
            ]

            if not liquidation_distances:
                return 0.2  # 默认低风险

            avg_distance = sum(liquidation_distances) / len(liquidation_distances)

            # 距离越近，风险越高
            if avg_distance <= 2.0:
                return 0.95
            elif avg_distance <= 5.0:
                return 0.8
            elif avg_distance <= 10.0:
                return 0.6
            elif avg_distance <= 20.0:
                return 0.4
            else:
                return 0.2

        except Exception as e:
            logger.error(f"计算仓位风险评分失败: {e}")
            return 0.3

    def _get_market_risk_score(self, market_condition: MarketCondition) -> float:
        """获取市场风险评分"""
        try:
            volatility_score = min(market_condition.volatility * 2, 1.0)
            liquidity_score = 1.0 - market_condition.liquidity_score

            return (volatility_score + liquidity_score) / 2

        except Exception:
            return 0.3

    def _determine_market_regime(self, analysis: Dict[str, Any]) -> MarketRegime:
        """确定市场状态"""
        try:
            volatility = analysis.get('volatility', 0.2)
            trend_strength = analysis.get('trend_strength', 0.5)

            if volatility > 0.4:
                return MarketRegime.HIGH_VOLATILITY
            elif volatility > 0.3:
                return MarketRegime.VOLATILE
            elif trend_strength > 0.7:
                return MarketRegime.TRENDING
            elif trend_strength < 0.3:
                return MarketRegime.RANGING
            else:
                return MarketRegime.CALM

        except Exception:
            return MarketRegime.CALM

    def _determine_liquidity_level(self, analysis: Dict[str, Any]) -> LiquidityLevel:
        """确定流动性等级"""
        try:
            liquidity_score = analysis.get('liquidity_score', 0.7)

            if liquidity_score >= 0.8:
                return LiquidityLevel.HIGH
            elif liquidity_score >= 0.6:
                return LiquidityLevel.MEDIUM
            elif liquidity_score >= 0.4:
                return LiquidityLevel.LOW
            else:
                return LiquidityLevel.VERY_LOW

        except Exception:
            return LiquidityLevel.MEDIUM

    def _determine_market_regime_from_enhanced_data(self, leverage_data: Dict[str, Any]) -> MarketRegime:
        """
        从增强市场数据中确定市场状态
        
        Args:
            leverage_data: 增强的市场数据
            
        Returns:
            MarketRegime: 市场状态枚举
        """
        try:
            enhanced_volatility = leverage_data.get('enhanced_volatility', {})
            market_sentiment = leverage_data.get('market_sentiment', {})
            volume_characteristics = leverage_data.get('volume_characteristics', {})
            
            # 获取关键指标
            volatility = enhanced_volatility.get('current', 0.2)
            volatility_regime = enhanced_volatility.get('regime', 'normal')
            fear_greed_index = market_sentiment.get('fear_greed_index', 50.0)
            volume_strength = volume_characteristics.get('strength', 0.5)
            abnormal_volume = volume_characteristics.get('abnormal_detected', False)
            
            # 基于多维数据判断市场状态
            if volatility > 0.4 or volatility_regime == 'high':
                if fear_greed_index < 25:  # 极端恐惧
                    return MarketRegime.CRISIS
                elif abnormal_volume:
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.STRESSED
            elif volatility > 0.25 or volatility_regime == 'medium':
                if volume_strength > 0.7:
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.NORMAL
            else:
                if fear_greed_index > 75 and volume_strength < 0.3:  # 极端贪婪且成交量低
                    return MarketRegime.COMPLACENT
                else:
                    return MarketRegime.CALM
                    
        except Exception as e:
            logger.error(f"从增强数据确定市场状态失败: {e}")
            return MarketRegime.NORMAL
    
    def _determine_liquidity_level_from_enhanced_data(self, leverage_data: Dict[str, Any]) -> LiquidityLevel:
        """
        从增强市场数据中确定流动性水平
        
        Args:
            leverage_data: 增强的市场数据
            
        Returns:
            LiquidityLevel: 流动性水平枚举
        """
        try:
            volume_characteristics = leverage_data.get('volume_characteristics', {})
            market_sentiment = leverage_data.get('market_sentiment', {})
            
            # 获取关键指标
            volume_strength = volume_characteristics.get('strength', 0.5)
            volume_trend = volume_characteristics.get('trend', 'stable')
            breakout_signal = volume_characteristics.get('breakout_signal', False)
            market_stress = market_sentiment.get('stress_indicators', {})
            
            # 计算综合流动性评分
            liquidity_score = volume_strength
            
            # 成交量趋势调整
            if volume_trend == 'increasing':
                liquidity_score *= 1.2
            elif volume_trend == 'decreasing':
                liquidity_score *= 0.8
            
            # 突破信号调整
            if breakout_signal:
                liquidity_score *= 1.1
            
            # 市场压力调整
            if market_stress:
                stress_level = sum(market_stress.values()) / max(len(market_stress), 1)
                liquidity_score *= (1.0 - stress_level * 0.3)
            
            # 确定流动性水平
            if liquidity_score > 0.8:
                return LiquidityLevel.HIGH
            elif liquidity_score > 0.6:
                return LiquidityLevel.MEDIUM
            elif liquidity_score > 0.3:
                return LiquidityLevel.LOW
            else:
                return LiquidityLevel.VERY_LOW
                
        except Exception as e:
            logger.error(f"从增强数据确定流动性水平失败: {e}")
            return LiquidityLevel.MEDIUM
    
    def _determine_market_regime_from_analysis(self, analysis) -> MarketRegime:
        """
        从基础市场分析中确定市场状态（回退方案）
        
        Args:
            analysis: 基础市场分析结果
            
        Returns:
            MarketRegime: 市场状态枚举
        """
        try:
            if hasattr(analysis, 'primary_condition'):
                condition_str = str(analysis.primary_condition).lower()
                if 'crisis' in condition_str or 'crash' in condition_str:
                    return MarketRegime.CRISIS
                elif 'volatile' in condition_str or 'unstable' in condition_str:
                    return MarketRegime.VOLATILE
                elif 'stressed' in condition_str or 'tension' in condition_str:
                    return MarketRegime.STRESSED
                elif 'bullish' in condition_str or 'trending_up' in condition_str:
                    return MarketRegime.TRENDING
                elif 'calm' in condition_str or 'stable' in condition_str:
                    return MarketRegime.CALM
                else:
                    return MarketRegime.NORMAL
            else:
                return MarketRegime.NORMAL
        except Exception as e:
            logger.error(f"从基础分析确定市场状态失败: {e}")
            return MarketRegime.NORMAL
    
    def _determine_liquidity_level_from_analysis(self, analysis) -> LiquidityLevel:
        """
        从基础市场分析中确定流动性水平（回退方案）
        
        Args:
            analysis: 基础市场分析结果
            
        Returns:
            LiquidityLevel: 流动性水平枚举
        """
        try:
            if hasattr(analysis, 'liquidity_analysis'):
                liquidity = analysis.liquidity_analysis.liquidity_level
                if hasattr(liquidity, 'value'):
                    return LiquidityLevel(liquidity.value)
                else:
                    return LiquidityLevel(str(liquidity).lower())
            else:
                return LiquidityLevel.MEDIUM
        except Exception as e:
            logger.error(f"从基础分析确定流动性水平失败: {e}")
            return LiquidityLevel.MEDIUM

    def _generate_recommendations(self, result: LeverageCalculationResult):
        """生成杠杆建议"""
        try:
            recommendations = []

            # 基于调整幅度的建议
            adjustment_pct = result.get_adjustment_percentage()

            if adjustment_pct < -30:
                recommendations.append("市场风险较高，建议降低仓位规模")
            elif adjustment_pct > 20:
                recommendations.append("市场条件良好，可考虑适度增加仓位")

            # 基于风险等级的建议
            if result.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
                recommendations.append("账户风险极高，建议立即减仓或平仓")
            elif result.risk_level == RiskLevel.HIGH:
                recommendations.append("账户风险偏高，建议谨慎操作")

            # 基于市场条件的建议
            if result.market_condition:
                if result.market_condition.volatility > 0.3:
                    recommendations.append("市场波动性较高，建议设置更严格的止损")
                if result.market_condition.liquidity_level == LiquidityLevel.LOW:
                    recommendations.append("流动性较差，建议避免大额交易")

            result.recommendation = "; ".join(recommendations) if recommendations else "当前参数合理"

        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            result.recommendation = "建议保持谨慎"

    def _validate_leverage_result(self, result: LeverageCalculationResult):
        """验证杠杆计算结果"""
        try:
            # 检查杠杆范围
            if result.applied_leverage <= 0:
                result.add_error("杠杆倍数必须大于0")
            elif result.applied_leverage > 200:
                result.add_warning("杠杆倍数异常高，请检查计算参数")

            # 检查调整幅度
            if result.is_adjustment_significant(threshold=0.5):
                result.add_warning("杠杆调整幅度较大，建议仔细评估")

            # 检查风险等级
            if result.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
                result.add_warning("当前风险等级较高，建议降低杠杆")

            # 设置置信度
            if result.has_errors():
                result.confidence_score = 0.0
            elif result.has_warnings():
                result.confidence_score = 0.7
            else:
                result.confidence_score = 0.9

        except Exception as e:
            logger.error(f"验证结果失败: {e}")
            result.add_error(f"结果验证失败: {str(e)}")

    def get_leverage_statistics(self) -> Dict[str, Any]:
        """获取杠杆控制统计信息"""
        return {
            "calculation_count": self._calculation_count,
            "adjustment_stats": self._adjustment_stats.copy(),
            "cache_size": {
                "market_conditions": len(self._market_conditions_cache),
                "limits": len(self._limits_cache),
                "calculations": len(self._calculation_cache)
            }
        }

    def clear_cache(self):
        """清空缓存"""
        self._market_conditions_cache.clear()
        self._limits_cache.clear()
        self._calculation_cache.clear()
        logger.info("杠杆控制器缓存已清空")

    def update_config(self, new_config: LeverageConfig):
        """更新配置"""
        self.config = new_config
        self.clear_cache()  # 清空缓存以应用新配置
        logger.info("杠杆控制器配置已更新")

    def add_symbol_limits(self, ticker: str, limits: LeverageLimits):
        """添加交易对特定限制"""
        self.config.add_symbol_limits(ticker, limits)
        # 更新缓存
        self._limits_cache[ticker] = limits
        logger.info(f"已添加{ticker}的杠杆限制")


# 导出主要类
__all__ = ['LeverageController', 'PortfolioRiskMetrics']