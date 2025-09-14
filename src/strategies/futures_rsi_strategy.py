from typing import Dict, Any
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode
from src.graph.utils import show_agent_reasoning
from indicators import (calculate_trend_signals,
                        calculate_mean_reversion_signals,
                        calculate_momentum_signals,
                        calculate_volatility_signals,
                        calculate_stat_arb_signals,
                        weighted_signal_combination,
                        normalize_pandas)

# 安全的统计计算函数
def safe_std(arr, ddof=0):
    """安全的标准差计算，处理NaN值"""
    if len(arr) == 0:
        return 0.0
    clean_arr = np.array(arr)[~np.isnan(arr)]
    if len(clean_arr) <= ddof:
        return 0.0
    return np.std(clean_arr, ddof=ddof)

def safe_mean(arr):
    """安全的均值计算，处理NaN值和空数组"""
    if len(arr) == 0:
        return 0.0
    clean_arr = np.array(arr)[~np.isnan(arr)]
    if len(clean_arr) == 0:
        return 0.0
    return np.mean(clean_arr)


class FuturesRSIStrategy(BaseNode):
    """
    期货版本的RSI策略，专为期货交易设计
    - 支持long/short/neutral信号输出语义
    - RSI阈值动态调整，适配期货高杠杆环境
    - 基于波动率的智能阈值调整
    - 增强的双向交易支持
    - 期货特有的风险管理指标
    """

    def __init__(self):
        """初始化期货RSI策略"""
        super().__init__()

        # 期货特有配置
        self.futures_config = {
            "dynamic_thresholds": True,     # 启用动态阈值调整
            "bidirectional_signals": True,  # 启用双向信号
            "leverage_sensitivity": True,   # 启用杠杆敏感性分析
            "enhanced_risk_metrics": True,  # 启用增强风险指标
            "signal_strength_multiplier": 1.15,  # 期货信号强度倍数
        }

        # 默认RSI阈值 (适用于标准波动率)
        self.default_thresholds = {
            "oversold": 30,
            "overbought": 70,
            "extreme_oversold": 20,
            "extreme_overbought": 80
        }

    def _convert_signal_to_futures(self, signal: str) -> str:
        """
        将现货交易信号转换为期货交易信号

        Args:
            signal: 原始信号 (buy/sell/hold)

        Returns:
            str: 期货信号 (long/short/neutral)
        """
        signal_mapping = {
            "buy": "long",
            "bullish": "long",
            "sell": "short",
            "bearish": "short",
            "hold": "neutral",
            "neutral": "neutral"
        }
        return signal_mapping.get(signal, "neutral")

    def _calculate_dynamic_rsi_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        根据市场波动率动态调整RSI阈值

        Args:
            df: 价格数据

        Returns:
            dict: 动态调整后的RSI阈值
        """
        try:
            # 计算波动率指标
            volatility_data = self._calculate_volatility_metrics(df)

            # 基于波动率的阈值调整
            volatility_percentile = volatility_data.get('volatility_percentile', 50)

            # 基础阈值
            base_oversold = self.default_thresholds["oversold"]
            base_overbought = self.default_thresholds["overbought"]

            # 根据波动率调整阈值
            if volatility_percentile >= 80:  # 高波动率
                # 扩展阈值范围，减少假信号
                oversold_threshold = max(20, base_oversold - 8)
                overbought_threshold = min(80, base_overbought + 8)
            elif volatility_percentile >= 60:  # 中高波动率
                oversold_threshold = max(22, base_oversold - 5)
                overbought_threshold = min(78, base_overbought + 5)
            elif volatility_percentile <= 20:  # 低波动率
                # 收紧阈值范围，提高灵敏度
                oversold_threshold = min(38, base_oversold + 8)
                overbought_threshold = max(62, base_overbought - 8)
            elif volatility_percentile <= 40:  # 中低波动率
                oversold_threshold = min(35, base_oversold + 5)
                overbought_threshold = max(65, base_overbought - 5)
            else:  # 标准波动率
                oversold_threshold = base_oversold
                overbought_threshold = base_overbought

            return {
                "oversold": oversold_threshold,
                "overbought": overbought_threshold,
                "extreme_oversold": max(15, oversold_threshold - 10),
                "extreme_overbought": min(85, overbought_threshold + 10),
                "volatility_adjustment": volatility_percentile
            }

        except Exception as e:
            print(f"Dynamic RSI threshold calculation error: {e}")
            return self.default_thresholds.copy()

    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        计算波动率相关指标

        Args:
            df: 价格数据

        Returns:
            dict: 波动率指标
        """
        try:
            if df.empty or len(df) < 20:
                return {'volatility_percentile': 50.0, 'current_volatility': 0.0}

            # 获取收盘价
            close_prices = df['close'] if 'close' in df.columns else df['Close']

            # 计算不同周期的波动率
            returns = close_prices.pct_change().dropna()

            # 当前波动率 (14天)
            current_volatility = returns.rolling(window=14).std().iloc[-1] * np.sqrt(252)

            # 历史波动率分布 (60天窗口)
            historical_vol = returns.rolling(window=14).std() * np.sqrt(252)
            historical_vol = historical_vol.dropna()

            if len(historical_vol) > 1:
                # 计算当前波动率的百分位数
                volatility_percentile = (historical_vol <= current_volatility).mean() * 100
            else:
                volatility_percentile = 50.0

            return {
                'volatility_percentile': float(volatility_percentile),
                'current_volatility': float(current_volatility) if not np.isnan(current_volatility) else 0.0
            }

        except Exception as e:
            print(f"Volatility metrics calculation error: {e}")
            return {'volatility_percentile': 50.0, 'current_volatility': 0.0}

    def _calculate_futures_rsi_signals(self, df: pd.DataFrame,
                                     dynamic_thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        基于动态阈值计算期货RSI信号

        Args:
            df: 价格数据
            dynamic_thresholds: 动态阈值

        Returns:
            dict: RSI信号数据
        """
        try:
            if df.empty or len(df) < 14:
                return self._get_default_rsi_signal()

            # 计算RSI
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            rsi_14 = ta.rsi(close_prices, length=14)

            if rsi_14.empty or rsi_14.isna().all():
                return self._get_default_rsi_signal()

            current_rsi = rsi_14.iloc[-1]
            previous_rsi = rsi_14.iloc[-2] if len(rsi_14) > 1 else current_rsi

            # 获取动态阈值
            oversold = dynamic_thresholds.get("oversold", 30)
            overbought = dynamic_thresholds.get("overbought", 70)
            extreme_oversold = dynamic_thresholds.get("extreme_oversold", 20)
            extreme_overbought = dynamic_thresholds.get("extreme_overbought", 80)

            # 信号生成逻辑
            signal = "neutral"
            confidence = 50
            signal_strength = "weak"

            # RSI超卖信号 (做多机会)
            if current_rsi <= extreme_oversold:
                signal = "long"
                confidence = 85
                signal_strength = "strong"
            elif current_rsi <= oversold:
                signal = "long"
                confidence = 70
                signal_strength = "moderate"

            # RSI超买信号 (做空机会)
            elif current_rsi >= extreme_overbought:
                signal = "short"
                confidence = 85
                signal_strength = "strong"
            elif current_rsi >= overbought:
                signal = "short"
                confidence = 70
                signal_strength = "moderate"

            # RSI背离分析
            divergence_info = self._analyze_rsi_divergence(df, rsi_14)
            if divergence_info["has_divergence"]:
                # 背离增强信号置信度
                if divergence_info["divergence_type"] == "bullish" and signal == "long":
                    confidence = min(95, confidence + 15)
                    signal_strength = "strong"
                elif divergence_info["divergence_type"] == "bearish" and signal == "short":
                    confidence = min(95, confidence + 15)
                    signal_strength = "strong"

            # RSI趋势分析
            rsi_trend = "neutral"
            if len(rsi_14) >= 5:
                recent_rsi = rsi_14.tail(5)
                if recent_rsi.is_monotonic_increasing:
                    rsi_trend = "rising"
                elif recent_rsi.is_monotonic_decreasing:
                    rsi_trend = "falling"

            return {
                "signal": signal,
                "confidence": confidence,
                "signal_strength": signal_strength,
                "rsi_current": float(current_rsi),
                "rsi_previous": float(previous_rsi),
                "dynamic_thresholds": dynamic_thresholds,
                "rsi_trend": rsi_trend,
                "divergence_info": divergence_info,
                "metrics": {
                    "rsi_14": float(current_rsi),
                    "rsi_change": float(current_rsi - previous_rsi),
                    "oversold_threshold": oversold,
                    "overbought_threshold": overbought,
                    "threshold_distance": {
                        "to_oversold": float(current_rsi - oversold),
                        "to_overbought": float(overbought - current_rsi)
                    }
                }
            }

        except Exception as e:
            print(f"Futures RSI signal calculation error: {e}")
            return self._get_default_rsi_signal()

    def _analyze_rsi_divergence(self, df: pd.DataFrame, rsi_series: pd.Series) -> Dict[str, Any]:
        """
        分析RSI背离情况

        Args:
            df: 价格数据
            rsi_series: RSI序列

        Returns:
            dict: 背离分析结果
        """
        try:
            if len(df) < 20 or len(rsi_series) < 20:
                return {"has_divergence": False, "divergence_type": None}

            close_prices = df['close'] if 'close' in df.columns else df['Close']

            # 寻找最近的高点和低点
            price_highs = close_prices.rolling(window=5).max()
            price_lows = close_prices.rolling(window=5).min()
            rsi_highs = rsi_series.rolling(window=5).max()
            rsi_lows = rsi_series.rolling(window=5).min()

            # 检查最近20个周期的背离
            lookback = 20
            recent_period = min(lookback, len(df))

            price_recent = close_prices.tail(recent_period)
            rsi_recent = rsi_series.tail(recent_period)

            # 寻找价格和RSI的极值
            price_max_idx = price_recent.idxmax()
            price_min_idx = price_recent.idxmin()
            rsi_max_idx = rsi_recent.idxmax()
            rsi_min_idx = rsi_recent.idxmin()

            # 检查看涨背离 (价格创新低，RSI不创新低)
            if (price_recent.iloc[-1] <= price_recent.iloc[-5:-1].min() and
                rsi_recent.iloc[-1] > rsi_recent.iloc[-5:-1].min()):
                return {
                    "has_divergence": True,
                    "divergence_type": "bullish",
                    "strength": "moderate"
                }

            # 检查看跌背离 (价格创新高，RSI不创新高)
            if (price_recent.iloc[-1] >= price_recent.iloc[-5:-1].max() and
                rsi_recent.iloc[-1] < rsi_recent.iloc[-5:-1].max()):
                return {
                    "has_divergence": True,
                    "divergence_type": "bearish",
                    "strength": "moderate"
                }

            return {"has_divergence": False, "divergence_type": None}

        except Exception as e:
            print(f"RSI divergence analysis error: {e}")
            return {"has_divergence": False, "divergence_type": None}

    def _calculate_futures_signal_strength(self, signal_data: Dict[str, Any],
                                         df: pd.DataFrame) -> Dict[str, Any]:
        """
        计算期货特有的信号强度，考虑杠杆和双向交易特点

        Args:
            signal_data: 原始信号数据
            df: 价格数据

        Returns:
            dict: 增强的信号强度数据
        """
        try:
            # 基础信号强度
            base_confidence = signal_data.get("confidence", 50) / 100.0

            # 期货特有因素
            futures_factors = {
                "volatility_factor": self._calculate_volatility_factor(df),
                "momentum_factor": self._calculate_momentum_factor(df),
                "volume_factor": self._calculate_volume_factor(df),
                "leverage_sensitivity": self._calculate_leverage_sensitivity(df),
                "mean_reversion_bias": self._calculate_mean_reversion_bias(signal_data, df)
            }

            # 综合期货信号强度
            futures_multiplier = (
                futures_factors["volatility_factor"] * 0.3 +
                futures_factors["momentum_factor"] * 0.25 +
                futures_factors["volume_factor"] * 0.2 +
                futures_factors["leverage_sensitivity"] * 0.15 +
                futures_factors["mean_reversion_bias"] * 0.1
            )

            # 应用期货配置倍数
            enhanced_confidence = base_confidence * futures_multiplier * self.futures_config["signal_strength_multiplier"]
            enhanced_confidence = max(0.1, min(0.95, enhanced_confidence))

            return {
                "enhanced_confidence": enhanced_confidence * 100,
                "futures_factors": futures_factors,
                "futures_multiplier": futures_multiplier
            }

        except Exception as e:
            print(f"Futures signal strength calculation error: {e}")
            return {"enhanced_confidence": base_confidence * 100, "futures_factors": {}, "futures_multiplier": 1.0}

    def _calculate_volatility_factor(self, df: pd.DataFrame) -> float:
        """计算波动率因子"""
        try:
            volatility_data = self._calculate_volatility_metrics(df)
            volatility_percentile = volatility_data.get('volatility_percentile', 50)

            # 期货交易中，适度的波动率有利于RSI策略
            if 30 <= volatility_percentile <= 70:
                return 1.2  # 最佳波动率范围
            elif volatility_percentile > 80:
                return 0.8  # 过高波动率，降低信号可靠性
            else:
                return 1.0  # 标准权重
        except:
            return 1.0

    def _calculate_momentum_factor(self, df: pd.DataFrame) -> float:
        """计算动量因子"""
        try:
            if len(df) < 10:
                return 1.0

            close_prices = df['close'] if 'close' in df.columns else df['Close']

            # 计算短期和中期动量
            short_momentum = (close_prices.iloc[-1] / close_prices.iloc[-5] - 1) * 100
            medium_momentum = (close_prices.iloc[-1] / close_prices.iloc[-10] - 1) * 100

            # RSI是均值回归策略，强动量可能影响其效果
            avg_momentum = abs((short_momentum + medium_momentum) / 2)

            if avg_momentum > 5:  # 强动量环境
                return 0.8
            elif avg_momentum < 1:  # 低动量环境，适合RSI
                return 1.3
            else:
                return 1.0
        except:
            return 1.0

    def _calculate_volume_factor(self, df: pd.DataFrame) -> float:
        """计算成交量因子"""
        try:
            if 'volume' not in df.columns and 'Volume' not in df.columns:
                return 1.0

            volume = df['volume'] if 'volume' in df.columns else df['Volume']
            if len(volume) < 10:
                return 1.0

            # 计算相对成交量
            avg_volume = volume.rolling(window=10).mean().iloc[-1]
            current_volume = volume.iloc[-1]

            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # 高成交量增强信号可靠性
            if volume_ratio > 1.5:
                return 1.2
            elif volume_ratio < 0.5:
                return 0.9
            else:
                return 1.0
        except:
            return 1.0

    def _calculate_leverage_sensitivity(self, df: pd.DataFrame) -> float:
        """计算杠杆敏感性因子"""
        try:
            # 期货杠杆环境下，需要考虑价格变动的放大效应
            volatility_data = self._calculate_volatility_metrics(df)
            current_volatility = volatility_data.get('current_volatility', 0)

            # 高波动率环境下，杠杆风险增加
            if current_volatility > 0.4:  # 40%年化波动率
                return 0.7  # 降低信号权重
            elif current_volatility > 0.25:  # 25%年化波动率
                return 0.85
            else:
                return 1.1  # 低波动率环境，可适度增加权重
        except:
            return 1.0

    def _calculate_mean_reversion_bias(self, signal_data: Dict[str, Any], df: pd.DataFrame) -> float:
        """计算均值回归偏向因子"""
        try:
            # RSI本质上是均值回归策略
            current_rsi = signal_data.get("rsi_current", 50)

            # RSI距离极值越远，均值回归概率越高
            distance_from_extreme = min(current_rsi - 20, 80 - current_rsi)

            if distance_from_extreme > 30:  # RSI在中间区域
                return 0.8  # 降低权重
            elif distance_from_extreme < 10:  # RSI在极值区域
                return 1.4  # 增强权重
            else:
                return 1.0
        except:
            return 1.0

    def _get_default_rsi_signal(self) -> Dict[str, Any]:
        """获取默认RSI信号"""
        return {
            "signal": "neutral",
            "confidence": 50,
            "signal_strength": "weak",
            "rsi_current": 50.0,
            "rsi_previous": 50.0,
            "dynamic_thresholds": self.default_thresholds.copy(),
            "rsi_trend": "neutral",
            "divergence_info": {"has_divergence": False, "divergence_type": None},
            "metrics": {
                "rsi_14": 50.0,
                "rsi_change": 0.0,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
                "threshold_distance": {
                    "to_oversold": 20.0,
                    "to_overbought": 20.0
                }
            }
        }

    # 继承现货RSI策略的分析方法
    def calculate_atr_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        计算ATR（平均真实波幅）指标

        Args:
            df: 包含OHLC数据的DataFrame，必须包含'high', 'low', 'close'列

        Returns:
            dict: 包含atr_14, atr_28, atr_percentile的字典
        """
        if df.empty or len(df) < 28:
            return {
                'atr_14': 0.0,
                'atr_28': 0.0,
                'atr_percentile': 0.0
            }

        # 确保列名正确（转换为小写）
        required_cols = ['high', 'low', 'close']
        df_cols = df.columns.str.lower()

        # 检查必需的列是否存在
        if not all(col in df_cols for col in required_cols):
            # 尝试大写列名
            if all(col.upper() in df.columns for col in required_cols):
                high = df[df.columns[df.columns.str.upper() == 'HIGH'][0]]
                low = df[df.columns[df.columns.str.upper() == 'LOW'][0]]
                close = df[df.columns[df.columns.str.upper() == 'CLOSE'][0]]
            else:
                return {
                    'atr_14': 0.0,
                    'atr_28': 0.0,
                    'atr_percentile': 0.0
                }
        else:
            high = df[df.columns[df_cols == 'high'][0]]
            low = df[df.columns[df_cols == 'low'][0]]
            close = df[df.columns[df_cols == 'close'][0]]

        try:
            # 使用pandas-ta计算ATR
            # 计算14期ATR (默认使用RMA移动平均)
            atr_14_series = ta.atr(high=high, low=low, close=close, length=14)

            # 计算28期ATR
            atr_28_series = ta.atr(high=high, low=low, close=close, length=28)

            # 获取最新的ATR值（去除NaN）
            atr_14 = atr_14_series.dropna().iloc[-1] if not atr_14_series.dropna().empty else 0.0
            atr_28 = atr_28_series.dropna().iloc[-1] if not atr_28_series.dropna().empty else 0.0

            # 计算ATR百分位数（使用14期ATR的历史分布）
            if not atr_14_series.dropna().empty and len(atr_14_series.dropna()) > 1:
                # 计算当前ATR在历史数据中的百分位数
                atr_percentile = (atr_14_series.dropna().rank(pct=True).iloc[-1] * 100)
            else:
                atr_percentile = 50.0  # 默认中位数

            return {
                'atr_14': round(float(atr_14), 6),
                'atr_28': round(float(atr_28), 6),
                'atr_percentile': round(float(atr_percentile), 2)
            }

        except Exception as e:
            print(f"ATR calculation error: {e}")
            return {
                'atr_14': 0.0,
                'atr_28': 0.0,
                'atr_percentile': 0.0
            }

    def identify_price_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        识别关键价位：支撑位、阻力位、枢轴点等

        Args:
            df: 包含OHLC数据的DataFrame

        Returns:
            dict: 包含各种价位信息的字典
        """
        try:
            if df.empty or len(df) < 20:
                return self._get_default_levels()

            # 数据预处理和列名标准化
            df_normalized = self._normalize_columns(df)

            if not self._validate_required_columns(df_normalized):
                return self._get_default_levels()

            current_price = float(df_normalized['close'].iloc[-1])

            # 计算枢轴点
            pivot_point = self._calculate_pivot_point(df_normalized)

            # 识别支撑位
            support_levels = self._identify_support_levels(df_normalized, current_price)

            # 识别阻力位
            resistance_levels = self._identify_resistance_levels(df_normalized, current_price)

            # 计算突破阈值
            breakout_threshold = self._calculate_breakout_threshold(df_normalized, current_price)

            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'pivot_point': pivot_point,
                'breakout_threshold': breakout_threshold,
                'current_price': current_price
            }

        except Exception as e:
            print(f"Price level identification error: {e}")
            return self._get_default_levels()

    def analyze_volatility_depth(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        深度波动率分析

        Args:
            df: 价格数据DataFrame

        Returns:
            dict: 波动率分析结果
        """
        try:
            if df.empty or len(df) < 30:
                return self._get_default_volatility_analysis()

            # 计算历史波动率数据
            volatility_data = self._calculate_historical_volatility(df)

            # 计算波动率百分位数
            volatility_percentile = self._calculate_volatility_percentile(volatility_data)

            # 识别波动率趋势
            volatility_trend = self._identify_volatility_trend(volatility_data)

            # 预测未来波动率
            volatility_forecast = self._forecast_volatility(volatility_data)

            # 计算波动率制度概率
            regime_probability = self._calculate_regime_probability(volatility_data)

            return {
                'volatility_percentile': volatility_percentile,
                'volatility_trend': volatility_trend,
                'volatility_forecast': volatility_forecast,
                'regime_probability': regime_probability,
                'historical_volatility': {
                    'daily_vol': float(volatility_data['daily_vol'][-1]) if len(volatility_data['daily_vol']) > 0 else 0.0,
                    'weekly_vol': float(volatility_data['weekly_vol'][-1]) if len(volatility_data['weekly_vol']) > 0 else 0.0,
                    'monthly_vol': float(volatility_data['monthly_vol'][-1]) if len(volatility_data['monthly_vol']) > 0 else 0.0
                }
            }

        except Exception as e:
            print(f"Volatility depth analysis error: {e}")
            return self._get_default_volatility_analysis()

    def _calculate_historical_volatility(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算多时间框架历史波动率"""
        try:
            close_prices = df['close'] if 'close' in df.columns else df['Close']
            returns = close_prices.pct_change().dropna()

            # 计算不同时间窗口的波动率
            daily_vol = returns.rolling(window=5).std() * np.sqrt(252)
            weekly_vol = returns.rolling(window=10).std() * np.sqrt(252)
            monthly_vol = returns.rolling(window=21).std() * np.sqrt(252)

            return {
                'returns': returns.values,
                'daily_vol': daily_vol.dropna().values,
                'weekly_vol': weekly_vol.dropna().values,
                'monthly_vol': monthly_vol.dropna().values
            }
        except Exception as e:
            print(f"Historical volatility calculation error: {e}")
            return {'returns': np.array([]), 'daily_vol': np.array([]), 'weekly_vol': np.array([]), 'monthly_vol': np.array([])}

    def _calculate_volatility_percentile(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """计算波动率百分位数"""
        try:
            daily_vol = volatility_data.get('daily_vol', np.array([]))
            if len(daily_vol) < 2:
                return 50.0

            current_vol = daily_vol[-1]
            percentile = (daily_vol <= current_vol).mean() * 100
            return float(percentile)
        except:
            return 50.0

    def _identify_volatility_trend(self, volatility_data: Dict[str, np.ndarray]) -> str:
        """识别波动率趋势"""
        try:
            daily_vol = volatility_data.get('daily_vol', np.array([]))
            if len(daily_vol) < 10:
                return 'stable'

            recent_vol = daily_vol[-5:]
            early_vol = daily_vol[-10:-5]

            recent_mean = np.mean(recent_vol)
            early_mean = np.mean(early_vol)

            change_pct = (recent_mean - early_mean) / early_mean * 100

            if change_pct > 10:
                return 'increasing'
            elif change_pct < -10:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'stable'

    def _forecast_volatility(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """预测未来波动率"""
        try:
            daily_vol = volatility_data.get('daily_vol', np.array([]))
            if len(daily_vol) < 5:
                return 0.0

            # 简单的指数平滑预测
            alpha = 0.3
            forecast = daily_vol[-1]
            for i in range(1, min(5, len(daily_vol))):
                forecast = alpha * daily_vol[-i-1] + (1 - alpha) * forecast

            return float(forecast)
        except:
            return 0.0

    def _calculate_regime_probability(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """计算波动率制度概率"""
        try:
            daily_vol = volatility_data.get('daily_vol', np.array([]))
            if len(daily_vol) < 20:
                return 0.5

            # 使用中位数作为高低波动率的分界点
            median_vol = np.median(daily_vol)
            current_vol = daily_vol[-1]

            # 计算当前处于高波动率制度的概率
            high_vol_prob = 1 / (1 + np.exp(-2 * (current_vol - median_vol) / median_vol))
            return float(high_vol_prob)
        except:
            return 0.5

    def _get_default_volatility_analysis(self) -> Dict[str, Any]:
        """获取默认波动率分析结果"""
        return {
            'volatility_percentile': 50.0,
            'volatility_trend': 'stable',
            'volatility_forecast': 0.0,
            'regime_probability': 0.5,
            'historical_volatility': {
                'daily_vol': 0.0,
                'weekly_vol': 0.0,
                'monthly_vol': 0.0
            }
        }

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()
        return df_copy

    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """验证必需的列是否存在"""
        required_cols = ['high', 'low', 'close']
        return all(col in df.columns for col in required_cols)

    def _get_default_levels(self) -> Dict[str, Any]:
        """获取默认价位信息"""
        return {
            'support_levels': [0.0, 0.0, 0.0],
            'resistance_levels': [0.0, 0.0, 0.0],
            'pivot_point': 0.0,
            'breakout_threshold': 0.0,
            'current_price': 0.0
        }

    def _calculate_pivot_point(self, df: pd.DataFrame) -> float:
        """计算枢轴点"""
        try:
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-2]  # 使用前一个收盘价
            return float((high + low + close) / 3)
        except:
            return 0.0

    def _identify_support_levels(self, df: pd.DataFrame, current_price: float) -> list:
        """识别支撑位"""
        try:
            low_prices = df['low'].rolling(window=5).min()
            support_candidates = low_prices[low_prices == df['low']]

            # 筛选显著支撑位
            significant_supports = []
            for price in support_candidates.values:
                if price < current_price * 0.98:  # 至少低于当前价格2%
                    significant_supports.append(float(price))

            # 排序并取前3个
            significant_supports.sort(reverse=True)
            supports = significant_supports[:3]

            # 填充至3个
            while len(supports) < 3:
                supports.append(0.0)

            return supports
        except:
            return [0.0, 0.0, 0.0]

    def _identify_resistance_levels(self, df: pd.DataFrame, current_price: float) -> list:
        """识别阻力位"""
        try:
            high_prices = df['high'].rolling(window=5).max()
            resistance_candidates = high_prices[high_prices == df['high']]

            # 筛选显著阻力位
            significant_resistances = []
            for price in resistance_candidates.values:
                if price > current_price * 1.02:  # 至少高于当前价格2%
                    significant_resistances.append(float(price))

            # 排序并取前3个
            significant_resistances.sort()
            resistances = significant_resistances[:3]

            # 填充至3个
            while len(resistances) < 3:
                resistances.append(0.0)

            return resistances
        except:
            return [0.0, 0.0, 0.0]

    def _calculate_breakout_threshold(self, df: pd.DataFrame, current_price: float) -> float:
        """计算突破阈值"""
        try:
            # 使用ATR计算动态突破阈值
            atr_data = self.calculate_atr_values(df)
            atr_14 = atr_data.get('atr_14', 0)

            # 突破阈值为当前价格 + 1.5倍ATR
            return float(current_price + 1.5 * atr_14)
        except:
            return 0.0

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        期货RSI策略主执行方法

        整合了原始RSI策略的所有复杂分析能力，但调整为期货交易语义：
        1. 动态RSI阈值调整 (根据波动率)
        2. long/short/neutral信号输出
        3. 期货特有的风险指标
        4. 双向交易支持
        5. 杠杆敏感性分析
        """

        data = state['data']
        data['name'] = "FuturesRSIStrategy"

        data = state.get("data", {})
        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        # Initialize analysis for each ticker
        technical_analysis = {}
        for ticker in tickers:
            technical_analysis[ticker] = {}

        # 期货RSI策略权重 (更偏重均值回归)
        strategy_weights = {
            "trend": 0.15,           # 降低趋势权重
            "mean_reversion": 0.35,  # 增加均值回归权重
            "momentum": 0.20,        # 保持动量权重
            "volatility": 0.15,      # 保持波动率权重
            "stat_arb": 0.15,        # 保持统计套利权重
        }

        for ticker in tickers:
            # 收集该ticker所有时间框架的信号数据（用于跨时间框架分析）
            timeframe_signals = {}

            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())

                # 如果数据为空，跳过该时间框架
                if df.empty:
                    continue

                # 计算动态RSI阈值
                dynamic_thresholds = self._calculate_dynamic_rsi_thresholds(df)

                # 计算期货RSI信号
                futures_rsi_signals = self._calculate_futures_rsi_signals(df, dynamic_thresholds)

                # 计算各种策略信号 (继承原始RSI策略)
                trend_signals = calculate_trend_signals(df)
                mean_reversion_signals = calculate_mean_reversion_signals(df)
                momentum_signals = calculate_momentum_signals(df)
                volatility_signals = calculate_volatility_signals(df)
                stat_arb_signals = calculate_stat_arb_signals(df)

                # 组合信号
                combined_signal = weighted_signal_combination(
                    {
                        "trend": trend_signals,
                        "mean_reversion": mean_reversion_signals,
                        "momentum": momentum_signals,
                        "volatility": volatility_signals,
                        "stat_arb": stat_arb_signals,
                    },
                    strategy_weights,
                )

                # 转换组合信号为期货信号格式
                combined_futures_signal = self._convert_signal_to_futures(combined_signal["signal"])

                # 融合RSI信号和组合信号
                final_signal = self._merge_signals(futures_rsi_signals, combined_signal, combined_futures_signal)

                # 构建策略信号结构
                strategy_signals_data = {
                    "futures_rsi": {
                        "signal": futures_rsi_signals["signal"],
                        "confidence": futures_rsi_signals["confidence"],
                        "metrics": futures_rsi_signals["metrics"],
                    },
                    "trend_following": {
                        "signal": self._convert_signal_to_futures(trend_signals["signal"]),
                        "confidence": round(trend_signals["confidence"] * 100),
                        "metrics": normalize_pandas(trend_signals["metrics"]),
                    },
                    "mean_reversion": {
                        "signal": self._convert_signal_to_futures(mean_reversion_signals["signal"]),
                        "confidence": round(mean_reversion_signals["confidence"] * 100),
                        "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                    },
                    "momentum": {
                        "signal": self._convert_signal_to_futures(momentum_signals["signal"]),
                        "confidence": round(momentum_signals["confidence"] * 100),
                        "metrics": normalize_pandas(momentum_signals["metrics"]),
                    },
                    "volatility": {
                        "signal": self._convert_signal_to_futures(volatility_signals["signal"]),
                        "confidence": round(volatility_signals["confidence"] * 100),
                        "metrics": normalize_pandas(volatility_signals["metrics"]),
                    },
                    "statistical_arbitrage": {
                        "signal": self._convert_signal_to_futures(stat_arb_signals["signal"]),
                        "confidence": round(stat_arb_signals["confidence"] * 100),
                        "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                    },
                }

                # 计算增强分析功能
                atr_values = {'atr_14': 0.0, 'atr_28': 0.0, 'atr_percentile': 0.0}
                price_levels = {
                    'support_levels': [0.0, 0.0, 0.0],
                    'resistance_levels': [0.0, 0.0, 0.0],
                    'pivot_point': 0.0,
                    'breakout_threshold': 0.0
                }
                volatility_analysis = {
                    'volatility_percentile': 50.0,
                    'volatility_trend': 'stable',
                    'volatility_forecast': 0.0,
                    'regime_probability': 0.5
                }

                try:
                    atr_values = self.calculate_atr_values(df)
                except:
                    pass

                try:
                    price_levels = self.identify_price_levels(df)
                except:
                    pass

                try:
                    volatility_analysis = self.analyze_volatility_depth(df)
                except:
                    pass

                # 计算期货特有的信号强度
                futures_signal_strength = self._calculate_futures_signal_strength(futures_rsi_signals, df)

                # 构建完整的时间框架信号数据
                timeframe_signal_data = {
                    "signal": final_signal["signal"],
                    "confidence": final_signal["confidence"],
                    "signal_strength": futures_rsi_signals["signal_strength"],
                    "futures_factors": futures_signal_strength.get("futures_factors", {}),
                    "rsi_data": futures_rsi_signals,
                    "dynamic_thresholds": dynamic_thresholds,
                    "atr_values": atr_values,
                    "price_levels": price_levels,
                    "volatility_analysis": volatility_analysis,
                    "strategy_signals": strategy_signals_data,
                    "interval": interval.value
                }

                timeframe_signals[interval.value] = timeframe_signal_data

                # 存储该时间框架的分析结果
                technical_analysis[ticker][interval.value] = {
                    "signal": final_signal["signal"],
                    "confidence": round(final_signal["confidence"], 2),
                    "signal_strength": futures_rsi_signals["signal_strength"],

                    # 期货RSI核心数据
                    "rsi_data": {
                        "current_rsi": futures_rsi_signals["rsi_current"],
                        "rsi_trend": futures_rsi_signals["rsi_trend"],
                        "dynamic_thresholds": dynamic_thresholds,
                        "divergence_info": futures_rsi_signals["divergence_info"]
                    },

                    # 期货特有因素
                    "futures_factors": futures_signal_strength.get("futures_factors", {}),

                    # 技术分析数据
                    "atr_values": atr_values,
                    "price_levels": price_levels,
                    "volatility_analysis": volatility_analysis,

                    # 策略组合数据
                    "strategy_signals": strategy_signals_data,

                    # 元数据
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "data_quality": "good" if not df.empty else "poor"
                }

            # 跨时间框架分析（继承原始RSI策略的方法）
            if timeframe_signals:
                cross_timeframe_analysis = self._perform_cross_timeframe_analysis(timeframe_signals)
                technical_analysis[ticker]["cross_timeframe_analysis"] = cross_timeframe_analysis

        # 最终输出
        output = {
            "technical_analysis": technical_analysis,
            "strategy_name": "FuturesRSIStrategy",
            "strategy_type": "futures_mean_reversion",
            "execution_timestamp": pd.Timestamp.now().isoformat(),
            "configuration": self.futures_config
        }

        # 创建期货RSI策略消息
        message = HumanMessage(
            content=json.dumps(output, ensure_ascii=False),
            name="futures_rsi_strategy_agent",
        )

        # 显示推理过程（如果启用）
        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "期货RSI策略分析师 (Futures RSI Strategy)")

        # 添加信号到分析信号列表
        state["data"]["analyst_signals"]["futures_rsi_strategy_agent"] = technical_analysis

        # 更新状态数据
        data["analysis"] = technical_analysis
        data["strategy_output"] = output

        return {
            "messages": [message],
            "data": data
        }

    def _merge_signals(self, rsi_signal: Dict[str, Any], combined_signal: Dict[str, Any],
                      combined_futures_signal: str) -> Dict[str, Any]:
        """
        融合RSI信号和组合信号

        Args:
            rsi_signal: RSI信号数据
            combined_signal: 组合信号数据
            combined_futures_signal: 转换后的期货组合信号

        Returns:
            dict: 融合后的最终信号
        """
        try:
            rsi_conf = rsi_signal.get("confidence", 50)
            combined_conf = combined_signal.get("confidence", 0.5) * 100

            # 如果RSI信号和组合信号一致，增强置信度
            if rsi_signal["signal"] == combined_futures_signal:
                final_confidence = min(95, (rsi_conf * 0.6 + combined_conf * 0.4) * 1.2)
                final_signal = rsi_signal["signal"]

            # 如果信号冲突，以RSI为主但降低置信度
            elif rsi_signal["signal"] != "neutral" and combined_futures_signal != "neutral":
                if rsi_conf > combined_conf:
                    final_signal = rsi_signal["signal"]
                    final_confidence = rsi_conf * 0.7
                else:
                    final_signal = combined_futures_signal
                    final_confidence = combined_conf * 0.7

            # 一个为中性时，采用非中性信号
            elif rsi_signal["signal"] == "neutral":
                final_signal = combined_futures_signal
                final_confidence = combined_conf * 0.8
            else:
                final_signal = rsi_signal["signal"]
                final_confidence = rsi_conf * 0.8

            return {
                "signal": final_signal,
                "confidence": max(30, min(95, final_confidence))
            }

        except Exception as e:
            print(f"Signal merging error: {e}")
            return {
                "signal": "neutral",
                "confidence": 50
            }

    def _perform_cross_timeframe_analysis(self, timeframe_signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行跨时间框架分析（简化版本）

        Args:
            timeframe_signals: 各时间框架信号数据

        Returns:
            dict: 跨时间框架分析结果
        """
        try:
            if not timeframe_signals:
                return self._get_default_cross_timeframe_analysis()

            # 计算信号一致性
            signals = [data.get("signal", "neutral") for data in timeframe_signals.values()]
            signal_counts = {signal: signals.count(signal) for signal in set(signals)}

            # 找到主导信号
            dominant_signal = max(signal_counts.items(), key=lambda x: x[1])

            # 计算平均置信度
            confidences = [data.get("confidence", 50) for data in timeframe_signals.values()]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 50

            # 评估整体信号强度
            signal_strength_values = [data.get("signal_strength", "weak") for data in timeframe_signals.values()]
            strong_count = signal_strength_values.count("strong")
            moderate_count = signal_strength_values.count("moderate")

            if strong_count > len(timeframe_signals) / 2:
                overall_strength = "strong"
            elif strong_count + moderate_count > len(timeframe_signals) / 2:
                overall_strength = "moderate"
            else:
                overall_strength = "weak"

            return {
                "dominant_signal": dominant_signal[0],
                "signal_consensus": dominant_signal[1] / len(timeframe_signals),
                "average_confidence": round(avg_confidence, 2),
                "overall_strength": overall_strength,
                "timeframe_count": len(timeframe_signals),
                "signal_distribution": signal_counts
            }

        except Exception as e:
            print(f"Cross timeframe analysis error: {e}")
            return self._get_default_cross_timeframe_analysis()

    def _get_default_cross_timeframe_analysis(self) -> Dict[str, Any]:
        """获取默认跨时间框架分析结果"""
        return {
            "dominant_signal": "neutral",
            "signal_consensus": 0.0,
            "average_confidence": 50.0,
            "overall_strength": "weak",
            "timeframe_count": 0,
            "signal_distribution": {}
        }