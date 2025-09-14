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


class FuturesMacdStrategy(BaseNode):
    """
    期货版本的MACD策略，专为期货交易设计
    - 支持long/short/neutral信号输出语义
    - 考虑双向交易特性
    - 包含杠杆敏感性分析
    - 增强的风险管理指标
    """

    def __init__(self):
        """初始化期货MACD策略"""
        super().__init__()

        # 期货特有配置
        self.futures_config = {
            "leverage_sensitivity": True,  # 启用杠杆敏感性分析
            "bidirectional_signals": True,  # 启用双向信号
            "enhanced_risk_metrics": True,  # 启用增强风险指标
            "signal_strength_multiplier": 1.2,  # 期货信号强度倍数
        }

    def _convert_signal_to_futures(self, signal: str) -> str:
        """
        将现货交易信号转换为期货交易信号

        Args:
            signal: 原始信号 (bullish/bearish/neutral)

        Returns:
            str: 期货信号 (long/short/neutral)
        """
        signal_mapping = {
            "bullish": "long",
            "bearish": "short",
            "neutral": "neutral"
        }
        return signal_mapping.get(signal, "neutral")

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
                "bidirectional_bias": self._calculate_bidirectional_bias(signal_data)
            }

            # 综合期货信号强度
            futures_multiplier = (
                futures_factors["volatility_factor"] * 0.25 +
                futures_factors["momentum_factor"] * 0.25 +
                futures_factors["volume_factor"] * 0.20 +
                futures_factors["leverage_sensitivity"] * 0.15 +
                futures_factors["bidirectional_bias"] * 0.15
            )

            # 应用期货特有的信号强度倍数
            enhanced_confidence = min(1.0, base_confidence * futures_multiplier *
                                    self.futures_config["signal_strength_multiplier"])

            # 信号强度等级
            if enhanced_confidence >= 0.8:
                strength_level = "very_strong"
            elif enhanced_confidence >= 0.65:
                strength_level = "strong"
            elif enhanced_confidence >= 0.45:
                strength_level = "moderate"
            elif enhanced_confidence >= 0.25:
                strength_level = "weak"
            else:
                strength_level = "very_weak"

            return {
                "enhanced_confidence": round(enhanced_confidence * 100, 2),
                "strength_level": strength_level,
                "futures_factors": futures_factors,
                "leverage_recommendation": self._get_leverage_recommendation(
                    enhanced_confidence, futures_factors["leverage_sensitivity"]
                )
            }

        except Exception as e:
            print(f"Error calculating futures signal strength: {e}")
            return {
                "enhanced_confidence": 50.0,
                "strength_level": "moderate",
                "futures_factors": {},
                "leverage_recommendation": "conservative"
            }

    def _calculate_volatility_factor(self, df: pd.DataFrame) -> float:
        """计算波动率因子，高波动率对期货更敏感"""
        try:
            if len(df) < 20:
                return 1.0

            # 计算短期和长期ATR
            high = df['high'] if 'high' in df.columns else df['High']
            low = df['low'] if 'low' in df.columns else df['Low']
            close = df['close'] if 'close' in df.columns else df['Close']

            atr_14 = ta.atr(high=high, low=low, close=close, length=14)
            atr_21 = ta.atr(high=high, low=low, close=close, length=21)

            if atr_14 is None or atr_21 is None:
                return 1.0

            # 波动率变化率
            current_atr = atr_14.iloc[-1] if not atr_14.empty else 0
            avg_atr = atr_21.iloc[-1] if not atr_21.empty else current_atr

            if avg_atr == 0:
                return 1.0

            volatility_ratio = current_atr / avg_atr

            # 期货对高波动率更敏感，提高信号强度
            if volatility_ratio > 1.3:
                return 1.4  # 高波动率增强信号
            elif volatility_ratio > 1.1:
                return 1.2
            elif volatility_ratio < 0.7:
                return 0.8  # 低波动率减弱信号
            else:
                return 1.0

        except Exception:
            return 1.0

    def _calculate_momentum_factor(self, df: pd.DataFrame) -> float:
        """计算动量因子，期货更适合趋势跟踪"""
        try:
            if len(df) < 20:
                return 1.0

            close = df['close'] if 'close' in df.columns else df['Close']

            # 计算多重时间框架的动量
            roc_5 = ta.roc(close, length=5)
            roc_10 = ta.roc(close, length=10)
            roc_20 = ta.roc(close, length=20)

            if roc_5 is None or roc_10 is None or roc_20 is None:
                return 1.0

            # 动量一致性检查
            current_roc_5 = roc_5.iloc[-1] if not roc_5.empty else 0
            current_roc_10 = roc_10.iloc[-1] if not roc_10.empty else 0
            current_roc_20 = roc_20.iloc[-1] if not roc_20.empty else 0

            # 动量方向一致性
            momentum_signals = [current_roc_5 > 0, current_roc_10 > 0, current_roc_20 > 0]
            consistency = sum(momentum_signals) / len(momentum_signals)

            # 动量强度
            momentum_strength = (abs(current_roc_5) + abs(current_roc_10) + abs(current_roc_20)) / 3

            # 期货偏好强动量
            if consistency >= 0.67 and momentum_strength > 2.0:
                return 1.5
            elif consistency >= 0.67:
                return 1.3
            elif consistency >= 0.33:
                return 1.0
            else:
                return 0.8

        except Exception:
            return 1.0

    def _calculate_volume_factor(self, df: pd.DataFrame) -> float:
        """计算成交量因子"""
        try:
            if 'volume' not in df.columns and 'Volume' not in df.columns:
                return 1.0

            volume = df['volume'] if 'volume' in df.columns else df['Volume']

            if len(volume) < 20:
                return 1.0

            # 成交量相对强度
            current_volume = volume.iloc[-1]
            avg_volume_20 = volume.rolling(20).mean().iloc[-1]

            if avg_volume_20 == 0:
                return 1.0

            volume_ratio = current_volume / avg_volume_20

            # 期货重视成交量确认
            if volume_ratio > 2.0:
                return 1.4
            elif volume_ratio > 1.5:
                return 1.2
            elif volume_ratio < 0.5:
                return 0.7
            else:
                return 1.0

        except Exception:
            return 1.0

    def _calculate_leverage_sensitivity(self, df: pd.DataFrame) -> float:
        """计算杠杆敏感性指标"""
        try:
            if len(df) < 14:
                return 0.5

            close = df['close'] if 'close' in df.columns else df['Close']
            high = df['high'] if 'high' in df.columns else df['High']
            low = df['low'] if 'low' in df.columns else df['Low']

            # 价格波动性 (ATR相对价格)
            atr = ta.atr(high=high, low=low, close=close, length=14)
            if atr is None or atr.empty:
                return 0.5

            current_price = close.iloc[-1]
            current_atr = atr.iloc[-1]

            if current_price == 0:
                return 0.5

            # ATR相对于价格的百分比
            atr_percentage = (current_atr / current_price) * 100

            # 杠杆敏感性评分 (波动率越低，杠杆越安全)
            if atr_percentage < 1.0:
                return 0.9  # 低波动，高杠杆安全性
            elif atr_percentage < 2.0:
                return 0.7
            elif atr_percentage < 3.0:
                return 0.5
            elif atr_percentage < 5.0:
                return 0.3
            else:
                return 0.1  # 高波动，低杠杆安全性

        except Exception:
            return 0.5

    def _calculate_bidirectional_bias(self, signal_data: Dict[str, Any]) -> float:
        """计算双向交易偏差，期货可以等效做多做空"""
        try:
            # 分析各策略信号的一致性
            strategy_signals = signal_data.get("strategy_signals", {})

            long_signals = 0
            short_signals = 0
            total_signals = 0

            for strategy, data in strategy_signals.items():
                signal = data.get("signal", "neutral")
                confidence = data.get("confidence", 0) / 100.0

                if signal == "bullish":
                    long_signals += confidence
                elif signal == "bearish":
                    short_signals += confidence

                total_signals += confidence

            if total_signals == 0:
                return 1.0

            # 计算方向性偏差
            long_bias = long_signals / total_signals
            short_bias = short_signals / total_signals

            # 期货双向平等性，强化占优方向
            max_bias = max(long_bias, short_bias)

            if max_bias > 0.7:
                return 1.3  # 强方向性
            elif max_bias > 0.6:
                return 1.2
            else:
                return 1.0

        except Exception:
            return 1.0

    def _get_leverage_recommendation(self, confidence: float, leverage_sensitivity: float) -> str:
        """根据信号强度和杠杆敏感性推荐杠杆水平"""
        try:
            # 综合评分
            leverage_score = confidence * leverage_sensitivity

            if leverage_score >= 0.8:
                return "aggressive"  # 5-10x
            elif leverage_score >= 0.6:
                return "moderate"   # 3-5x
            elif leverage_score >= 0.4:
                return "conservative" # 2-3x
            elif leverage_score >= 0.2:
                return "minimal"    # 1-2x
            else:
                return "avoid"      # 现货或极低杠杆

        except Exception:
            return "conservative"

    def calculate_atr_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        计算ATR（平均真实波幅）指标 - 继承自原版本
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
            atr_14_series = ta.atr(high=high, low=low, close=close, length=14)
            atr_28_series = ta.atr(high=high, low=low, close=close, length=28)

            # 获取最新的ATR值（去除NaN）
            atr_14 = atr_14_series.dropna().iloc[-1] if not atr_14_series.dropna().empty else 0.0
            atr_28 = atr_28_series.dropna().iloc[-1] if not atr_28_series.dropna().empty else 0.0

            # 计算ATR百分位数
            if not atr_14_series.dropna().empty and len(atr_14_series.dropna()) > 1:
                atr_percentile = (atr_14_series.dropna().rank(pct=True).iloc[-1] * 100)
            else:
                atr_percentile = 50.0

            return {
                'atr_14': round(float(atr_14), 6),
                'atr_28': round(float(atr_28), 6),
                'atr_percentile': round(float(atr_percentile), 2)
            }

        except Exception as e:
            print(f"Error calculating ATR values: {e}")
            return {
                'atr_14': 0.0,
                'atr_28': 0.0,
                'atr_percentile': 0.0
            }

    def identify_price_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        识别关键价位 - 继承自原版本但优化用于期货交易
        """
        if df.empty or len(df) < 20:
            return self._get_default_levels()

        try:
            # 数据标准化
            df = self._normalize_columns(df)

            # 验证必需列
            if not self._validate_required_columns(df):
                return self._get_default_levels()

            current_price = df['close'].iloc[-1]

            # 计算枢轴点
            pivot_point = self._calculate_pivot_point(df)

            # 识别支撑位（期货做空时的目标位）
            support_levels = self._identify_support_levels(df, current_price)

            # 识别阻力位（期货做多时的目标位）
            resistance_levels = self._identify_resistance_levels(df, current_price)

            # 计算突破阈值（期货更关注突破交易）
            breakout_threshold = self._calculate_breakout_threshold(df, current_price)

            # 验证价位有效性
            support_levels = self._validate_support_levels(support_levels, current_price)
            resistance_levels = self._validate_resistance_levels(resistance_levels, current_price)

            return {
                'support_levels': support_levels[:3],  # 取前3个最强支撑
                'resistance_levels': resistance_levels[:3],  # 取前3个最强阻力
                'pivot_point': round(pivot_point, 6),
                'breakout_threshold': round(breakout_threshold, 6),
                'current_price': round(current_price, 6),
                # 期货特有：距离关键位的百分比
                'distance_to_support': round(((current_price - support_levels[0]) / current_price * 100) if support_levels else 0, 2),
                'distance_to_resistance': round(((resistance_levels[0] - current_price) / current_price * 100) if resistance_levels else 0, 2)
            }

        except Exception as e:
            print(f"Error identifying price levels: {e}")
            return self._get_default_levels()

    def analyze_volatility_depth(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        波动率深度分析 - 继承自原版本但增加期货特有指标
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

            # 波动率预测
            volatility_forecast = self._forecast_volatility(volatility_data)

            # 市场状态概率
            regime_probability = self._calculate_regime_probability(volatility_data)

            # 期货特有：杠杆调整后的波动率风险
            leverage_adjusted_vol = self._calculate_leverage_adjusted_volatility(volatility_data)

            return {
                'volatility_percentile': round(volatility_percentile, 2),
                'volatility_trend': volatility_trend,
                'volatility_forecast': round(volatility_forecast, 6),
                'regime_probability': round(regime_probability, 3),
                'leverage_adjusted_volatility': leverage_adjusted_vol,  # 期货特有
                'volatility_risk_level': self._assess_volatility_risk_level(volatility_percentile, leverage_adjusted_vol)  # 期货特有
            }

        except Exception as e:
            print(f"Error in volatility analysis: {e}")
            return self._get_default_volatility_analysis()

    def _calculate_leverage_adjusted_volatility(self, volatility_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算杠杆调整后的波动率指标 - 期货特有"""
        try:
            current_vol = volatility_data.get('daily_volatility', np.array([0.02]))[-1]

            # 不同杠杆水平的风险调整
            leverage_scenarios = {
                '2x': current_vol * 2,
                '5x': current_vol * 5,
                '10x': current_vol * 10,
                '20x': current_vol * 20
            }

            # 计算各杠杆水平的风险评分
            risk_scores = {}
            for leverage, adjusted_vol in leverage_scenarios.items():
                if adjusted_vol < 0.05:  # < 5% 日波动
                    risk_scores[leverage] = 'low'
                elif adjusted_vol < 0.10:  # < 10% 日波动
                    risk_scores[leverage] = 'moderate'
                elif adjusted_vol < 0.20:  # < 20% 日波动
                    risk_scores[leverage] = 'high'
                else:
                    risk_scores[leverage] = 'extreme'

            return {
                'current_daily_vol': round(current_vol * 100, 2),  # 百分比
                'leverage_scenarios': {k: round(v * 100, 2) for k, v in leverage_scenarios.items()},
                'risk_assessment': risk_scores
            }

        except Exception:
            return {
                'current_daily_vol': 2.0,
                'leverage_scenarios': {'2x': 4.0, '5x': 10.0, '10x': 20.0, '20x': 40.0},
                'risk_assessment': {'2x': 'low', '5x': 'moderate', '10x': 'high', '20x': 'extreme'}
            }

    def _assess_volatility_risk_level(self, vol_percentile: float, leverage_adjusted_vol: Dict[str, float]) -> str:
        """评估波动率风险等级 - 期货特有"""
        try:
            current_vol = leverage_adjusted_vol.get('current_daily_vol', 2.0)

            # 综合波动率百分位和绝对波动率
            if vol_percentile > 80 and current_vol > 5.0:
                return 'very_high'
            elif vol_percentile > 60 and current_vol > 3.0:
                return 'high'
            elif vol_percentile > 40 or current_vol > 2.0:
                return 'moderate'
            elif vol_percentile > 20:
                return 'low'
            else:
                return 'very_low'

        except Exception:
            return 'moderate'

    # 以下方法继承自原版本，保持技术分析能力
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        df = df.copy()
        df.columns = df.columns.str.lower()
        return df

    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """验证必需列"""
        required = ['high', 'low', 'close']
        return all(col in df.columns for col in required)

    def _get_default_levels(self) -> Dict[str, Any]:
        """默认价位数据"""
        return {
            'support_levels': [0.0, 0.0, 0.0],
            'resistance_levels': [0.0, 0.0, 0.0],
            'pivot_point': 0.0,
            'breakout_threshold': 0.0,
            'current_price': 0.0,
            'distance_to_support': 0.0,
            'distance_to_resistance': 0.0
        }

    def _calculate_pivot_point(self, df: pd.DataFrame) -> float:
        """计算枢轴点"""
        try:
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            close = df['close'].iloc[-2]  # 前一日收盘价
            return (high + low + close) / 3
        except:
            return 0.0

    def _identify_support_levels(self, df: pd.DataFrame, current_price: float) -> list:
        """识别支撑位"""
        try:
            low_prices = df['low']
            support_levels = []

            # 寻找局部最低点
            for i in range(2, len(low_prices) - 2):
                if (low_prices.iloc[i] < low_prices.iloc[i-1] and
                    low_prices.iloc[i] < low_prices.iloc[i-2] and
                    low_prices.iloc[i] < low_prices.iloc[i+1] and
                    low_prices.iloc[i] < low_prices.iloc[i+2] and
                    low_prices.iloc[i] < current_price):
                    support_levels.append(low_prices.iloc[i])

            # 按照距离当前价格排序
            support_levels.sort(reverse=True)  # 最近的支撑在前
            return support_levels[:5] if support_levels else [current_price * 0.95]

        except Exception:
            return [current_price * 0.95]

    def _identify_resistance_levels(self, df: pd.DataFrame, current_price: float) -> list:
        """识别阻力位"""
        try:
            high_prices = df['high']
            resistance_levels = []

            # 寻找局部最高点
            for i in range(2, len(high_prices) - 2):
                if (high_prices.iloc[i] > high_prices.iloc[i-1] and
                    high_prices.iloc[i] > high_prices.iloc[i-2] and
                    high_prices.iloc[i] > high_prices.iloc[i+1] and
                    high_prices.iloc[i] > high_prices.iloc[i+2] and
                    high_prices.iloc[i] > current_price):
                    resistance_levels.append(high_prices.iloc[i])

            # 按照距离当前价格排序
            resistance_levels.sort()  # 最近的阻力在前
            return resistance_levels[:5] if resistance_levels else [current_price * 1.05]

        except Exception:
            return [current_price * 1.05]

    def _calculate_breakout_threshold(self, df: pd.DataFrame, current_price: float) -> float:
        """计算突破阈值"""
        try:
            # 使用ATR计算动态突破阈值
            atr = ta.atr(high=df['high'], low=df['low'], close=df['close'], length=14)
            if atr is not None and not atr.empty:
                return atr.iloc[-1] * 1.5  # 1.5倍ATR作为突破阈值
            else:
                return current_price * 0.02  # 2%作为默认突破阈值
        except Exception:
            return current_price * 0.02

    def _validate_support_levels(self, support_levels: list, current_price: float) -> list:
        """验证支撑位有效性"""
        return [level for level in support_levels if 0 < level < current_price * 0.99]

    def _validate_resistance_levels(self, resistance_levels: list, current_price: float) -> list:
        """验证阻力位有效性"""
        return [level for level in resistance_levels if level > current_price * 1.01]

    def _calculate_historical_volatility(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算历史波动率"""
        try:
            returns = df['close'].pct_change().dropna()

            # 不同时间窗口的波动率
            vol_5 = returns.rolling(5).std() * np.sqrt(252)
            vol_20 = returns.rolling(20).std() * np.sqrt(252)
            vol_60 = returns.rolling(60).std() * np.sqrt(252)

            return {
                'daily_volatility': returns.rolling(20).std().values,
                'weekly_volatility': vol_5.values,
                'monthly_volatility': vol_20.values,
                'quarterly_volatility': vol_60.values
            }
        except Exception:
            return {
                'daily_volatility': np.array([0.02]),
                'weekly_volatility': np.array([0.02]),
                'monthly_volatility': np.array([0.02]),
                'quarterly_volatility': np.array([0.02])
            }

    def _calculate_volatility_percentile(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """计算波动率百分位数"""
        try:
            vol_array = volatility_data['monthly_volatility']
            vol_array = vol_array[~np.isnan(vol_array)]
            if len(vol_array) < 2:
                return 50.0
            current_vol = vol_array[-1]
            percentile = (vol_array <= current_vol).sum() / len(vol_array) * 100
            return float(percentile)
        except Exception:
            return 50.0

    def _identify_volatility_trend(self, volatility_data: Dict[str, np.ndarray]) -> str:
        """识别波动率趋势"""
        try:
            vol_array = volatility_data['monthly_volatility']
            vol_array = vol_array[~np.isnan(vol_array)]
            if len(vol_array) < 10:
                return 'stable'

            recent_vol = vol_array[-5:].mean()
            past_vol = vol_array[-10:-5].mean()

            if recent_vol > past_vol * 1.1:
                return 'increasing'
            elif recent_vol < past_vol * 0.9:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return 'stable'

    def _forecast_volatility(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """波动率预测"""
        try:
            vol_array = volatility_data['monthly_volatility']
            vol_array = vol_array[~np.isnan(vol_array)]
            if len(vol_array) < 3:
                return 0.02
            return float(vol_array[-3:].mean())
        except Exception:
            return 0.02

    def _calculate_regime_probability(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """计算市场状态概率"""
        try:
            vol_array = volatility_data['monthly_volatility']
            vol_array = vol_array[~np.isnan(vol_array)]
            if len(vol_array) < 2:
                return 0.5
            current_vol = vol_array[-1]
            median_vol = np.median(vol_array)
            # 高于中位数表示高波动状态
            return float(min(1.0, max(0.0, current_vol / median_vol - 0.5)))
        except Exception:
            return 0.5

    def _get_default_volatility_analysis(self) -> Dict[str, Any]:
        """默认波动率分析"""
        return {
            'volatility_percentile': 50.0,
            'volatility_trend': 'stable',
            'volatility_forecast': 0.02,
            'regime_probability': 0.5,
            'leverage_adjusted_volatility': {
                'current_daily_vol': 2.0,
                'leverage_scenarios': {'2x': 4.0, '5x': 10.0, '10x': 20.0, '20x': 40.0},
                'risk_assessment': {'2x': 'low', '5x': 'moderate', '10x': 'high', '20x': 'extreme'}
            },
            'volatility_risk_level': 'moderate'
        }

    def generate_signal_metadata(self, df: pd.DataFrame, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成信号元数据 - 基于原版本但优化用于期货交易
        """
        try:
            # 计算期货特有的信号强度
            futures_strength = self._calculate_futures_signal_strength(signal_data, df)

            # 使用期货增强的置信度
            enhanced_confidence = futures_strength["enhanced_confidence"] / 100.0
            signal_strength = futures_strength["strength_level"]

            # 信号衰减时间（期货市场更快节奏）
            signal_decay_time = self._calculate_signal_decay_time(df, is_futures=True)

            # 信号可靠性评估
            signal_reliability = self._evaluate_signal_reliability(df, signal_data, signal_strength)

            # 确认状态
            confirmation_status = self._determine_confirmation_status(
                signal_strength, signal_reliability, signal_data
            )

            return {
                'signal_strength': signal_strength,
                'enhanced_confidence': futures_strength["enhanced_confidence"],
                'signal_decay_time': signal_decay_time,
                'signal_reliability': round(signal_reliability, 3),
                'confirmation_status': confirmation_status,
                'futures_factors': futures_strength["futures_factors"],
                'leverage_recommendation': futures_strength["leverage_recommendation"]
            }

        except Exception as e:
            print(f"Error generating signal metadata: {e}")
            return self._get_default_signal_metadata()

    def _calculate_signal_decay_time(self, df: pd.DataFrame, is_futures: bool = False) -> int:
        """计算信号衰减时间，期货版本调整为更快节奏"""
        try:
            if len(df) < 20:
                return 30 if is_futures else 60

            # 计算市场活跃度
            close = df['close'] if 'close' in df.columns else df['Close']
            volatility = close.pct_change().rolling(10).std().iloc[-1]

            if np.isnan(volatility):
                return 30 if is_futures else 60

            # 期货市场基础衰减时间更短
            base_time = 30 if is_futures else 60

            # 根据波动率调整
            if volatility > 0.03:  # 高波动
                return int(base_time * 0.7)  # 信号衰减更快
            elif volatility > 0.02:
                return int(base_time * 0.85)
            elif volatility < 0.01:  # 低波动
                return int(base_time * 1.5)  # 信号持续更久
            else:
                return base_time

        except Exception:
            return 30 if is_futures else 60

    def _evaluate_signal_reliability(self, df: pd.DataFrame, signal_data: Dict[str, Any], signal_strength: str) -> float:
        """评估信号可靠性"""
        try:
            # 市场环境评估
            market_score = self._assess_market_conditions(df)

            # 指标稳定性评估
            stability_score = self._assess_indicator_stability(df, signal_data)

            # 历史模式评估
            pattern_score = self._assess_historical_patterns(df)

            # 成交量确认评估
            volume_score = self._assess_volume_confirmation(df)

            # 期货特有：信号强度权重调整
            strength_weights = {
                'very_strong': 1.2,
                'strong': 1.1,
                'moderate': 1.0,
                'weak': 0.8,
                'very_weak': 0.6
            }

            strength_multiplier = strength_weights.get(signal_strength, 1.0)

            # 综合可靠性评分
            reliability = (
                market_score * 0.3 +
                stability_score * 0.25 +
                pattern_score * 0.25 +
                volume_score * 0.2
            ) * strength_multiplier

            return min(1.0, max(0.0, reliability))

        except Exception:
            return 0.5

    def _assess_market_conditions(self, df: pd.DataFrame) -> float:
        """评估市场环境条件"""
        try:
            if len(df) < 20:
                return 0.5

            close = df['close'] if 'close' in df.columns else df['Close']

            # 趋势强度
            sma_20 = close.rolling(20).mean()
            trend_strength = abs(close.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]

            # 波动率适中性
            volatility = close.pct_change().rolling(20).std().iloc[-1]
            vol_score = 1.0 - min(1.0, max(0.0, abs(volatility - 0.02) / 0.05))

            # 综合评分
            market_score = (
                min(1.0, trend_strength * 20) * 0.6 +  # 趋势强度
                vol_score * 0.4  # 波动率适中性
            )

            return market_score

        except Exception:
            return 0.5

    def _assess_indicator_stability(self, df: pd.DataFrame, signal_data: Dict[str, Any]) -> float:
        """评估技术指标稳定性"""
        try:
            # 检查各策略信号的一致性
            strategy_signals = signal_data.get('strategy_signals', {})

            if not strategy_signals:
                return 0.5

            # 信号一致性检查
            signals = []
            confidences = []

            for strategy, data in strategy_signals.items():
                signal = data.get('signal', 'neutral')
                confidence = data.get('confidence', 0) / 100.0

                if signal != 'neutral':
                    signals.append(signal)
                    confidences.append(confidence)

            if not signals:
                return 0.3  # 全部中性信号，稳定性较低

            # 计算信号一致性
            if len(set(signals)) == 1:
                consistency = 1.0  # 完全一致
            else:
                # 计算主导信号比例
                from collections import Counter
                signal_counts = Counter(signals)
                dominant_count = signal_counts.most_common(1)[0][1]
                consistency = dominant_count / len(signals)

            # 计算平均置信度
            avg_confidence = np.mean(confidences) if confidences else 0.5

            # 综合稳定性评分
            stability = consistency * 0.7 + avg_confidence * 0.3

            return stability

        except Exception:
            return 0.5

    def _assess_historical_patterns(self, df: pd.DataFrame) -> float:
        """评估历史模式匹配度"""
        try:
            if len(df) < 50:
                return 0.5

            close = df['close'] if 'close' in df.columns else df['Close']

            # 计算多重时间框架的移动平均线排列
            sma_5 = close.rolling(5).mean()
            sma_10 = close.rolling(10).mean()
            sma_20 = close.rolling(20).mean()

            current_price = close.iloc[-1]

            # 均线排列评分
            if (current_price > sma_5.iloc[-1] > sma_10.iloc[-1] > sma_20.iloc[-1]):
                ma_score = 1.0  # 完美多头排列
            elif (current_price < sma_5.iloc[-1] < sma_10.iloc[-1] < sma_20.iloc[-1]):
                ma_score = 1.0  # 完美空头排列
            elif (current_price > sma_20.iloc[-1]):
                ma_score = 0.7  # 部分多头
            elif (current_price < sma_20.iloc[-1]):
                ma_score = 0.7  # 部分空头
            else:
                ma_score = 0.3  # 混乱排列

            # 价格动量一致性
            returns = close.pct_change()
            recent_momentum = returns.iloc[-5:].mean()
            medium_momentum = returns.iloc[-20:].mean()

            momentum_consistency = 1.0 - abs(recent_momentum - medium_momentum) / max(abs(medium_momentum), 0.001)
            momentum_consistency = max(0.0, min(1.0, momentum_consistency))

            # 综合历史模式评分
            pattern_score = ma_score * 0.6 + momentum_consistency * 0.4

            return pattern_score

        except Exception:
            return 0.5

    def _assess_volume_confirmation(self, df: pd.DataFrame) -> float:
        """评估成交量确认"""
        try:
            if 'volume' not in df.columns and 'Volume' not in df.columns:
                return 0.6  # 无成交量数据时给予中等评分

            volume = df['volume'] if 'volume' in df.columns else df['Volume']

            if len(volume) < 20:
                return 0.6

            # 成交量趋势分析
            volume_sma = volume.rolling(20).mean()
            current_volume = volume.iloc[-1]
            avg_volume = volume_sma.iloc[-1]

            if avg_volume == 0:
                return 0.6

            volume_ratio = current_volume / avg_volume

            # 成交量确认评分
            if volume_ratio > 1.5:
                volume_score = 1.0  # 强成交量确认
            elif volume_ratio > 1.2:
                volume_score = 0.8  # 良好成交量确认
            elif volume_ratio > 0.8:
                volume_score = 0.6  # 正常成交量
            elif volume_ratio > 0.5:
                volume_score = 0.4  # 成交量不足
            else:
                volume_score = 0.2  # 成交量严重不足

            return volume_score

        except Exception:
            return 0.6

    def _determine_confirmation_status(self, signal_strength: str, signal_reliability: float, signal_data: Dict[str, Any]) -> str:
        """确定确认状态"""
        try:
            confidence = signal_data.get('confidence', 50) / 100.0

            # 期货交易的确认标准更严格
            if signal_strength in ['very_strong', 'strong'] and signal_reliability >= 0.7 and confidence >= 0.7:
                return 'confirmed'
            elif signal_strength in ['strong', 'moderate'] and signal_reliability >= 0.6 and confidence >= 0.6:
                return 'likely'
            elif signal_strength == 'moderate' and signal_reliability >= 0.5:
                return 'tentative'
            else:
                return 'unconfirmed'

        except Exception:
            return 'pending'

    def _get_default_signal_metadata(self) -> Dict[str, Any]:
        """默认信号元数据"""
        return {
            'signal_strength': 'moderate',
            'enhanced_confidence': 50.0,
            'signal_decay_time': 30,
            'signal_reliability': 0.5,
            'confirmation_status': 'pending',
            'futures_factors': {},
            'leverage_recommendation': 'conservative'
        }

    def cross_timeframe_analysis(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        跨时间框架综合分析 - 基于原版本，适用于期货交易
        """
        try:
            if not timeframe_signals:
                return self._get_default_cross_timeframe_analysis()

            # 时间框架共识度
            consensus_score = self._calculate_timeframe_consensus(timeframe_signals, weights={
                '1m': 0.05, '5m': 0.10, '15m': 0.15, '1h': 0.25, '4h': 0.25, '1d': 0.20
            })

            # 主导时间框架识别
            dominant_timeframe, dominant_confidence = self._identify_dominant_timeframe(
                timeframe_signals, weights={
                    '1m': 0.05, '5m': 0.10, '15m': 0.15, '1h': 0.25, '4h': 0.25, '1d': 0.20
                }
            )

            # 冲突区域识别
            conflict_areas = self._identify_conflict_areas(timeframe_signals)

            # 趋势对齐评估
            trend_alignment, alignment_score = self._assess_trend_alignment(
                timeframe_signals, weights={
                    '1m': 0.05, '5m': 0.10, '15m': 0.15, '1h': 0.25, '4h': 0.25, '1d': 0.20
                }
            )

            # 综合信号强度评估
            overall_strength, overall_confidence = self._evaluate_overall_signal_strength(
                timeframe_signals, consensus_score, alignment_score
            )

            return {
                'timeframe_consensus': round(consensus_score, 3),
                'dominant_timeframe': dominant_timeframe,
                'dominant_confidence': round(dominant_confidence, 3),
                'conflict_areas': conflict_areas,
                'trend_alignment': trend_alignment,
                'alignment_score': round(alignment_score, 3),
                'overall_signal_strength': overall_strength,
                'overall_confidence': round(overall_confidence, 3),
                # 期货特有的时间框架建议
                'futures_timeframe_recommendation': self._get_futures_timeframe_recommendation(
                    overall_strength, dominant_timeframe, conflict_areas
                )
            }

        except Exception as e:
            print(f"Error in cross-timeframe analysis: {e}")
            return self._get_default_cross_timeframe_analysis()

    def _get_futures_timeframe_recommendation(self, overall_strength: str,
                                           dominant_timeframe: str,
                                           conflict_areas: list) -> Dict[str, Any]:
        """期货特有的时间框架交易建议"""
        try:
            # 根据信号强度和主导时间框架给出建议
            if overall_strength in ['very_strong', 'strong']:
                if dominant_timeframe in ['1d', '4h']:
                    recommendation = {
                        'primary_entry': dominant_timeframe,
                        'confirmation': '1h' if dominant_timeframe == '4h' else '4h',
                        'stop_loss': '15m',
                        'position_size': 'aggressive' if overall_strength == 'very_strong' else 'moderate',
                        'hold_duration': 'long_term'
                    }
                elif dominant_timeframe in ['1h', '15m']:
                    recommendation = {
                        'primary_entry': dominant_timeframe,
                        'confirmation': '5m' if dominant_timeframe == '15m' else '15m',
                        'stop_loss': '5m',
                        'position_size': 'moderate',
                        'hold_duration': 'medium_term'
                    }
                else:
                    recommendation = {
                        'primary_entry': dominant_timeframe,
                        'confirmation': '1m',
                        'stop_loss': '1m',
                        'position_size': 'conservative',
                        'hold_duration': 'short_term'
                    }
            else:
                recommendation = {
                    'primary_entry': '1h',  # 默认中期框架
                    'confirmation': '15m',
                    'stop_loss': '5m',
                    'position_size': 'conservative',
                    'hold_duration': 'wait_for_clarity'
                }

            # 考虑冲突区域调整
            if len(conflict_areas) > 2:
                recommendation['position_size'] = 'minimal'
                recommendation['risk_warning'] = 'high_conflict_detected'

            return recommendation

        except Exception:
            return {
                'primary_entry': '1h',
                'confirmation': '15m',
                'stop_loss': '5m',
                'position_size': 'conservative',
                'hold_duration': 'medium_term'
            }

    # 以下方法继承自原版本
    def _calculate_timeframe_consensus(self, timeframe_signals: Dict[str, Dict[str, Any]],
                                     weights: Dict[str, float]) -> float:
        """计算时间框架共识度"""
        try:
            total_weight = 0
            consensus_sum = 0

            # 收集所有信号
            all_signals = []
            weighted_confidences = []

            for timeframe, data in timeframe_signals.items():
                weight = weights.get(timeframe, 0.1)
                signal = data.get('signal', 'neutral')
                confidence = data.get('confidence', 50) / 100.0

                all_signals.append(signal)
                weighted_confidences.append(confidence * weight)
                total_weight += weight

            if not all_signals or total_weight == 0:
                return 0.5

            # 计算信号一致性
            from collections import Counter
            signal_counts = Counter(all_signals)

            if len(signal_counts) == 1:
                # 完全一致
                consensus_base = 1.0
            else:
                # 计算主导信号的比例
                most_common_count = signal_counts.most_common(1)[0][1]
                consensus_base = most_common_count / len(all_signals)

            # 加权置信度
            avg_weighted_confidence = sum(weighted_confidences) / total_weight

            # 综合共识度
            consensus = consensus_base * 0.7 + avg_weighted_confidence * 0.3

            return min(1.0, max(0.0, consensus))

        except Exception:
            return 0.5

    def _identify_dominant_timeframe(self, timeframe_signals: Dict[str, Dict[str, Any]],
                                   weights: Dict[str, float]) -> tuple:
        """识别主导时间框架"""
        try:
            timeframe_scores = {}

            for timeframe, data in timeframe_signals.items():
                weight = weights.get(timeframe, 0.1)
                confidence = data.get('confidence', 50) / 100.0
                signal = data.get('signal', 'neutral')

                # 非中性信号获得额外分数
                signal_bonus = 0.2 if signal != 'neutral' else 0.0

                score = (confidence + signal_bonus) * weight
                timeframe_scores[timeframe] = score

            if not timeframe_scores:
                return '1h', 0.5

            # 找到得分最高的时间框架
            dominant = max(timeframe_scores.items(), key=lambda x: x[1])

            return dominant[0], min(1.0, dominant[1])

        except Exception:
            return '1h', 0.5

    def _identify_conflict_areas(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> list:
        """识别冲突区域"""
        try:
            conflicts = []
            timeframes = list(timeframe_signals.keys())

            for i, tf1 in enumerate(timeframes):
                for tf2 in timeframes[i+1:]:
                    signal1 = timeframe_signals[tf1].get('signal', 'neutral')
                    signal2 = timeframe_signals[tf2].get('signal', 'neutral')

                    conf1 = timeframe_signals[tf1].get('confidence', 50)
                    conf2 = timeframe_signals[tf2].get('confidence', 50)

                    if self._check_signal_conflict(signal1, signal2, conf1, conf2):
                        conflicts.append({
                            'timeframes': [tf1, tf2],
                            'signals': [signal1, signal2],
                            'confidences': [conf1, conf2],
                            'conflict_severity': self._calculate_conflict_severity(signal1, signal2, conf1, conf2)
                        })

            # 按冲突严重程度排序
            conflicts.sort(key=lambda x: x['conflict_severity'], reverse=True)

            return conflicts[:5]  # 返回最严重的5个冲突

        except Exception:
            return []

    def _check_signal_conflict(self, signal1: str, signal2: str, conf1: float, conf2: float) -> bool:
        """检查两个信号是否冲突"""
        try:
            # 定义冲突的信号对
            conflict_pairs = [
                ('bullish', 'bearish'),
                ('bearish', 'bullish')
            ]

            # 检查是否为冲突对，且置信度都不低
            signal_pair = (signal1, signal2)
            reverse_pair = (signal2, signal1)

            is_conflicting = (signal_pair in conflict_pairs or reverse_pair in conflict_pairs)
            both_confident = min(conf1, conf2) > 40  # 两个信号的置信度都不太低

            return is_conflicting and both_confident

        except Exception:
            return False

    def _calculate_conflict_severity(self, signal1: str, signal2: str, conf1: float, conf2: float) -> float:
        """计算冲突严重程度"""
        try:
            # 基础冲突强度
            if {signal1, signal2} == {'bullish', 'bearish'}:
                base_severity = 1.0  # 最高冲突
            elif 'neutral' in [signal1, signal2]:
                base_severity = 0.3  # 中性信号冲突较轻
            else:
                base_severity = 0.0

            # 置信度影响冲突严重程度
            confidence_factor = (conf1 + conf2) / 200.0  # 归一化到0-1

            severity = base_severity * confidence_factor

            return min(1.0, max(0.0, severity))

        except Exception:
            return 0.0

    def _assess_trend_alignment(self, timeframe_signals: Dict[str, Dict[str, Any]],
                              weights: Dict[str, float]) -> tuple:
        """评估趋势对齐度"""
        try:
            # 按时间框架长度分组
            short_term = ['1m', '5m', '15m']
            medium_term = ['1h', '4h']
            long_term = ['1d', '1w']

            groups = {
                'short': short_term,
                'medium': medium_term,
                'long': long_term
            }

            group_signals = {}

            # 计算各组的主导信号
            for group_name, timeframes in groups.items():
                group_votes = []
                group_confidences = []

                for tf in timeframes:
                    if tf in timeframe_signals:
                        signal = timeframe_signals[tf].get('signal', 'neutral')
                        confidence = timeframe_signals[tf].get('confidence', 50)
                        weight = weights.get(tf, 0.1)

                        group_votes.append(signal)
                        group_confidences.append(confidence * weight)

                if group_votes:
                    # 找到该组的主导信号
                    from collections import Counter
                    signal_counts = Counter(group_votes)
                    dominant_signal = signal_counts.most_common(1)[0][0]
                    avg_confidence = sum(group_confidences) / len(group_confidences)

                    group_signals[group_name] = {
                        'signal': dominant_signal,
                        'confidence': avg_confidence
                    }

            # 评估组间对齐度
            if len(group_signals) < 2:
                return 'insufficient_data', 0.5

            signals = [data['signal'] for data in group_signals.values()]
            confidences = [data['confidence'] for data in group_signals.values()]

            # 检查信号一致性
            unique_signals = set(signals)

            if len(unique_signals) == 1:
                if signals[0] == 'neutral':
                    alignment = 'neutral'
                    score = 0.6  # 全部中性，适中对齐
                else:
                    alignment = 'strong_aligned'
                    score = 0.9  # 强对齐
            elif len(unique_signals) == 2:
                if 'neutral' in unique_signals:
                    alignment = 'partial_aligned'
                    score = 0.7  # 部分对齐
                else:
                    alignment = 'conflicted'
                    score = 0.2  # 冲突
            else:
                alignment = 'mixed'
                score = 0.4  # 混合状态

            # 置信度调整
            avg_confidence = sum(confidences) / len(confidences) if confidences else 50
            confidence_factor = avg_confidence / 100.0
            final_score = score * confidence_factor

            return alignment, min(1.0, max(0.0, final_score))

        except Exception:
            return 'mixed', 0.5

    def _evaluate_overall_signal_strength(self, timeframe_signals: Dict[str, Dict[str, Any]],
                                        consensus_score: float, alignment_score: float) -> tuple:
        """评估整体信号强度"""
        try:
            # 收集所有非中性信号
            active_signals = []
            active_confidences = []

            for timeframe, data in timeframe_signals.items():
                signal = data.get('signal', 'neutral')
                confidence = data.get('confidence', 50)

                if signal != 'neutral':
                    active_signals.append(signal)
                    active_confidences.append(confidence)

            if not active_signals:
                return 'neutral', 50.0

            # 计算信号方向一致性
            from collections import Counter
            signal_counts = Counter(active_signals)
            dominant_signal = signal_counts.most_common(1)[0][0]
            dominant_count = signal_counts.most_common(1)[0][1]

            direction_consistency = dominant_count / len(active_signals)

            # 平均置信度
            avg_confidence = sum(active_confidences) / len(active_confidences)

            # 综合强度评估
            base_strength = (
                direction_consistency * 0.4 +
                consensus_score * 0.3 +
                alignment_score * 0.3
            )

            # 转换为强度等级
            if base_strength >= 0.8 and avg_confidence >= 75:
                strength_level = 'very_strong'
                final_confidence = min(95, avg_confidence * 1.2)
            elif base_strength >= 0.65 and avg_confidence >= 65:
                strength_level = 'strong'
                final_confidence = min(90, avg_confidence * 1.1)
            elif base_strength >= 0.5 and avg_confidence >= 50:
                strength_level = 'moderate'
                final_confidence = avg_confidence
            elif base_strength >= 0.35:
                strength_level = 'weak'
                final_confidence = avg_confidence * 0.9
            else:
                strength_level = 'very_weak'
                final_confidence = avg_confidence * 0.8

            return strength_level, final_confidence

        except Exception:
            return 'moderate', 50.0

    def _get_default_cross_timeframe_analysis(self) -> Dict[str, Any]:
        """默认跨时间框架分析"""
        return {
            'timeframe_consensus': 0.5,
            'dominant_timeframe': '1h',
            'dominant_confidence': 0.5,
            'conflict_areas': [],
            'trend_alignment': 'mixed',
            'alignment_score': 0.5,
            'overall_signal_strength': 'moderate',
            'overall_confidence': 50.0,
            'futures_timeframe_recommendation': {
                'primary_entry': '1h',
                'confirmation': '15m',
                'stop_loss': '5m',
                'position_size': 'conservative',
                'hold_duration': 'medium_term'
            }
        }

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        期货版本的MACD策略主函数
        - 将所有bullish/bearish信号转换为long/short
        - 增加期货特有的分析指标
        - 保持与现货版本的兼容性
        """
        data = state['data']
        data['name'] = "FuturesMacdStrategy"

        data = state.get("data", {})
        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        # Initialize analysis for each ticker
        technical_analysis = {}
        for ticker in tickers:
            technical_analysis[ticker] = {}

        # 期货特有的策略权重（更注重动量和波动率）
        strategy_weights = {
            "trend": 0.30,           # 期货更重视趋势
            "mean_reversion": 0.15,  # 期货较少均值回归
            "momentum": 0.30,        # 期货重视动量
            "volatility": 0.15,      # 期货重视波动率
            "stat_arb": 0.10,        # 期货较少统计套利
        }

        for ticker in tickers:
            # 收集该ticker所有时间框架的信号数据
            timeframe_signals = {}

            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())

                if df.empty:
                    continue

                # 计算各种策略信号（使用原始信号）
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

                # 转换为期货信号语义
                futures_signal = self._convert_signal_to_futures(combined_signal["signal"])

                # 构建期货版本的策略信号结构
                futures_strategy_signals = {
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
                try:
                    # ATR值计算
                    atr_values = self.calculate_atr_values(df)

                    # 关键价位识别
                    price_levels = self.identify_price_levels(df)

                    # 波动率深度分析（包含期货特有指标）
                    volatility_analysis = self.analyze_volatility_depth(df)

                    # 准备完整的信号数据用于元数据生成
                    current_signal_data = {
                        "signal": futures_signal,
                        "confidence": round(combined_signal["confidence"] * 100),
                        "strategy_signals": futures_strategy_signals
                    }

                    # 生成期货特有的信号元数据
                    signal_metadata = self.generate_signal_metadata(df, current_signal_data)

                except Exception as e:
                    print(f"Error calculating futures analysis for {ticker}_{interval.value}: {e}")
                    # 使用默认值
                    atr_values = {'atr_14': 0.0, 'atr_28': 0.0, 'atr_percentile': 0.0}
                    price_levels = self._get_default_levels()
                    volatility_analysis = self._get_default_volatility_analysis()
                    signal_metadata = self._get_default_signal_metadata()

                # 生成该时间框架的完整分析结果
                technical_analysis[ticker][interval.value] = {
                    # === 期货信号字段 ===
                    "signal": futures_signal,  # long/short/neutral
                    "confidence": round(combined_signal["confidence"] * 100),
                    "strategy_signals": futures_strategy_signals,

                    # === 期货特有增强字段 ===
                    "atr_values": atr_values,
                    "price_levels": price_levels,
                    "volatility_analysis": volatility_analysis,
                    "signal_metadata": signal_metadata,

                    # === 期货交易建议 ===
                    "futures_trading_recommendation": {
                        "direction": futures_signal,
                        "leverage_recommendation": signal_metadata.get("leverage_recommendation", "conservative"),
                        "risk_level": volatility_analysis.get("volatility_risk_level", "moderate"),
                        "suggested_timeframe": signal_metadata.get("futures_factors", {}).get("dominant_timeframe", interval.value)
                    }
                }

                # 收集信号数据用于跨时间框架分析（使用期货信号）
                timeframe_signals[interval.value] = {
                    "signal": futures_signal,
                    "confidence": round(combined_signal["confidence"] * 100),
                    "strategy_signals": futures_strategy_signals
                }

            # 跨时间框架综合分析（期货版本）
            try:
                if timeframe_signals:
                    cross_timeframe_result = self.cross_timeframe_analysis(timeframe_signals)
                else:
                    cross_timeframe_result = self._get_default_cross_timeframe_analysis()

                technical_analysis[ticker]["cross_timeframe_analysis"] = cross_timeframe_result

            except Exception as e:
                print(f"Error in futures cross-timeframe analysis for {ticker}: {e}")
                technical_analysis[ticker]["cross_timeframe_analysis"] = self._get_default_cross_timeframe_analysis()

        # 创建期货技术分析师消息
        message = HumanMessage(
            content=json.dumps(technical_analysis),
            name="futures_technical_analyst_agent",
        )

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "期货技术分析师 (Futures Technical Analyst)")

        # 添加信号到分析信号列表
        state["data"]["analyst_signals"]["futures_macd_strategy_agent"] = technical_analysis

        return {
            "messages": [message],
            "data": data,
        }