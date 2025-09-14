from typing import Dict, Any
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from langchain_core.messages import HumanMessage
from src.graph import AgentState, BaseNode, show_agent_reasoning
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
    return safe_std(clean_arr, ddof=ddof)

def safe_mean(arr):
    """安全的均值计算，处理NaN值和空数组"""
    if len(arr) == 0:
        return 0.0
    clean_arr = np.array(arr)[~np.isnan(arr)]
    if len(clean_arr) == 0:
        return 0.0
    return safe_mean(clean_arr)


class RSIStrategy(BaseNode):
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
            # 在出错时返回默认值
            print(f"ATR calculation error: {e}")
            return {
                'atr_14': 0.0,
                'atr_28': 0.0,
                'atr_percentile': 0.0
            }

    def identify_price_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        识别关键价格水平：支撑位、阻力位、枢轴点和突破临界点
        
        使用经典技术分析算法：
        - 支撑阻力位：基于历史高低点、成交量确认
        - 枢轴点：使用标准枢轴点公式 (H+L+C)/3
        - 突破临界点：基于波动率和价格区间
        
        Args:
            df: 包含OHLC和成交量数据的DataFrame，必须包含'high', 'low', 'close', 'volume'列
            
        Returns:
            Dict[str, Any]: 包含support_levels, resistance_levels, pivot_point, breakout_threshold的字典
        """
        if df.empty or len(df) < 50:  # 至少需要50个数据点进行可靠分析
            return {
                'support_levels': [0.0, 0.0, 0.0],
                'resistance_levels': [0.0, 0.0, 0.0],
                'pivot_point': 0.0,
                'breakout_threshold': 0.0
            }
        
        try:
            # 标准化列名（处理大小写）
            df_normalized = self._normalize_columns(df)
            
            if not self._validate_required_columns(df_normalized):
                return self._get_default_levels()
            
            # 获取当前价格
            current_price = float(df_normalized['close'].iloc[-1])
            
            # 1. 计算枢轴点（使用标准公式） - 完全内联避免递归
            pivot_point = 0.0
            try:
                # 获取最近的高点、低点、收盘价
                recent_high = float(df_normalized['high'].tail(20).max())
                recent_low = float(df_normalized['low'].tail(20).min())
                recent_close = float(df_normalized['close'].iloc[-1])
                
                # 标准枢轴点公式
                pivot_point = (recent_high + recent_low + recent_close) / 3.0
            except:
                pivot_point = current_price
            
            # 2. 识别支撑和阻力位 - 完全内联避免递归
            support_levels = [0.0, 0.0, 0.0]
            resistance_levels = [0.0, 0.0, 0.0]
            
            try:
                # 寻找局部高点和低点
                high_prices = df_normalized['high'].values
                low_prices = df_normalized['low'].values
                close_prices = df_normalized['close'].values
                
                if len(high_prices) >= 50:
                    # 寻找局部极值点
                    window = 5  # 窗口大小
                    local_highs = []
                    local_lows = []
                    
                    for i in range(window, len(high_prices) - window):
                        # 检查局部高点
                        is_local_high = True
                        for j in range(i - window, i + window + 1):
                            if j != i and high_prices[j] >= high_prices[i]:
                                is_local_high = False
                                break
                        if is_local_high:
                            local_highs.append(high_prices[i])
                        
                        # 检查局部低点
                        is_local_low = True
                        for j in range(i - window, i + window + 1):
                            if j != i and low_prices[j] <= low_prices[i]:
                                is_local_low = False
                                break
                        if is_local_low:
                            local_lows.append(low_prices[i])
                    
                    # 选择最相关的支撑阻力位
                    if local_lows:
                        # 按照与当前价格的接近程度排序
                        local_lows.sort(key=lambda x: abs(x - current_price))
                        support_levels = local_lows[:3]
                        # 确保有3个值
                        while len(support_levels) < 3:
                            support_levels.append(support_levels[-1] if support_levels else current_price * 0.95)
                    else:
                        # 使用简单的百分比支撑位
                        support_levels = [
                            current_price * 0.98,
                            current_price * 0.95,
                            current_price * 0.92
                        ]
                    
                    if local_highs:
                        # 按照与当前价格的接近程度排序
                        local_highs.sort(key=lambda x: abs(x - current_price))
                        resistance_levels = local_highs[:3]
                        # 确保有3个值
                        while len(resistance_levels) < 3:
                            resistance_levels.append(resistance_levels[-1] if resistance_levels else current_price * 1.05)
                    else:
                        # 使用简单的百分比阻力位
                        resistance_levels = [
                            current_price * 1.02,
                            current_price * 1.05,
                            current_price * 1.08
                        ]
                else:
                    # 数据不足，使用简单方法
                    support_levels = [
                        current_price * 0.98,
                        current_price * 0.95,
                        current_price * 0.92
                    ]
                    resistance_levels = [
                        current_price * 1.02,
                        current_price * 1.05,
                        current_price * 1.08
                    ]
                
            except:
                # 计算失败，使用简单的百分比方法
                support_levels = [
                    current_price * 0.98,
                    current_price * 0.95,
                    current_price * 0.92
                ]
                resistance_levels = [
                    current_price * 1.02,
                    current_price * 1.05,
                    current_price * 1.08
                ]
            
            # 3. 计算突破临界点 - 完全内联避免递归
            breakout_threshold = 0.0
            try:
                # 基于ATR计算突破阈值
                if len(df_normalized) >= 14:
                    import pandas_ta as ta
                    atr = ta.atr(df_normalized['high'], df_normalized['low'], df_normalized['close'], length=14)
                    if not atr.empty and not pd.isna(atr.iloc[-1]):
                        current_atr = float(atr.iloc[-1])
                        breakout_threshold = current_atr * 1.5  # 1.5倍ATR作为突破阈值
                    else:
                        breakout_threshold = current_price * 0.02  # 默认2%
                else:
                    breakout_threshold = current_price * 0.02  # 默认2%
            except:
                breakout_threshold = current_price * 0.02  # 默认2%
            
            # 4. 验证价位逻辑一致性 - 完全内联避免递归
            # 确保支撑位都低于当前价格
            validated_support = []
            for level in support_levels:
                if level < current_price:
                    validated_support.append(level)
                else:
                    validated_support.append(current_price * 0.98)  # 调整为合理值
            
            # 确保阻力位都高于当前价格
            validated_resistance = []
            for level in resistance_levels:
                if level > current_price:
                    validated_resistance.append(level)
                else:
                    validated_resistance.append(current_price * 1.02)  # 调整为合理值
            
            return {
                'support_levels': [round(float(level), 8) for level in validated_support],
                'resistance_levels': [round(float(level), 8) for level in validated_resistance],
                'pivot_point': round(float(pivot_point), 8),
                'breakout_threshold': round(float(breakout_threshold), 8)
            }
            
        except Exception as e:
            print(f"Price levels identification error: {e}")
            return self._get_default_levels()

    def analyze_volatility_depth(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        深度波动率分析功能
        
        基于历史OHLC数据进行全面的波动率分析，包括：
        1. 波动率历史百分位数计算
        2. 波动率趋势识别（增加/减少/稳定）
        3. 波动率预期值预测
        4. 当前波动率状态概率评估
        
        Args:
            df: 包含OHLC数据的DataFrame，必须包含'high', 'low', 'close'列
            
        Returns:
            Dict[str, Any]: 包含以下字段的字典：
                - volatility_percentile: float - 波动率历史百分位(0-100)
                - volatility_trend: str - "increasing|decreasing|stable" 
                - volatility_forecast: float - 预期波动率
                - regime_probability: float - 当前波动率状态概率(0-1)
        """
        if df.empty or len(df) < 60:  # 至少需要60个数据点进行可靠分析
            return {
                'volatility_percentile': 50.0,
                'volatility_trend': 'stable',
                'volatility_forecast': 0.0,
                'regime_probability': 0.5
            }
        
        try:
            # 标准化列名
            df_normalized = self._normalize_columns(df)
            
            if not self._validate_required_columns(df_normalized):
                return self._get_default_volatility_analysis()
            
            # 1. 计算多个时间窗口的历史波动率 - 完全内联避免递归
            volatility_data = {}
            try:
                # 计算价格变化率
                returns = df_normalized['close'].pct_change().dropna()
                
                if len(returns) >= 30:
                    # 计算不同窗口的波动率
                    windows = [10, 21, 63]  # 约2周、1月、3月
                    volatilities = {}
                    
                    for window in windows:
                        if len(returns) >= window:
                            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)  # 年化
                            volatilities[f'vol_{window}'] = rolling_vol.dropna()
                    
                    volatility_data = volatilities
                else:
                    # 数据不足，使用简单方法
                    overall_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.05
                    volatility_data = {'vol_21': pd.Series([overall_vol] * len(returns))}
            except:
                # 计算失败，使用默认值
                volatility_data = {'vol_21': pd.Series([0.05] * 30)}
            
            # 2. 计算波动率百分位数 - 完全内联避免递归
            volatility_percentile = 50.0
            try:
                if 'vol_21' in volatility_data and len(volatility_data['vol_21']) > 0:
                    current_vol = volatility_data['vol_21'].iloc[-1]
                    historical_vols = volatility_data['vol_21']
                    
                    # 计算百分位数
                    if len(historical_vols) >= 10:
                        percentile = (historical_vols <= current_vol).sum() / len(historical_vols) * 100
                        volatility_percentile = float(percentile)
                    else:
                        volatility_percentile = 50.0
            except:
                volatility_percentile = 50.0
            
            # 3. 识别波动率趋势 - 完全内联避免递归
            volatility_trend = 'stable'
            try:
                if 'vol_21' in volatility_data and len(volatility_data['vol_21']) >= 10:
                    recent_vols = volatility_data['vol_21'].tail(10)
                    
                    if len(recent_vols) >= 5:
                        # 计算趋势斜率
                        x = np.arange(len(recent_vols))
                        y = recent_vols.values
                        
                        if len(x) == len(y) and len(x) > 1:
                            # 简单线性趋势
                            slope = (y[-1] - y[0]) / (len(y) - 1)
                            relative_slope = slope / np.mean(y) if np.mean(y) > 0 else 0
                            
                            if relative_slope > 0.05:  # 5%以上的相对增长
                                volatility_trend = 'increasing'
                            elif relative_slope < -0.05:  # 5%以上的相对下降
                                volatility_trend = 'decreasing'
                            else:
                                volatility_trend = 'stable'
            except:
                volatility_trend = 'stable'
            
            # 4. 预测未来波动率 - 完全内联避免递归
            volatility_forecast = 0.0
            try:
                if 'vol_21' in volatility_data and len(volatility_data['vol_21']) >= 5:
                    recent_vols = volatility_data['vol_21'].tail(5)
                    
                    # 简单的移动平均预测
                    volatility_forecast = float(recent_vols.mean())
                else:
                    volatility_forecast = 0.05  # 默认5%年化波动率
            except:
                volatility_forecast = 0.05
            
            # 5. 计算波动率状态概率 - 完全内联避免递归
            regime_probability = 0.5
            try:
                if 'vol_21' in volatility_data and len(volatility_data['vol_21']) >= 20:
                    current_vol = volatility_data['vol_21'].iloc[-1]
                    historical_vols = volatility_data['vol_21']
                    
                    # 计算当前波动率相对于历史分布的位置
                    mean_vol = historical_vols.mean()
                    std_vol = historical_vols.std()
                    
                    if std_vol > 0:
                        # 计算标准化位置
                        z_score = (current_vol - mean_vol) / std_vol
                        
                        # 使用累积分布函数估计概率
                        from scipy import stats
                        regime_probability = float(stats.norm.cdf(z_score))
                    else:
                        regime_probability = 0.5
            except:
                regime_probability = 0.5
            
            return {
                'volatility_percentile': round(float(volatility_percentile), 2),
                'volatility_trend': volatility_trend,
                'volatility_forecast': round(float(volatility_forecast), 6),
                'regime_probability': round(float(regime_probability), 4)
            }
            
        except Exception as e:
            print(f"Volatility depth analysis error: {e}")
            return self._get_default_volatility_analysis()
    
    def _calculate_historical_volatility(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        计算多个时间窗口的历史波动率
        
        使用对数收益率计算真实波动率，考虑多个时间框架：
        - 14日短期波动率
        - 30日中期波动率  
        - 60日长期波动率
        """
        close_prices = df['close'].values
        
        # 计算对数收益率
        log_returns = np.diff(np.log(close_prices))
        
        # 定义时间窗口
        windows = [14, 30, 60]
        volatility_data = {}
        
        for window in windows:
            # 滚动窗口计算波动率（年化）
            rolling_vol = []
            for i in range(window, len(log_returns) + 1):
                window_returns = log_returns[i-window:i]
                # 计算标准差并年化（假设252个交易日）
                vol = safe_std(window_returns, ddof=1) * np.sqrt(252)
                rolling_vol.append(vol)
            
            volatility_data[f'vol_{window}'] = np.array(rolling_vol)
        
        # 添加当前波动率
        if len(log_returns) >= 14:
            current_vol = safe_std(log_returns[-14:], ddof=1) * np.sqrt(252)
            volatility_data['current_vol'] = current_vol
        else:
            volatility_data['current_vol'] = 0.0
        
        return volatility_data
    
    def _calculate_volatility_percentile(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """
        计算当前波动率在历史分布中的百分位数
        
        使用30日波动率作为参考基准
        """
        if 'vol_30' not in volatility_data or len(volatility_data['vol_30']) == 0:
            return 50.0
        
        vol_30 = volatility_data['vol_30']
        current_vol = volatility_data.get('current_vol', vol_30[-1])
        
        # 计算当前波动率在历史分布中的百分位
        if len(vol_30) > 1:
            percentile = np.sum(vol_30 <= current_vol) / len(vol_30) * 100
        else:
            percentile = 50.0
        
        # 确保百分位在合理范围内
        return max(0.0, min(100.0, percentile))
    
    def _identify_volatility_trend(self, volatility_data: Dict[str, np.ndarray]) -> str:
        """
        识别波动率趋势
        
        通过比较不同时间窗口的波动率和线性回归分析来判断趋势
        """
        if 'vol_14' not in volatility_data or len(volatility_data['vol_14']) < 10:
            return 'stable'
        
        vol_14 = volatility_data['vol_14']
        vol_30 = volatility_data.get('vol_30', vol_14)
        
        # 方法1: 比较短期和中期波动率
        if len(vol_14) >= 5 and len(vol_30) >= 5:
            recent_short = safe_mean(vol_14[-5:])
            recent_medium = safe_mean(vol_30[-5:])
            
            # 计算相对差异
            relative_diff = (recent_short - recent_medium) / recent_medium
            
            if relative_diff > 0.1:  # 10%以上差异认为是增加
                trend_signal_1 = 'increasing'
            elif relative_diff < -0.1:
                trend_signal_1 = 'decreasing'
            else:
                trend_signal_1 = 'stable'
        else:
            trend_signal_1 = 'stable'
        
        # 方法2: 线性回归分析最近20个数据点的趋势
        if len(vol_14) >= 20:
            recent_vol = vol_14[-20:]
            x = np.arange(len(recent_vol))
            
            # 使用最小二乘法计算趋势斜率
            slope = np.polyfit(x, recent_vol, 1)[0]
            
            # 标准化斜率（相对于平均波动率）
            avg_vol = safe_mean(recent_vol)
            normalized_slope = slope / avg_vol if avg_vol > 0 else 0
            
            if normalized_slope > 0.02:  # 斜率大于2%认为是增加趋势
                trend_signal_2 = 'increasing'
            elif normalized_slope < -0.02:
                trend_signal_2 = 'decreasing'
            else:
                trend_signal_2 = 'stable'
        else:
            trend_signal_2 = 'stable'
        
        # 综合两种方法的结果
        if trend_signal_1 == trend_signal_2:
            return trend_signal_1
        elif trend_signal_1 == 'stable' or trend_signal_2 == 'stable':
            return 'stable'
        else:
            # 如果两种方法结果不一致，倾向于保守判断
            return 'stable'
    
    def _forecast_volatility(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """
        预测未来波动率
        
        使用GARCH类型的方法和历史模式分析
        """
        if 'vol_30' not in volatility_data or len(volatility_data['vol_30']) < 30:
            return volatility_data.get('current_vol', 0.0)
        
        vol_30 = volatility_data['vol_30']
        current_vol = volatility_data.get('current_vol', vol_30[-1])
        
        # 方法1: 指数加权移动平均(EWMA)
        # 使用RiskMetrics建议的lambda=0.94
        lambda_ewma = 0.94
        ewma_vol = vol_30[0]
        
        for vol in vol_30[1:]:
            ewma_vol = lambda_ewma * ewma_vol + (1 - lambda_ewma) * vol**2
        
        ewma_forecast = np.sqrt(ewma_vol)
        
        # 方法2: 历史平均回归
        historical_mean = safe_mean(vol_30)
        mean_reversion_speed = 0.1  # 均值回归速度
        mean_reversion_forecast = current_vol + mean_reversion_speed * (historical_mean - current_vol)
        
        # 方法3: 趋势延续
        if len(vol_30) >= 10:
            recent_trend = safe_mean(vol_30[-5:]) - safe_mean(vol_30[-10:-5])
            trend_forecast = current_vol + 0.5 * recent_trend  # 50%的趋势延续
        else:
            trend_forecast = current_vol
        
        # 综合预测（权重分配）
        weights = [0.4, 0.4, 0.2]  # EWMA, 均值回归, 趋势延续
        forecast = (weights[0] * ewma_forecast + 
                   weights[1] * mean_reversion_forecast + 
                   weights[2] * trend_forecast)
        
        # 确保预测值在合理范围内（年化波动率通常在5%-200%之间）
        forecast = max(0.05, min(2.0, forecast))
        
        return forecast
    
    def _calculate_regime_probability(self, volatility_data: Dict[str, np.ndarray]) -> float:
        """
        计算当前波动率状态概率
        
        基于统计分布分析，判断当前处于高波动率或低波动率状态的概率
        """
        if 'vol_30' not in volatility_data or len(volatility_data['vol_30']) < 30:
            return 0.5
        
        vol_30 = volatility_data['vol_30']
        current_vol = volatility_data.get('current_vol', vol_30[-1])
        
        # 使用分位数定义波动率状态
        # 低波动率：<33%分位数
        # 中波动率：33%-67%分位数  
        # 高波动率：>67%分位数
        
        percentiles = np.percentile(vol_30, [33, 67])
        low_vol_threshold = percentiles[0]
        high_vol_threshold = percentiles[1]
        
        # 计算当前状态概率
        if current_vol <= low_vol_threshold:
            # 低波动率状态概率
            # 使用正态分布近似计算概率
            mean_low = safe_mean(vol_30[vol_30 <= low_vol_threshold])
            std_low = safe_std(vol_30[vol_30 <= low_vol_threshold]) if len(vol_30[vol_30 <= low_vol_threshold]) > 1 else low_vol_threshold * 0.1
            
            # 计算在低波动率分布中的概率密度
            if std_low > 0:
                z_score = abs(current_vol - mean_low) / std_low
                probability = np.exp(-0.5 * z_score**2)  # 简化的正态分布概率
            else:
                probability = 1.0
                
        elif current_vol >= high_vol_threshold:
            # 高波动率状态概率
            mean_high = safe_mean(vol_30[vol_30 >= high_vol_threshold])
            std_high = safe_std(vol_30[vol_30 >= high_vol_threshold]) if len(vol_30[vol_30 >= high_vol_threshold]) > 1 else high_vol_threshold * 0.1
            
            if std_high > 0:
                z_score = abs(current_vol - mean_high) / std_high
                probability = np.exp(-0.5 * z_score**2)
            else:
                probability = 1.0
                
        else:
            # 中等波动率状态
            # 计算距离边界的相对位置
            mid_range = high_vol_threshold - low_vol_threshold
            relative_position = (current_vol - low_vol_threshold) / mid_range
            
            # 中间状态的概率（距离中心越近概率越高）
            probability = 1.0 - 2 * abs(relative_position - 0.5)
        
        # 确保概率在0-1范围内
        probability = max(0.0, min(1.0, probability))
        
        return probability
    
    def _get_default_volatility_analysis(self) -> Dict[str, Any]:
        """返回默认的波动率分析结果"""
        return {
            'volatility_percentile': 50.0,
            'volatility_trend': 'stable', 
            'volatility_forecast': 0.0,
            'regime_probability': 0.5
        }
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化DataFrame列名"""
        df_copy = df.copy()
        
        # 创建列名映射（不区分大小写）
        column_mapping = {}
        for col in df_copy.columns:
            lower_col = col.lower()
            if lower_col in ['high', 'low', 'close', 'open', 'volume']:
                column_mapping[col] = lower_col
        
        if column_mapping:
            df_copy = df_copy.rename(columns=column_mapping)
        
        return df_copy
    
    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """验证必需的列是否存在"""
        required_cols = ['high', 'low', 'close']
        return all(col in df.columns for col in required_cols)
    
    def _get_default_levels(self) -> Dict[str, Any]:
        """返回默认的价格水平"""
        return {
            'support_levels': [0.0, 0.0, 0.0],
            'resistance_levels': [0.0, 0.0, 0.0],
            'pivot_point': 0.0,
            'breakout_threshold': 0.0
        }
    
    def _calculate_pivot_point(self, df: pd.DataFrame) -> float:
        """计算标准枢轴点 (H+L+C)/3"""
        # 使用最近的高、低、收盘价
        high = df['high'].iloc[-1]
        low = df['low'].iloc[-1]
        close = df['close'].iloc[-1]
        
        pivot_point = (high + low + close) / 3.0
        return pivot_point
    
    def _identify_support_levels(self, df: pd.DataFrame, current_price: float) -> list:
        """
        识别支撑位
        基于历史低点、成交量确认和技术指标
        """
        # 1. 找到历史低点（局部最小值）
        window = 10  # 窗口大小
        lows = df['low'].values
        volumes = df.get('volume', pd.Series([1] * len(df))).values
        
        # 识别局部最小值
        local_lows = []
        for i in range(window, len(lows) - window):
            if all(lows[i] <= lows[i-j] for j in range(1, window+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, window+1)):
                # 成交量确认（高成交量的低点更重要）
                volume_factor = volumes[i] / safe_mean(volumes[max(0, i-20):i+20])
                local_lows.append((lows[i], volume_factor))
        
        if not local_lows:
            # 如果没有找到局部低点，使用最近的低点
            recent_lows = sorted(df['low'].tail(50).tolist())
            return recent_lows[:3] if len(recent_lows) >= 3 else [current_price * 0.95, current_price * 0.9, current_price * 0.85]
        
        # 2. 按重要性排序（考虑价格水平和成交量）
        # 只考虑低于当前价格的支撑位
        valid_supports = [(price, volume) for price, volume in local_lows if price < current_price]
        
        if len(valid_supports) < 3:
            # 补充支撑位：使用移动平均和价格百分比
            ma_20 = df['close'].rolling(20).mean().iloc[-1]
            ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_20
            
            additional_supports = []
            if ma_20 < current_price:
                additional_supports.append((ma_20, 1.0))
            if ma_50 < current_price and ma_50 != ma_20:
                additional_supports.append((ma_50, 1.0))
            
            # 添加基于百分比的支撑位
            percentage_supports = [current_price * 0.98, current_price * 0.95, current_price * 0.9]
            for sup in percentage_supports:
                if len(additional_supports) < 3:
                    additional_supports.append((sup, 0.5))
            
            valid_supports.extend(additional_supports)
        
        # 3. 选择最重要的3个支撑位
        # 按成交量权重排序，然后按价格接近度排序
        valid_supports.sort(key=lambda x: x[1], reverse=True)  # 按成交量权重排序
        top_supports = valid_supports[:5]  # 取前5个
        
        # 从中选择最接近当前价格的3个
        top_supports.sort(key=lambda x: abs(current_price - x[0]))
        support_prices = [x[0] for x in top_supports[:3]]
        
        # 确保支撑位按从高到低排序
        support_prices.sort(reverse=True)
        
        # 如果支撑位不足3个，补充
        while len(support_prices) < 3:
            if len(support_prices) == 0:
                support_prices.append(current_price * 0.95)
            elif len(support_prices) == 1:
                support_prices.append(support_prices[-1] * 0.95)
            else:
                support_prices.append(support_prices[-1] * 0.95)
        
        return support_prices
    
    def _identify_resistance_levels(self, df: pd.DataFrame, current_price: float) -> list:
        """
        识别阻力位
        基于历史高点、成交量确认和技术指标
        """
        # 1. 找到历史高点（局部最大值）
        window = 10  # 窗口大小
        highs = df['high'].values
        volumes = df.get('volume', pd.Series([1] * len(df))).values
        
        # 识别局部最大值
        local_highs = []
        for i in range(window, len(highs) - window):
            if all(highs[i] >= highs[i-j] for j in range(1, window+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, window+1)):
                # 成交量确认
                volume_factor = volumes[i] / safe_mean(volumes[max(0, i-20):i+20])
                local_highs.append((highs[i], volume_factor))
        
        if not local_highs:
            # 如果没有找到局部高点，使用最近的高点
            recent_highs = sorted(df['high'].tail(50).tolist(), reverse=True)
            return recent_highs[:3] if len(recent_highs) >= 3 else [current_price * 1.05, current_price * 1.1, current_price * 1.15]
        
        # 2. 按重要性排序
        # 只考虑高于当前价格的阻力位
        valid_resistances = [(price, volume) for price, volume in local_highs if price > current_price]
        
        if len(valid_resistances) < 3:
            # 补充阻力位：使用移动平均和价格百分比
            ma_20 = df['close'].rolling(20).mean().iloc[-1]
            ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_20
            
            additional_resistances = []
            if ma_20 > current_price:
                additional_resistances.append((ma_20, 1.0))
            if ma_50 > current_price and ma_50 != ma_20:
                additional_resistances.append((ma_50, 1.0))
            
            # 添加基于百分比的阻力位
            percentage_resistances = [current_price * 1.02, current_price * 1.05, current_price * 1.1]
            for res in percentage_resistances:
                if len(additional_resistances) < 3:
                    additional_resistances.append((res, 0.5))
            
            valid_resistances.extend(additional_resistances)
        
        # 3. 选择最重要的3个阻力位
        valid_resistances.sort(key=lambda x: x[1], reverse=True)  # 按成交量权重排序
        top_resistances = valid_resistances[:5]  # 取前5个
        
        # 从中选择最接近当前价格的3个
        top_resistances.sort(key=lambda x: abs(current_price - x[0]))
        resistance_prices = [x[0] for x in top_resistances[:3]]
        
        # 确保阻力位按从低到高排序
        resistance_prices.sort()
        
        # 如果阻力位不足3个，补充
        while len(resistance_prices) < 3:
            if len(resistance_prices) == 0:
                resistance_prices.append(current_price * 1.05)
            elif len(resistance_prices) == 1:
                resistance_prices.append(resistance_prices[-1] * 1.05)
            else:
                resistance_prices.append(resistance_prices[-1] * 1.05)
        
        return resistance_prices
    
    def _calculate_breakout_threshold(self, df: pd.DataFrame, current_price: float) -> float:
        """
        计算突破临界点
        基于ATR（平均真实波幅）和近期价格波动
        """
        # 使用ATR计算
        atr_values = self.calculate_atr_values(df)
        atr_14 = atr_values.get('atr_14', 0.0)
        
        if atr_14 == 0.0:
            # 如果ATR计算失败，使用简单的价格波动率
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                volatility = returns.std() * np.sqrt(len(returns))
                breakout_threshold = current_price * volatility * 2.0
            else:
                breakout_threshold = current_price * 0.02  # 默认2%
        else:
            # 使用ATR的1.5倍作为突破阈值
            breakout_threshold = atr_14 * 1.5
        
        # 确保突破阈值合理（不少于0.1%，不多于5%）
        min_threshold = current_price * 0.001
        max_threshold = current_price * 0.05
        
        breakout_threshold = max(min_threshold, min(breakout_threshold, max_threshold))
        
        return breakout_threshold
    
    def _validate_support_levels(self, support_levels: list, current_price: float) -> list:
        """验证支撑位的逻辑一致性"""
        # 确保所有支撑位都低于当前价格
        valid_supports = [level for level in support_levels if level < current_price]
        
        # 按从高到低排序
        valid_supports.sort(reverse=True)
        
        # 确保支撑位之间有合理的间距
        filtered_supports = []
        for i, level in enumerate(valid_supports):
            if i == 0:
                filtered_supports.append(level)
            else:
                # 确保至少有0.5%的价格差异
                if abs(level - filtered_supports[-1]) / current_price >= 0.005:
                    filtered_supports.append(level)
        
        # 补充到3个支撑位
        while len(filtered_supports) < 3:
            if len(filtered_supports) == 0:
                filtered_supports.append(current_price * 0.98)
            else:
                new_level = filtered_supports[-1] * 0.97
                filtered_supports.append(new_level)
        
        return filtered_supports[:3]
    
    def _validate_resistance_levels(self, resistance_levels: list, current_price: float) -> list:
        """验证阻力位的逻辑一致性"""
        # 确保所有阻力位都高于当前价格
        valid_resistances = [level for level in resistance_levels if level > current_price]
        
        # 按从低到高排序
        valid_resistances.sort()
        
        # 确保阻力位之间有合理的间距
        filtered_resistances = []
        for i, level in enumerate(valid_resistances):
            if i == 0:
                filtered_resistances.append(level)
            else:
                # 确保至少有0.5%的价格差异
                if abs(level - filtered_resistances[-1]) / current_price >= 0.005:
                    filtered_resistances.append(level)
        
        # 补充到3个阻力位
        while len(filtered_resistances) < 3:
            if len(filtered_resistances) == 0:
                filtered_resistances.append(current_price * 1.02)
            else:
                new_level = filtered_resistances[-1] * 1.03
                filtered_resistances.append(new_level)
        
        return filtered_resistances[:3]

    def generate_signal_metadata(self, df: pd.DataFrame, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成信号元数据，包括信号强度、衰减时间、可靠性和确认状态
        
        完全内联实现，避免任何方法调用导致的递归问题
        """
        try:
            # 基础数据验证 - 完全内联
            if df is None or df.empty or len(df) < 10:
                return {
                    'signal_strength': 'weak',
                    'signal_decay_time': 240,
                    'signal_reliability': 0.3,
                    'confirmation_status': 'weak'
                }
            
            if not signal_data or 'strategy_signals' not in signal_data:
                return {
                    'signal_strength': 'weak',
                    'signal_decay_time': 240,
                    'signal_reliability': 0.3,
                    'confirmation_status': 'weak'
                }
            
            # 1. 计算信号强度 - 完全内联实现
            strategy_signals = signal_data.get('strategy_signals', {})
            signal_strength = 'weak'
            
            if strategy_signals:
                signal_values = []
                confidences = []
                
                for strategy_name, strategy_info in strategy_signals.items():
                    if not isinstance(strategy_info, dict):
                        continue
                        
                    signal = strategy_info.get('signal', 'neutral')
                    confidence = strategy_info.get('confidence', 0)
                    
                    # 确保confidence是数值
                    try:
                        confidence = float(confidence) / 100.0 if confidence > 1 else float(confidence)
                    except:
                        confidence = 0.0
                    
                    # 信号转数值
                    if signal == 'bullish':
                        signal_values.append(1)
                    elif signal == 'bearish':
                        signal_values.append(-1)
                    else:
                        signal_values.append(0)
                    
                    confidences.append(confidence)
                
                if signal_values and confidences:
                    # 计算平均置信度
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    # 计算信号一致性
                    if len(signal_values) > 1:
                        mean_signal = sum(signal_values) / len(signal_values)
                        variance = sum((x - mean_signal) ** 2 for x in signal_values) / len(signal_values)
                        consistency = max(0.0, 1.0 - variance / 4.0)  # 标准化到0-1
                    else:
                        consistency = 1.0
                    
                    # 综合评分
                    composite_score = avg_confidence * consistency * 0.8
                    
                    if composite_score >= 0.7:
                        signal_strength = 'strong'
                    elif composite_score >= 0.4:
                        signal_strength = 'moderate'
                    else:
                        signal_strength = 'weak'
            
            # 2. 计算信号衰减时间 - 完全内联实现
            signal_decay_time = 240  # 默认4小时
            
            try:
                if len(df) >= 14 and 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
                    # 简单的波动率估算
                    recent_prices = df['close'].tail(14)
                    if len(recent_prices) >= 10:
                        price_changes = recent_prices.pct_change().dropna()
                        if len(price_changes) > 0:
                            volatility = price_changes.std()
                            if not pd.isna(volatility) and volatility > 0:
                                # 根据波动率调整衰减时间
                                if volatility >= 0.03:  # 高波动
                                    signal_decay_time = 120  # 2小时
                                elif volatility >= 0.015:  # 中等波动
                                    signal_decay_time = 180  # 3小时
                                else:  # 低波动
                                    signal_decay_time = 300  # 5小时
            except:
                signal_decay_time = 240
            
            # 3. 计算信号可靠性 - 完全内联实现
            base_reliability = {
                'strong': 0.8,
                'moderate': 0.6,
                'weak': 0.3
            }.get(signal_strength, 0.3)
            
            # 简单的市场稳定性评估
            market_stability = 0.5
            try:
                if len(df) >= 20 and 'close' in df.columns:
                    recent_prices = df['close'].tail(20)
                    if len(recent_prices) >= 10:
                        returns = recent_prices.pct_change().dropna()
                        if len(returns) > 0:
                            mean_return = returns.mean()
                            return_volatility = returns.std()
                            
                            if not pd.isna(mean_return) and not pd.isna(return_volatility):
                                # 趋势一致性
                                trend_score = min(1.0, abs(mean_return) * 100) if abs(mean_return) < 1 else 0.5
                                # 波动率合理性
                                vol_score = max(0.0, 1.0 - return_volatility) if return_volatility < 1 else 0.0
                                market_stability = (trend_score + vol_score) / 2
            except:
                market_stability = 0.5
            
            # 综合可靠性
            signal_reliability = base_reliability * 0.7 + market_stability * 0.3
            signal_reliability = max(0.0, min(1.0, signal_reliability))
            
            # 4. 确定确认状态 - 完全内联实现
            if signal_strength == 'strong' and signal_reliability >= 0.7:
                confirmation_status = 'confirmed'
            elif signal_strength == 'moderate' and signal_reliability >= 0.5:
                confirmation_status = 'pending'
            else:
                confirmation_status = 'weak'
            
            return {
                'signal_strength': signal_strength,
                'signal_decay_time': int(signal_decay_time),
                'signal_reliability': round(float(signal_reliability), 4),
                'confirmation_status': confirmation_status
            }
            
        except Exception as e:
            print(f"Signal metadata generation error: {e}")
            return {
                'signal_strength': 'weak',
                'signal_decay_time': 240,
                'signal_reliability': 0.3,
                'confirmation_status': 'weak'
            }
    
    def _calculate_signal_strength(self, signal_data: Dict[str, Any]) -> str:
        """
        计算信号强度，基于多个技术指标的一致性和置信度
        
        算法逻辑：
        1. 统计各策略信号的一致性（bullish/bearish比例）
        2. 计算加权平均置信度
        3. 评估信号分散度
        4. 综合判断强度等级
        
        Args:
            signal_data: 包含策略信号数据的字典
            
        Returns:
            str: "weak"|"moderate"|"strong"
        """
        try:
            strategy_signals = signal_data.get('strategy_signals', {})
            if not strategy_signals:
                return 'weak'
            
            # 提取各策略信号和置信度
            signals = []
            confidences = []
            strategy_weights = {
                'trend_following': 0.25,
                'mean_reversion': 0.20,
                'momentum': 0.25,
                'volatility': 0.15,
                'statistical_arbitrage': 0.15
            }
            
            for strategy_name, strategy_info in strategy_signals.items():
                signal = strategy_info.get('signal', 'neutral')
                confidence = strategy_info.get('confidence', 0) / 100.0  # 转换为0-1范围
                
                # 将信号转换为数值 (-1: bearish, 0: neutral, 1: bullish)
                signal_value = 0
                if signal == 'bullish':
                    signal_value = 1
                elif signal == 'bearish':
                    signal_value = -1
                
                # 获取策略权重
                weight = strategy_weights.get(strategy_name, 0.1)
                
                signals.append((signal_value, weight))
                confidences.append((confidence, weight))
            
            if not signals:
                return 'weak'
            
            # 1. 计算加权信号方向一致性
            weighted_signal = sum(signal * weight for signal, weight in signals)
            total_weight = sum(weight for _, weight in signals)
            normalized_signal = weighted_signal / total_weight if total_weight > 0 else 0
            
            # 2. 计算加权平均置信度
            weighted_confidence = sum(conf * weight for conf, weight in confidences)
            avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0
            
            # 3. 计算信号一致性（所有策略信号方向的标准差）
            signal_values = [signal for signal, _ in signals]
            signal_consistency = 1.0 - (safe_std(signal_values) / 1.0) if len(signal_values) > 1 else 1.0
            
            # 4. 计算综合强度评分
            # 考虑信号方向强度、置信度、一致性
            direction_strength = abs(normalized_signal)  # 0-1，信号方向的强度
            
            # 综合评分 = 方向强度 × 置信度 × 一致性
            composite_score = direction_strength * avg_confidence * signal_consistency
            
            # 5. 根据评分分级
            if composite_score >= 0.7:
                return 'strong'
            elif composite_score >= 0.4:
                return 'moderate'
            else:
                return 'weak'
                
        except Exception as e:
            print(f"Signal strength calculation error: {e}")
            return 'weak'
    
    def _calculate_signal_decay_time(self, df: pd.DataFrame) -> int:
        """
        计算信号衰减时间（分钟），基于ATR和波动率
        
        算法逻辑：
        1. 使用ATR评估价格波动幅度
        2. 计算历史波动率
        3. 考虑时间框架因素
        4. 综合计算信号有效期
        
        Args:
            df: 标准化后的OHLCV数据
            
        Returns:
            int: 信号有效期（分钟）
        """
        try:
            # 1. 计算ATR值
            atr_values = self.calculate_atr_values(df)
            atr_14 = atr_values.get('atr_14', 0.0)
            atr_percentile = atr_values.get('atr_percentile', 50.0)
            
            # 2. 计算历史波动率
            volatility_analysis = self.analyze_volatility_depth(df)
            volatility_percentile = volatility_analysis.get('volatility_percentile', 50.0)
            volatility_trend = volatility_analysis.get('volatility_trend', 'stable')
            
            # 3. 基础衰减时间（以ATR为基准）
            if atr_14 > 0:
                current_price = float(df['close'].iloc[-1])
                atr_ratio = atr_14 / current_price
                
                # ATR越大，价格变化越快，信号衰减越快
                base_decay_minutes = 240  # 基础4小时
                
                # 根据ATR比例调整
                if atr_ratio > 0.05:  # 高波动
                    base_decay_minutes = 120  # 2小时
                elif atr_ratio > 0.03:  # 中等波动
                    base_decay_minutes = 180  # 3小时
                elif atr_ratio > 0.01:  # 低波动
                    base_decay_minutes = 300  # 5小时
                else:  # 极低波动
                    base_decay_minutes = 480  # 8小时
            else:
                base_decay_minutes = 240
            
            # 4. 根据波动率百分位数调整
            # 高波动率环境下信号衰减更快
            volatility_factor = 1.0
            if volatility_percentile >= 80:  # 极高波动率
                volatility_factor = 0.6
            elif volatility_percentile >= 60:  # 高波动率
                volatility_factor = 0.8
            elif volatility_percentile >= 40:  # 中等波动率
                volatility_factor = 1.0
            elif volatility_percentile >= 20:  # 低波动率
                volatility_factor = 1.3
            else:  # 极低波动率
                volatility_factor = 1.6
            
            # 5. 根据波动率趋势调整
            trend_factor = 1.0
            if volatility_trend == 'increasing':
                trend_factor = 0.8  # 波动率上升，信号衰减加快
            elif volatility_trend == 'decreasing':
                trend_factor = 1.2  # 波动率下降，信号持续更久
            
            # 6. 计算最终衰减时间
            final_decay_time = int(base_decay_minutes * volatility_factor * trend_factor)
            
            # 7. 约束在合理范围内（30分钟 - 24小时）
            final_decay_time = max(30, min(1440, final_decay_time))
            
            return final_decay_time
            
        except Exception as e:
            print(f"Signal decay time calculation error: {e}")
            return 240  # 默认4小时
    
    def _evaluate_signal_reliability(self, df: pd.DataFrame, signal_data: Dict[str, Any], signal_strength: str) -> float:
        """
        评估信号可靠性，综合考虑历史准确率、市场条件、指标稳定性
        
        算法逻辑：
        1. 基于信号强度的基础可靠性
        2. 市场条件评估（趋势稳定性、波动率合理性）
        3. 简化的指标稳定性评估（避免递归调用）
        4. 简化的历史模式匹配评估（避免递归调用）
        
        Args:
            df: 标准化后的OHLCV数据
            signal_data: 策略信号数据
            signal_strength: 计算出的信号强度
            
        Returns:
            float: 可靠性评分 (0-1)
        """
        try:
            # 1. 基于信号强度的基础评分
            base_reliability = {
                'strong': 0.8,
                'moderate': 0.6,
                'weak': 0.3
            }.get(signal_strength, 0.3)
            
            # 2. 简化的市场条件评估（不调用其他可能递归的方法）
            if len(df) >= 20:
                # 价格趋势稳定性
                recent_prices = df['close'].tail(20)
                price_trend = recent_prices.pct_change().mean()
                price_volatility = recent_prices.pct_change().std()
                
                # 趋势一致性评分
                trend_consistency = min(1.0, abs(price_trend) * 100) if not np.isnan(price_trend) else 0.5
                volatility_score = max(0.0, 1.0 - price_volatility) if not np.isnan(price_volatility) and price_volatility > 0 else 0.5
                
                market_condition_score = (trend_consistency + volatility_score) / 2
            else:
                market_condition_score = 0.5
            
            # 3. 简化的指标稳定性评估（不调用其他方法）
            strategy_signals = signal_data.get('strategy_signals', {})
            if strategy_signals:
                confidences = []
                for strategy_info in strategy_signals.values():
                    confidence = strategy_info.get('confidence', 0) / 100.0
                    confidences.append(confidence)
                
                avg_confidence = safe_mean(confidences) if confidences else 0.5
                confidence_consistency = 1.0 - safe_std(confidences) if len(confidences) > 1 else 1.0
                indicator_stability_score = (avg_confidence + confidence_consistency) / 2
            else:
                indicator_stability_score = 0.5
            
            # 4. 简化的模式匹配评估（不调用identify_price_levels）
            if len(df) >= 30:
                # 简单的价格模式评估
                recent_returns = df['close'].pct_change().tail(10).fillna(0)
                pattern_stability = 1.0 - safe_std(recent_returns.values) if len(recent_returns) > 0 else 0.5
                pattern_matching_score = min(1.0, max(0.0, pattern_stability))
            else:
                pattern_matching_score = 0.5
            
            # 5. 简化的成交量确认评估
            if 'volume' in df.columns and len(df) >= 10:
                recent_volume = df['volume'].tail(10).mean()
                historical_volume = df['volume'].mean()
                volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
                
                # 理想的成交量比例在0.8-2.0之间
                if 0.8 <= volume_ratio <= 2.0:
                    volume_confirmation_score = 0.8
                elif 0.5 <= volume_ratio < 0.8:
                    volume_confirmation_score = 0.6
                else:
                    volume_confirmation_score = 0.4
            else:
                volume_confirmation_score = 0.5
            
            # 6. 加权综合评分
            weights = {
                'base': 0.3,
                'market_condition': 0.25,
                'indicator_stability': 0.2,
                'pattern_matching': 0.15,
                'volume_confirmation': 0.1
            }
            
            composite_reliability = (
                base_reliability * weights['base'] +
                market_condition_score * weights['market_condition'] +
                indicator_stability_score * weights['indicator_stability'] +
                pattern_matching_score * weights['pattern_matching'] +
                volume_confirmation_score * weights['volume_confirmation']
            )
            
            # 7. 约束在合理范围
            composite_reliability = max(0.0, min(1.0, composite_reliability))
            
            return composite_reliability
            
        except Exception as e:
            print(f"Signal reliability evaluation error: {e}")
            return 0.5  # 默认中等可靠性  # 默认中等可靠性
    
    def _assess_market_conditions(self, df: pd.DataFrame) -> float:
        """
        评估市场条件质量
        
        考虑因素：
        1. 趋势一致性
        2. 波动率合理性
        3. 价格结构健康度
        """
        try:
            # 1. 趋势一致性评估
            # 使用多个移动平均线判断趋势稳定性
            ma_5 = df['close'].rolling(5).mean()
            ma_20 = df['close'].rolling(20).mean()
            ma_50 = df['close'].rolling(50).mean() if len(df) >= 50 else ma_20
            
            # 计算趋势一致性（移动平均线排列）
            current_price = df['close'].iloc[-1]
            trend_alignment = 0
            
            if len(ma_5) > 0 and len(ma_20) > 0:
                if current_price > ma_5.iloc[-1] > ma_20.iloc[-1] > ma_50.iloc[-1]:  # 完全上升趋势
                    trend_alignment = 1.0
                elif current_price < ma_5.iloc[-1] < ma_20.iloc[-1] < ma_50.iloc[-1]:  # 完全下降趋势
                    trend_alignment = 1.0
                elif abs(current_price - ma_20.iloc[-1]) / current_price < 0.02:  # 横盘整理
                    trend_alignment = 0.7
                else:  # 趋势混乱
                    trend_alignment = 0.3
            
            # 2. 波动率合理性
            volatility_analysis = self.analyze_volatility_depth(df)
            volatility_percentile = volatility_analysis.get('volatility_percentile', 50.0)
            
            # 极端波动率降低可靠性
            if 20 <= volatility_percentile <= 80:  # 正常波动率范围
                volatility_score = 1.0
            elif 10 <= volatility_percentile < 20 or 80 < volatility_percentile <= 90:
                volatility_score = 0.7
            else:  # 极端波动率
                volatility_score = 0.3
            
            # 3. 价格结构健康度
            # 检查是否有异常的价格跳跃或者间隙
            price_changes = df['close'].pct_change().dropna()
            extreme_moves = (abs(price_changes) > 0.05).sum()  # 单日变动超过5%
            structure_score = max(0.3, 1.0 - extreme_moves / len(price_changes))
            
            # 综合评分
            market_score = (trend_alignment * 0.4 + volatility_score * 0.4 + structure_score * 0.2)
            
            return max(0.0, min(1.0, market_score))
            
        except Exception as e:
            print(f"Market conditions assessment error: {e}")
            return 0.5
    
    def _assess_indicator_stability(self, df: pd.DataFrame, signal_data: Dict[str, Any]) -> float:
        """
        评估技术指标的稳定性
        
        考虑因素：
        1. 指标值的连续性
        2. 信号的持续性
        3. 指标间的协调性
        """
        try:
            # 1. 计算关键指标的稳定性
            stability_scores = []
            
            # RSI稳定性
            if len(df) >= 14:
                import pandas_ta as ta
                high = df['high']
                low = df['low']
                close = df['close']
                
                # 计算RSI
                rsi = ta.rsi(close, length=14)
                if len(rsi.dropna()) >= 10:
                    # RSI连续性检查（避免频繁跳跃）
                    rsi_changes = rsi.dropna().diff().abs()
                    avg_rsi_change = rsi_changes.mean()
                    rsi_stability = max(0.0, 1.0 - avg_rsi_change / 10.0)  # 标准化
                    stability_scores.append(rsi_stability)
            
            # MACD稳定性
            try:
                macd_line = ta.macd(df['close'])['MACD_12_26_9']
                if len(macd_line.dropna()) >= 10:
                    macd_changes = macd_line.dropna().diff().abs()
                    avg_macd_change = macd_changes.mean()
                    current_price = df['close'].iloc[-1]
                    normalized_macd_change = avg_macd_change / current_price if current_price > 0 else 0
                    macd_stability = max(0.0, 1.0 - normalized_macd_change * 100)
                    stability_scores.append(macd_stability)
            except:
                pass
            
            # 2. 信号持续性评估
            strategy_signals = signal_data.get('strategy_signals', {})
            signal_consistency = []
            
            for strategy_name, strategy_info in strategy_signals.items():
                confidence = strategy_info.get('confidence', 0) / 100.0
                
                # 高置信度的信号通常更稳定
                if confidence >= 0.7:
                    signal_consistency.append(1.0)
                elif confidence >= 0.5:
                    signal_consistency.append(0.7)
                else:
                    signal_consistency.append(0.3)
            
            consistency_score = safe_mean(signal_consistency) if signal_consistency else 0.5
            
            # 3. 综合稳定性评分
            if stability_scores:
                indicator_stability = safe_mean(stability_scores)
            else:
                indicator_stability = 0.5
            
            # 加权组合
            final_stability = indicator_stability * 0.6 + consistency_score * 0.4
            
            return max(0.0, min(1.0, final_stability))
            
        except Exception as e:
            print(f"Indicator stability assessment error: {e}")
            return 0.5
    
    def _assess_historical_patterns(self, df: pd.DataFrame) -> float:
        """
        评估历史模式匹配度
        
        通过分析相似的历史价格模式来评估当前信号的可靠性
        """
        try:
            if len(df) < 50:
                return 0.5
            
            # 1. 计算当前价格模式特征
            # 使用最近20个周期的价格相对变化
            recent_returns = df['close'].pct_change().tail(20).fillna(0)
            
            # 2. 寻找历史相似模式
            # 滑动窗口匹配历史模式
            pattern_matches = []
            window_size = 20
            
            if len(df) >= window_size * 3:  # 确保有足够的历史数据
                all_returns = df['close'].pct_change().fillna(0)
                
                # 计算与历史模式的相关性
                for i in range(window_size, len(all_returns) - window_size):
                    historical_pattern = all_returns.iloc[i-window_size:i]
                    
                    # 计算皮尔逊相关系数
                    # 验证数据质量，避免numpy警告
                    recent_vals = recent_returns.values
                    hist_vals = historical_pattern.values
                    
                    # 检查数据有效性
                    if (len(recent_vals) == len(hist_vals) and 
                        not np.any(np.isnan(recent_vals)) and not np.any(np.isnan(hist_vals)) and
                        safe_std(recent_vals) > 1e-8 and safe_std(hist_vals) > 1e-8):
                        correlation = np.corrcoef(recent_vals, hist_vals)[0, 1]
                    else:
                        correlation = np.nan
                    
                    if not np.isnan(correlation) and abs(correlation) > 0.6:  # 高相关性
                        pattern_matches.append(abs(correlation))
            
            # 3. 评估模式匹配质量
            if pattern_matches:
                avg_correlation = safe_mean(pattern_matches)
                pattern_score = avg_correlation
            else:
                pattern_score = 0.5  # 无明显模式匹配
            
            # 4. 简化的关键位置检查（移除递归调用）
            # 直接计算简单的支撑阻力位而不调用其他方法
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            
            # 检查当前价格是否在关键位置附近
            near_key_level = False
            if abs(current_price - recent_high) / current_price < 0.02:  # 接近近期高点
                near_key_level = True
            elif abs(current_price - recent_low) / current_price < 0.02:  # 接近近期低点
                near_key_level = True
            
            level_bonus = 0.1 if near_key_level else 0.0
            
            final_pattern_score = min(1.0, pattern_score + level_bonus)
            
            return max(0.0, final_pattern_score)
            
        except Exception as e:
            print(f"Historical pattern assessment error: {e}")
            return 0.5
    
    def _assess_volume_confirmation(self, df: pd.DataFrame) -> float:
        """
        评估成交量确认度
        
        考虑因素：
        1. 成交量趋势一致性
        2. 价量配合度
        3. 成交量相对强度
        """
        try:
            if 'volume' not in df.columns or len(df) < 10:
                return 0.5  # 无成交量数据时返回中性评分
            
            volume = df['volume']
            close = df['close']
            
            # 1. 计算价量相关性
            price_changes = close.pct_change().dropna()
            volume_changes = volume.pct_change().dropna()
            
            if len(price_changes) >= 10 and len(volume_changes) >= 10:
                # 对齐数据长度
                min_length = min(len(price_changes), len(volume_changes))
                price_changes = price_changes.tail(min_length)
                volume_changes = volume_changes.tail(min_length)
                
                # 计算价量相关性
                # 验证数据质量，避免numpy警告
                abs_price = abs(price_changes).values
                abs_volume = abs(volume_changes).values
                
                # 检查数据有效性
                if (len(abs_price) == len(abs_volume) and
                    not np.any(np.isnan(abs_price)) and not np.any(np.isnan(abs_volume)) and
                    safe_std(abs_price) > 1e-8 and safe_std(abs_volume) > 1e-8):
                    correlation = np.corrcoef(abs_price, abs_volume)[0, 1]
                else:
                    correlation = np.nan
                correlation = 0.0 if np.isnan(correlation) else abs(correlation)
            else:
                correlation = 0.0
            
            # 2. 成交量相对强度
            recent_volume = volume.tail(10).mean()
            historical_volume = volume.tail(50).mean() if len(volume) >= 50 else recent_volume
            
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
            
            # 理想的成交量比例在1.2-2.0之间（放量但不过度）
            if 1.2 <= volume_ratio <= 2.0:
                volume_strength = 1.0
            elif 1.0 <= volume_ratio < 1.2:
                volume_strength = 0.7
            elif 0.8 <= volume_ratio < 1.0:
                volume_strength = 0.5
            elif volume_ratio > 2.0:
                # 过度放量可能是反转信号，降低确认度
                volume_strength = 0.6
            else:
                volume_strength = 0.3
            
            # 3. 综合评分
            volume_confirmation = correlation * 0.4 + volume_strength * 0.6
            
            return max(0.0, min(1.0, volume_confirmation))
            
        except Exception as e:
            print(f"Volume confirmation assessment error: {e}")
            return 0.5
    
    def _determine_confirmation_status(self, signal_strength: str, signal_reliability: float, signal_data: Dict[str, Any]) -> str:
        """
        确定信号确认状态，基于信号强度、可靠性和多重确认机制
        
        确认逻辑：
        1. "confirmed": 强信号 + 高可靠性 + 多指标确认
        2. "pending": 中等信号 + 中等可靠性，等待更多确认
        3. "weak": 弱信号 + 低可靠性，谨慎对待
        
        Args:
            signal_strength: 信号强度
            signal_reliability: 信号可靠性
            signal_data: 策略信号数据
            
        Returns:
            str: "confirmed"|"pending"|"weak"
        """
        try:
            # 1. 基于强度和可靠性的初步判断
            if signal_strength == 'strong' and signal_reliability >= 0.7:
                base_status = 'confirmed'
            elif signal_strength == 'moderate' and signal_reliability >= 0.5:
                base_status = 'pending'
            elif signal_strength == 'strong' and signal_reliability >= 0.5:
                base_status = 'pending'
            else:
                base_status = 'weak'
            
            # 2. 多重确认检查
            strategy_signals = signal_data.get('strategy_signals', {})
            
            # 统计不同信号方向和高置信度信号
            bullish_count = 0
            bearish_count = 0
            high_confidence_count = 0
            total_strategies = len(strategy_signals)
            
            for strategy_info in strategy_signals.values():
                signal = strategy_info.get('signal', 'neutral')
                confidence = strategy_info.get('confidence', 0)
                
                if signal == 'bullish':
                    bullish_count += 1
                elif signal == 'bearish':
                    bearish_count += 1
                    
                if confidence >= 70:  # 高置信度阈值
                    high_confidence_count += 1
            
            # 3. 计算确认因子
            if total_strategies > 0:
                # 方向一致性
                direction_consensus = max(bullish_count, bearish_count) / total_strategies
                
                # 高置信度比例
                high_confidence_ratio = high_confidence_count / total_strategies
                
                # 综合确认分数
                confirmation_score = (direction_consensus * 0.6 + high_confidence_ratio * 0.4)
            else:
                confirmation_score = 0.0
            
            # 4. 基于确认分数调整状态
            if base_status == 'confirmed':
                if confirmation_score >= 0.7:
                    return 'confirmed'
                elif confirmation_score >= 0.5:
                    return 'pending'
                else:
                    return 'weak'
                    
            elif base_status == 'pending':
                if confirmation_score >= 0.8:
                    return 'confirmed'  # 多重确认提升状态
                elif confirmation_score >= 0.4:
                    return 'pending'
                else:
                    return 'weak'
                    
            else:  # base_status == 'weak'
                if confirmation_score >= 0.9:
                    return 'pending'  # 极强确认可以提升弱信号
                else:
                    return 'weak'
                    
        except Exception as e:
            print(f"Confirmation status determination error: {e}")
            return 'weak'
    
    def _get_default_signal_metadata(self) -> Dict[str, Any]:
        """
        返回默认的信号元数据（用于错误情况或数据不足时）
        
        Returns:
            Dict[str, Any]: 默认的信号元数据
        """
        return {
            'signal_strength': 'weak',
            'signal_decay_time': 240,  # 4小时
            'signal_reliability': 0.3,
            'confirmation_status': 'weak'
        }

    def cross_timeframe_analysis(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        跨时间框架综合分析，评估多个时间框架信号的一致性和冲突
        
        完全内联实现，避免任何方法调用导致的递归问题
        """
        try:
            # 基础数据验证
            if not timeframe_signals or len(timeframe_signals) < 2:
                return {
                    'timeframe_consensus': 0.5,
                    'dominant_timeframe': '1h',
                    'conflict_areas': [],
                    'trend_alignment': 'mixed',
                    'overall_signal_strength': 'weak'
                }
            
            # 时间框架权重配置 - 完全内联
            timeframe_weights = {
                '5m': 0.10,
                '15m': 0.15,
                '30m': 0.20,
                '1h': 0.25,
                '4h': 0.20,
                '1d': 0.10
            }
            
            # 1. 计算时间框架一致性 - 完全内联实现
            signal_values = []
            weights = []
            confidence_values = []
            
            for timeframe, signal_data in timeframe_signals.items():
                if not isinstance(signal_data, dict):
                    continue
                    
                signal = signal_data.get('signal', 'neutral')
                confidence = signal_data.get('confidence', 0)
                
                # 确保confidence是数值
                try:
                    confidence = float(confidence) / 100.0 if confidence > 1 else float(confidence)
                except:
                    confidence = 0.0
                
                # 信号方向数值化
                signal_value = 0
                if signal == 'bullish':
                    signal_value = 1
                elif signal == 'bearish': 
                    signal_value = -1
                    
                # 获取时间框架权重
                timeframe_weight = timeframe_weights.get(timeframe, 0.1)
                
                # 综合权重 = 时间框架权重 × 置信度
                combined_weight = timeframe_weight * (0.5 + 0.5 * confidence)
                
                signal_values.append(signal_value)
                weights.append(combined_weight)
                confidence_values.append(confidence)
            
            # 计算一致性评分
            timeframe_consensus = 0.5
            if signal_values and weights and len(signal_values) >= 2:
                try:
                    # 加权平均信号方向
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weighted_mean_signal = sum(sv * w for sv, w in zip(signal_values, weights)) / total_weight
                        
                        # 计算信号方向的一致性（加权方差）
                        weighted_variance = sum(w * (sv - weighted_mean_signal) ** 2 for sv, w in zip(signal_values, weights)) / total_weight
                        
                        # 将方差转换为一致性评分
                        consistency_score = max(0.0, 1.0 - weighted_variance / 4.0)
                        
                        # 考虑信号强度
                        non_neutral_signals = sum(1 for sv in signal_values if sv != 0)
                        signal_strength_factor = non_neutral_signals / len(signal_values)
                        
                        # 考虑置信度因子
                        avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
                        
                        # 综合一致性评分
                        timeframe_consensus = consistency_score * (0.6 + 0.2 * signal_strength_factor + 0.2 * avg_confidence)
                        timeframe_consensus = max(0.0, min(1.0, timeframe_consensus))
                except:
                    timeframe_consensus = 0.5
            
            # 2. 识别主导时间框架 - 完全内联实现
            dominant_timeframe = '1h'  # 默认值
            try:
                timeframe_scores = {}
                for timeframe, signal_data in timeframe_signals.items():
                    confidence = signal_data.get('confidence', 0)
                    try:
                        confidence = float(confidence) / 100.0 if confidence > 1 else float(confidence)
                    except:
                        confidence = 0.0
                    
                    base_weight = timeframe_weights.get(timeframe, 0.1)
                    
                    # 主导度 = 基础权重 × 置信度 × 信号强度
                    signal = signal_data.get('signal', 'neutral')
                    signal_strength = 1.0 if signal in ['bullish', 'bearish'] else 0.3
                    
                    dominance_score = base_weight * confidence * signal_strength
                    timeframe_scores[timeframe] = dominance_score
                
                if timeframe_scores:
                    dominant_timeframe = max(timeframe_scores, key=timeframe_scores.get)
            except:
                dominant_timeframe = '1h'
            
            # 3. 识别冲突区域 - 完全内联实现
            conflict_areas = []
            try:
                timeframes = list(timeframe_signals.keys())
                for i in range(len(timeframes)):
                    for j in range(i + 1, len(timeframes)):
                        tf1, tf2 = timeframes[i], timeframes[j]
                        signal1 = timeframe_signals[tf1].get('signal', 'neutral')
                        signal2 = timeframe_signals[tf2].get('signal', 'neutral')
                        
                        # 检查信号冲突（一个看多，一个看空）
                        if (signal1 == 'bullish' and signal2 == 'bearish') or \
                           (signal1 == 'bearish' and signal2 == 'bullish'):
                            conflict_pair = f"{tf1}-{tf2}"
                            if conflict_pair not in conflict_areas:
                                conflict_areas.append(conflict_pair)
            except:
                conflict_areas = []
            
            # 4. 评估趋势对齐 - 完全内联实现
            trend_alignment = 'mixed'
            try:
                bullish_count = 0
                bearish_count = 0
                
                for signals in timeframe_signals.values():
                    signal = signals.get('signal', 'neutral')
                    if signal == 'bullish':
                        bullish_count += 1
                    elif signal == 'bearish':
                        bearish_count += 1
                
                total_signals = len(timeframe_signals)
                if total_signals > 0:
                    if bullish_count >= total_signals * 0.6:
                        trend_alignment = 'bullish'
                    elif bearish_count >= total_signals * 0.6:
                        trend_alignment = 'bearish'
                    elif (total_signals - bullish_count - bearish_count) >= total_signals * 0.5:
                        trend_alignment = 'neutral'
                    else:
                        trend_alignment = 'mixed'
            except:
                trend_alignment = 'mixed'
            
            # 5. 评估整体信号强度 - 完全内联实现
            overall_signal_strength = 'weak'
            try:
                dominant_confidence = 0.0
                if dominant_timeframe in timeframe_signals:
                    conf = timeframe_signals[dominant_timeframe].get('confidence', 0)
                    try:
                        dominant_confidence = float(conf) / 100.0 if conf > 1 else float(conf)
                    except:
                        dominant_confidence = 0.0
                
                avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
                
                # 综合强度评分
                overall_strength_score = (
                    timeframe_consensus * 0.4 +  # 一致性权重40%
                    dominant_confidence * 0.3 +   # 主导时间框架置信度30%
                    avg_confidence * 0.3          # 平均置信度30%
                )
                
                if overall_strength_score >= 0.7:
                    overall_signal_strength = 'strong'
                elif overall_strength_score >= 0.5:
                    overall_signal_strength = 'moderate'
                else:
                    overall_signal_strength = 'weak'
            except:
                overall_signal_strength = 'weak'
            
            return {
                'timeframe_consensus': round(timeframe_consensus, 4),
                'dominant_timeframe': dominant_timeframe,
                'conflict_areas': conflict_areas,
                'trend_alignment': trend_alignment,
                'overall_signal_strength': overall_signal_strength
            }
            
        except Exception as e:
            print(f"Cross-timeframe analysis error: {e}")
            return {
                'timeframe_consensus': 0.5,
                'dominant_timeframe': '1h',
                'conflict_areas': [],
                'trend_alignment': 'mixed',
                'overall_signal_strength': 'weak'
            }
    
    def _calculate_timeframe_consensus(self, timeframe_signals: Dict[str, Dict[str, Any]], 
                                     timeframe_weights: Dict[str, float]) -> float:
        """
        计算时间框架一致性评分
        
        基于信号方向一致性和置信度的加权计算：
        1. 将信号转换为数值(-1: bearish, 0: neutral, 1: bullish)
        2. 根据置信度和时间框架权重计算加权信号强度
        3. 计算信号方向的方差以评估一致性
        4. 综合考虑信号强度和一致性得出最终评分
        
        Args:
            timeframe_signals: 时间框架信号数据
            timeframe_weights: 时间框架权重映射
            
        Returns:
            float: 一致性评分 (0-1)，1表示完全一致，0表示完全分歧
        """
        try:
            signal_values = []
            weights = []
            confidence_values = []
            
            for timeframe, signal_data in timeframe_signals.items():
                if not isinstance(signal_data, dict):
                    continue
                    
                signal = signal_data.get('signal', 'neutral')
                confidence = signal_data.get('confidence', 0) / 100.0  # 转换为0-1范围
                
                # 信号方向数值化
                signal_value = 0
                if signal == 'bullish':
                    signal_value = 1
                elif signal == 'bearish': 
                    signal_value = -1
                    
                # 获取时间框架权重
                timeframe_weight = timeframe_weights.get(timeframe, 0.1)
                
                # 综合权重 = 时间框架权重 × 置信度
                combined_weight = timeframe_weight * (0.5 + 0.5 * confidence)  # 置信度影响权重
                
                signal_values.append(signal_value)
                weights.append(combined_weight)
                confidence_values.append(confidence)
            
            if not signal_values or len(signal_values) < 2:
                return 0.5
            
            # 转换为numpy数组便于计算
            signal_array = np.array(signal_values)
            weight_array = np.array(weights)
            
            # 1. 计算加权平均信号方向
            total_weight = np.sum(weight_array)
            if total_weight == 0:
                return 0.5
                
            weighted_mean_signal = np.sum(signal_array * weight_array) / total_weight
            
            # 2. 计算信号方向的一致性
            # 使用加权方差来衡量信号的分散程度
            weighted_variance = np.sum(weight_array * (signal_array - weighted_mean_signal) ** 2) / total_weight
            
            # 将方差转换为一致性评分（方差越小，一致性越高）
            # 最大可能方差为4（全是+1和-1的混合），标准化到0-1范围
            consistency_score = max(0.0, 1.0 - weighted_variance / 4.0)
            
            # 3. 考虑信号强度（非中性信号的比例）
            non_neutral_signals = np.sum(signal_array != 0)
            signal_strength_factor = non_neutral_signals / len(signal_array)
            
            # 4. 考虑置信度因子
            avg_confidence = safe_mean(confidence_values)
            confidence_factor = avg_confidence
            
            # 5. 综合计算一致性评分
            # 一致性 = 基础一致性 × 信号强度因子 × 置信度因子
            final_consensus = consistency_score * (0.6 + 0.2 * signal_strength_factor + 0.2 * confidence_factor)
            
            return max(0.0, min(1.0, final_consensus))
            
        except Exception as e:
            print(f"Timeframe consensus calculation error: {e}")
            return 0.5
    
    def _identify_dominant_timeframe(self, timeframe_signals: Dict[str, Dict[str, Any]], 
                                   timeframe_weights: Dict[str, float]) -> str:
        """
        识别主导时间框架
        
        基于以下因素综合评估：
        1. 信号置信度
        2. 时间框架权重（长期 > 短期）
        3. 信号强度（非中性信号优先）
        4. 策略信号一致性
        
        Args:
            timeframe_signals: 时间框架信号数据
            timeframe_weights: 时间框架权重映射
            
        Returns:
            str: 主导时间框架 ("5m"|"15m"|"30m"|"1h"|"4h")
        """
        try:
            if not timeframe_signals:
                return "1h"  # 默认返回中期时间框架
            
            timeframe_scores = {}
            
            for timeframe, signal_data in timeframe_signals.items():
                if not isinstance(signal_data, dict):
                    continue
                    
                signal = signal_data.get('signal', 'neutral')
                confidence = signal_data.get('confidence', 0) / 100.0
                strategy_signals = signal_data.get('strategy_signals', {})
                
                # 1. 基础评分：置信度 × 时间框架权重
                base_score = confidence * timeframe_weights.get(timeframe, 0.1)
                
                # 2. 信号强度加成（非中性信号获得加成）
                signal_strength_bonus = 0.0
                if signal != 'neutral':
                    signal_strength_bonus = 0.2
                
                # 3. 策略信号一致性评估
                strategy_consistency_bonus = self._calculate_strategy_consistency_bonus(strategy_signals)
                
                # 4. 长期时间框架加成
                time_horizon_bonus = 0.0
                if timeframe in ['4h', '1h']:
                    time_horizon_bonus = 0.15
                elif timeframe in ['30m']:
                    time_horizon_bonus = 0.1
                
                # 5. 高置信度加成
                high_confidence_bonus = 0.0
                if confidence >= 0.8:
                    high_confidence_bonus = 0.2
                elif confidence >= 0.6:
                    high_confidence_bonus = 0.1
                
                # 6. 综合评分
                total_score = (base_score + 
                             signal_strength_bonus + 
                             strategy_consistency_bonus + 
                             time_horizon_bonus + 
                             high_confidence_bonus)
                
                timeframe_scores[timeframe] = total_score
            
            # 返回评分最高的时间框架
            if timeframe_scores:
                dominant_tf = max(timeframe_scores, key=timeframe_scores.get)
                return dominant_tf
            else:
                return "1h"  # 默认返回中期时间框架
                
        except Exception as e:
            print(f"Dominant timeframe identification error: {e}")
            return "1h"
    
    def _calculate_strategy_consistency_bonus(self, strategy_signals: Dict[str, Any]) -> float:
        """
        计算策略信号一致性加成
        
        评估单个时间框架内各策略信号的一致性程度
        
        Args:
            strategy_signals: 策略信号数据
            
        Returns:
            float: 一致性加成分数 (0-0.3)
        """
        try:
            if not strategy_signals:
                return 0.0
            
            # 提取各策略的信号方向
            signals = []
            confidences = []
            
            for strategy_name, strategy_data in strategy_signals.items():
                if isinstance(strategy_data, dict):
                    signal = strategy_data.get('signal', 'neutral')
                    confidence = strategy_data.get('confidence', 0) / 100.0
                    
                    # 转换信号为数值
                    signal_value = 0
                    if signal == 'bullish':
                        signal_value = 1
                    elif signal == 'bearish':
                        signal_value = -1
                    
                    signals.append(signal_value)
                    confidences.append(confidence)
            
            if len(signals) < 2:
                return 0.0
            
            # 计算策略信号的标准差（一致性指标）
            signal_std = safe_std(signals)
            
            # 标准差越小，一致性越高，加成越大
            # 最大可能标准差约为1（全是+1和-1），标准化处理
            consistency_factor = max(0.0, 1.0 - signal_std / 1.0)
            
            # 考虑平均置信度
            avg_confidence = safe_mean(confidences) if confidences else 0.0
            
            # 综合计算加成分数
            bonus = consistency_factor * avg_confidence * 0.3  # 最大加成0.3
            
            return bonus
            
        except Exception as e:
            print(f"Strategy consistency bonus calculation error: {e}")
            return 0.0
    
    def _identify_conflict_areas(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> list:
        """
        识别时间框架信号冲突区域
        
        找出信号方向相互冲突的时间框架组合，用于风险评估和决策调整
        
        Args:
            timeframe_signals: 时间框架信号数据
            
        Returns:
            list: 冲突的时间框架对列表，格式如["5m_vs_4h", "15m_vs_1h"]
        """
        try:
            timeframes = list(timeframe_signals.keys())
            conflicts = []
            
            # 两两比较时间框架信号
            for i in range(len(timeframes)):
                for j in range(i + 1, len(timeframes)):
                    tf1, tf2 = timeframes[i], timeframes[j]
                    
                    # 获取信号数据
                    signal1_data = timeframe_signals.get(tf1, {})
                    signal2_data = timeframe_signals.get(tf2, {})
                    
                    if not isinstance(signal1_data, dict) or not isinstance(signal2_data, dict):
                        continue
                    
                    signal1 = signal1_data.get('signal', 'neutral')
                    signal2 = signal2_data.get('signal', 'neutral')
                    confidence1 = signal1_data.get('confidence', 0)
                    confidence2 = signal2_data.get('confidence', 0)
                    
                    # 检查是否存在信号冲突
                    is_conflict = self._check_signal_conflict(
                        signal1, signal2, confidence1, confidence2
                    )
                    
                    if is_conflict:
                        # 按时间框架优先级排序（长期在前）
                        tf_priority = {'4h': 4, '1h': 3, '30m': 2, '15m': 1, '5m': 0}
                        
                        if tf_priority.get(tf1, 0) > tf_priority.get(tf2, 0):
                            conflict_pair = f"{tf1}_vs_{tf2}"
                        else:
                            conflict_pair = f"{tf2}_vs_{tf1}"
                        
                        conflicts.append(conflict_pair)
            
            return conflicts
            
        except Exception as e:
            print(f"Conflict areas identification error: {e}")
            return []
    
    def _check_signal_conflict(self, signal1: str, signal2: str, 
                             confidence1: float, confidence2: float) -> bool:
        """
        检查两个信号是否冲突
        
        冲突定义：
        1. 信号方向完全相反（bullish vs bearish）
        2. 至少一个信号的置信度 >= 50%（避免低置信度噪音）
        
        Args:
            signal1, signal2: 信号方向
            confidence1, confidence2: 信号置信度
            
        Returns:
            bool: 是否存在冲突
        """
        try:
            # 1. 检查信号方向是否相反
            opposite_signals = (
                (signal1 == 'bullish' and signal2 == 'bearish') or
                (signal1 == 'bearish' and signal2 == 'bullish')
            )
            
            if not opposite_signals:
                return False
            
            # 2. 检查置信度阈值
            # 只有当至少一个信号的置信度达到50%时才认为是有意义的冲突
            meaningful_confidence = confidence1 >= 50 or confidence2 >= 50
            
            return meaningful_confidence
            
        except Exception as e:
            print(f"Signal conflict check error: {e}")
            return False
    
    def _assess_trend_alignment(self, timeframe_signals: Dict[str, Dict[str, Any]], 
                              consensus_score: float) -> str:
        """
        评估趋势对齐情况
        
        基于信号方向分布和一致性评分判断趋势对齐状态：
        1. "aligned": 大部分时间框架信号方向一致
        2. "divergent": 时间框架信号严重分歧
        3. "mixed": 部分一致部分分歧的混合状态
        
        Args:
            timeframe_signals: 时间框架信号数据
            consensus_score: 已计算的一致性评分
            
        Returns:
            str: 趋势对齐状态 ("aligned"|"divergent"|"mixed")
        """
        try:
            if not timeframe_signals:
                return "mixed"
            
            # 统计不同信号方向的数量
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            total_signals = 0
            
            # 统计高置信度信号
            high_confidence_bullish = 0
            high_confidence_bearish = 0
            
            for signal_data in timeframe_signals.values():
                if not isinstance(signal_data, dict):
                    continue
                    
                signal = signal_data.get('signal', 'neutral')
                confidence = signal_data.get('confidence', 0)
                
                total_signals += 1
                
                if signal == 'bullish':
                    bullish_count += 1
                    if confidence >= 70:
                        high_confidence_bullish += 1
                elif signal == 'bearish':
                    bearish_count += 1
                    if confidence >= 70:
                        high_confidence_bearish += 1
                else:
                    neutral_count += 1
            
            if total_signals == 0:
                return "mixed"
            
            # 计算信号方向比例
            bullish_ratio = bullish_count / total_signals
            bearish_ratio = bearish_count / total_signals
            neutral_ratio = neutral_count / total_signals
            dominant_ratio = max(bullish_ratio, bearish_ratio)
            
            # 基于一致性评分和信号分布判断对齐状态
            if consensus_score >= 0.75:
                # 高一致性情况下
                if dominant_ratio >= 0.7:  # 70%以上的信号方向一致
                    return "aligned"
                elif neutral_ratio >= 0.5:  # 大部分信号为中性
                    return "mixed"
                else:
                    return "aligned"
                    
            elif consensus_score >= 0.4:
                # 中等一致性情况下
                if dominant_ratio >= 0.8:  # 需要更高的比例才认为对齐
                    return "aligned"
                elif abs(bullish_ratio - bearish_ratio) <= 0.2:  # 多空相对平衡
                    return "mixed"
                else:
                    return "mixed"
                    
            else:
                # 低一致性情况下
                if dominant_ratio >= 0.9:  # 需要极高比例才认为对齐
                    return "aligned"
                elif abs(bullish_ratio - bearish_ratio) <= 0.1:  # 多空严重分歧
                    return "divergent"
                else:
                    return "divergent"
                    
        except Exception as e:
            print(f"Trend alignment assessment error: {e}")
            return "mixed"
    
    def _evaluate_overall_signal_strength(self, timeframe_signals: Dict[str, Dict[str, Any]], 
                                        consensus_score: float, trend_alignment: str, 
                                        conflict_areas: list) -> str:
        """
        综合评估整体信号强度
        
        基于多个因子综合判断：
        1. 时间框架一致性评分
        2. 趋势对齐状态
        3. 冲突区域数量  
        4. 平均置信度
        5. 主导信号比例
        
        Args:
            timeframe_signals: 时间框架信号数据
            consensus_score: 一致性评分
            trend_alignment: 趋势对齐状态
            conflict_areas: 冲突区域列表
            
        Returns:
            str: 整体信号强度 ("weak"|"moderate"|"strong")
        """
        try:
            if not timeframe_signals:
                return "weak"
            
            # 1. 计算基础强度分数
            base_score = 0.0
            
            # 一致性评分贡献 (0-0.4)
            consensus_contribution = consensus_score * 0.4
            base_score += consensus_contribution
            
            # 2. 趋势对齐状态贡献 (0-0.25) 
            alignment_contribution = 0.0
            if trend_alignment == "aligned":
                alignment_contribution = 0.25
            elif trend_alignment == "mixed":
                alignment_contribution = 0.15
            else:  # divergent
                alignment_contribution = 0.05
            base_score += alignment_contribution
            
            # 3. 冲突惩罚 (0到-0.2)
            conflict_penalty = min(0.2, len(conflict_areas) * 0.05)
            base_score -= conflict_penalty
            
            # 4. 平均置信度贡献 (0-0.2)
            total_confidence = 0
            confidence_count = 0
            high_confidence_count = 0
            
            for signal_data in timeframe_signals.values():
                if isinstance(signal_data, dict):
                    confidence = signal_data.get('confidence', 0)
                    total_confidence += confidence
                    confidence_count += 1
                    
                    if confidence >= 75:
                        high_confidence_count += 1
            
            if confidence_count > 0:
                avg_confidence = total_confidence / confidence_count / 100.0  # 转换为0-1
                confidence_contribution = avg_confidence * 0.2
                base_score += confidence_contribution
            
            # 5. 高置信度信号比例加成 (0-0.15)
            if confidence_count > 0:
                high_confidence_ratio = high_confidence_count / confidence_count
                high_confidence_bonus = high_confidence_ratio * 0.15
                base_score += high_confidence_bonus
            
            # 6. 信号参与度评估（非中性信号比例）
            non_neutral_count = 0
            total_count = 0
            
            for signal_data in timeframe_signals.values():
                if isinstance(signal_data, dict):
                    signal = signal_data.get('signal', 'neutral')
                    total_count += 1
                    if signal != 'neutral':
                        non_neutral_count += 1
            
            if total_count > 0:
                signal_participation = non_neutral_count / total_count
                participation_bonus = signal_participation * 0.1
                base_score += participation_bonus
            
            # 7. 根据最终评分确定强度等级
            # 总分范围: 0-1.0（理论最大值）
            if base_score >= 0.75:
                return "strong"
            elif base_score >= 0.45:
                return "moderate"  
            else:
                return "weak"
                
        except Exception as e:
            print(f"Overall signal strength evaluation error: {e}")
            return "weak"
    
    def _get_default_cross_timeframe_analysis(self) -> Dict[str, Any]:
        """
        返回默认的跨时间框架分析结果
        
        在数据不足或发生错误时使用的默认值
        
        Returns:
            Dict[str, Any]: 默认分析结果
        """
        return {
            'timeframe_consensus': 0.5,
            'dominant_timeframe': '1h',
            'conflict_areas': [],
            'trend_alignment': 'mixed',
            'overall_signal_strength': 'weak'
        }
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """
        Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
        1. Trend Following
        2. Mean Reversion
        3. Momentum
        4. Volatility Analysis
        5. Statistical Arbitrage Signals
        """

        data = state['data']
        data['name'] = "RSIStrategy"

        data = state.get("data", {})
        tickers = data.get("tickers", [])
        intervals = data.get("intervals", [])

        # Initialize analysis for each ticker
        technical_analysis = {}
        for ticker in tickers:
            technical_analysis[ticker] = {}

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        for ticker in tickers:
            # 收集该ticker所有时间框架的信号数据（用于跨时间框架分析）
            timeframe_signals = {}
            
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())
                
                # 如果数据为空，跳过该时间框架
                if df.empty:
                    continue

                # 计算各种策略信号
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

                # 构建策略信号结构（用于signal_metadata生成和跨时间框架分析）
                strategy_signals_data = {
                    "trend_following": {
                        "signal": trend_signals["signal"],
                        "confidence": round(trend_signals["confidence"] * 100),
                        "metrics": normalize_pandas(trend_signals["metrics"]),
                    },
                    "mean_reversion": {
                        "signal": mean_reversion_signals["signal"],
                        "confidence": round(mean_reversion_signals["confidence"] * 100),
                        "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                    },
                    "momentum": {
                        "signal": momentum_signals["signal"],
                        "confidence": round(momentum_signals["confidence"] * 100),
                        "metrics": normalize_pandas(momentum_signals["metrics"]),
                    },
                    "volatility": {
                        "signal": volatility_signals["signal"],
                        "confidence": round(volatility_signals["confidence"] * 100),
                        "metrics": normalize_pandas(volatility_signals["metrics"]),
                    },
                    "statistical_arbitrage": {
                        "signal": stat_arb_signals["signal"],
                        "confidence": round(stat_arb_signals["confidence"] * 100),
                        "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                    },
                }

                # 计算新增分析功能 - 使用完全隔离的try-except避免递归
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
                signal_metadata = {
                    'signal_strength': 'weak',
                    'signal_decay_time': 240,
                    'signal_reliability': 0.3,
                    'confirmation_status': 'weak'
                }
                
                try:
                    # ATR值计算 - 独立异常处理
                    atr_values = self.calculate_atr_values(df)
                except:
                    # 静默使用默认值，避免任何递归可能
                    pass
                    
                try:
                    # 关键价位识别 - 独立异常处理
                    price_levels = self.identify_price_levels(df)
                except:
                    # 静默使用默认值，避免任何递归可能
                    pass
                    
                try:
                    # 波动率深度分析 - 独立异常处理
                    volatility_analysis = self.analyze_volatility_depth(df)
                except:
                    # 静默使用默认值，避免任何递归可能
                    pass
                    
                try:
                    # 为signal_metadata准备完整的信号数据
                    current_signal_data = {
                        "signal": combined_signal["signal"],
                        "confidence": round(combined_signal["confidence"] * 100),
                        "strategy_signals": strategy_signals_data
                    }
                    
                    # 生成信号元数据 - 独立异常处理
                    signal_metadata = self.generate_signal_metadata(df, current_signal_data)
                except:
                    # 静默使用默认值，避免任何递归可能
                    pass

                # 生成该时间框架的完整分析结果
                technical_analysis[ticker][interval.value] = {
                    # === 现有字段保持不变（向后兼容性） ===
                    "signal": combined_signal["signal"],
                    "confidence": round(combined_signal["confidence"] * 100),
                    "strategy_signals": strategy_signals_data,
                    
                    # === 新增：合约交易字段 ===
                    # ATR值计算基础
                    "atr_values": atr_values,
                    
                    # 关键价位识别
                    "price_levels": price_levels,
                    
                    # 波动率深度分析
                    "volatility_analysis": volatility_analysis,
                    
                    # 信号时效性
                    "signal_metadata": signal_metadata,
                }
                
                # 收集该时间框架的信号数据用于跨时间框架分析
                timeframe_signals[interval.value] = {
                    "signal": combined_signal["signal"],
                    "confidence": round(combined_signal["confidence"] * 100),
                    "strategy_signals": strategy_signals_data
                }
            
            # === 新增：跨时间框架综合分析（ticker级别字段） ===
            cross_timeframe_result = {
                'timeframe_consensus': 0.5,
                'dominant_timeframe': '1h',  # 默认中期时间框架
                'conflict_areas': [],
                'trend_alignment': 'mixed',
                'overall_signal_strength': 'weak'
            }
            
            try:
                if timeframe_signals:  # 确保有数据才进行分析
                    cross_timeframe_result = self.cross_timeframe_analysis(timeframe_signals)
            except:
                # 静默使用默认值，完全避免任何递归可能
                pass
            
            # 将跨时间框架分析结果添加到该ticker的顶级分析结果中
            technical_analysis[ticker]["cross_timeframe_analysis"] = cross_timeframe_result

        # Create the technical analyst message
        message = HumanMessage(
            content=json.dumps(technical_analysis),
            name="technical_analyst_agent",
        )

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "Technical Analyst")

        # Add the signal to the analyst_signals list
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

        return {
            "messages": [message],
            "data": data,
        }
