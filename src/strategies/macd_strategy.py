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


class MacdStrategy(BaseNode):
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
            
            # 1. 计算枢轴点（使用标准公式）
            pivot_point = self._calculate_pivot_point(df_normalized)
            
            # 2. 识别支撑和阻力位
            support_levels = self._identify_support_levels(df_normalized, current_price)
            resistance_levels = self._identify_resistance_levels(df_normalized, current_price)
            
            # 3. 计算突破临界点
            breakout_threshold = self._calculate_breakout_threshold(df_normalized, current_price)
            
            # 4. 验证价位逻辑一致性
            support_levels = self._validate_support_levels(support_levels, current_price)
            resistance_levels = self._validate_resistance_levels(resistance_levels, current_price)
            
            return {
                'support_levels': [round(float(level), 8) for level in support_levels],
                'resistance_levels': [round(float(level), 8) for level in resistance_levels],
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
            
            # 1. 计算多个时间窗口的历史波动率
            volatility_data = self._calculate_historical_volatility(df_normalized)
            
            # 2. 计算波动率百分位数
            volatility_percentile = self._calculate_volatility_percentile(volatility_data)
            
            # 3. 识别波动率趋势
            volatility_trend = self._identify_volatility_trend(volatility_data)
            
            # 4. 预测未来波动率
            volatility_forecast = self._forecast_volatility(volatility_data)
            
            # 5. 计算波动率状态概率
            regime_probability = self._calculate_regime_probability(volatility_data)
            
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
                vol = np.std(window_returns, ddof=1) * np.sqrt(252)
                rolling_vol.append(vol)
            
            volatility_data[f'vol_{window}'] = np.array(rolling_vol)
        
        # 添加当前波动率
        if len(log_returns) >= 14:
            current_vol = np.std(log_returns[-14:], ddof=1) * np.sqrt(252)
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
            recent_short = np.mean(vol_14[-5:])
            recent_medium = np.mean(vol_30[-5:])
            
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
            avg_vol = np.mean(recent_vol)
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
        historical_mean = np.mean(vol_30)
        mean_reversion_speed = 0.1  # 均值回归速度
        mean_reversion_forecast = current_vol + mean_reversion_speed * (historical_mean - current_vol)
        
        # 方法3: 趋势延续
        if len(vol_30) >= 10:
            recent_trend = np.mean(vol_30[-5:]) - np.mean(vol_30[-10:-5])
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
            mean_low = np.mean(vol_30[vol_30 <= low_vol_threshold])
            std_low = np.std(vol_30[vol_30 <= low_vol_threshold]) if len(vol_30[vol_30 <= low_vol_threshold]) > 1 else low_vol_threshold * 0.1
            
            # 计算在低波动率分布中的概率密度
            if std_low > 0:
                z_score = abs(current_vol - mean_low) / std_low
                probability = np.exp(-0.5 * z_score**2)  # 简化的正态分布概率
            else:
                probability = 1.0
                
        elif current_vol >= high_vol_threshold:
            # 高波动率状态概率
            mean_high = np.mean(vol_30[vol_30 >= high_vol_threshold])
            std_high = np.std(vol_30[vol_30 >= high_vol_threshold]) if len(vol_30[vol_30 >= high_vol_threshold]) > 1 else high_vol_threshold * 0.1
            
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
                volume_factor = volumes[i] / np.mean(volumes[max(0, i-20):i+20])
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
                volume_factor = volumes[i] / np.mean(volumes[max(0, i-20):i+20])
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
        data['name'] = "MacdStrategy"

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
            for interval in intervals:
                df = data.get(f"{ticker}_{interval.value}", pd.DataFrame())

                trend_signals = calculate_trend_signals(df)
                mean_reversion_signals = calculate_mean_reversion_signals(df)
                momentum_signals = calculate_momentum_signals(df)

                volatility_signals = calculate_volatility_signals(df)
                stat_arb_signals = calculate_stat_arb_signals(df)

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

                # Generate detailed analysis report for this ticker
                technical_analysis[ticker][interval.value] = {
                    "signal": combined_signal["signal"],
                    "confidence": round(combined_signal["confidence"] * 100),
                    "strategy_signals": {
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
                    },
                }

        # Create the technical analyst message
        message = HumanMessage(
            content=json.dumps(technical_analysis),
            name="technical_analyst_agent",
        )

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(technical_analysis, "Technical Analyst")

        # Add the signal to the analyst_signals list
        state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

        # return state
        # # print(state)

        return {
            "messages": [message],
            "data": data,
        }
