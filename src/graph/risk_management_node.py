import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from langchain_core.messages import HumanMessage
from scipy.stats import norm
from .state import AgentState, show_agent_reasoning
from .base_node import BaseNode
from src.utils import Interval


class RiskManagementNode(BaseNode):
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Controls position sizing based on real-world risk factors for multiple tickers."""
        data = state.get('data', {})
        data['name'] = "RiskManagementNode"

        portfolio = data.get("portfolio", {})
        tickers = data.get("tickers", [])
        primary_interval: Optional[Interval] = data.get("primary_interval")

        risk_analysis = {}
        current_prices = {}  # Store prices here to avoid redundant API calls

        for ticker in tickers:

            price_df = data.get(f"{ticker}_{primary_interval.value}")

            # Calculate portfolio value
            current_price = price_df["close"].iloc[-1]
            current_prices[ticker] = current_price  # Store the current price

            # Calculate current position value for this ticker
            current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0.0)

            # Calculate total portfolio value using stored prices
            total_portfolio_value = portfolio.get("cash", 0.0) + sum(
                portfolio.get("cost_basis", {}).get(t, 0.0) for t in portfolio.get("cost_basis", {}))

            # Base limit is 20% of portfolio for any single position
            position_limit = total_portfolio_value * 0.20

            # For existing positions, subtract current position value from limit
            remaining_position_limit = position_limit - current_position_value

            # Ensure we don't exceed available cash
            max_position_size = min(remaining_position_limit, portfolio.get("cash", 0.0))

            risk_analysis[ticker] = {
                "remaining_position_limit": float(max_position_size),
                "current_price": float(current_price),
                "reasoning": {
                    "portfolio_value": float(total_portfolio_value),
                    "current_position": float(current_position_value),
                    "position_limit": float(position_limit),
                    "remaining_limit": float(remaining_position_limit),
                    "available_cash": float(portfolio.get("cash", 0.0)),
                },
            }

        # 添加杠杆分析功能
        leverage_analysis = self.analyze_leverage(state)
        
        # 将杠杆分析结果整合到风险分析中
        for ticker in tickers:
            if ticker in leverage_analysis:
                risk_analysis[ticker].update(leverage_analysis[ticker])
        
        # 添加保证金管理分析功能
        margin_management_analysis = self.calculate_margin_management(state)
        
        # 将保证金管理结果整合到风险分析中
        for ticker in tickers:
            if ticker in margin_management_analysis:
                risk_analysis[ticker].update(margin_management_analysis[ticker])
        
        # 添加动态风险指标计算功能
        dynamic_risk_analysis = self.calculate_dynamic_risk_metrics(state)
        
        # 将动态风险指标结果整合到风险分析中
        for ticker in tickers:
            if ticker in dynamic_risk_analysis:
                risk_analysis[ticker].update(dynamic_risk_analysis[ticker])
        
        # 添加强平风险分析功能
        liquidation_risk_analysis = self.analyze_liquidation_risk(state)
        
        # 将强平风险分析结果整合到风险分析中
        for ticker in tickers:
            if ticker in liquidation_risk_analysis:
                risk_analysis[ticker].update(liquidation_risk_analysis[ticker])

        message = HumanMessage(
            content=json.dumps(risk_analysis),
            name="risk_management_agent",
        )

        if state["metadata"]["show_reasoning"]:
            show_agent_reasoning(risk_analysis, "Risk Management Agent")

        # Add the signal to the analyst_signals list
        data["analyst_signals"]["risk_management_agent"] = risk_analysis

        return {
            "messages": [message],
            "data": data,
        }

    def analyze_leverage(self, state: AgentState) -> Dict[str, Dict[str, Any]]:
        """
        基于实际市场波动率和当前资金状况分析杠杆倍数建议
        
        Args:
            state: AgentState包含技术分析数据、投资组合信息等
            
        Returns:
            Dict[str, Dict[str, Any]]: 每个ticker的杠杆分析结果
        """
        data = state.get('data', {})
        portfolio = data.get("portfolio", {})
        tickers = data.get("tickers", [])
        primary_interval = data.get("primary_interval")
        
        # 获取技术分析数据
        technical_signals = data.get("analyst_signals", {}).get("technical_analyst_agent", {})
        
        leverage_analysis = {}
        
        for ticker in tickers:
            try:
                # 获取价格数据
                price_df = data.get(f"{ticker}_{primary_interval.value}")
                if price_df is None or price_df.empty:
                    leverage_analysis[ticker] = self._get_default_leverage_analysis()
                    continue
                
                current_price = price_df["close"].iloc[-1]
                
                # 获取技术分析数据
                tech_data = technical_signals.get(ticker, {})
                interval_data = tech_data.get(primary_interval.value, {})
                
                # 提取关键技术指标
                atr_values = interval_data.get("atr_values", {})
                volatility_analysis = interval_data.get("volatility_analysis", {})
                cross_timeframe = tech_data.get("cross_timeframe_analysis", {})
                
                # 计算投资组合相关指标
                total_portfolio_value = portfolio.get("cash", 0.0) + sum(
                    portfolio.get("cost_basis", {}).get(t, 0.0) for t in portfolio.get("cost_basis", {}))
                available_cash = portfolio.get("cash", 0.0)
                current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0.0)
                
                # 基于波动率和资金状况计算杠杆建议
                leverage_metrics = self._calculate_leverage_metrics(
                    atr_values=atr_values,
                    volatility_analysis=volatility_analysis,
                    cross_timeframe=cross_timeframe,
                    portfolio_value=total_portfolio_value,
                    available_cash=available_cash,
                    current_position=current_position_value,
                    current_price=current_price
                )
                
                leverage_analysis[ticker] = {
                    "leverage_analysis": leverage_metrics,
                    "current_price": float(current_price),
                    "reasoning": {
                        "atr_14": atr_values.get("atr_14", 0.0),
                        "atr_percentile": atr_values.get("atr_percentile", 50.0),
                        "volatility_percentile": volatility_analysis.get("volatility_percentile", 50.0),
                        "volatility_trend": volatility_analysis.get("volatility_trend", "stable"),
                        "timeframe_consensus": cross_timeframe.get("timeframe_consensus", 0.5),
                        "portfolio_value": float(total_portfolio_value),
                        "available_cash": float(available_cash)
                    }
                }
                
            except Exception as e:
                print(f"Error analyzing leverage for {ticker}: {e}")
                leverage_analysis[ticker] = {
                    "leverage_analysis": self._get_default_leverage_analysis(),
                    "current_price": float(price_df["close"].iloc[-1]) if price_df is not None and not price_df.empty else 0.0,
                    "reasoning": {"error": str(e)}
                }
        
        return leverage_analysis
    
    def _calculate_leverage_metrics(self, atr_values: Dict, volatility_analysis: Dict, 
                                  cross_timeframe: Dict, portfolio_value: float, 
                                  available_cash: float, current_position: float, 
                                  current_price: float) -> Dict[str, Any]:
        """计算杠杆相关指标"""
        
        # 提取关键指标
        atr_14 = atr_values.get("atr_14", 0.0)
        atr_percentile = atr_values.get("atr_percentile", 50.0)
        volatility_percentile = volatility_analysis.get("volatility_percentile", 50.0)
        volatility_trend = volatility_analysis.get("volatility_trend", "stable")
        timeframe_consensus = cross_timeframe.get("timeframe_consensus", 0.5)
        
        # 计算基础风险分数 (0-1，越高风险越大)
        base_risk_score = self._calculate_base_risk_score(
            atr_percentile, volatility_percentile, volatility_trend, timeframe_consensus
        )
        
        # 计算资金利用率风险
        position_ratio = current_position / max(portfolio_value, 1.0) if portfolio_value > 0 else 0.0
        cash_ratio = available_cash / max(portfolio_value, 1.0) if portfolio_value > 0 else 0.0
        
        # 资金风险调整
        capital_risk_adjustment = min(1.0, position_ratio * 2.0)  # 已有持仓越多，风险越高
        cash_availability_factor = max(0.1, cash_ratio)  # 可用现金越少，杠杆越保守
        
        # 综合风险评分
        final_risk_score = min(1.0, base_risk_score + capital_risk_adjustment * 0.3)
        
        # 基于风险评分计算杠杆倍数
        max_safe_leverage = self._calculate_max_safe_leverage(final_risk_score, cash_availability_factor)
        recommended_leverage = self._calculate_recommended_leverage(final_risk_score, max_safe_leverage)
        volatility_adjusted_leverage = self._calculate_volatility_adjusted_leverage(
            recommended_leverage, volatility_percentile, atr_percentile
        )
        
        # 生成杠杆选项
        leverage_options = self._generate_leverage_options(recommended_leverage, max_safe_leverage)
        
        return {
            "recommended_leverage": int(recommended_leverage),
            "max_safe_leverage": int(max_safe_leverage),
            "leverage_options": leverage_options,
            "leverage_risk_score": round(final_risk_score, 4),
            "volatility_adjusted_leverage": int(volatility_adjusted_leverage)
        }
    
    def _calculate_base_risk_score(self, atr_percentile: float, volatility_percentile: float, 
                                 volatility_trend: str, timeframe_consensus: float) -> float:
        """计算基础风险分数"""
        # ATR百分位数风险 (越高越危险)
        atr_risk = (atr_percentile / 100.0) * 0.3
        
        # 波动率百分位数风险
        vol_risk = (volatility_percentile / 100.0) * 0.3
        
        # 波动率趋势风险
        trend_risk_map = {"increasing": 0.3, "stable": 0.1, "decreasing": 0.0}
        trend_risk = trend_risk_map.get(volatility_trend, 0.1)
        
        # 时间框架共识风险 (共识度低表示信号不明确，风险更高)
        consensus_risk = (1.0 - timeframe_consensus) * 0.1
        
        return min(1.0, atr_risk + vol_risk + trend_risk + consensus_risk)
    
    def _calculate_max_safe_leverage(self, risk_score: float, cash_factor: float) -> int:
        """计算最大安全杠杆倍数"""
        # 基础最大杠杆：低风险125x，高风险1x
        base_max = max(1, int(125 * (1.0 - risk_score)))
        
        # 根据可用现金调整
        cash_adjusted = max(1, int(base_max * cash_factor))
        
        # 确保在合理范围内
        return min(125, max(1, cash_adjusted))
    
    def _calculate_recommended_leverage(self, risk_score: float, max_safe: int) -> int:
        """计算推荐杠杆倍数（通常是最大安全杠杆的60-80%）"""
        safety_factor = 0.7  # 保守系数
        recommended = int(max_safe * safety_factor)
        return min(max_safe, max(1, recommended))
    
    def _calculate_volatility_adjusted_leverage(self, base_leverage: int, 
                                              vol_percentile: float, atr_percentile: float) -> int:
        """根据波动率调整杠杆倍数"""
        # 计算波动率调整因子
        vol_adjustment = 1.0 - (vol_percentile / 100.0) * 0.3  # 高波动率降低杠杆
        atr_adjustment = 1.0 - (atr_percentile / 100.0) * 0.2   # 高ATR降低杠杆
        
        combined_adjustment = min(vol_adjustment, atr_adjustment)
        adjusted = int(base_leverage * combined_adjustment)
        
        return min(125, max(1, adjusted))
    
    def _generate_leverage_options(self, recommended: int, max_safe: int) -> list:
        """生成3个杠杆选项：保守、推荐、激进"""
        conservative = max(1, int(recommended * 0.7))
        aggressive = min(max_safe, int(recommended * 1.3))
        
        options = [conservative, recommended, aggressive]
        # 去重并排序
        options = sorted(list(set(options)))
        
        # 确保有3个选项
        while len(options) < 3:
            if options[-1] < 125:
                options.append(min(125, options[-1] + 1))
            elif options[0] > 1:
                options.insert(0, max(1, options[0] - 1))
            else:
                break
        
        return options[:3]  # 返回前3个选项
    
    def _get_default_leverage_analysis(self) -> Dict[str, Any]:
        """返回默认的杠杆分析结果"""
        return {
            "recommended_leverage": 5,
            "max_safe_leverage": 10,
            "leverage_options": [3, 5, 8],
            "leverage_risk_score": 0.5,
            "volatility_adjusted_leverage": 4
        }

    def calculate_margin_management(self, state: AgentState) -> Dict[str, Dict[str, Any]]:
        """
        基于实际账户资金和真实保证金要求进行保证金管理计算
        
        Args:
            state: AgentState包含投资组合信息、杠杆分析等
            
        Returns:
            Dict[str, Dict[str, Any]]: 每个ticker的保证金管理结果
        """
        data = state.get('data', {})
        portfolio = data.get("portfolio", {})
        tickers = data.get("tickers", [])
        primary_interval = data.get("primary_interval")
        
        # 获取杠杆分析结果
        analyst_signals = data.get("analyst_signals", {})
        risk_signals = analyst_signals.get("risk_management_agent", {})
        
        margin_management = {}
        
        for ticker in tickers:
            try:
                # 获取价格数据
                price_df = data.get(f"{ticker}_{primary_interval.value}")
                if price_df is None or price_df.empty:
                    margin_management[ticker] = self._get_default_margin_management()
                    continue
                
                current_price = price_df["close"].iloc[-1]
                
                # 获取杠杆分析数据
                ticker_risk_data = risk_signals.get(ticker, {})
                leverage_analysis = ticker_risk_data.get("leverage_analysis", {})
                
                # 获取投资组合数据
                total_portfolio_value = portfolio.get("cash", 0.0) + sum(
                    portfolio.get("cost_basis", {}).get(t, 0.0) for t in portfolio.get("cost_basis", {}))
                available_cash = portfolio.get("cash", 0.0)
                current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0.0)
                base_margin_requirement = portfolio.get("margin_requirement", 0.0)
                
                # 获取推荐杠杆
                recommended_leverage = leverage_analysis.get("recommended_leverage", 10)
                max_safe_leverage = leverage_analysis.get("max_safe_leverage", 20)
                
                # 计算保证金管理指标
                margin_metrics = self._calculate_margin_metrics(
                    current_price=current_price,
                    portfolio_value=total_portfolio_value,
                    available_cash=available_cash,
                    current_position=current_position_value,
                    recommended_leverage=recommended_leverage,
                    max_safe_leverage=max_safe_leverage,
                    base_margin_requirement=base_margin_requirement,
                    ticker=ticker
                )
                
                margin_management[ticker] = {
                    "margin_management": margin_metrics
                }
                
            except Exception as e:
                print(f"Error calculating margin management for {ticker}: {e}")
                margin_management[ticker] = {
                    "margin_management": self._get_default_margin_management()
                }
        
        return margin_management
    
    def _calculate_margin_metrics(self, current_price: float, portfolio_value: float,
                                available_cash: float, current_position: float,
                                recommended_leverage: int, max_safe_leverage: int,
                                base_margin_requirement: float, ticker: str) -> Dict[str, float]:
        """计算保证金管理相关指标"""
        
        # 基于杠杆计算保证金比例
        if recommended_leverage > 1:
            # 杠杆越高，保证金比例越低 (initial_margin = 1 / leverage)
            initial_margin_ratio = 1.0 / recommended_leverage
        else:
            initial_margin_ratio = max(0.1, base_margin_requirement)  # 至少10%保证金
        
        # 维持保证金通常是初始保证金的60-80%
        maintenance_margin_ratio = initial_margin_ratio * 0.75
        
        # 假设单笔交易最大仓位价值（基于风险控制，不超过总资产的20%）
        max_position_value = min(portfolio_value * 0.2, available_cash * recommended_leverage)
        
        # 计算基于最大仓位的保证金需求
        initial_margin = max_position_value * initial_margin_ratio
        maintenance_margin = max_position_value * maintenance_margin_ratio
        
        # 保证金缓冲区 (维持保证金基础上额外20%的缓冲)
        margin_buffer = maintenance_margin * 0.2
        
        # 计算保证金使用率
        # 当前已用保证金 = 当前持仓价值 / 杠杆
        current_margin_used = current_position / max(recommended_leverage, 1) if current_position > 0 else 0.0
        # 保证金使用率 = 已用保证金 / 总可用资金
        total_funds = portfolio_value  # 使用总投资组合价值作为基准
        margin_utilization = current_margin_used / max(total_funds, 1.0) if total_funds > 0 else 0.0
        
        # 可用保证金
        available_margin = available_cash - current_margin_used
        
        # 追保门槛 (维持保证金 + 缓冲区)
        margin_call_threshold = maintenance_margin + margin_buffer
        
        # 强平门槛 (维持保证金)
        liquidation_threshold = maintenance_margin
        
        # 强平价格计算（针对多头仓位）
        if current_position > 0 and recommended_leverage > 1:
            # 强平价格 = 开仓价格 * (1 - 1/杠杆 + 手续费)
            # 这里简化计算，假设开仓价格为当前价格
            liquidation_price_ratio = 1.0 - (1.0 / recommended_leverage) + 0.001  # 加上0.1%的手续费缓冲
            liquidation_price = current_price * liquidation_price_ratio
        else:
            liquidation_price = 0.0
        
        return {
            "initial_margin": round(initial_margin, 2),
            "maintenance_margin": round(maintenance_margin, 2),
            "margin_buffer": round(margin_buffer, 2),
            "margin_utilization": round(margin_utilization, 4),
            "available_margin": round(available_margin, 2),
            "margin_call_threshold": round(margin_call_threshold, 2),
            "liquidation_threshold": round(liquidation_threshold, 2)
        }
    
    def _get_default_margin_management(self) -> Dict[str, float]:
        """返回默认的保证金管理结果"""
        return {
            "initial_margin": 100.0,
            "maintenance_margin": 75.0,
            "margin_buffer": 15.0,
            "margin_utilization": 0.0,
            "available_margin": 1000.0,
            "margin_call_threshold": 90.0,
            "liquidation_threshold": 75.0
        }

    def assess_position_risk(self, state: AgentState) -> Dict[str, Dict[str, Any]]:
        """
        基于当前真实持仓和资金状况进行风险评估，计算仓位风险控制参数
        
        Args:
            state: AgentState包含投资组合信息、技术分析数据等
            
        Returns:
            Dict[str, Dict[str, Any]]: 每个ticker的仓位风险控制结果
        """
        data = state.get('data', {})
        portfolio = data.get("portfolio", {})
        tickers = data.get("tickers", [])
        primary_interval = data.get("primary_interval")
        
        # 获取技术分析和风险管理数据
        analyst_signals = data.get("analyst_signals", {})
        technical_signals = analyst_signals.get("technical_analyst_agent", {})
        risk_signals = analyst_signals.get("risk_management_agent", {})
        
        position_risk_results = {}
        
        for ticker in tickers:
            try:
                # 获取价格数据
                price_df = data.get(f"{ticker}_{primary_interval.value}")
                if price_df is None or price_df.empty:
                    position_risk_results[ticker] = {
                        "position_risk_control": self._get_default_position_risk_control()
                    }
                    continue
                
                current_price = price_df["close"].iloc[-1]
                
                # 获取投资组合相关数据
                total_portfolio_value = portfolio.get("cash", 0.0) + sum(
                    portfolio.get("cost_basis", {}).get(t, 0.0) for t in portfolio.get("cost_basis", {}))
                available_cash = portfolio.get("cash", 0.0)
                current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0.0)
                
                # 获取技术分析数据
                tech_data = technical_signals.get(ticker, {})
                interval_data = tech_data.get(primary_interval.value, {})
                volatility_analysis = interval_data.get("volatility_analysis", {})
                atr_values = interval_data.get("atr_values", {})
                
                # 获取杠杆分析数据
                ticker_risk_data = risk_signals.get(ticker, {})
                leverage_analysis = ticker_risk_data.get("leverage_analysis", {})
                
                # 计算仓位风险控制参数
                position_risk_control = self._calculate_position_risk_control(
                    portfolio_value=total_portfolio_value,
                    available_cash=available_cash,
                    current_position=current_position_value,
                    current_price=current_price,
                    volatility_analysis=volatility_analysis,
                    atr_values=atr_values,
                    leverage_analysis=leverage_analysis,
                    ticker=ticker
                )
                
                position_risk_results[ticker] = {
                    "position_risk_control": position_risk_control
                }
                
            except Exception as e:
                print(f"Error assessing position risk for {ticker}: {e}")
                position_risk_results[ticker] = {
                    "position_risk_control": self._get_default_position_risk_control()
                }
        
        return position_risk_results
    
    def _calculate_position_risk_control(self, portfolio_value: float, available_cash: float,
                                       current_position: float, current_price: float,
                                       volatility_analysis: Dict, atr_values: Dict,
                                       leverage_analysis: Dict, ticker: str) -> Dict[str, float]:
        """计算仓位风险控制相关参数"""
        
        # 1. 计算最大仓位规模 (max_position_size)
        # 基于投资组合总价值的20%作为单个仓位上限
        position_limit_ratio = 0.20
        max_position_size = portfolio_value * position_limit_ratio
        
        # 考虑已有仓位，计算剩余可用仓位限额
        remaining_position_limit = max_position_size - current_position
        # 确保不超过可用现金
        max_position_size = min(remaining_position_limit, available_cash)
        max_position_size = max(0.0, max_position_size)  # 确保非负
        
        # 2. 计算仓位规模因子 (position_sizing_factor)
        # 基于风险评分和波动率调整仓位规模
        risk_score = leverage_analysis.get("leverage_risk_score", 0.5)
        volatility_percentile = volatility_analysis.get("volatility_percentile", 50.0)
        
        # 风险越高，仓位规模因子越小
        base_sizing_factor = 1.0 - risk_score * 0.6  # 最高风险时为0.4
        
        # 波动率调整：高波动率降低仓位规模
        volatility_adjustment = 1.0 - (volatility_percentile / 100.0) * 0.3
        
        position_sizing_factor = base_sizing_factor * volatility_adjustment
        position_sizing_factor = max(0.1, min(1.0, position_sizing_factor))  # 限制在0.1-1.0之间
        
        # 3. 计算单笔交易风险 (risk_per_trade)
        # 基于ATR计算潜在损失
        atr_14 = atr_values.get("atr_14", current_price * 0.02)  # 默认2%
        atr_percentage = (atr_14 / current_price) if current_price > 0 else 0.02
        
        # 单笔交易风险 = 仓位规模 * ATR百分比 * 风险系数
        risk_multiplier = 1.5  # 风险放大系数
        estimated_position_size = max_position_size * position_sizing_factor
        risk_per_trade = estimated_position_size * atr_percentage * risk_multiplier
        
        # 限制单笔交易风险不超过总资产的2%
        max_trade_risk = portfolio_value * 0.02
        risk_per_trade = min(risk_per_trade, max_trade_risk)
        
        # 4. 计算日最大风险 (max_daily_risk)
        # 假设一天最多5笔交易，日最大风险不超过总资产的5%
        daily_trade_limit = 5
        theoretical_daily_risk = risk_per_trade * daily_trade_limit
        max_daily_risk_limit = portfolio_value * 0.05
        max_daily_risk = min(theoretical_daily_risk, max_daily_risk_limit)
        
        # 5. 计算仓位集中度 (position_concentration)
        # 当前仓位价值占总投资组合的比例
        position_concentration = (current_position / portfolio_value) if portfolio_value > 0 else 0.0
        
        # 6. 计算分散化评分 (diversification_score)
        # 基于仓位集中度和风险评分计算分散化程度
        # 评分越高表示分散化越好
        concentration_penalty = position_concentration * 2.0  # 集中度越高，分散化越差
        risk_penalty = risk_score * 0.5  # 高风险资产降低分散化评分
        
        # 基础分散化评分
        base_diversification = 1.0 - concentration_penalty - risk_penalty
        diversification_score = max(0.0, min(1.0, base_diversification))
        
        # 如果仓位集中度低于10%，给予分散化奖励
        if position_concentration < 0.10:
            diversification_bonus = (0.10 - position_concentration) * 2.0
            diversification_score = min(1.0, diversification_score + diversification_bonus)
        
        return {
            "max_position_size": round(max_position_size, 2),
            "position_sizing_factor": round(position_sizing_factor, 4),
            "risk_per_trade": round(risk_per_trade, 2),
            "max_daily_risk": round(max_daily_risk, 2),
            "position_concentration": round(position_concentration, 4),
            "diversification_score": round(diversification_score, 4)
        }
    
    def _get_default_position_risk_control(self) -> Dict[str, float]:
        """返回默认的仓位风险控制参数"""
        return {
            "max_position_size": 1000.0,
            "position_sizing_factor": 0.5,
            "risk_per_trade": 50.0,
            "max_daily_risk": 250.0,
            "position_concentration": 0.0,
            "diversification_score": 0.8
        }

    def calculate_dynamic_risk_metrics(self, state: AgentState) -> Dict[str, Dict[str, Any]]:
        """
        基于实际历史数据和当前持仓计算动态风险指标
        
        Args:
            state: AgentState包含历史价格数据、投资组合信息等
            
        Returns:
            Dict[str, Dict[str, Any]]: 每个ticker的动态风险指标
        """
        data = state.get('data', {})
        portfolio = data.get("portfolio", {})
        tickers = data.get("tickers", [])
        primary_interval = data.get("primary_interval")
        
        dynamic_risk_results = {}
        
        for ticker in tickers:
            try:
                # 获取价格数据
                price_df = data.get(f"{ticker}_{primary_interval.value}")
                if price_df is None or price_df.empty or len(price_df) < 30:
                    dynamic_risk_results[ticker] = {
                        "dynamic_risk_metrics": self._get_default_risk_metrics()
                    }
                    continue
                
                # 获取当前持仓和投资组合信息
                current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0.0)
                total_portfolio_value = portfolio.get("cash", 0.0) + sum(
                    portfolio.get("cost_basis", {}).get(t, 0.0) for t in portfolio.get("cost_basis", {}))
                
                # 计算历史收益率
                returns = self._calculate_returns(price_df)
                
                # 计算各项风险指标
                var_1day = self._calculate_var(returns, confidence=0.95, days=1)
                var_7day = self._calculate_var(returns, confidence=0.95, days=7)
                expected_shortfall = self._calculate_expected_shortfall(returns, confidence=0.95)
                maximum_drawdown = self._calculate_maximum_drawdown(price_df)
                sharpe_ratio_impact = self._calculate_sharpe_ratio_impact(
                    returns, current_position_value, total_portfolio_value
                )
                risk_adjusted_return = self._calculate_risk_adjusted_return(
                    returns, maximum_drawdown, var_1day
                )
                
                dynamic_risk_results[ticker] = {
                    "dynamic_risk_metrics": {
                        "var_1day": float(var_1day),
                        "var_7day": float(var_7day),
                        "expected_shortfall": float(expected_shortfall),
                        "maximum_drawdown": float(maximum_drawdown),
                        "sharpe_ratio_impact": float(sharpe_ratio_impact),
                        "risk_adjusted_return": float(risk_adjusted_return)
                    }
                }
                
            except Exception as e:
                print(f"Error calculating dynamic risk metrics for {ticker}: {e}")
                dynamic_risk_results[ticker] = {
                    "dynamic_risk_metrics": self._get_default_risk_metrics()
                }
        
        return dynamic_risk_results

    def _calculate_returns(self, price_df: pd.DataFrame) -> pd.Series:
        """计算日收益率"""
        return price_df["close"].pct_change().dropna()

    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95, days: int = 1) -> float:
        """
        计算风险价值(Value at Risk)
        
        Args:
            returns: 历史收益率序列
            confidence: 置信水平 (如0.95表示95%置信度)
            days: 时间跨度（天数）
        
        Returns:
            VaR值（负数，表示潜在损失）
        """
        if len(returns) < 10:
            return -0.02  # 默认2%的日VaR
        
        # 使用历史模拟法计算VaR
        sorted_returns = returns.sort_values()
        percentile = (1 - confidence) * 100
        var_1day_base = np.percentile(sorted_returns, percentile)
        
        # 调整到指定天数
        # VaR_n = VaR_1 * sqrt(n) (假设收益率独立同分布)
        var_n_days = var_1day_base * np.sqrt(days)
        
        return var_n_days

    def _calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算期望损失(Expected Shortfall/CVaR)
        
        Args:
            returns: 历史收益率序列
            confidence: 置信水平
        
        Returns:
            期望损失值（负数）
        """
        if len(returns) < 10:
            return -0.035  # 默认3.5%的期望损失
        
        # 计算VaR
        var_level = self._calculate_var(returns, confidence, days=1)
        
        # 计算超过VaR的损失的平均值
        tail_losses = returns[returns <= var_level]
        
        if len(tail_losses) == 0:
            return var_level * 1.5  # 如果没有尾部损失，返回VaR的1.5倍
        
        expected_shortfall = tail_losses.mean()
        return expected_shortfall

    def _calculate_maximum_drawdown(self, price_df: pd.DataFrame) -> float:
        """
        计算最大回撤
        
        Args:
            price_df: 价格数据DataFrame
        
        Returns:
            最大回撤百分比（正数）
        """
        if len(price_df) < 2:
            return 0.1  # 默认10%回撤
        
        close_prices = price_df["close"]
        
        # 计算累计最大值
        peak = close_prices.expanding(min_periods=1).max()
        
        # 计算回撤
        drawdown = (close_prices - peak) / peak
        
        # 最大回撤（转为正数）
        max_drawdown = abs(drawdown.min())
        
        return max_drawdown

    def _calculate_sharpe_ratio_impact(self, returns: pd.Series, position_value: float, 
                                     portfolio_value: float) -> float:
        """
        计算该资产对投资组合夏普比率的影响
        
        Args:
            returns: 历史收益率序列
            position_value: 当前持仓价值
            portfolio_value: 总投资组合价值
        
        Returns:
            夏普比率影响评分（-1到1之间，正数表示正面影响）
        """
        if len(returns) < 10 or portfolio_value <= 0:
            return 0.0
        
        # 计算基础夏普比率
        excess_returns = returns  # 简化处理，不减去无风险利率
        sharpe_ratio = excess_returns.mean() / (excess_returns.std() + 1e-6) * np.sqrt(252)  # 年化
        
        # 计算仓位权重
        position_weight = position_value / portfolio_value if portfolio_value > 0 else 0.0
        
        # 影响评分：夏普比率 * 仓位权重
        # 限制在-1到1之间
        impact_score = sharpe_ratio * position_weight
        impact_score = max(-1.0, min(1.0, impact_score))
        
        return impact_score

    def _calculate_risk_adjusted_return(self, returns: pd.Series, max_drawdown: float, 
                                       var_1day: float) -> float:
        """
        计算风险调整后收益率
        
        Args:
            returns: 历史收益率序列
            max_drawdown: 最大回撤
            var_1day: 1日VaR
        
        Returns:
            风险调整后收益率
        """
        if len(returns) < 10:
            return 0.0
        
        # 计算年化收益率
        annualized_return = returns.mean() * 252
        
        # 计算复合风险指标
        # 风险惩罚 = 最大回撤 + |VaR|
        risk_penalty = max_drawdown + abs(var_1day)
        
        # 风险调整后收益 = 年化收益 / (1 + 风险惩罚)
        risk_adjusted_return = annualized_return / (1 + risk_penalty)
        
        return risk_adjusted_return

    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """返回默认的风险指标"""
        return {
            "var_1day": -0.02,          # 2%的1日VaR
            "var_7day": -0.053,         # 约5.3%的7日VaR（sqrt(7) * 2%）
            "expected_shortfall": -0.035, # 3.5%的期望损失
            "maximum_drawdown": 0.1,     # 10%的最大回撤
            "sharpe_ratio_impact": 0.0,  # 中性的夏普比率影响
            "risk_adjusted_return": 0.0  # 零风险调整收益
        }

    def analyze_liquidation_risk(self, state: AgentState) -> Dict[str, Dict[str, Any]]:
        """
        基于实际价格和真实杠杆进行强平风险分析
        
        Args:
            state: AgentState包含投资组合信息、杠杆分析、技术分析等数据
            
        Returns:
            Dict[str, Dict[str, Any]]: 每个ticker的强平风险分析结果
        """
        data = state.get('data', {})
        portfolio = data.get("portfolio", {})
        tickers = data.get("tickers", [])
        primary_interval = data.get("primary_interval")
        
        # 获取已有分析结果
        analyst_signals = data.get("analyst_signals", {})
        risk_signals = analyst_signals.get("risk_management_agent", {})
        technical_signals = analyst_signals.get("technical_analyst_agent", {})
        
        liquidation_risk_results = {}
        
        for ticker in tickers:
            try:
                # 获取价格数据
                price_df = data.get(f"{ticker}_{primary_interval.value}")
                if price_df is None or price_df.empty:
                    liquidation_risk_results[ticker] = {
                        "liquidation_analysis": self._get_default_liquidation_analysis()
                    }
                    continue
                
                current_price = price_df["close"].iloc[-1]
                
                # 获取投资组合相关数据
                total_portfolio_value = portfolio.get("cash", 0.0) + sum(
                    portfolio.get("cost_basis", {}).get(t, 0.0) for t in portfolio.get("cost_basis", {}))
                available_cash = portfolio.get("cash", 0.0)
                current_position_value = portfolio.get("cost_basis", {}).get(ticker, 0.0)
                
                # 获取杠杆和保证金数据
                ticker_risk_data = risk_signals.get(ticker, {})
                leverage_analysis = ticker_risk_data.get("leverage_analysis", {})
                margin_management = ticker_risk_data.get("margin_management", {})
                
                # 获取技术分析数据
                tech_data = technical_signals.get(ticker, {})
                interval_data = tech_data.get(primary_interval.value, {})
                volatility_analysis = interval_data.get("volatility_analysis", {})
                atr_values = interval_data.get("atr_values", {})
                
                # 计算强平风险分析
                liquidation_analysis = self._calculate_liquidation_analysis(
                    current_price=current_price,
                    price_df=price_df,
                    current_position=current_position_value,
                    portfolio_value=total_portfolio_value,
                    leverage_analysis=leverage_analysis,
                    margin_management=margin_management,
                    volatility_analysis=volatility_analysis,
                    atr_values=atr_values,
                    ticker=ticker
                )
                
                liquidation_risk_results[ticker] = {
                    "liquidation_analysis": liquidation_analysis
                }
                
            except Exception as e:
                print(f"Error analyzing liquidation risk for {ticker}: {e}")
                liquidation_risk_results[ticker] = {
                    "liquidation_analysis": self._get_default_liquidation_analysis()
                }
        
        return liquidation_risk_results
    
    def _calculate_liquidation_analysis(self, current_price: float, price_df: pd.DataFrame,
                                      current_position: float, portfolio_value: float,
                                      leverage_analysis: Dict, margin_management: Dict,
                                      volatility_analysis: Dict, atr_values: Dict,
                                      ticker: str) -> Dict[str, float]:
        """计算强平风险相关指标"""
        
        # 获取杠杆倍数
        recommended_leverage = leverage_analysis.get("recommended_leverage", 10)
        max_safe_leverage = leverage_analysis.get("max_safe_leverage", 20)
        
        # 获取保证金信息
        maintenance_margin_ratio = 0.75 / max(recommended_leverage, 1)  # 维持保证金比例
        
        # 1. 计算强平价格 (liquidation_price)
        liquidation_price = self._calculate_liquidation_price(
            current_price, recommended_leverage, maintenance_margin_ratio, current_position
        )
        
        # 2. 计算距强平距离 (liquidation_distance)
        liquidation_distance = self._calculate_liquidation_distance(
            current_price, liquidation_price
        )
        
        # 3. 计算强平概率 (liquidation_probability)
        liquidation_probability = self._calculate_liquidation_probability(
            price_df, current_price, liquidation_price, volatility_analysis, atr_values
        )
        
        # 4. 计算安全杠杆门槛 (safe_leverage_threshold)
        safe_leverage_threshold = self._calculate_safe_leverage_threshold(
            volatility_analysis, atr_values, portfolio_value, current_position
        )
        
        # 5. 预计强平时间 (time_to_liquidation)
        time_to_liquidation = self._calculate_time_to_liquidation(
            current_price, liquidation_price, volatility_analysis, atr_values, liquidation_probability
        )
        
        return {
            "liquidation_price": round(liquidation_price, 4),
            "liquidation_distance": round(liquidation_distance, 4),
            "liquidation_probability": round(liquidation_probability, 4),
            "safe_leverage_threshold": int(safe_leverage_threshold),
            "time_to_liquidation": round(time_to_liquidation, 2)
        }
    
    def _calculate_liquidation_price(self, current_price: float, leverage: int, 
                                   maintenance_margin_ratio: float, position_value: float) -> float:
        """
        计算强平价格
        
        强平价格计算公式（多头仓位）:
        对于多头仓位，当 (开仓价格 - 当前价格) / 开仓价格 >= (1 - 维持保证金率) 时触发强平
        即: 强平价格 = 开仓价格 * 维持保证金率
        
        维持保证金率通常等于 1/杠杆倍数
        """
        if position_value <= 0 or leverage <= 1:
            return 0.0  # 无持仓或无杠杆时不存在强平风险
        
        # 使用当前价格作为开仓价格的近似值
        # 在实际应用中，这应该是真实的开仓价格
        entry_price = current_price
        
        # 计算维持保证金率
        # 对于不同杠杆，维持保证金率不同：
        # - 10x杠杆: 维持保证金率约为8-10%
        # - 20x杠杆: 维持保证金率约为5%
        # - 50x杠杆: 维持保证金率约为2%
        
        if leverage >= 50:
            maintenance_margin_rate = 0.02  # 2%
        elif leverage >= 20:
            maintenance_margin_rate = 0.05  # 5%
        elif leverage >= 10:
            maintenance_margin_rate = 0.08  # 8%
        elif leverage >= 5:
            maintenance_margin_rate = 0.15  # 15%
        else:
            maintenance_margin_rate = 0.20  # 20%
        
        # 考虑手续费影响（约0.1%）
        fee_buffer = 0.001
        
        # 强平价格 = 开仓价格 * (维持保证金率 + 手续费缓冲)
        # 这表示价格下跌到这个水平时，净值刚好等于维持保证金要求
        liquidation_price = entry_price * (1.0 - (1.0/leverage - maintenance_margin_rate - fee_buffer))
        
        # 确保强平价格为正数且小于开仓价格
        liquidation_price = max(0.0, min(liquidation_price, entry_price * 0.99))
        
        return liquidation_price
    
    def _calculate_liquidation_distance(self, current_price: float, liquidation_price: float) -> float:
        """
        计算距强平距离（百分比）
        
        Returns:
            距离百分比，正数表示距离强平还有多远，负数表示已触发强平
        """
        if liquidation_price <= 0 or current_price <= 0:
            return 1.0  # 无强平风险时返回100%距离
        
        # 距离百分比 = (当前价格 - 强平价格) / 当前价格
        distance_percentage = (current_price - liquidation_price) / current_price
        
        return distance_percentage
    
    def _calculate_liquidation_probability(self, price_df: pd.DataFrame, current_price: float,
                                         liquidation_price: float, volatility_analysis: Dict,
                                         atr_values: Dict) -> float:
        """
        基于历史波动率和蒙特卡洛模拟计算强平概率
        
        Returns:
            24小时内强平概率 (0-1之间)
        """
        if liquidation_price <= 0 or current_price <= 0:
            return 0.0
        
        # 获取历史收益率标准差
        if len(price_df) >= 30:
            returns = price_df["close"].pct_change().dropna()
            daily_volatility = returns.std()
        else:
            # 使用ATR估算波动率
            atr_14 = atr_values.get("atr_14", current_price * 0.02)
            daily_volatility = atr_14 / current_price / 2.0  # ATR约为2倍标准差
        
        # 确保波动率不为零
        daily_volatility = max(daily_volatility, 0.001)
        
        # 计算需要的价格变动幅度来触发强平
        required_move = (current_price - liquidation_price) / current_price
        
        # 使用正态分布计算概率
        # Z = required_move / daily_volatility
        z_score = required_move / daily_volatility
        
        # 使用标准正态分布累积分布函数
        # 计算价格下跌到强平价格的概率
        liquidation_prob = norm.cdf(-z_score) if z_score > 0 else 0.95
        
        # 调整概率：考虑波动率聚集效应
        volatility_percentile = volatility_analysis.get("volatility_percentile", 50.0)
        volatility_adjustment = 1.0 + (volatility_percentile - 50.0) / 200.0  # ±25%调整
        
        adjusted_probability = liquidation_prob * volatility_adjustment
        
        # 限制在0-1之间
        return max(0.0, min(1.0, adjusted_probability))
    
    def _calculate_safe_leverage_threshold(self, volatility_analysis: Dict, atr_values: Dict,
                                         portfolio_value: float, current_position: float) -> int:
        """
        计算安全杠杆门槛，避免强平风险
        
        Returns:
            建议的最大安全杠杆倍数
        """
        # 获取波动率指标
        volatility_percentile = volatility_analysis.get("volatility_percentile", 50.0)
        atr_percentile = atr_values.get("atr_percentile", 50.0)
        volatility_trend = volatility_analysis.get("volatility_trend", "stable")
        
        # 基础安全杠杆：根据波动率确定
        if volatility_percentile >= 80 or atr_percentile >= 80:
            base_safe_leverage = 3  # 高波动率时使用低杠杆
        elif volatility_percentile >= 60 or atr_percentile >= 60:
            base_safe_leverage = 5  # 中等波动率
        elif volatility_percentile <= 20 and atr_percentile <= 20:
            base_safe_leverage = 20  # 低波动率时可以使用较高杠杆
        else:
            base_safe_leverage = 10  # 默认中等杠杆
        
        # 趋势调整
        trend_adjustment = {
            "increasing": 0.7,  # 波动率上升时降低杠杆
            "stable": 1.0,      # 稳定时保持
            "decreasing": 1.2   # 波动率下降时可适当提高
        }.get(volatility_trend, 1.0)
        
        # 仓位规模调整
        position_ratio = (current_position / portfolio_value) if portfolio_value > 0 else 0.0
        if position_ratio > 0.15:  # 仓位超过15%时更保守
            position_adjustment = 0.8
        elif position_ratio < 0.05:  # 小仓位时可以稍微激进
            position_adjustment = 1.1
        else:
            position_adjustment = 1.0
        
        # 计算最终安全杠杆
        safe_leverage = base_safe_leverage * trend_adjustment * position_adjustment
        
        # 限制在合理范围内
        safe_leverage = max(1, min(50, int(safe_leverage)))
        
        return safe_leverage
    
    def _calculate_time_to_liquidation(self, current_price: float, liquidation_price: float,
                                     volatility_analysis: Dict, atr_values: Dict,
                                     liquidation_probability: float) -> float:
        """
        预估到达强平价格的时间（小时）
        
        Returns:
            预计强平时间（小时），999.0表示风险极低
        """
        if liquidation_price <= 0 or liquidation_probability <= 0.01:
            return 999.0  # 几乎无强平风险
        
        # 计算需要的价格变动幅度
        price_move_needed = abs(current_price - liquidation_price) / current_price
        
        # 获取日波动率
        atr_14 = atr_values.get("atr_14", current_price * 0.02)
        daily_volatility = atr_14 / current_price
        
        # 估算每小时的价格变动标准差
        hourly_volatility = daily_volatility / np.sqrt(24)
        
        # 基于随机游走模型估算时间
        # 使用几何布朗运动的首次通过时间期望
        if hourly_volatility > 0:
            # 简化模型：期望时间 ≈ (变动幅度)² / (2 * 小时波动率²)
            expected_hours = (price_move_needed ** 2) / (2 * hourly_volatility ** 2)
        else:
            expected_hours = 999.0
        
        # 根据强平概率调整
        # 概率越高，预期时间越短
        probability_adjustment = 1.0 / max(liquidation_probability, 0.01)
        adjusted_hours = expected_hours / probability_adjustment
        
        # 考虑波动率趋势
        volatility_trend = volatility_analysis.get("volatility_trend", "stable")
        if volatility_trend == "increasing":
            trend_factor = 0.7  # 波动率上升，预期时间缩短
        elif volatility_trend == "decreasing":
            trend_factor = 1.3  # 波动率下降，预期时间延长
        else:
            trend_factor = 1.0
        
        final_hours = adjusted_hours * trend_factor
        
        # 限制在合理范围内：最少1小时，最多999小时
        return max(1.0, min(999.0, final_hours))
    
    def _get_default_liquidation_analysis(self) -> Dict[str, float]:
        """返回默认的强平风险分析结果"""
        return {
            "liquidation_price": 0.0,      # 无强平价格
            "liquidation_distance": 1.0,   # 100%距离
            "liquidation_probability": 0.0, # 无强平概率
            "safe_leverage_threshold": 10,  # 默认安全杠杆
            "time_to_liquidation": 999.0    # 极长时间
        }
