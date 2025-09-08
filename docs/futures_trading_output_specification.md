# AI Hedge Fund Crypto - 合约交易输出字段规范

*版本: v2.0 | 更新时间: 2024-09-06 | 分支: feature/futures-trading*

## 📋 概述

本文档定义了AI量化交易系统中各Agent针对合约交易的完整输出字段规范。系统包含3个核心Agent，每个Agent负责不同的分析维度，最终由Portfolio Management Agent整合所有信息做出交易决策。

---

## 🔬 Agent 1: Technical Analyst（技术分析师）

### 当前输出字段
```json
{
  "BTCUSDT": {
    "30m|1h|4h|15m|5m": {
      "signal": "bearish|bullish|neutral",
      "confidence": 0-100,
      "strategy_signals": {
        "trend_following": {
          "signal": "bearish|bullish|neutral",
          "confidence": 0-100,
          "metrics": {
            "adx": float,
            "trend_strength": float
          }
        },
        "mean_reversion": {
          "signal": "bearish|bullish|neutral", 
          "confidence": 0-100,
          "metrics": {
            "z_score": float,
            "price_vs_bb": float,
            "rsi_14": float,
            "rsi_28": float
          }
        },
        "momentum": {
          "signal": "bearish|bullish|neutral",
          "confidence": 0-100,
          "metrics": {
            "momentum_1m": float,
            "momentum_3m": float,
            "momentum_6m": float,
            "volume_momentum": float
          }
        },
        "volatility": {
          "signal": "bearish|bullish|neutral",
          "confidence": 0-100,
          "metrics": {
            "historical_volatility": float,
            "volatility_regime": float,
            "volatility_z_score": float,
            "atr_ratio": float
          }
        },
        "statistical_arbitrage": {
          "signal": "bearish|bullish|neutral",
          "confidence": 0-100,
          "metrics": {
            "hurst_exponent": float,
            "skewness": float,
            "kurtosis": float
          }
        }
      }
    }
  }
}
```

### 合约交易新增字段
```json
{
  "BTCUSDT": {
    "timeframe": {
      // === 现有字段保持不变 ===
      
      // === 新增：止损止盈计算基础 ===
      "atr_values": {
        "atr_14": float,              // 14期平均真实波幅
        "atr_28": float,              // 28期平均真实波幅
        "atr_percentile": float       // ATR历史百分位
      },
      
      // === 新增：关键价位识别 ===
      "price_levels": {
        "support_levels": [float, float, float],    // 支撑位数组
        "resistance_levels": [float, float, float], // 阻力位数组
        "pivot_point": float,                       // 枢轴点
        "breakout_threshold": float                 // 突破临界点
      },
      
      // === 新增：波动率深度分析 ===
      "volatility_analysis": {
        "volatility_percentile": float,    // 波动率历史百分位
        "volatility_trend": "increasing|decreasing|stable",
        "volatility_forecast": float,      // 预期波动率
        "regime_probability": float        // 当前波动率状态概率
      },
      
      // === 新增：信号时效性 ===
      "signal_metadata": {
        "signal_strength": "weak|moderate|strong",
        "signal_decay_time": int,          // 信号有效期（分钟）
        "signal_reliability": float,       // 信号可靠性评分
        "confirmation_status": "confirmed|pending|weak"
      }
    },
    
    // === 新增：跨时间框架综合分析 ===
    "cross_timeframe_analysis": {
      "timeframe_consensus": float,        // 时间框架一致性 0-1
      "dominant_timeframe": "5m|15m|30m|1h|4h",
      "conflict_areas": ["timeframe_pairs"],
      "trend_alignment": "aligned|divergent|mixed",
      "overall_signal_strength": "weak|moderate|strong"
    }
  }
}
```

---

## ⚖️ Agent 2: Risk Management（风险管理）

### 当前输出字段
```json
{
  "BTCUSDT": {
    "remaining_position_limit": float,
    "current_price": float,
    "reasoning": {
      "portfolio_value": float,
      "current_position": float,
      "position_limit": float,
      "remaining_limit": float,
      "available_cash": float
    }
  }
}
```

### 合约交易新增字段
```json
{
  "BTCUSDT": {
    // === 现有字段保持不变 ===
    
    // === 新增：杠杆管理 ===
    "leverage_analysis": {
      "recommended_leverage": int,         // 推荐杠杆倍数
      "max_safe_leverage": int,           // 最大安全杠杆
      "leverage_options": [int, int, int], // 可选杠杆倍数
      "leverage_risk_score": float,       // 杠杆风险评分
      "volatility_adjusted_leverage": int  // 波动率调整后杠杆
    },
    
    // === 新增：保证金管理 ===
    "margin_management": {
      "initial_margin": float,            // 初始保证金
      "maintenance_margin": float,        // 维持保证金
      "margin_buffer": float,             // 保证金缓冲区
      "margin_utilization": float,        // 保证金使用率
      "available_margin": float,          // 可用保证金
      "margin_call_threshold": float,     // 追保门槛
      "liquidation_threshold": float      // 强平门槛
    },
    
    // === 新增：仓位风险控制 ===
    "position_risk_control": {
      "max_position_size": float,         // 最大仓位规模
      "position_sizing_factor": float,    // 仓位规模因子
      "risk_per_trade": float,           // 单笔交易风险
      "max_daily_risk": float,           // 日最大风险
      "position_concentration": float,    // 仓位集中度
      "diversification_score": float     // 分散化评分
    },
    
    // === 新增：动态风险指标 ===
    "dynamic_risk_metrics": {
      "var_1day": float,                 // 1日风险价值
      "var_7day": float,                 // 7日风险价值
      "expected_shortfall": float,       // 期望损失
      "maximum_drawdown": float,         // 最大回撤
      "sharpe_ratio_impact": float,      // 对夏普比率的影响
      "risk_adjusted_return": float      // 风险调整后收益
    },
    
    // === 新增：强平风险评估 ===
    "liquidation_analysis": {
      "liquidation_price": float,        // 强平价格
      "liquidation_distance": float,     // 距强平距离
      "liquidation_probability": float,  // 强平概率
      "safe_leverage_threshold": int,    // 安全杠杆门槛
      "time_to_liquidation": float       // 预计强平时间（小时）
    }
  }
}
```

---

## 🎯 Agent 3: Portfolio Management（投资组合管理）

### 当前输出字段
```json
{
  "BTCUSDT": {
    "action": "short|long|hold",
    "quantity": float,
    "confidence": 0-100,
    "reasoning": "string"
  }
}
```

### 合约交易完整输出字段
```json
{
  "BTCUSDT": {
    // === 基础交易参数 ===
    "basic_params": {
      "direction": "long|short",           // 交易方向
      "operation": "open|close|add|reduce", // 操作类型
      "leverage": int,                     // 杠杆倍数
      "position_size": float,              // 名义价值（USDT）
      "position_ratio": float,             // 仓位比例
      "current_price": float,              // 当前价格
      "contract_value": float,             // 实际投入资金
      "contract_quantity": float,          // 合约数量
      "entry_price_target": float,         // 目标入场价
      "order_type": "market|limit|stop_limit" // 订单类型
    },
    
    // === 风险管理参数 ===
    "risk_management": {
      "stop_loss": float,                  // 止损价
      "take_profit": float,                // 止盈价
      "trailing_stop": float,              // 跟踪止损
      "liquidation_price": float,          // 强平价
      "margin_required": float,            // 保证金占用
      "risk_percentage": float,            // 风险百分比
      "risk_reward_ratio": float,          // 风险收益比
      "atr_based_stop": float,            // 基于ATR的止损
      "volatility_adjusted_size": float,   // 波动率调整仓位
      "max_loss_amount": float,           // 最大亏损金额
      "position_hold_time": int           // 预期持仓时间（小时）
    },
    
    // === 多时间框架信号强度 ===
    "timeframe_analysis": {
      "consensus_score": float,            // 共识评分 0-1
      "dominant_timeframe": "5m|15m|30m|1h|4h",
      "signal_alignment": "strong|moderate|weak",
      "conflicting_signals": int,         // 冲突信号数量
      "timeframe_weights": {
        "5m": float, "15m": float, "30m": float,
        "1h": float, "4h": float
      },
      "overall_direction_confidence": float
    },
    
    // === 技术指标风险评估 ===
    "technical_risk_assessment": {
      "volatility_risk": "low|moderate|high|extreme",
      "trend_strength": "weak|moderate|strong",
      "mean_reversion_risk": "low|moderate|high",
      "statistical_edge": "weak|moderate|strong",
      "momentum_alignment": boolean,
      "support_resistance_proximity": "far|near|at_level",
      "breakout_probability": float,
      "false_breakout_risk": float
    },
    
    // === 成本收益分析 ===
    "cost_benefit_analysis": {
      "estimated_trading_fee": float,      // 预计交易手续费
      "funding_rate": float,              // 当前资金费率
      "funding_cost_8h": float,           // 8小时资金费用
      "funding_cost_daily": float,        // 日资金费用
      "holding_cost_total": float,        // 总持仓成本
      "break_even_price": float,          // 盈亏平衡价
      "target_profit": float,             // 目标利润
      "expected_return": float,           // 期望收益率
      "profit_probability": float,        // 获利概率
      "loss_probability": float,          // 亏损概率
      "expected_value": float,            // 期望值
      "roi_annualized": float            // 年化收益率
    },
    
    // === 市场环境评估 ===
    "market_environment": {
      "trend_regime": "trending|ranging|transitional",
      "volatility_regime": "low|normal|elevated|extreme",
      "liquidity_assessment": "poor|fair|good|excellent",
      "market_structure": "bullish|bearish|neutral",
      "market_phase": "accumulation|markup|distribution|decline",
      "sentiment_indicator": float,       // 情绪指标
      "fear_greed_index": int,           // 恐慌贪婪指数
      "funding_rate_trend": "increasing|decreasing|stable",
      "open_interest_trend": "increasing|decreasing|stable"
    },
    
    // === 执行策略建议 ===
    "execution_strategy": {
      "entry_strategy": "immediate|gradual|wait_for_dip",
      "entry_timing": "now|wait_5m|wait_15m|wait_pullback",
      "order_placement": "aggressive|passive|hidden",
      "position_building": "single_entry|scale_in|dca",
      "exit_strategy": "target_based|signal_based|time_based",
      "partial_profit_taking": boolean,
      "scale_out_levels": [float, float, float],
      "emergency_exit_conditions": ["condition1", "condition2"]
    },
    
    // === 情景分析 ===
    "scenario_analysis": {
      "best_case": {
        "price_target": float,
        "profit_potential": float,
        "probability": float,
        "timeframe": int
      },
      "base_case": {
        "price_target": float,
        "profit_potential": float,
        "probability": float,
        "timeframe": int
      },
      "worst_case": {
        "price_target": float,
        "loss_potential": float,
        "probability": float,
        "timeframe": int
      },
      "black_swan": {
        "trigger_conditions": ["condition1", "condition2"],
        "max_loss": float,
        "probability": float,
        "mitigation_strategy": "string"
      }
    },
    
    // === AI决策元数据 ===
    "decision_metadata": {
      "confidence": int,                   // 总体置信度 0-100
      "confidence_breakdown": {
        "technical_analysis": float,      // 技术分析置信度
        "risk_assessment": float,         // 风险评估置信度
        "market_conditions": float,       // 市场条件置信度
        "cost_benefit": float,            // 成本效益置信度
        "execution_feasibility": float    // 执行可行性置信度
      },
      "decision_factors": {
        "primary_drivers": ["factor1", "factor2", "factor3"],
        "supporting_factors": ["factor1", "factor2"],
        "risk_factors": ["factor1", "factor2"],
        "uncertainty_factors": ["factor1", "factor2"]
      },
      "alternative_scenarios": [
        {
          "condition": "string",
          "alternative_action": "string",
          "probability": float,
          "impact": "low|moderate|high"
        }
      ],
      "decision_tree_path": ["node1", "node2", "node3"],
      "reasoning_chain": ["step1", "step2", "step3"],
      "supporting_evidence": ["evidence1", "evidence2"],
      "contrary_evidence": ["evidence1", "evidence2"]
    },
    
    // === 监控和告警 ===
    "monitoring_alerts": {
      "price_alerts": {
        "stop_loss_alert": float,
        "take_profit_alert": float,
        "liquidation_warning": float,
        "margin_call_warning": float
      },
      "risk_alerts": {
        "max_drawdown_alert": float,
        "volatility_spike_alert": float,
        "correlation_breakdown_alert": float,
        "funding_rate_spike_alert": float
      },
      "signal_alerts": {
        "signal_reversal_alert": boolean,
        "trend_change_alert": boolean,
        "momentum_divergence_alert": boolean,
        "volume_anomaly_alert": boolean
      },
      "system_alerts": {
        "api_latency_warning": boolean,
        "data_freshness_warning": boolean,
        "execution_slippage_warning": float,
        "liquidity_warning": boolean
      }
    },
    
    // === 最终决策摘要 ===
    "decision_summary": {
      "action_type": "open_long|open_short|close_long|close_short|hold|reduce|add",
      "urgency": "immediate|high|medium|low|wait",
      "expected_holding_period": "scalp|short_term|medium_term|long_term",
      "strategy_category": "trend_following|mean_reversion|breakout|arbitrage",
      "risk_category": "conservative|moderate|aggressive|speculative",
      "execution_complexity": "simple|moderate|complex",
      "reasoning": "详细的中文推理说明，包含关键技术指标、风险评估、成本分析和执行建议"
    }
  }
}
```

---

## 🔄 Agent工作流整合

### 信息流向图
```
Technical Analyst
├─ 多时间框架技术分析
├─ 关键价位识别  
├─ 波动率分析
└─ 信号时效评估
    ↓
Risk Management  
├─ 杠杆优化建议
├─ 保证金管理
├─ 强平风险评估
└─ 动态风险控制
    ↓
Portfolio Management
├─ 综合所有信息
├─ LLM智能决策
├─ 执行策略制定
└─ 完整输出生成
```

### 关键集成点
1. **价格数据共享**：Risk Management的current_price → Portfolio Management
2. **技术指标传递**：Technical Analyst的ATR值 → Portfolio Management的止损计算
3. **风险约束应用**：Risk Management的杠杆建议 → Portfolio Management的仓位规划
4. **信号强度评估**：Technical Analyst的多时间框架共识 → Portfolio Management的置信度

---

## 📊 实施建议

### Phase 1: 核心功能（优先级：高）
- ✅ 基础交易参数（杠杆、仓位、方向）
- ✅ 风险管理参数（止损、止盈、强平）
- ✅ 多时间框架分析整合
- ✅ 基础成本收益计算

### Phase 2: 增强功能（优先级：中）
- 🔄 技术风险评估
- 🔄 市场环境分析
- 🔄 执行策略建议
- 🔄 情景分析

### Phase 3: 高级功能（优先级：低）
- ⏳ 动态风险调整
- ⏳ 复杂监控告警
- ⏳ 高级成本分析
- ⏳ 多策略组合优化

---

## 📈 数据验证和测试

### 字段验证规则
1. **数值范围验证**：杠杆倍数1-125，置信度0-100，风险收益比>0
2. **逻辑一致性验证**：stop_loss与方向的逻辑关系
3. **风险约束验证**：position_size不超过risk limits
4. **成本合理性验证**：所有费用计算的准确性

### 回测验证指标
- 信号准确率 >65%
- 风险收益比 >1.5
- 最大回撤 <15%
- 年化收益率 >20%

---

*该规范将随着系统迭代持续更新和完善*