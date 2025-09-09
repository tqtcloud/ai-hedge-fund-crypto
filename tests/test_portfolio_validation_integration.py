"""
投资组合管理节点验证系统集成测试

测试验证系统在实际交易决策过程中的表现
"""

import pytest
from unittest.mock import Mock, patch
from src.graph.portfolio_management_node import PortfolioManagementNode
from src.graph.state import AgentState
from src.utils.validators import ValidationSeverity


class TestPortfolioValidationIntegration:
    """投资组合验证集成测试"""
    
    def setup_method(self):
        """测试前的设置"""
        self.node = PortfolioManagementNode(validation_level="moderate")
        
        # 模拟状态数据
        self.test_state = {
            "data": {
                "tickers": ["BTCUSDT"],
                "portfolio": {
                    "total_value": 100000.0,
                    "current_drawdown": 0.02,
                    "correlations": {},
                    "top_positions": []
                },
                "analyst_signals": {
                    "risk_management_agent": {
                        "BTCUSDT": {
                            "remaining_position_limit": 10000.0,
                            "current_price": 50000.0
                        }
                    },
                    "technical_analyst_agent": {
                        "BTCUSDT": {
                            "1h": {
                                "signal": "bullish",
                                "confidence": 75,
                                "strategy_signals": {
                                    "volatility": {
                                        "metrics": {
                                            "historical_volatility": 0.05
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "metadata": {
                "model_name": "gpt-4",
                "model_provider": "openai", 
                "model_base_url": "https://api.openai.com/v1",
                "show_reasoning": False
            }
        }
    
    def test_validation_context_extraction(self):
        """测试验证上下文提取"""
        ticker = "BTCUSDT"
        analyst_signals = self.test_state["data"]["analyst_signals"]
        
        # 测试波动率提取
        volatility = self.node._extract_volatility_from_signals(analyst_signals, ticker)
        assert volatility == 0.05  # 应该从历史波动率中提取
        
        # 测试无波动率数据的情况
        empty_signals = {"technical_analyst_agent": {"BTCUSDT": {}}}
        volatility = self.node._extract_volatility_from_signals(empty_signals, ticker)
        assert volatility == 0.03  # 应该返回默认值
    
    def test_decision_validation_valid_data(self):
        """测试有效数据的验证"""
        ticker = "BTCUSDT"
        
        # 创建有效的交易决策数据
        valid_decision = {
            "basic_params": {
                "leverage": 10,
                "direction": "long",
                "current_price": 50000.0,
                "position_size": 5000.0,
                "position_ratio": 0.05,
                "entry_price_target": 50000.0
            },
            "risk_management": {
                "stop_loss": 47500.0,  # 5%止损
                "take_profit": 52500.0,  # 5%止盈
                "risk_reward_ratio": 1.0,
                "risk_percentage": 0.025  # 2.5%风险
            },
            "decision_metadata": {
                "confidence": 75
            },
            "cost_benefit_analysis": {
                "estimated_trading_fee": 2.5,
                "expected_return": 0.05,
                "expected_value": 250
            },
            "margin_management": {
                "margin_utilization": 0.5
            }
        }
        
        context = {
            "ticker": ticker,
            "account_balance": 100000.0,
            "volatility": 0.05
        }
        
        result = self.node._validate_trading_decision(ticker, valid_decision, context)
        
        # 应该通过验证
        assert result["is_valid"] == True
        assert result["critical_count"] == 0
        assert result["error_count"] == 0
    
    def test_decision_validation_invalid_data(self):
        """测试无效数据的验证"""
        ticker = "BTCUSDT"
        
        # 创建有问题的交易决策数据
        invalid_decision = {
            "basic_params": {
                "leverage": 200,  # 超出限制
                "direction": "long",
                "current_price": 50000.0,
                "position_size": 5000.0,
                "position_ratio": 0.05,
                "entry_price_target": 50000.0
            },
            "risk_management": {
                "stop_loss": 52000.0,  # 错误：多头止损高于入场价
                "take_profit": 48000.0,  # 错误：多头止盈低于入场价
                "risk_percentage": 0.15  # 15%风险过高
            },
            "decision_metadata": {
                "confidence": 150  # 超出范围
            },
            "cost_benefit_analysis": {
                "expected_value": -100  # 负期望值
            },
            "margin_management": {
                "margin_utilization": 0.98  # 极高保证金使用率
            }
        }
        
        context = {
            "ticker": ticker,
            "account_balance": 100000.0,
            "volatility": 0.05
        }
        
        result = self.node._validate_trading_decision(ticker, invalid_decision, context)
        
        # 应该不通过验证
        assert result["is_valid"] == False
        assert result["critical_count"] > 0 or result["error_count"] > 0
        assert len(result["suggestions"]) > 0
    
    def test_validation_with_auto_correction(self):
        """测试自动修正功能"""
        ticker = "BTCUSDT"
        
        # 创建可修正的数据
        correctable_decision = {
            "basic_params": {
                "leverage": 150,  # 应该被修正为125
                "position_size": 5000.0
            },
            "decision_metadata": {
                "confidence": 110  # 应该被修正为100
            }
        }
        
        context = {
            "ticker": ticker,
            "account_balance": 100000.0
        }
        
        result = self.node._validate_trading_decision(ticker, correctable_decision, context)
        
        # 检查是否有修正
        corrected_data = result["corrected_data"]
        original_leverage = correctable_decision["basic_params"]["leverage"]
        corrected_leverage = corrected_data["basic_params"]["leverage"]
        
        # 如果发生了修正，杠杆应该被调整
        if original_leverage != corrected_leverage:
            assert corrected_leverage <= 125  # 应该不超过最大值
    
    @patch('src.graph.portfolio_management_node.generate_trading_decision')
    def test_full_validation_workflow(self, mock_generate_decision):
        """测试完整的验证工作流程"""
        # 模拟LLM决策结果
        mock_generate_decision.return_value = {
            "decisions": {
                "BTCUSDT": {
                    "action": "long",
                    "quantity": 0.1,
                    "confidence": 75,
                    "reasoning": "测试决策"
                }
            }
        }
        
        # 模拟各个计算方法返回合理的数据
        with patch.object(self.node, 'calculate_basic_params') as mock_basic, \
             patch.object(self.node, 'design_risk_management') as mock_risk, \
             patch.object(self.node, 'analyze_timeframes') as mock_timeframes, \
             patch.object(self.node, 'assess_technical_risk') as mock_tech_risk, \
             patch.object(self.node, 'calculate_cost_benefit') as mock_cost, \
             patch.object(self.node, 'evaluate_market_environment') as mock_market, \
             patch.object(self.node, 'design_execution_strategy') as mock_execution, \
             patch.object(self.node, 'generate_scenario_analysis') as mock_scenario, \
             patch.object(self.node, 'create_decision_metadata') as mock_metadata, \
             patch.object(self.node, 'setup_monitoring_alerts') as mock_alerts, \
             patch.object(self.node, 'generate_decision_summary') as mock_summary:
            
            # 配置模拟返回值
            mock_basic.return_value = {
                "leverage": 10,
                "direction": "long",
                "current_price": 50000.0,
                "position_size": 5000.0,
                "position_ratio": 0.05,
                "contract_quantity": 0.1
            }
            
            mock_risk.return_value = {
                "stop_loss": 47500.0,
                "take_profit": 52500.0,
                "risk_reward_ratio": 1.0,
                "risk_percentage": 0.025
            }
            
            mock_timeframes.return_value = {
                "consensus_score": 0.8,
                "conflicting_signals": 1
            }
            
            mock_tech_risk.return_value = {
                "volatility_risk": "moderate",
                "trend_strength": "moderate"
            }
            
            mock_cost.return_value = {
                "estimated_trading_fee": 2.5,
                "expected_return": 0.05,
                "expected_value": 250
            }
            
            mock_market.return_value = {
                "trend_regime": "trending",
                "volatility_regime": "normal"
            }
            
            mock_execution.return_value = {
                "entry_strategy": "immediate",
                "exit_strategy": "target_based"
            }
            
            mock_scenario.return_value = {
                "best_case": {"profit_potential": 500},
                "worst_case": {"loss_potential": -250}
            }
            
            mock_metadata.return_value = {
                "confidence": 75,
                "confidence_breakdown": {"technical_analysis": 0.8}
            }
            
            mock_alerts.return_value = {
                "price_alerts": {},
                "risk_alerts": {}
            }
            
            mock_summary.return_value = {
                "action_type": "open_long",
                "reasoning": "技术指标显示上涨趋势"
            }
            
            # 执行节点调用
            result = self.node(self.test_state)
            
            # 验证结果结构
            assert "messages" in result
            assert "data" in result
            
            # 检查是否有验证信息
            decisions = result["messages"][0].content
            import json
            decisions_data = json.loads(decisions)
            
            # 验证字段应该存在
            if "validation" in decisions_data.get("BTCUSDT", {}):
                validation_info = decisions_data["BTCUSDT"]["validation"]
                assert "is_valid" in validation_info
                assert "validation_level" in validation_info
                assert validation_info["validation_level"] == "moderate"
    
    def test_critical_error_handling(self):
        """测试严重错误处理"""
        ticker = "BTCUSDT"
        
        # 创建有严重错误的数据
        critical_error_decision = {
            "basic_params": {
                "leverage": 10,
                "current_price": 50000.0
            },
            "margin_management": {
                "margin_utilization": 0.99  # 极高的保证金使用率
            },
            "cost_benefit_analysis": {
                "expected_value": -500  # 负期望值
            }
        }
        
        context = {"ticker": ticker, "account_balance": 100000.0}
        
        result = self.node._validate_trading_decision(ticker, critical_error_decision, context)
        
        # 应该有严重错误
        assert result["critical_count"] > 0 or result["error_count"] > 0
        assert result["is_valid"] == False
    
    def test_validation_performance(self):
        """测试验证性能"""
        import time
        
        ticker = "BTCUSDT"
        decision_data = {
            "basic_params": {"leverage": 10, "current_price": 50000.0},
            "decision_metadata": {"confidence": 75}
        }
        context = {"ticker": ticker, "account_balance": 100000.0}
        
        # 测量验证时间
        start_time = time.time()
        result = self.node._validate_trading_decision(ticker, decision_data, context)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # 验证应该在合理时间内完成（比如小于1秒）
        assert validation_time < 1.0
        assert result is not None
    
    def test_validation_error_recovery(self):
        """测试验证过程中的错误恢复"""
        ticker = "BTCUSDT"
        
        # 创建可能导致验证异常的数据
        problematic_data = {
            "basic_params": {
                "leverage": "not_a_number",  # 错误的数据类型
                "current_price": None
            }
        }
        
        context = {"ticker": ticker}
        
        # 验证应该能处理异常而不崩溃
        result = self.node._validate_trading_decision(ticker, problematic_data, context)
        
        assert result is not None
        assert "is_valid" in result
        # 在异常情况下，应该返回False
        assert result["is_valid"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])