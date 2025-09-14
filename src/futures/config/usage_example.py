"""
期货配置管理使用示例

演示如何使用期货配置验证器、API管理器和环境检查器的功能。
"""

import asyncio
import logging
from pathlib import Path

from .config_validator import FuturesConfigValidator
from .api_manager import FuturesAPIManager, APIEnvironment
from .environment_checker import EnvironmentChecker


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demonstrate_config_validation():
    """演示配置验证功能"""
    logger.info("=== 期货配置验证示例 ===")
    
    validator = FuturesConfigValidator()
    
    # 1. 从文件验证配置
    try:
        config_path = "config.yaml"
        futures_config = validator.validate_from_file(config_path)
        logger.info(f"配置验证成功: {futures_config.environment}")
        logger.info(f"交易对: {futures_config.symbols}")
        logger.info(f"最大杠杆: {futures_config.risk.max_leverage}")
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
    
    # 2. 生成默认配置
    try:
        default_config = validator.generate_default_config()
        logger.info("默认配置生成成功")
        logger.info(f"默认环境: {default_config['futures']['environment']}")
    except Exception as e:
        logger.error(f"默认配置生成失败: {e}")


async def demonstrate_api_management():
    """演示API管理功能"""
    logger.info("=== 期货API管理示例 ===")
    
    # 1. Testnet环境API管理
    testnet_api = FuturesAPIManager(APIEnvironment.TESTNET)
    
    try:
        # 从环境变量加载凭证
        credentials = testnet_api.load_credentials_from_env()
        logger.info(f"API凭证加载成功: {credentials.environment.value}")
        
        # 验证API凭证
        validation_result = await testnet_api.validate_api_credentials()
        logger.info(f"API验证结果: {validation_result['valid']}")
        
        # 检查权限
        permissions = await testnet_api.check_api_permissions()
        logger.info(f"API权限: {permissions}")
        
        # 获取凭证状态
        status = testnet_api.get_credentials_status()
        logger.info(f"凭证状态: {status}")
        
    except Exception as e:
        logger.error(f"API管理演示失败: {e}")


async def demonstrate_environment_checking():
    """演示环境检查功能"""
    logger.info("=== 环境检查示例 ===")
    
    checker = EnvironmentChecker()
    
    try:
        # 1. 检查单个环境健康状态
        testnet_health = await checker.check_environment_health(APIEnvironment.TESTNET)
        logger.info(f"Testnet健康状态: {testnet_health.status.value}")
        logger.info(f"API响应延迟: {testnet_health.latency_ms:.1f}ms")
        
        # 2. 生成环境报告
        testnet_report = await checker.generate_environment_report(APIEnvironment.TESTNET, "config.yaml")
        logger.info(f"Testnet报告: {len(testnet_report.recommendations)}个建议, {len(testnet_report.warnings)}个警告")
        
        for rec in testnet_report.recommendations[:3]:  # 显示前3个建议
            logger.info(f"建议: {rec}")
        
        for warn in testnet_report.warnings[:3]:  # 显示前3个警告
            logger.info(f"警告: {warn}")
        
        # 3. 全面环境检查
        all_reports = await checker.comprehensive_environment_check()
        logger.info(f"环境检查完成，共检查{len(all_reports)}个环境")
        
        for env, report in all_reports.items():
            logger.info(f"{env.value}: {report.health.status.value}")
        
        # 4. 自动选择最佳环境
        optimal_env = await checker.auto_switch_environment()
        logger.info(f"推荐环境: {optimal_env.value}")
        
        # 5. 保存环境报告
        report_path = "testnet_environment_report.yaml"
        checker.save_environment_report(testnet_report, report_path)
        logger.info(f"环境报告已保存到: {report_path}")
        
    except Exception as e:
        logger.error(f"环境检查演示失败: {e}")


async def demonstrate_integrated_workflow():
    """演示集成工作流程"""
    logger.info("=== 集成工作流程示例 ===")
    
    try:
        # 1. 环境检查和选择
        checker = EnvironmentChecker()
        optimal_env = await checker.auto_switch_environment()
        logger.info(f"选择的环境: {optimal_env.value}")
        
        # 2. 配置验证
        validator = FuturesConfigValidator()
        config = validator.validate_from_file("config.yaml")
        
        # 检查配置与选择环境的一致性
        if config.environment != optimal_env.value:
            logger.warning(f"配置环境 {config.environment} 与选择环境 {optimal_env.value} 不一致")
        
        # 3. API管理
        api_manager = FuturesAPIManager(optimal_env)
        credentials = api_manager.load_credentials_from_env()
        
        # 4. API验证
        api_result = await api_manager.validate_api_credentials()
        if api_result['valid']:
            logger.info("API验证成功，系统已准备就绪")
        else:
            logger.error("API验证失败，请检查配置")
        
        # 5. 获取API端点配置
        endpoints = api_manager.get_api_endpoints()
        logger.info(f"期货API端点: {endpoints.futures_url}")
        logger.info(f"WebSocket端点: {endpoints.websocket_futures_url}")
        
        return {
            'environment': optimal_env,
            'config': config,
            'api_manager': api_manager,
            'api_valid': api_result['valid']
        }
        
    except Exception as e:
        logger.error(f"集成工作流程失败: {e}")
        return None


def demonstrate_configuration_templates():
    """演示配置模板功能"""
    logger.info("=== 配置模板示例 ===")
    
    checker = EnvironmentChecker()
    
    # 生成不同环境的配置模板
    for env in APIEnvironment:
        template = checker.get_environment_config_template(env)
        logger.info(f"{env.value}环境配置模板:")
        logger.info(f"  最大杠杆: {template['futures']['risk']['max_leverage']}")
        logger.info(f"  最大仓位: {template['futures']['risk']['max_position_size']}")
        logger.info(f"  日损失限额: {template['futures']['risk']['daily_loss_limit']}")


async def main():
    """主函数"""
    logger.info("期货配置管理系统演示开始")
    
    # 1. 配置验证演示
    await demonstrate_config_validation()
    
    # 2. API管理演示
    await demonstrate_api_management()
    
    # 3. 环境检查演示  
    await demonstrate_environment_checking()
    
    # 4. 配置模板演示
    demonstrate_configuration_templates()
    
    # 5. 集成工作流程演示
    result = await demonstrate_integrated_workflow()
    
    if result and result['api_valid']:
        logger.info("演示完成，系统配置正确且API可用")
    else:
        logger.warning("演示完成，但系统配置或API存在问题")
    
    logger.info("期货配置管理系统演示结束")


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())