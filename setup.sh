#!/bin/bash

echo "🚀 AI Hedge Fund Crypto 项目设置脚本"
echo "=================================="

# 检查uv是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ uv 未安装，正在安装..."
    curl -fsSL https://install.lunarvim.org/uv.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "✅ uv 已安装"
fi

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo "📦 创建虚拟环境..."
    uv venv --python 3.12
else
    echo "✅ 虚拟环境已存在"
fi

# 激活虚拟环境并安装依赖
echo "📚 安装项目依赖..."
source .venv/bin/activate

# 尝试多种安装方法
echo "尝试使用 uv pip install..."
if ! uv pip install -e .; then
    echo "尝试使用传统 pip install..."
    pip install -e .
fi

# 检查关键依赖
echo "🔍 检查关键依赖..."
if python -c "import langchain, langgraph, pandas" 2>/dev/null; then
    echo "✅ 核心依赖安装成功"
else
    echo "❌ 依赖安装失败，尝试单独安装核心包..."
    pip install langchain langgraph pandas matplotlib pyyaml python-dotenv
fi

# 复制配置文件
if [ ! -f "config.yaml" ]; then
    echo "📋 复制配置文件..."
    cp config.example.yaml config.yaml
else
    echo "✅ 配置文件已存在"
fi

if [ ! -f ".env" ]; then
    echo "🔑 复制环境变量文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件添加你的 API 密钥"
else
    echo "✅ 环境变量文件已存在"
fi

echo ""
echo "🎉 设置完成！"
echo "接下来的步骤："
echo "1. 编辑 .env 文件，添加你的币安 API 密钥"
echo "2. 运行: source .venv/bin/activate"
echo "3. 运行: python main.py (实时模式) 或 python backtest.py (回测模式)"