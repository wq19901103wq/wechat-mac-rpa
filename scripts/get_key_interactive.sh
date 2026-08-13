#!/bin/bash
# 交互式获取微信数据库密钥

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "           微信数据库密钥获取工具 (交互式)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 检查微信是否运行
if ! pgrep WeChat > /dev/null; then
    echo "⚠️  微信未运行，正在启动..."
    open -a WeChat
    echo "等待微信启动..."
    sleep 5
fi

echo "✅ 微信正在运行"
echo ""

# 方法1: 尝试使用 sudo strings
echo "方法1: 使用 strings 命令从微信内存中提取密钥"
echo "----------------------------------------------"
echo "这将执行: sudo strings /Applications/WeChat.app/Contents/MacOS/WeChat | grep ..."
echo ""
read -p "请输入您的Mac密码 (输入时不会显示): " -s password
echo ""
echo ""

# 使用密码执行sudo
echo "正在提取密钥..."
result=$(echo "$password" | sudo -S strings /Applications/WeChat.app/Contents/MacOS/WeChat | grep -E "[0-9a-f]{64}" | head -5) || true

if [ -n "$result" ]; then
    echo "✅ 找到可能的密钥:"
    echo ""
    echo "$result" | while read -r line; do
        # 取前32位作为db_key
        db_key=$(echo "$line" | cut -c1-32)
        echo "  完整: $line"
        echo "  db_key (前32位): $db_key"
        echo ""
    done
    
    echo "═══════════════════════════════════════════════════════════════"
    echo "请复制上面的一个 db_key (32位十六进制字符串)"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "⚠️  未找到密钥，尝试方法2..."
fi

echo ""
read -p "按回车键继续配置..."

# 运行配置向导
cd "$(dirname "$0")"
python3 setup_auto.py
