#!/bin/bash
# 双击运行此脚本获取微信数据库密钥

cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════════"
echo "           获取微信数据库密钥"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 确保微信运行
if ! pgrep WeChat > /dev/null; then
    echo "启动微信..."
    open -a WeChat
    sleep 3
fi

echo "正在从微信中提取密钥..."
echo "请输入您的Mac登录密码（输入时不会显示）:"
echo ""

# 获取密钥
sudo strings /Applications/WeChat.app/Contents/MacOS/WeChat | grep -E "[0-9a-f]{64}" | head -5 | while read -r line; do
    echo "找到: $line"
    echo "db_key (取前32位): ${line:0:32}"
    echo ""
done

echo "═══════════════════════════════════════════════════════════════"
echo "请复制上面的 db_key (32位十六进制)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

read -p "获取到密钥了吗? (y/n): " has_key

if [ "$has_key" = "y" ]; then
    echo ""
    read -p "请输入32位db_key: " db_key
    
    # 更新配置文件
    sed -i.bak "s/db_key: \"YOUR_DB_KEY_HERE\"/db_key: \"$db_key\"/" config/config.yaml
    echo ""
    echo "✅ 配置文件已更新!"
    echo ""
    echo "现在可以启动机器人:"
    echo "  ./run_auto.sh"
fi

echo ""
read -p "按回车键关闭..."
