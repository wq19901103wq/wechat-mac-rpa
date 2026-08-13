#!/bin/bash
# 微信RPA完整配置脚本

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "           微信RPA - 完整配置向导"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查微信运行
if ! pgrep WeChat > /dev/null; then
    echo -e "${YELLOW}⚠️  微信未运行，正在启动...${NC}"
    open -a WeChat
    sleep 5
fi

echo -e "${GREEN}✅ 微信正在运行${NC}"
echo ""

# 查找数据库
DB_PATH=$(find ~/Library/Containers/com.tencent.xinWeChat -name "msg_*.db" 2>/dev/null | head -1)
if [ -n "$DB_PATH" ]; then
    MSG_DIR=$(dirname "$DB_PATH")
    echo -e "${GREEN}✅ 找到数据库目录: $MSG_DIR${NC}"
else
    echo -e "${RED}❌ 未找到数据库目录${NC}"
fi
echo ""

# 获取db_key
echo "═══════════════════════════════════════════════════════════════"
echo "步骤: 获取数据库密钥 (db_key)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "db_key 是 32 位十六进制字符串，用于解密微信数据库。"
echo ""
echo "获取方法："
echo ""
echo "方法1 (推荐) - 使用 strings 命令:"
echo "  请在另一个终端窗口执行："
echo ""
echo "    sudo strings /Applications/WeChat.app/Contents/MacOS/WeChat | grep -E '[0-9a-f]{64}' | head -5"
echo ""
echo "  然后输入Mac密码，复制输出的 64 位字符串的前 32 位。"
echo ""
echo "方法2 - 使用第三方工具:"
echo "  - 安装微信小助手等插件"
echo "  - 在插件设置中查看 db_key"
echo ""
echo "-"  | head -50
echo ""

# 读取db_key
read -p "请输入获取到的 db_key (32位十六进制): " DB_KEY

# 验证格式
if [ ${#DB_KEY} -ne 32 ]; then
    echo -e "${RED}❌ 错误: db_key 应为 32 位，当前 ${#DB_KEY} 位${NC}"
    exit 1
fi

if ! [[ "$DB_KEY" =~ ^[0-9a-fA-F]{32}$ ]]; then
    echo -e "${RED}❌ 错误: db_key 应只包含十六进制字符 (0-9, a-f)${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ db_key 格式正确${NC}"
echo ""

# 更新配置文件
CONFIG_FILE="$(dirname "$0")/config/config.yaml"

if [ -f "$CONFIG_FILE" ]; then
    # 备份原配置
    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
    
    # 替换db_key
    sed -i '' "s/db_key: \"YOUR_DB_KEY_HERE\"/db_key: \"$DB_KEY\"/" "$CONFIG_FILE"
    
    # 如果有数据库路径也更新
    if [ -n "$MSG_DIR" ]; then
        # 检查是否已有db_path配置
        if grep -q "db_path:" "$CONFIG_FILE"; then
            sed -i '' "s|db_path: .*|db_path: \"$MSG_DIR\"|" "$CONFIG_FILE"
        fi
    fi
    
    echo -e "${GREEN}✅ 配置文件已更新: $CONFIG_FILE${NC}"
else
    echo -e "${RED}❌ 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    配置完成!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "配置摘要:"
echo "  - db_key: $DB_KEY"
echo "  - 数据库: $MSG_DIR"
echo "  - 配置: $CONFIG_FILE"
echo ""
echo "下一步:"
echo ""
echo "1. 授权辅助功能 (只需一次):"
echo "   系统设置 → 隐私与安全 → 辅助功能 → 添加终端/iTerm"
echo ""
echo "2. 启动机器人:"
echo "   cd ~/wechat-mac-rpa"
echo "   ./run_auto.sh"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

read -p "是否现在启动机器人? (y/n): " START_NOW

if [ "$START_NOW" = "y" ] || [ "$START_NOW" = "Y" ]; then
    echo ""
    echo "启动机器人..."
    "$(dirname "$0")/run_auto.sh"
fi
