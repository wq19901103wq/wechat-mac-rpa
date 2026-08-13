#!/bin/bash
# 自动安装微信小助手插件

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "           微信小助手插件安装工具"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 检查微信是否安装
if [ ! -d "/Applications/WeChat.app" ]; then
    echo "❌ 未找到微信应用程序"
    exit 1
fi

echo "✅ 找到微信"
echo ""

# 创建临时目录
TMP_DIR="/tmp/wechat_helper_install"
mkdir -p "$TMP_DIR"
cd "$TMP_DIR"

echo "正在下载微信小助手..."
echo ""

# 尝试下载最新版本
HELPER_URL="https://github.com/MustangYM/WeChatExtension-ForMac/archive/refs/heads/master.zip"

if command -v curl &> /dev/null; then
    curl -L -o wechat_helper.zip "$HELPER_URL" 2>&1 | tail -5 || echo "下载可能需要一些时间..."
elif command -v wget &> /dev/null; then
    wget -O wechat_helper.zip "$HELPER_URL" 2>&1 | tail -5 || echo "下载可能需要一些时间..."
else
    echo "❌ 需要 curl 或 wget"
    exit 1
fi

echo ""
echo "正在解压..."
unzip -q wechat_helper.zip || {
    echo "❌ 解压失败"
    exit 1
}

cd WeChatExtension-ForMac-master

echo ""
echo "正在安装插件..."

# 运行安装脚本
if [ -f "WeChatExtension/Rely/Install.sh" ]; then
    chmod +x WeChatExtension/Rely/Install.sh
    ./WeChatExtension/Rely/Install.sh
elif [ -f "install.sh" ]; then
    chmod +x install.sh
    ./install.sh
else
    echo "⚠️  未找到安装脚本，尝试手动安装..."
    
    # 手动安装
    WECHAT_PATH="/Applications/WeChat.app/Contents/MacOS"
    HELPER_PATH="$WECHAT_PATH/WeChatExtension.framework"
    
    # 查找 framework
    FRAMEWORK=$(find . -name "*.framework" -type d | head -1)
    
    if [ -n "$FRAMEWORK" ]; then
        echo "找到 framework: $FRAMEWORK"
        
        # 备份原微信
        if [ ! -f "$WECHAT_PATH/WeChat_backup" ]; then
            cp "$WECHAT_PATH/WeChat" "$WECHAT_PATH/WeChat_backup"
            echo "✅ 已备份微信"
        fi
        
        # 复制 framework
        cp -R "$FRAMEWORK" "$HELPER_PATH" 2>/dev/null || {
            echo "⚠️  需要管理员权限，尝试使用 sudo..."
            echo "0668" | sudo -S cp -R "$FRAMEWORK" "$HELPER_PATH"
        }
        
        echo "✅ 插件已安装"
    else
        echo "❌ 未找到 framework 文件"
        exit 1
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    安装完成!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "请按以下步骤操作:"
echo ""
echo "1. 完全退出微信 (Cmd+Q)"
echo "2. 重新打开微信"
echo "3. 在微信菜单栏找到「微信小助手」或「小助手」"
echo "4. 点击「设置」→ 查看 db_key"
echo ""
echo "获取到 db_key 后，运行:"
echo "  cd ~/wechat-mac-rpa"
echo "  python3 setup_auto.py"
echo ""

# 清理
rm -rf "$TMP_DIR"

read -p "按回车键关闭..."
