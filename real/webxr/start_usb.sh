#!/bin/bash
# USB 模式启动脚本

echo "=========================================="
echo "  Quest 3 手柄数据读取 - USB 模式"
echo "=========================================="
echo ""

# 检查 adb
if ! command -v adb &> /dev/null; then
    echo "❌ 未安装 adb"
    echo "请运行: sudo apt install android-tools-adb"
    exit 1
fi

# 检查依赖
if ! python3 -c "import websockets" 2>/dev/null; then
    echo "❌ 缺少依赖: websockets"
    echo "正在安装..."
    pip3 install websockets
    echo ""
fi

echo "步骤 1: 检查 Quest 3 连接..."
ADB_DEVICES=$(adb devices | grep -w "device" | wc -l)

if [ "$ADB_DEVICES" -eq 0 ]; then
    echo "❌ 未检测到 Quest 3 设备"
    echo ""
    echo "请确保:"
    echo "  1. Quest 3 已开启开发者模式"
    echo "  2. USB 线已连接"
    echo "  3. Quest 3 上点击了'允许 USB 调试'"
    echo ""
    echo "然后运行: adb devices"
    exit 1
fi

echo "✓ Quest 3 已连接"
echo ""

echo "步骤 2: 设置端口转发..."
# 反向端口转发 (Quest → Ubuntu)
adb reverse tcp:8765 tcp:8765
adb reverse tcp:8080 tcp:8080

if [ $? -eq 0 ]; then
    echo "✓ 端口转发设置成功"
else
    echo "❌ 端口转发失败"
    exit 1
fi

echo ""
echo "步骤 3: 检查 WebSocket 服务器..."
echo ""

# 检查 WebSocket 服务器是否已在运行（应该在终端1运行）
if lsof -i:8765 >/dev/null 2>&1; then
    echo "✓ WebSocket 服务器已在运行（端口 8765）"
    SERVER_PID=""
else
    echo "⚠ 警告: 未检测到 WebSocket 服务器"
    echo "  请确保在终端 1 运行了："
    echo "    python3 robot_control.py"
    echo ""
    echo "  将在3秒后继续启动 HTTP 服务器..."
    sleep 3
    SERVER_PID=""
fi

echo ""
echo "步骤 4: 启动 HTTP 服务器..."

# 只启动 HTTP 服务器
python3 -m http.server 8080 &
HTTP_PID=$!

echo ""
echo "=========================================="
echo "✓ USB 模式已启动!"
echo "=========================================="
echo ""
echo "在 Quest 3 浏览器中访问:"
echo ""
echo "   http://localhost:8080/index.html"
echo ""
echo "注意: 使用 localhost,不要用 IP 地址!"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo "=========================================="
echo ""

# 清理函数
cleanup() {
    echo ""
    echo "正在停止服务..."
    # 只停止 HTTP 服务器，不停止 WebSocket 服务器（在终端1）
    if [ ! -z "$HTTP_PID" ]; then
        kill $HTTP_PID 2>/dev/null
        echo "  HTTP 服务器已停止"
    fi
    echo "正在移除端口转发..."
    adb reverse --remove tcp:8765 2>/dev/null
    adb reverse --remove tcp:8080 2>/dev/null
    echo "✓ 服务已停止"
    echo ""
    echo "注意: WebSocket 服务器（终端1）仍在运行"
    echo "      如需停止，请在终端 1 按 Ctrl+C"
    exit 0
}

# 捕获 Ctrl+C
trap cleanup INT

# 保持脚本运行
wait
