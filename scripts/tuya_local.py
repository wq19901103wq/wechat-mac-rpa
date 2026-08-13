#!/usr/bin/env python3
"""
Tuya Local LAN Control - 本地局域网控制
直接通过 WiFi 控制设备，不需要 Cloud API
"""

import os
import sys
import json

try:
    import tinytuya
except ImportError:
    print("❌ tinytuya 未安装")
    print("   运行: pip3 install tinytuya")
    sys.exit(1)

# 设备配置文件
DEVICES_FILE = os.path.expanduser('~/.openclaw/tuya_local_devices.json')

def load_devices():
    """加载设备列表"""
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, 'r') as f:
            return json.load(f)
    return []

def save_devices(devices):
    """保存设备列表"""
    with open(DEVICES_FILE, 'w') as f:
        json.dump(devices, f, indent=2, ensure_ascii=False)

def find_device(name_or_id):
    """通过名称或ID查找设备"""
    devices = load_devices()
    for device in devices:
        if name_or_id.lower() in device.get('name', '').lower():
            return device
        if name_or_id == device.get('id'):
            return device
    return None

def control_device(device_info, action):
    """控制设备"""
    try:
        # 创建设备对象
        device = tinytuya.Device(
            device_info['id'],
            device_info['ip'],
            device_info['key'],
            version=device_info.get('version', '3.1')
        )
        
        # 连接设备
        device.set_dpsUsed({"1": None})  # 开关状态
        
        if action == 'on':
            device.turn_on()
            print(f"✅ {device_info['name']} 已打开")
        elif action == 'off':
            device.turn_off()
            print(f"✅ {device_info['name']} 已关闭")
        elif action == 'status':
            status = device.status()
            print(f"📊 {device_info['name']} 状态:")
            print(f"   在线: {'🟢' if status else '⚫'}")
            if status:
                for k, v in status.items():
                    print(f"   {k}: {v}")
        
        return True
        
    except Exception as e:
        print(f"❌ 控制失败: {e}")
        return False

def list_devices():
    """列出所有设备"""
    devices = load_devices()
    if not devices:
        print("⚠️ 没有配置的设备")
        print("\n请先添加设备信息到:")
        print(f"  {DEVICES_FILE}")
        print("\n格式:")
        print('''  [
    {
      "name": "客厅灯",
      "id": "bfxxxxxxxxxxxxxxxx",
      "ip": "192.168.2.100",
      "key": "xxxxxxxxxxxxxxxx",
      "version": "3.1"
    }
  ]''')
        return
    
    print(f"\n📱 已配置 {len(devices)} 个设备:\n")
    for i, device in enumerate(devices, 1):
        print(f"{i}. {device.get('name', 'Unknown')}")
        print(f"   IP: {device.get('ip', 'N/A')}")
        print(f"   ID: {device.get('id', 'N/A')}")
        print()

def add_device(name, device_id, ip, local_key, version='3.1'):
    """添加新设备"""
    devices = load_devices()
    
    new_device = {
        "name": name,
        "id": device_id,
        "ip": ip,
        "key": local_key,
        "version": version,
        "category": "unknown"
    }
    
    devices.append(new_device)
    save_devices(devices)
    print(f"✅ 已添加设备: {name}")

def quick_control(name, action):
    """快速控制设备"""
    device = find_device(name)
    if device:
        return control_device(device, action)
    else:
        print(f"❌ 未找到设备: {name}")
        print("\n可用设备:")
        list_devices()
        return False

def main():
    if len(sys.argv) < 2:
        print("""
🏠 Tuya 本地智能家居控制

用法:
  python3 tuya_local.py list                           # 列出设备
  python3 tuya_local.py on <设备名>                    # 打开设备
  python3 tuya_local.py off <设备名>                   # 关闭设备
  python3 tuya_local.py status <设备名>                # 查看状态
  python3 tuya_local.py add <名称> <ID> <IP> <密钥>    # 添加设备

示例:
  python3 tuya_local.py on 客厅灯
  python3 tuya_local.py off 卧室插座
        """)
        return
    
    cmd = sys.argv[1]
    
    if cmd == 'list':
        list_devices()
    elif cmd == 'on' and len(sys.argv) > 2:
        quick_control(sys.argv[2], 'on')
    elif cmd == 'off' and len(sys.argv) > 2:
        quick_control(sys.argv[2], 'off')
    elif cmd == 'status' and len(sys.argv) > 2:
        quick_control(sys.argv[2], 'status')
    elif cmd == 'add' and len(sys.argv) > 5:
        add_device(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print("❌ 未知命令或参数不足")

if __name__ == '__main__':
    main()
