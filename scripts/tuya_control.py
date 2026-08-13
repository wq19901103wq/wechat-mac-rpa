#!/usr/bin/env python3
"""
Tuya Smart Home Control - Cloud API Version
控制涂鸦智能家居设备
"""

import os
import sys
import json

# 尝试导入 tuya SDK
try:
    from tuya_connector import TuyaOpenAPI
    TUYA_SDK_AVAILABLE = True
except ImportError:
    TUYA_SDK_AVAILABLE = False
    print("❌ tuya-connector-python 未安装")
    print("   运行: pip3 install tuya-connector-python")
    sys.exit(1)

# 配置文件路径
CONFIG_FILE = os.path.expanduser('~/.openclaw/tuya_config.json')
DEVICES_FILE = os.path.expanduser('~/.openclaw/tuya_devices.json')

def load_config():
    """加载配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_devices(devices):
    """保存设备列表"""
    with open(DEVICES_FILE, 'w') as f:
        json.dump(devices, f, indent=2, ensure_ascii=False)

def load_devices():
    """加载设备列表"""
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, 'r') as f:
            return json.load(f)
    return []

def get_api():
    """获取 Tuya API 连接"""
    config = load_config()
    access_id = config.get('access_id', '')
    access_secret = config.get('access_secret', '')
    endpoint = config.get('api_endpoint', 'https://openapi.tuyacn.com')
    
    if not access_id or not access_secret:
        print("❌ 缺少 Tuya API 凭证")
        return None, None
    
    api = TuyaOpenAPI(endpoint, access_id, access_secret)
    api.connect()
    return api, config.get('uid', '')

def list_devices():
    """列出所有设备"""
    api, uid = get_api()
    if not api:
        return
    
    # 使用 UID 获取设备列表
    if uid:
        response = api.get(f"/v1.0/users/{uid}/devices")
    else:
        response = api.get("/v1.0/users/-/devices")
    
    if response.get('success'):
        devices = response.get('result', [])
        print(f"\n📱 发现 {len(devices)} 个设备:\n")
        
        device_list = []
        for i, device in enumerate(devices, 1):
            device_info = {
                "name": device.get('name', 'Unknown'),
                "id": device.get('id'),
                "category": device.get('category'),
                "online": device.get('online', False)
            }
            device_list.append(device_info)
            
            status_icon = "🟢" if device.get('online') else "⚫"
            category_map = {
                'dj': '灯', 'kt': '空调', 'cl': '窗帘', 
                'qn': '地暖', 'wg2': '网关', 'xfj': '新风'
            }
            category = category_map.get(device.get('category'), device.get('category', '设备'))
            
            print(f"{i}. {status_icon} {device.get('name', 'Unknown')} ({category})")
            print(f"   ID: {device.get('id')}")
        
        save_devices(device_list)
        print(f"\n✅ 设备列表已保存")
    else:
        print(f"❌ 获取设备失败: {response.get('msg', 'Unknown error')}")

def device_status(device_id):
    """获取设备状态"""
    api, _ = get_api()
    if not api:
        return
    
    response = api.get(f"/v1.0/devices/{device_id}")
    if response.get('success'):
        device = response.get('result', {})
        print(f"\n📟 {device.get('name', 'Unknown')}:")
        print(f"   在线: {'🟢 是' if device.get('online') else '⚫ 否'}")
        if device.get('status'):
            print(f"   状态:")
            for status in device.get('status', []):
                code = status.get('code', '')
                value = status.get('value', '')
                print(f"     • {code}: {value}")
    else:
        print(f"❌ 获取状态失败")

def control_device(device_id, command):
    """控制设备"""
    api, _ = get_api()
    if not api:
        return False
    
    commands = []
    if command == 'on':
        commands = [{"code": "switch_led", "value": True}]
    elif command == 'off':
        commands = [{"code": "switch_led", "value": False}]
    elif command.startswith('temp:'):
        value = int(command.split(':')[1])
        commands = [{"code": "temp_set", "value": value}]
    
    if commands:
        response = api.post(f"/v1.0/devices/{device_id}/commands", {"commands": commands})
        if response.get('success'):
            print(f"✅ 命令已发送: {command}")
            return True
        else:
            print(f"❌ 命令失败: {response.get('msg', 'Unknown')}")
            return False
    return False

def quick_control(name, action):
    """通过设备名称快速控制"""
    devices = load_devices()
    
    # 查找匹配的设备
    for device in devices:
        if name.lower() in device.get('name', '').lower():
            print(f"找到设备: {device.get('name')}")
            return control_device(device.get('id'), action)
    
    print(f"❌ 未找到设备: {name}")
    print("\n可用设备:")
    list_devices()
    return False

def main():
    if len(sys.argv) < 2:
        print("""
🏠 Tuya 智能家居控制 (Cloud API)

用法:
  python3 tuya_control.py list                    # 列出所有设备
  python3 tuya_control.py status <device_id>      # 查看设备状态
  python3 tuya_control.py on <设备名>              # 打开设备
  python3 tuya_control.py off <设备名>             # 关闭设备

示例:
  python3 tuya_control.py on 客厅主灯
  python3 tuya_control.py off 主卧灯带
        """)
        return
    
    cmd = sys.argv[1]
    
    if cmd == 'list':
        list_devices()
    elif cmd == 'status' and len(sys.argv) > 2:
        device_status(sys.argv[2])
    elif cmd == 'on' and len(sys.argv) > 2:
        quick_control(sys.argv[2], 'on')
    elif cmd == 'off' and len(sys.argv) > 2:
        quick_control(sys.argv[2], 'off')
    else:
        print("❌ 未知命令或参数不足")

if __name__ == '__main__':
    main()
