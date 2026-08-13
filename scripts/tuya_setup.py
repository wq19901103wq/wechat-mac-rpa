#!/usr/bin/env python3
"""
Tuya Smart Home - Automated Setup
Auto-configures Tuya Cloud API connection with minimal user input.
"""

import os
import sys
import json
import subprocess

# Configuration paths
CONFIG_DIR = os.path.expanduser("~/.openclaw")
CONFIG_FILE = os.path.join(CONFIG_DIR, "tuya_config.json")
DEVICES_FILE = os.path.join(CONFIG_DIR, "tuya_devices.json")

def print_step(step_num, message):
    """Print formatted step message"""
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {message}")
    print('='*60)

def check_python_dependencies():
    """Check and install required Python packages"""
    print_step(1, "Checking Python Dependencies")
    
    required = ["tuya-connector-python"]
    
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} already installed")
        except ImportError:
            print(f"📦 Installing {package}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                check=True
            )
            print(f"✅ {package} installed")

def get_user_credentials():
    """Get Access ID and Secret from user"""
    print_step(2, "Tuya API Credentials")
    
    print("\nPlease enter your Tuya IoT Platform credentials:")
    print("(Get them from: https://iot.tuya.com/ → Project → Overview)")
    print()
    
    access_id = input("Access ID: ").strip()
    access_secret = input("Access Secret: ").strip()
    
    if not access_id or not access_secret:
        print("❌ Access ID and Secret are required!")
        sys.exit(1)
    
    return access_id, access_secret

def auto_detect_uid(access_id, access_secret):
    """Auto-detect user UID from Tuya API"""
    print_step(3, "Auto-Detecting User UID")
    
    try:
        from tuya_connector import TuyaOpenAPI
    except ImportError:
        print("❌ tuya-connector-python not installed!")
        sys.exit(1)
    
    # Try different endpoints
    endpoints = {
        "China": "https://openapi.tuyacn.com",
        "US": "https://openapi.tuyaus.com",
        "EU": "https://openapi.tuyaeu.com",
    }
    
    for region, endpoint in endpoints.items():
        print(f"\n🌐 Trying {region} endpoint...")
        try:
            api = TuyaOpenAPI(endpoint, access_id, access_secret)
            api.connect()
            
            # Try to get token info (contains uid)
            response = api.get("/v1.0/token")
            
            if response.get('success'):
                result = response.get('result', {})
                uid = result.get('uid', '')
                
                if uid:
                    print(f"✅ UID found: {uid}")
                    return endpoint, uid
                else:
                    print(f"⚠️  Token success but no UID")
            else:
                error = response.get('msg', 'Unknown error')
                if 'sign' in error.lower():
                    print(f"   ❌ Invalid credentials")
                elif 'permission' in error.lower():
                    print(f"   ❌ Permission denied - check app authorization")
                else:
                    print(f"   ❌ {error}")
                    
        except Exception as e:
            print(f"   ❌ Connection failed: {str(e)[:50]}")
    
    print("\n❌ Could not auto-detect UID!")
    print("\nTroubleshooting:")
    print("1. Verify your Access ID and Secret are correct")
    print("2. Check that your Tuya/Smart Life app is linked to the project")
    print("3. Go to https://iot.tuya.com/ → API Test to verify connectivity")
    sys.exit(1)

def fetch_devices(api, uid):
    """Fetch all devices from Tuya API"""
    print_step(4, "Fetching Device List")
    
    response = api.get(f"/v1.0/users/{uid}/devices")
    
    if not response.get('success'):
        print(f"❌ Failed to fetch devices: {response.get('msg')}")
        return []
    
    devices = response.get('result', [])
    print(f"✅ Found {len(devices)} devices!")
    
    # Parse and display devices
    device_list = []
    category_map = {
        'dj': '💡 Light',
        'kt': '❄️ AC',
        'cl': '🪟 Curtain',
        'qn': '🔥 Floor Heating',
        'xfj': '🌀 Fresh Air',
        'wg2': '📡 Gateway',
    }
    
    print("\n📱 Devices:")
    for i, device in enumerate(devices, 1):
        name = device.get('name', 'Unknown')
        category = device.get('category', 'unknown')
        device_id = device.get('id', '')
        online = device.get('online', False)
        
        device_info = {
            "name": name,
            "id": device_id,
            "category": category,
            "online": online
        }
        device_list.append(device_info)
        
        icon = category_map.get(category, '📟 Device')
        status = "🟢" if online else "⚫"
        print(f"  {i}. {status} {icon} {name}")
    
    return device_list

def save_config(access_id, access_secret, endpoint, uid, devices):
    """Save configuration to files"""
    print_step(5, "Saving Configuration")
    
    # Ensure config directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Save API config
    config = {
        "access_id": access_id,
        "access_secret": access_secret,
        "api_endpoint": endpoint,
        "uid": uid
    }
    
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✅ Config saved to: {CONFIG_FILE}")
    
    # Save device list
    with open(DEVICES_FILE, 'w') as f:
        json.dump(devices, f, indent=2, ensure_ascii=False)
    print(f"✅ Device list saved to: {DEVICES_FILE}")

def test_connection():
    """Test the connection by listing devices"""
    print_step(6, "Testing Connection")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tuya_control = os.path.join(script_dir, "tuya_control.py")
    
    try:
        result = subprocess.run(
            [sys.executable, tuya_control, "list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print(result.stdout)
        if result.returncode == 0:
            print("✅ Connection test passed!")
        else:
            print(f"⚠️  Test returned: {result.stderr}")
    except Exception as e:
        print(f"⚠️  Test failed: {e}")

def main():
    """Main setup流程"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Tuya Smart Home - Automated Setup                        ║
║     涂鸦智能家居 - 自动化配置                                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Step 1: Check dependencies
        check_python_dependencies()
        
        # Step 2: Get credentials
        access_id, access_secret = get_user_credentials()
        
        # Step 3: Auto-detect UID
        endpoint, uid = auto_detect_uid(access_id, access_secret)
        
        # Step 4: Fetch devices
        from tuya_connector import TuyaOpenAPI
        api = TuyaOpenAPI(endpoint, access_id, access_secret)
        api.connect()
        devices = fetch_devices(api, uid)
        
        if not devices:
            print("❌ No devices found!")
            print("Make sure your Tuya/Smart Life app is linked and has devices.")
            sys.exit(1)
        
        # Step 5: Save config
        save_config(access_id, access_secret, endpoint, uid, devices)
        
        # Step 6: Test connection
        test_connection()
        
        # Success message
        print("""
╔══════════════════════════════════════════════════════════════╗
║     ✅ Setup Complete!                                        ║
║     ✅ 配置完成！                                             ║
╚══════════════════════════════════════════════════════════════╝

🎉 You can now control your Tuya devices!

Quick Commands:
  python3 scripts/tuya_control.py list
  python3 scripts/tuya_control.py on "客厅主灯"
  python3 scripts/tuya_control.py off "主卧灯带"

Or use in OpenClaw/Feishu:
  "列出所有设备"
  "打开客厅主灯"
  "关闭主卧空调"
        """)
        
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
