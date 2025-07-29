#!/usr/bin/env python3
"""
AI Trading Bot Advanced - Launcher Script
Rezz AI Developer Team - V.6

Simple launcher with error handling and restart capability
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_banner():
    """Print startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🤖 AI TRADING BOT ADVANCED - REZZ V.6                    ║
║                                                              ║
║    👨‍💻 Developer: Rezz AI Team                               ║
║    ⚡ Ultra Fast: < 5 seconds response                       ║
║    🧠 Triple AI: RF + XGB + GB Models                       ║
║    🎯 Accuracy: 85%+ Ensemble Voting                        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'MetaTrader5',
        'pandas',
        'numpy',
        'telegram',
        'sklearn',
        'xgboost',
        'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_config():
    """Check if configuration files exist"""
    required_files = [
        'ai_trading_bot_advanced.py',
        'config.json',
        'requirements.txt'
    ]
    
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All configuration files found")
    return True

def run_bot():
    """Run the main bot with error handling"""
    try:
        print("🚀 Starting AI Trading Bot...")
        print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run the main bot
        result = subprocess.run([
            sys.executable, 
            'ai_trading_bot_advanced.py'
        ], check=True)
        
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user (Ctrl+C)")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Bot crashed with error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main launcher function"""
    print_banner()
    
    # Check system requirements
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return
    
    if not check_config():
        input("\nPress Enter to exit...")
        return
    
    # Auto-restart configuration
    auto_restart = True
    max_restarts = 5
    restart_count = 0
    
    while auto_restart and restart_count < max_restarts:
        success = run_bot()
        
        if success:
            print("✅ Bot finished successfully")
            break
        else:
            restart_count += 1
            if restart_count < max_restarts:
                print(f"\n🔄 Restarting bot... (Attempt {restart_count}/{max_restarts})")
                print("⏳ Waiting 10 seconds before restart...")
                time.sleep(10)
            else:
                print(f"\n❌ Maximum restart attempts ({max_restarts}) reached")
                print("🛑 Bot stopped permanently")
    
    print("\n" + "=" * 60)
    print("🏁 Bot launcher finished")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()
