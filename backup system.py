#!/usr/bin/env python3
"""
AI Trading Bot Advanced - Backup & Recovery System
Automatic backup and recovery mechanisms for reliability
"""

import os
import json
import shutil
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class BackupRecoverySystem:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.backup_dir = "backups"
        self.models_dir = "models"
        self.logs_dir = "logs"
        
        # Create directories
        for directory in [self.backup_dir, self.models_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup backup system logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.logs_dir}/backup_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BackupSystem')
    
    def create_full_backup(self) -> bool:
        """Create complete system backup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_folder = os.path.join(self.backup_dir, f"backup_{timestamp}")
            os.makedirs(backup_folder, exist_ok=True)
            
            # Backup configuration
            if os.path.exists(self.config_path):
                shutil.copy2(self.config_path, backup_folder)
                self.logger.info("✅ Configuration backed up")
            
            # Backup models
            model_files = [f for f in os.listdir('.') if f.startswith('model_') or f.startswith('scaler_')]
            for model_file in model_files:
                shutil.copy2(model_file, backup_folder)
            
            if model_files:
                self.logger.info(f"✅ {len(model_files)} model files backed up")
            
            # Backup logs
            log_files = [f for f in os.listdir('.') if f.endswith('.log')]
            for log_file in log_files:
                try:
                    shutil.copy2(log_file, backup_folder)
                except:
                    pass  # Skip if log file is in use
            
            # Backup performance data
            perf_files = [f for f in os.listdir('.') if 'performance' in f or 'stats' in f]
            for perf_file in perf_files:
                try:
                    shutil.copy2(perf_file, backup_folder)
                except:
                    pass
            
            # Create backup manifest
            manifest = {
                'timestamp': timestamp,
                'backup_type': 'full',
                'files_backed_up': os.listdir(backup_folder),
                'created_by': 'AI Trading Bot Advanced V.6'
            }
            
            with open(os.path.join(backup_folder, 'manifest.json'), 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"✅ Full backup created: {backup_folder}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed: {e}")
            return False
    
    def restore_from_backup(self, backup_name: str) -> bool:
        """Restore system from backup"""
        try:
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            if not os.path.exists(backup_path):
                self.logger.error(f"❌ Backup not found: {backup_name}")
                return False
            
            # Read manifest
            manifest_path = os.path.join(backup_path, 'manifest.json')
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                self.logger.info(f"📋 Restoring backup from {manifest['timestamp']}")
            
            # Restore configuration
            config_backup = os.path.join(backup_path, 'config.json')
            if os.path.exists(config_backup):
                shutil.copy2(config_backup, '.')
                self.logger.info("✅ Configuration restored")
            
            # Restore models
            model_files = [f for f in os.listdir(backup_path) if f.startswith('model_') or f.startswith('scaler_')]
            for model_file in model_files:
                shutil.copy2(os.path.join(backup_path, model_file), '.')
            
            if model_files:
                self.logger.info(f"✅ {len(model_files)} model files restored")
            
            self.logger.info(f"✅ System restored from backup: {backup_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Restore failed: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days: int = 7) -> int:
        """Remove backups older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            removed_count = 0
            
            for backup_folder in os.listdir(self.backup_dir):
                backup_path = os.path.join(self.backup_dir, backup_folder)
                
                if os.path.isdir(backup_path):
                    # Extract timestamp from folder name
                    try:
                        timestamp_str = backup_folder.replace('backup_', '')
                        backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        
                        if backup_date < cutoff_date:
                            shutil.rmtree(backup_path)
                            removed_count += 1
                            self.logger.info(f"🗑️ Removed old backup: {backup_folder}")
                    except:
                        continue
            
            self.logger.info(f"✅ Cleanup complete: {removed_count} old backups removed")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")
            return 0
    
    def list_backups(self) -> List[Dict]:
        """List all available backups"""
        backups = []
        
        try:
            for backup_folder in sorted(os.listdir(self.backup_dir), reverse=True):
                backup_path = os.path.join(self.backup_dir, backup_folder)
                
                if os.path.isdir(backup_path):
                    manifest_path = os.path.join(backup_path, 'manifest.json')
                    
                    backup_info = {
                        'name': backup_folder,
                        'path': backup_path,
                        'size': self.get_folder_size(backup_path),
                        'created': datetime.fromtimestamp(os.path.getctime(backup_path))
                    }
                    
                    if os.path.exists(manifest_path):
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                            backup_info.update(manifest)
                    
                    backups.append(backup_info)
            
            return backups
            
        except Exception as e:
            self.logger.error(f"❌ Failed to list backups: {e}")
            return []
    
    def get_folder_size(self, folder_path: str) -> int:
        """Get total size of folder in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
        except:
            pass
        return total_size
    
    def verify_system_integrity(self) -> Dict[str, bool]:
        """Verify system integrity"""
        checks = {
            'config_file': os.path.exists('config.json'),
            'main_bot_file': os.path.exists('ai_trading_bot_advanced.py'),
            'requirements_file': os.path.exists('requirements.txt'),
            'models_present': len([f for f in os.listdir('.') if f.startswith('model_')]) > 0,
            'backup_system': os.path.exists(self.backup_dir),
            'logs_directory': os.path.exists(self.logs_dir)
        }
        
        all_good = all(checks.values())
        
        if all_good:
            self.logger.info("✅ System integrity check passed")
        else:
            failed_checks = [k for k, v in checks.items() if not v]
            self.logger.warning(f"⚠️ System integrity issues: {failed_checks}")
        
        return checks
    
    def auto_backup_schedule(self, interval_hours: int = 6) -> bool:
        """Check if auto backup is needed"""
        try:
            last_backup_file = os.path.join(self.backup_dir, 'last_backup.txt')
            
            if os.path.exists(last_backup_file):
                with open(last_backup_file, 'r') as f:
                    last_backup_str = f.read().strip()
                    last_backup = datetime.fromisoformat(last_backup_str)
                
                if datetime.now() - last_backup < timedelta(hours=interval_hours):
                    return False  # No backup needed yet
            
            # Create backup
            if self.create_full_backup():
                with open(last_backup_file, 'w') as f:
                    f.write(datetime.now().isoformat())
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Auto backup failed: {e}")
            return False
    
    def emergency_recovery(self) -> bool:
        """Emergency recovery from latest backup"""
        try:
            backups = self.list_backups()
            
            if not backups:
                self.logger.error("❌ No backups available for emergency recovery")
                return False
            
            latest_backup = backups[0]
            self.logger.info(f"🚨 Starting emergency recovery from: {latest_backup['name']}")
            
            return self.restore_from_backup(latest_backup['name'])
            
        except Exception as e:
            self.logger.error(f"❌ Emergency recovery failed: {e}")
            return False

def main():
    """Main backup system interface"""
    backup_system = BackupRecoverySystem()
    
    print("🔧 AI Trading Bot - Backup & Recovery System")
    print("=" * 50)
    
    while True:
        print("\n📋 Available Options:")
        print("1. Create Full Backup")
        print("2. List Backups")
        print("3. Restore from Backup")
        print("4. System Integrity Check")
        print("5. Cleanup Old Backups")
        print("6. Emergency Recovery")
        print("0. Exit")
        
        choice = input("\n🔹 Select option (0-6): ").strip()
        
        if choice == '1':
            print("\n🔄 Creating full backup...")
            if backup_system.create_full_backup():
                print("✅ Backup created successfully!")
            else:
                print("❌ Backup failed!")
        
        elif choice == '2':
            print("\n📋 Available Backups:")
            backups = backup_system.list_backups()
            if backups:
                for i, backup in enumerate(backups, 1):
                    size_mb = backup['size'] / (1024 * 1024)
                    print(f"{i}. {backup['name']} ({size_mb:.1f} MB) - {backup['created']}")
            else:
                print("No backups found.")
        
        elif choice == '3':
            backups = backup_system.list_backups()
            if backups:
                print("\n📋 Select backup to restore:")
                for i, backup in enumerate(backups, 1):
                    print(f"{i}. {backup['name']}")
                
                try:
                    selection = int(input("Enter backup number: ")) - 1
                    if 0 <= selection < len(backups):
                        backup_name = backups[selection]['name']
                        print(f"\n🔄 Restoring from {backup_name}...")
                        if backup_system.restore_from_backup(backup_name):
                            print("✅ Restore completed!")
                        else:
                            print("❌ Restore failed!")
                    else:
                        print("❌ Invalid selection!")
                except ValueError:
                    print("❌ Invalid input!")
            else:
                print("No backups available for restore.")
        
        elif choice == '4':
            print("\n🔍 Checking system integrity...")
            checks = backup_system.verify_system_integrity()
            for check, status in checks.items():
                status_icon = "✅" if status else "❌"
                print(f"{status_icon} {check.replace('_', ' ').title()}")
        
        elif choice == '5':
            days = input("\nKeep backups for how many days? (default: 7): ").strip()
            try:
                days = int(days) if days else 7
                removed = backup_system.cleanup_old_backups(days)
                print(f"✅ Removed {removed} old backups")
            except ValueError:
                print("❌ Invalid number of days!")
        
        elif choice == '6':
            print("\n🚨 Starting emergency recovery...")
            if backup_system.emergency_recovery():
                print("✅ Emergency recovery completed!")
            else:
                print("❌ Emergency recovery failed!")
        
        elif choice == '0':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option!")

if __name__ == "__main__":
    main()
