import os
import subprocess
import time

def set_maintenance_mode(on=True):
    if on:
        with open("../data/maintenance.flag", "w") as f:
            f.write("maintenance")
        print("Website set to maintenance mode.")
    else:
        if os.path.exists("../data/maintenance.flag"):
            os.remove("../data/maintenance.flag")
            print("Maintenance mode ended.")
        else:
            print("No maintenance flag found.")

def run_script(script_path):
    try:
        result = subprocess.run(["python", script_path], check=True)
        print(f"{script_path} finished successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error while running {script_path}: {e}")

if __name__ == "__main__":
    try:
        print("Starting maintenance cronjob...")
        
        # 1. Set maintenance mode
        set_maintenance_mode(True)
        
        # 2. Run scraper.py
        run_script("scraper.py")
        
        # 3. Run preprocessing.py
        run_script("preprocessing.py")
        
        # 4. Run processing.py
        run_script("processing.py")

    except Exception as e:
        print(f"Error during cronjob execution: {e}")

    finally:
        # 5. Always end maintenance
        set_maintenance_mode(False)
        print("Cronjob completed.")
