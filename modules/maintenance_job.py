import os
import subprocess
import time

def run_maintenance():
    # 1. Nyalakan maintenance mode
    open("maintenance.flag", "w").close()
    print("Maintenance mode ON")

    try:
        # 2. Jalankan 3 script kamu berurutan
        print("Running scraper...")
        subprocess.run(["python", "scraper.py"], check=True)

        print("Running preprocessing...")
        subprocess.run(["python", "preprocessing.py"], check=True)

        print("Running processing...")
        subprocess.run(["python", "processing.py"], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error while running maintenance scripts: {e}")
    finally:
        # 3. Matikan maintenance mode
        if os.path.exists("maintenance.flag"):
            os.remove("maintenance.flag")
            print("Maintenance mode OFF")

if __name__ == "__main__":
    run_maintenance()
