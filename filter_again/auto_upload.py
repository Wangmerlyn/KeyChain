import time
import subprocess

while True:
    print(f"[{time.ctime()}] Running ckpt_upload.sh...")
    subprocess.run(["bash", "/home/aiscuser/LongContextDataSynth/filter_again/ckpt_upload.sh"])
    print(f"[{time.ctime()}] Done. Sleeping for 30 minutes.\n")
    time.sleep(30 * 60)  # Sleep for 30 minutes