import shutil
import os

total, used, free = shutil.disk_usage("E:\\")
print(f"E: Total={total/(1024**3):.1f} GB, Used={used/(1024**3):.1f} GB, Free={free/(1024**3):.1f} GB")
try:
    print("E: root entries:", os.listdir("E:\\")[:10])
except Exception as e:
    print("Could not list E:", e)
