"""Stop minikeyvalue background servers."""
import os

pid_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_pids.txt")
if not os.path.exists(pid_file):
    print("No PID file found. Servers may not be running.")
    exit(0)

with open(pid_file) as f:
    pids = [int(line.strip()) for line in f if line.strip()]

import ctypes
for pid in pids:
    try:
        ctypes.windll.kernel32.TerminateProcess(
            ctypes.windll.kernel32.OpenProcess(1, False, pid), 0
        )
        print(f"Killed PID {pid}")
    except Exception:
        print(f"PID {pid} already stopped")

os.remove(pid_file)
print("Done")
