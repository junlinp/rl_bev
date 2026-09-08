"""Start minikeyvalue master + volume server in background."""
import subprocess, sys, os, time, signal

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "kv_data")
VOL_DIR = os.path.join(DATA_DIR, "vol1")
INDEX_DIR = os.path.join(DATA_DIR, "indexdb")
os.makedirs(VOL_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# kill old servers if running
import urllib.request
for port in [3000, 3001]:
    try:
        urllib.request.urlopen(f"http://localhost:{port}/", timeout=0.5)
    except Exception:
        pass  # not running

procs = []

# volume server
p1 = subprocess.Popen(
    [sys.executable, os.path.join(BASE, "kv_volume_server.py"), "--port", "3001", "--dir", VOL_DIR],
    creationflags=subprocess.CREATE_NO_WINDOW,
)
procs.append(("volume", p1))

# master server
mkv = os.path.join(BASE, "mkv.exe")
p2 = subprocess.Popen(
    [mkv, "-port", "3000", "-volumes", "localhost:3001", "-db", INDEX_DIR, "-replicas", "1", "server"],
    creationflags=subprocess.CREATE_NO_WINDOW,
)
procs.append(("master", p2))

# wait for ports
time.sleep(2)
for name, p in procs:
    if p.poll() is not None:
        print(f"[ERROR] {name} exited with code {p.returncode}")
        sys.exit(1)

# verify
import socket
for port in [3001, 3000]:
    s = socket.socket()
    s.settimeout(2)
    try:
        s.connect(("localhost", port))
        s.close()
    except Exception:
        print(f"[ERROR] port {port} not listening")
        sys.exit(1)

print(f"minikeyvalue running: master=:3000 volume=:3001")
print(f"PIDs: volume={p1.pid} master={p2.pid}")
print(f"Data: {DATA_DIR}")

# write PID file for easy cleanup
with open(os.path.join(BASE, "kv_pids.txt"), "w") as f:
    f.write(f"{p1.pid}\n{p2.pid}\n")
