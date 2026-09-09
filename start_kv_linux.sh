#!/bin/bash
# Start minikeyvalue master + volume server in background (Linux).
set -e

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$BASE/kv_data"
VOL_DIR="$DATA_DIR/vol1"
INDEX_DIR="$DATA_DIR/indexdb"
MKV_BIN="$BASE/mkv_repo/src/mkv"
PID_FILE="$BASE/kv_pids.txt"

mkdir -p "$VOL_DIR" "$INDEX_DIR"

# stop any previously started servers from our own PID file
if [ -f "$PID_FILE" ]; then
    while read -r pid; do
        [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
    done < "$PID_FILE"
    rm -f "$PID_FILE"
fi

# build mkv if the binary is missing
if [ ! -x "$MKV_BIN" ]; then
    echo "building mkv..."
    export PATH="$PATH:/usr/local/go/bin"
    ( cd "$BASE/mkv_repo/src" && go build -o mkv . )
fi

# volume server (Python, cross-platform)
setsid nohup python3 "$BASE/kv_volume_server.py" --port 3001 --dir "$VOL_DIR" \
    > "$BASE/kv_stdout.txt" 2> "$BASE/kv_stderr.txt" < /dev/null &
VOL_PID=$!
disown

sleep 1

# master server
setsid nohup "$MKV_BIN" -port 3000 -volumes localhost:3001 -db "$INDEX_DIR" -replicas 1 server \
    >> "$BASE/kv_stdout.txt" 2>> "$BASE/kv_stderr.txt" < /dev/null &
MASTER_PID=$!
disown

sleep 1

# verify
for port in 3001 3000; do
    if ! (exec 3<>"/dev/tcp/localhost/$port") 2>/dev/null; then
        echo "[ERROR] port $port not listening"
        exit 1
    fi
    exec 3<&- 3>&-
done

echo "minikeyvalue running: master=:3000 volume=:3001"
echo "PIDs: volume=$VOL_PID master=$MASTER_PID"
echo "Data: $DATA_DIR"

printf '%s\n%s\n' "$VOL_PID" "$MASTER_PID" > "$PID_FILE"
