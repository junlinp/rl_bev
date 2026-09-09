#!/bin/bash
# Stop minikeyvalue background servers (Linux).

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$BASE/kv_pids.txt"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. Servers may not be running."
    exit 0
fi

while read -r pid; do
    if [ -n "$pid" ]; then
        if kill -9 "$pid" 2>/dev/null; then
            echo "Killed PID $pid"
        else
            echo "PID $pid already stopped"
        fi
    fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "Done"
