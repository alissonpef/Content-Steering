#!/bin/bash

error_exit() {
    echo "ERROR: $1" >&2
    exit 1
}

if [ "$EUID" -ne 0 ]; then
  error_exit "This script must be run with sudo (e.g., sudo ./starting_streaming.sh)."
fi

echo "--- Starting/Verifying Cache Servers (Docker Compose) ---"
if [ -f ./streaming-service/docker-compose.yml ]; then
    docker compose -f ./streaming-service/docker-compose.yml up -d
    if [ $? -ne 0 ]; then
        error_exit "Failed to start cache containers with Docker Compose."
    fi
    echo "Docker Compose command executed. Verifying container status..."
    sleep 3
else
    error_exit "File ./streaming-service/docker-compose.yml not found."
fi

EXPECTED_CONTAINERS=("video-streaming-cache-1" "video-streaming-cache-2" "video-streaming-cache-3")
for container_name in "${EXPECTED_CONTAINERS[@]}"; do
    if ! docker ps --filter "name=^/${container_name}$" --filter "status=running" --format "{{.Names}}" | grep -Fxq "${container_name}"; then
        error_exit "Container ${container_name} is not running or not found. Check Docker logs."
    fi
done
echo "All expected cache containers are running."

echo ""
echo "--- Obtaining IP Addresses of Cache Containers ---"
IP_CACHE_1=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-1 2>/dev/null)
IP_CACHE_2=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-2 2>/dev/null)
IP_CACHE_3=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-3 2>/dev/null)

if [ -z "$IP_CACHE_1" ] || [ -z "$IP_CACHE_2" ] || [ -z "$IP_CACHE_3" ]; then
    error_exit "Could not obtain IP for one or more caches. Check 'docker ps' and container names."
fi
echo "IPs obtained: Cache1=$IP_CACHE_1, Cache2=$IP_CACHE_2, Cache3=$IP_CACHE_3"

HOST_CACHE_1="video-streaming-cache-1"
HOST_CACHE_2="video-streaming-cache-2"
HOST_CACHE_3="video-streaming-cache-3"
HOST_STEERING="steering-service"

ETC_HOSTS_FILE="/etc/hosts"
BACKUP_ETC_HOSTS_FILE="/etc/hosts.bak_content_steering_$(date +%Y%m%d_%H%M%S)"
TMP_HOSTS_FILE="${ETC_HOSTS_FILE}.tmp_content_steering_$(date +%s)"

echo ""
echo "--- Updating /etc/hosts File ---"

echo "Backing up $ETC_HOSTS_FILE to $BACKUP_ETC_HOSTS_FILE..."
cp "$ETC_HOSTS_FILE" "$BACKUP_ETC_HOSTS_FILE" || error_exit "Failed to create backup of $ETC_HOSTS_FILE."

echo "Removing old content-steering entries (if they exist)..."
awk '
    BEGIN { in_cs_block=0 }
    /# START Content Steering entries by starting_streaming\.sh/ { in_cs_block=1; next }
    /# END Content Steering entries/ { in_cs_block=0; next }
    !in_cs_block { print }
' "$BACKUP_ETC_HOSTS_FILE" > "$TMP_HOSTS_FILE"

if [ $? -ne 0 ]; then
    cp "$BACKUP_ETC_HOSTS_FILE" "$ETC_HOSTS_FILE"
    error_exit "Failed to process $ETC_HOSTS_FILE with awk to remove old entries."
fi

echo "Adding new entries..."
{
    echo ""
    echo "# START Content Steering entries by starting_streaming.sh - $(date)"
    echo "$IP_CACHE_1    $HOST_CACHE_1"
    echo "$IP_CACHE_2    $HOST_CACHE_2"
    echo "$IP_CACHE_3    $HOST_CACHE_3"
    echo "127.0.0.1    $HOST_STEERING"
    echo "# END Content Steering entries"
} >> "$TMP_HOSTS_FILE"

echo "Applying changes to $ETC_HOSTS_FILE..."
mv "$TMP_HOSTS_FILE" "$ETC_HOSTS_FILE" || { cp "$BACKUP_ETC_HOSTS_FILE" "$ETC_HOSTS_FILE"; error_exit "Failed to move $TMP_HOSTS_FILE to $ETC_HOSTS_FILE. Backup restored."; }

echo ""
echo "--- Initial Setup Completed Successfully ---"