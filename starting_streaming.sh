error_exit() {
    echo "ERRO: $1" >&2
    exit 1
}

if [ "$EUID" -ne 0 ]; then
  error_exit "Este script precisa ser executado com sudo (ex: sudo ./starting_streaming.sh)."
fi

echo "--- Iniciando/Verificando Servidores de Cache (Docker Compose) ---"
if [ -f ./streaming-service/docker-compose.yml ]; then
    docker compose -f ./streaming-service/docker-compose.yml up -d
    if [ $? -ne 0 ]; then
        error_exit "Falha ao iniciar os containers de cache com Docker Compose."
    fi
    echo "Comando Docker Compose executado. Verificando status dos containers..."
    sleep 3 
else
    error_exit "Arquivo ./streaming-service/docker-compose.yml não encontrado."
fi

EXPECTED_CONTAINERS=("video-streaming-cache-1" "video-streaming-cache-2" "video-streaming-cache-3")
for container_name in "${EXPECTED_CONTAINERS[@]}"; do
    if ! docker ps --filter "name=^/${container_name}$" --filter "status=running" --format "{{.Names}}" | grep -Fxq "${container_name}"; then
        error_exit "Container ${container_name} não está rodando ou não foi encontrado. Verifique os logs do Docker."
    fi
done
echo "Todos os containers de cache esperados estão rodando."

echo ""
echo "--- Obtendo Endereços IP dos Containers de Cache ---"
IP_CACHE_1=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-1 2>/dev/null)
IP_CACHE_2=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-2 2>/dev/null)
IP_CACHE_3=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' video-streaming-cache-3 2>/dev/null)

if [ -z "$IP_CACHE_1" ] || [ -z "$IP_CACHE_2" ] || [ -z "$IP_CACHE_3" ]; then
    error_exit "Não foi possível obter o IP de um ou mais caches. Verifique 'docker ps' e os nomes dos containers."
fi
echo "IPs obtidos: Cache1=$IP_CACHE_1, Cache2=$IP_CACHE_2, Cache3=$IP_CACHE_3"

HOST_CACHE_1="video-streaming-cache-1"
HOST_CACHE_2="video-streaming-cache-2"
HOST_CACHE_3="video-streaming-cache-3"
HOST_STEERING="steering-service"

ETC_HOSTS_FILE="/etc/hosts"
BACKUP_ETC_HOSTS_FILE="/etc/hosts.bak_content_steering_$(date +%Y%m%d_%H%M%S)"
TMP_HOSTS_FILE="${ETC_HOSTS_FILE}.tmp_content_steering_$(date +%s)"

echo ""
echo "--- Atualizando o Arquivo /etc/hosts ---"

echo "Fazendo backup de $ETC_HOSTS_FILE para $BACKUP_ETC_HOSTS_FILE..."
cp "$ETC_HOSTS_FILE" "$BACKUP_ETC_HOSTS_FILE" || error_exit "Falha ao criar o backup de $ETC_HOSTS_FILE."

echo "Removendo entradas antigas de content-steering (se existirem)..."
awk '
    BEGIN { in_cs_block=0 }
    /# Entradas adicionadas por content-steering starting_streaming\.sh/ { in_cs_block=1; next }
    /# Fim das entradas de content-steering/ { in_cs_block=0; next }
    !in_cs_block { print }
' "$BACKUP_ETC_HOSTS_FILE" > "$TMP_HOSTS_FILE"

if [ $? -ne 0 ]; then
    cp "$BACKUP_ETC_HOSTS_FILE" "$ETC_HOSTS_FILE" 
    error_exit "Falha ao processar $ETC_HOSTS_FILE com awk para remover entradas antigas."
fi

echo "Adicionando novas entradas..."
{
    echo "" 
    echo "$IP_CACHE_1    $HOST_CACHE_1"
    echo "$IP_CACHE_2    $HOST_CACHE_2"
    echo "$IP_CACHE_3    $HOST_CACHE_3"
    echo "127.0.0.1    $HOST_STEERING"
} >> "$TMP_HOSTS_FILE"

echo "Aplicando alterações em $ETC_HOSTS_FILE..."
mv "$TMP_HOSTS_FILE" "$ETC_HOSTS_FILE" || { cp "$BACKUP_ETC_HOSTS_FILE" "$ETC_HOSTS_FILE"; error_exit "Falha ao mover $TMP_HOSTS_FILE para $ETC_HOSTS_FILE. Backup restaurado."; }

echo ""
echo "--- Configuração Inicial Concluída com Sucesso ---"
