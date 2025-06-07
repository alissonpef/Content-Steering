import time
import random
import threading
import numpy as np
import logging
import math

logger = logging.getLogger("LatencyOracle")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING) 

# Calcula a distância Haversine em km entre dois pontos geográficos (latitude, longitude)
def calculate_haversine_distance(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    if None in [lat1, lon1, lat2, lon2]:
        return 0.0
    dLat, dLon, lat1_rad, lat2_rad = map(math.radians, [lat2 - lat1, lon2 - lon1, lat1, lat2])
    a = math.sin(dLat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dLon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

class DynamicLatencyOracle:
    # Simula dinamicamente a latência de rede para servidores, considerando distância do cliente,
    # latências base, ruído aleatório e eventos de modificação de latência
    DEFAULT_USE_DISTANCE_PENALTY: bool = True
    DEFAULT_MS_PER_KM_FACTOR: float = 0.0250
    DEFAULT_INITIAL_CLIENT_LAT: float = -23.0 
    DEFAULT_INITIAL_CLIENT_LON: float = -47.0 


    # Inicializa o oráculo de latência dinâmica
    def __init__(self, monitor, update_interval_seconds: int = 2):
        self.monitor = monitor
        self.server_latencies = {}
        self.server_base_latencies_config = {
            "video-streaming-cache-1": 30, 
            "video-streaming-cache-2": 25,
            "video-streaming-cache-3": 50 
        }
        self.server_geo_coords = {}
        self.client_latitude = DynamicLatencyOracle.DEFAULT_INITIAL_CLIENT_LAT
        self.client_longitude = DynamicLatencyOracle.DEFAULT_INITIAL_CLIENT_LON
        self.server_event_modifiers = {}
        self.update_interval_seconds = update_interval_seconds
        self.ms_per_km_factor = DynamicLatencyOracle.DEFAULT_MS_PER_KM_FACTOR
        self.use_distance_penalty = DynamicLatencyOracle.DEFAULT_USE_DISTANCE_PENALTY
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.noise_std_dev_factor = 0.15
        self.min_simulated_latency = 5
        self._update_server_geo_coordinates()

    # Obtém e armazena as coordenadas geográficas dos servidores a partir do monitor
    def _update_server_geo_coordinates(self):
        if self.monitor:
            coords = self.monitor.get_node_coordinates()
            with self.lock:
                self.server_geo_coords = coords

    # Atualiza a localização (latitude, longitude) do cliente
    def update_client_location(self, lat: float, lon: float):
        if lat is None or lon is None:
            return
        with self.lock:
            self.client_latitude = float(lat)
            self.client_longitude = float(lon)

    # Sincroniza a lista de servidores com o monitor e inicializa/remove seus estados de latência
    def _initialize_server_states(self):
        current_nodes_info = self.monitor.getNodes()
        if not current_nodes_info:
            return
        current_node_names = [info[0] for info in current_nodes_info if info and info[0]]
        self._update_server_geo_coordinates()

        with self.lock:
            for name in current_node_names:
                if name not in self.server_latencies:
                    initial_lat = self.server_base_latencies_config.get(name, random.uniform(10, 30))
                    self.server_latencies[name] = initial_lat
                    self.server_event_modifiers[name] = (1.0, 0) 
                    logger.info(f"Oráculo: Servidor {name} adicionado (lat inicial: {initial_lat:.2f}ms).")

            removed_servers = [name for name in list(self.server_latencies.keys()) if name not in current_node_names]
            for name in removed_servers:
                del self.server_latencies[name]
                if name in self.server_event_modifiers:
                    del self.server_event_modifiers[name]
            if removed_servers:
                logger.info(f"Oráculo: Servidores removidos: {', '.join(removed_servers)}")

    # Calcula e atualiza a latência simulada para cada servidor, aplicando penalidade de distância,
    # modificadores de evento e ruído
    def _update_latencies(self):
        self._initialize_server_states()

        with self.lock:
            current_time_server = time.time()
            can_calculate_distance = self.use_distance_penalty and \
                                     self.client_latitude is not None and \
                                     self.client_longitude is not None

            for server_name in list(self.server_latencies.keys()):
                base_latency_config = self.server_base_latencies_config.get(server_name, 30)
                distance_penalty = 0

                if can_calculate_distance:
                    server_coords = self.server_geo_coords.get(server_name)
                    if server_coords and server_coords.get('lat') is not None and server_coords.get('lon') is not None:
                        distance_km = calculate_haversine_distance(
                            self.client_latitude, self.client_longitude,
                            server_coords['lat'], server_coords['lon']
                        )
                        distance_penalty = distance_km * self.ms_per_km_factor

                effective_base_latency = base_latency_config + distance_penalty
                current_modifier_factor, current_expiry_time = self.server_event_modifiers.get(server_name, (1.0, 0))
                final_modifier_to_apply = current_modifier_factor

                if current_expiry_time != 0 and current_time_server >= current_expiry_time:
                    final_modifier_to_apply = 1.0
                    if self.server_event_modifiers.get(server_name) != (1.0,0):
                        logger.info(f"Oráculo: Modificador para {server_name} (fator={current_modifier_factor}, exp={current_expiry_time}) expirou em {current_time_server:.2f}. Resetando para (1.0, 0).")
                        self.server_event_modifiers[server_name] = (1.0, 0)

                noise = np.random.normal(loc=0, scale=max(1, effective_base_latency) * self.noise_std_dev_factor)
                simulated_latency_before_modifier = max(self.min_simulated_latency, effective_base_latency + noise)
                
                calculated_final_latency = simulated_latency_before_modifier * final_modifier_to_apply
                self.server_latencies[server_name] = calculated_final_latency
    def get_current_latency(self, server_name: str) -> float:
        with self.lock:
            if server_name not in self.server_latencies:
                self._initialize_server_states() 
            latency = self.server_latencies.get(server_name)
            if latency is None: 
                logger.warning(f"Oráculo: Latência não encontrada para {server_name} após inicialização. Retornando default.")
                return random.uniform(50, 150)
            return latency

    # Retorna um dicionário com as latências simuladas atuais de todos os servidores conhecidos
    def get_all_current_latencies(self) -> dict:
        with self.lock:
            if not self.server_latencies and self.monitor and self.monitor.getNodes():
                 self._initialize_server_states() 
            return dict(self.server_latencies) 

    # Aplica um fator multiplicativo à latência de um servidor por uma duração específica
    def apply_event_modifier(self, server_name: str, factor: float, duration_seconds: int):
        with self.lock:
            if server_name in self.server_latencies:
                expiry_timestamp = time.time() + duration_seconds if duration_seconds > 0 else 0 
                self.server_event_modifiers[server_name] = (factor, expiry_timestamp)
                logger.info(f"Oráculo: Evento em {server_name}. Fator: {factor:.2f}, Duração: {duration_seconds}s.")
            else:
                logger.warning(f"Oráculo: Tentativa de aplicar evento a servidor desconhecido '{server_name}'.")

    # Loop principal executado em uma thread para atualizar periodicamente as latências
    def run_update_loop(self):
        logger.info("Oráculo: Iniciando loop de atualização de latências.")
        try:
            while self.running:
                self._update_latencies()
                for _ in range(int(self.update_interval_seconds * 10)): 
                    if not self.running:
                        break
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Oráculo: Erro no loop de atualização: {e}", exc_info=True)
        finally:
            logger.info("Oráculo: Loop de atualização de latências encerrado.")

    # Inicia a thread de atualização periódica de latências
    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self._update_server_geo_coordinates() #
            self.thread = threading.Thread(target=self.run_update_loop, daemon=True)
            self.thread.start()
            logger.info("Oráculo: Thread de atualização iniciada.")

    # Para a thread de atualização de latências
    def stop(self):
        logger.info("Oráculo: Solicitando parada da thread de atualização.")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=self.update_interval_seconds + 1) 
        if self.thread and self.thread.is_alive():
            logger.warning("Oráculo: Thread de atualização não encerrou no tempo esperado.")
        else:
            logger.info("Oráculo: Thread de atualização parada.")
        self.thread = None