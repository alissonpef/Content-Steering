import time
import os
import csv
import json
import logging
import argparse

from flask import Flask, request, jsonify
from flask_cors import CORS

from dash_parser import DashParser
from monitor import ContainerMonitor
from selector import EpsilonGreedy, RandomSelector, NoSteeringSelector, UCB1Selector, OracleBestChoiceSelector
from dynamic_latency_oracle import DynamicLatencyOracle

STEERING_PORT = 30500
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "Graphics", "Logs")
CSV_HEADERS = [
    "timestamp_server", "sim_time_client", "client_lat", "client_lon",
    "server_used_for_latency", "experienced_latency_ms_CLIENT",
    "experienced_latency_ms_ORACLE", "experienced_latency_ms",
    "all_servers_oracle_latency_json", "steering_decision_main_server",
    "rl_strategy", "rl_counts_json", "rl_values_json",
]

selector_instance = None
selector_initialized = False
last_steering_main_server_decision = "N/A"
current_strategy_name = "N/A"
latency_oracle = None
active_log_filename = None

app_logger = logging.getLogger("SteeringApp")
oracle_logger = logging.getLogger("LatencyOracle")
monitor_logger = logging.getLogger("ContainerMonitor")

# Configura um logger com um manipulador de stream e formatador padrão
def _configure_logger(logger_instance, default_level=logging.WARNING):
    if not logger_instance.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
        logger_instance.setLevel(default_level)

_configure_logger(app_logger)
_configure_logger(oracle_logger)
_configure_logger(monitor_logger)

# Cria o diretório de logs e inicializa o arquivo CSV com cabeçalhos
def setup_csv_logging(filename: str):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file); writer.writerow(CSV_HEADERS)
        app_logger.info(f"Log CSV configurado: {filename}")
    except Exception as e:
        app_logger.critical(f"Erro na configuração do log CSV para {filename}: {e}", exc_info=True)

# Anexa uma linha de dados ao arquivo CSV especificado
def log_data_to_csv(data_dict: dict, filename: str):
    row = [data_dict.get(h) for h in CSV_HEADERS]
    try:
        with open(filename, mode="a", newline="") as file:
            csv.writer(file).writerow(row)
    except Exception as e:
        app_logger.error(f"Erro ao escrever no CSV {filename}: {e}", exc_info=True)

# Gera um nome de arquivo único para o log CSV, adicionando um contador se o nome base já existir
def get_unique_log_filename(base_name: str, suffix: str, directory: str = LOG_DIR) -> str:
    full_base = f"{base_name}{suffix}"
    cnt = 1
    while True:
        numbered_path = os.path.join(directory, f"{full_base}_{cnt}.csv")
        if not os.path.exists(numbered_path):
            return numbered_path
        cnt += 1

class Main:
    # Encapsula a aplicação Flask para o serviço de Content Steering
    def __init__(self, sel_inst, strategy_arg: str, log_file: str):
        global selector_instance, current_strategy_name, active_log_filename
        selector_instance, current_strategy_name, active_log_filename = sel_inst, strategy_arg, log_file
        self.app = Flask(__name__)
        CORS(self.app)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self._register_routes()

    # Inicializa ou re-inicializa o seletor de RL se necessário, obtendo nós do monitor
    # Retorna True se o seletor estiver pronto, False caso contrário
    def _initialize_selector_if_needed(self) -> bool:
        global selector_initialized, selector_instance
        if not selector_initialized or not selector_instance.nodes:
            nodes_info = monitor.getNodes()
            if nodes_info:
                node_names = [info[0] for info in nodes_info if info and info[0]]
                if node_names:
                    selector_instance.initialize(node_names)
                    selector_initialized = True
                    return True
                else:
                    app_logger.warning("Nenhum nome de nó obtido do monitor para inicializar o seletor.")
            else:
                app_logger.warning("Nenhuma informação de nó obtida do monitor para inicializar o seletor.")
            return False
        return True

    # Registra as rotas da aplicação Flask
    def _register_routes(self):
        # Rota principal para o cliente solicitar decisões de steering
        # Retorna uma lista ordenada de servidores baseada na estratégia de RL
        @self.app.route("/<path:name>", methods=["GET", "POST"])
        def do_remote_steering(name: str):
            global last_steering_main_server_decision
            if not self._initialize_selector_if_needed():
                return jsonify({"error": "Serviço não pronto (falha na inicialização do seletor)."}), 503

            ordered_nodes = selector_instance.select_arm()
            last_steering_main_server_decision = ordered_nodes[0] if ordered_nodes else "N/A_NO_NODES"
            if not ordered_nodes:
                app_logger.error("Nenhum servidor selecionado pelo RL.")
                return jsonify({"error": "Nenhum servidor selecionável"}), 503

            nodes_p = [(n, n) for n in ordered_nodes]
            uri = f"https://steering-service:{STEERING_PORT}"
            target = request.args.get("_DASH_pathway", "", str)
            resp = dash_parser.build(target=target, nodes=nodes_p, uri=uri, request=request)
            return jsonify(resp), 200

        # Rota para o cliente enviar feedback (coordenadas, latência experimentada)
        # Usado para atualizar o oráculo de latência, logar dados e atualizar o modelo de RL
        @self.app.route("/coords", methods=["POST"])
        def coords_update():
            global last_steering_main_server_decision, selector_instance, latency_oracle, active_log_filename
            if not request.json:
                return "Requisição inválida: Corpo JSON ausente", 400
            data = request.json
            s_t, lat, lon, rt_c, srv_u = (data.get(k) for k in ["time", "lat", "long", "rt", "server_used"])

            if latency_oracle and lat is not None and lon is not None:
                latency_oracle.update_client_location(lat, lon)

            all_oracle_lats = latency_oracle.get_all_current_latencies() if latency_oracle else {}
            all_srv_json = json.dumps(all_oracle_lats)
            log_base = {"timestamp_server": time.time(), "sim_time_client": s_t, "client_lat": lat, "client_lon": lon,
                          "all_servers_oracle_latency_json": all_srv_json,
                          "steering_decision_main_server": last_steering_main_server_decision,
                          "rl_strategy": current_strategy_name,
                          "rl_counts_json": json.dumps(getattr(selector_instance, "counts", {})),
                          "rl_values_json": json.dumps(getattr(selector_instance, "values", {}))}

            if srv_u and rt_c is not None and latency_oracle:
                oracle_lat = all_oracle_lats.get(srv_u, latency_oracle.get_current_latency(srv_u))
                log_entry = {**log_base, "server_used_for_latency": srv_u,
                             "experienced_latency_ms_CLIENT": rt_c,
                             "experienced_latency_ms_ORACLE": oracle_lat, "experienced_latency_ms": oracle_lat}
                log_data_to_csv(log_entry, filename=active_log_filename)

                if not self._initialize_selector_if_needed():
                    return "Serviço não pronto (seletor em /coords)", 503
                if hasattr(selector_instance, "update"):
                    if srv_u not in selector_instance.nodes:
                        app_logger.warning(f"Servidor {srv_u} não está no seletor. Re-inicializando.")
                        nodes = [n for n, _ in monitor.getNodes() if n]
                        if srv_u in nodes:
                            selector_instance.initialize(nodes)
                            if srv_u in selector_instance.nodes:
                                selector_instance.update(srv_u, float(oracle_lat))
                            else:
                                return "Servidor não reconhecido após re-inicialização", 400
                        else:
                            return "Servidor não está na lista do monitor", 400
                    else:
                        selector_instance.update(srv_u, float(oracle_lat))
                    return "RL atualizado e logado", 200
                return "Dados logados (sem atualização de RL)", 200
            elif lat is not None and lon is not None:
                log_entry = {**log_base, "server_used_for_latency": srv_u, "experienced_latency_ms_CLIENT": rt_c,
                             "experienced_latency_ms_ORACLE": None, "experienced_latency_ms": None}
                log_data_to_csv(log_entry, filename=active_log_filename)
                return "Dados de localização logados", 200
            else:
                return "Dados inválidos: Localização ou informação crítica ausente", 400

        # Rota para simular um evento de latência (e.g., congestionamento) em um servidor específico
        @self.app.route("/latency_event", methods=["POST"])
        def latency_event_route():
            global latency_oracle
            if not request.json:
                return "Requisição inválida: Corpo JSON ausente", 400
            data = request.json
            server = data.get("server_name")
            factor = data.get("factor", 2.0)
            duration = data.get("duration_seconds", 10)
            app_logger.info(f"Evento de Latência: Servidor={server}, Fator={factor}, Duração={duration}s")
            if not server:
                return "Nome do servidor (server_name) ausente", 400
            if not latency_oracle:
                return "Oráculo de latência não está pronto", 503
            try:
                latency_oracle.apply_event_modifier(server, float(factor), int(duration))
                return f"Evento de latência para {server} aplicado", 200
            except ValueError:
                return "Formato inválido para fator ou duração", 400
            except Exception as e:
                app_logger.error(f"Erro em /latency_event: {e}", exc_info=True)
                return "Erro ao aplicar evento", 500

    # Inicia a aplicação Flask, tentando HTTPS e recorrendo a HTTP em caso de falha
    def run(self):
        global current_strategy_name
        s_dir = os.path.dirname(os.path.abspath(__file__))
        certs_dir = os.path.join(s_dir, "..", "certs")
        cert = os.path.join(certs_dir, "steering-service.pem")
        key = os.path.join(certs_dir, "steering-service-key.pem")
        try:
            if not (os.path.exists(cert) and os.path.exists(key)):
                raise FileNotFoundError("Certificado/chave SSL não encontrado.")
            app_logger.info(f"Serviço (Estratégia: {current_strategy_name}) em https://0.0.0.0:{STEERING_PORT}")
            self.app.run(host="0.0.0.0", port=STEERING_PORT, debug=False, ssl_context=(cert, key))
        except Exception as e:
            app_logger.critical(f"Falha ao iniciar SSL: {e}", exc_info=True)
            app_logger.info(f"Fallback para HTTP. Serviço (Estratégia: {current_strategy_name}) em http://0.0.0.0:{STEERING_PORT}")
            self.app.run(host="0.0.0.0", port=STEERING_PORT, debug=False)

dash_parser = DashParser()
monitor = ContainerMonitor()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serviço de Content Steering com RL.")
    parser.add_argument("--strategy", type=str, default="epsilon_greedy",
                        choices=["epsilon_greedy", "no_steering", "random", "ucb1", "oracle_best_choice"],
                        help="Estratégia de steering.")
    parser.add_argument("--log_suffix", type=str, default="", help="Sufixo opcional para o nome do arquivo de log CSV.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Habilita logging DEBUG.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    app_logger.setLevel(log_level)
    oracle_logger.setLevel(log_level)
    monitor_logger.setLevel(log_level)
    if args.verbose:
        app_logger.info("Logging verboso (DEBUG) habilitado.")

    current_strategy_name = args.strategy
    app_logger.info(f"Estratégia selecionada: {current_strategy_name}")
    log_base = f"log_{current_strategy_name}"
    active_log_filename = get_unique_log_filename(log_base, args.log_suffix, directory=LOG_DIR)
    app_logger.info(f"Arquivo de log ativo: {active_log_filename}")

    app_logger.info("Iniciando monitor de contêineres...")
    monitor.start_collecting()
    app_logger.info("Inicializando oráculo de latência...")
    latency_oracle = DynamicLatencyOracle(monitor, update_interval_seconds=1)
    latency_oracle.start()

    if args.strategy == "epsilon_greedy":
        selector_instance = EpsilonGreedy(epsilon=0.1, counts={}, values={}, monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "no_steering":
        selector_instance = NoSteeringSelector(monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "random":
        selector_instance = RandomSelector(monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "ucb1":
        selector_instance = UCB1Selector(monitor=monitor, latency_oracle=latency_oracle)
    elif args.strategy == "oracle_best_choice":
        selector_instance = OracleBestChoiceSelector(monitor=monitor, latency_oracle=latency_oracle)
    else:
        app_logger.critical(f"Estratégia desconhecida: {args.strategy}. Usando EpsilonGreedy como padrão.")
        current_strategy_name = "epsilon_greedy"
        log_base = "log_epsilon_greedy"
        active_log_filename = get_unique_log_filename(log_base, args.log_suffix, directory=LOG_DIR) 
        selector_instance = EpsilonGreedy(epsilon=0.1, counts={}, values={}, monitor=monitor, latency_oracle=latency_oracle)

    setup_csv_logging(filename=active_log_filename)
    app_logger.info("Criando instância da aplicação Flask...")
    main_app = Main(selector_instance, current_strategy_name, active_log_filename)

    try:
        main_app.run()
    except KeyboardInterrupt:
        app_logger.info("Serviço encerrando (Ctrl+C).")
    except Exception as e:
        app_logger.critical(f"Erro em tempo de execução: {e}", exc_info=True)
    finally:
        app_logger.info("Procedimentos de encerramento...")
        if latency_oracle:
            app_logger.info("Parando oráculo de latência...")
            latency_oracle.stop()
        if monitor:
            app_logger.info("Parando monitor de contêineres...")
            if hasattr(monitor, 'stop_collecting'):
                monitor.stop_collecting()
            elif hasattr(monitor, 'running'):
                monitor.running = False
        app_logger.info(f"Serviço (Estratégia: {current_strategy_name}) parado.")