import time
import os
import csv
import json
import logging 

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

from dash_parser import DashParser
from monitor import ContainerMonitor
from selector import EpsilonGreedy

STEERING_PORT = 30500

dash_parser = DashParser()
monitor = ContainerMonitor()

selector = EpsilonGreedy(
    epsilon=0.3,
    counts={},
    values={},
    monitor=monitor
)
selector_initialized = False
last_steering_main_server_decision = "N/A"

PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "Files", "Data")
CSV_FILENAME = os.path.join(LOG_DIR, "simulation_log.csv")
CSV_HEADERS = [
    "timestamp_server", "sim_time_client", "client_lat", "client_lon",
    "server_used_for_latency", "experienced_latency_ms",
    "steering_decision_main_server", "rl_counts_json", "rl_values_json"
]

def setup_csv_logging():
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_exists = os.path.exists(CSV_FILENAME)
        is_empty = False
        if file_exists:
            is_empty = os.path.getsize(CSV_FILENAME) == 0
        if not file_exists or is_empty:
            with open(CSV_FILENAME, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(CSV_HEADERS)
            print(f"[LOGGING_APP] Initialized CSV log: {CSV_FILENAME}")
    except Exception as e:
        print(f"[LOGGING_APP] CRITICAL ERROR setting up CSV logging: {e}")

def log_data_to_csv(data_dict):
    row_to_write = [data_dict.get(header) for header in CSV_HEADERS]
    try:
        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row_to_write)
    except Exception as e:
        print(f"[LOGGING_APP] Error writing to CSV {CSV_FILENAME}: {e}")

class Main:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)

        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR) 

        @self.app.route("/<path:name>", methods=["GET", "POST"])
        def do_remote_steering(name):
            global selector_initialized, last_steering_main_server_decision
            tar = request.args.get("_DASH_pathway", default="", type=str)

            if not selector_initialized:
                available_nodes_info = monitor.getNodes()
                if available_nodes_info:
                    node_names = [info[0] for info in available_nodes_info if info and info[0]]
                    if node_names:
                        selector.initialize(node_names)
                        selector_initialized = True
                    else: return jsonify({"error": "Cache server names not resolved"}), 503
                else: return jsonify({"error": "No cache servers available or monitor not ready"}), 503
            
            if not selector.nodes:
                 available_nodes_info = monitor.getNodes()
                 if available_nodes_info:
                    node_names = [info[0] for info in available_nodes_info if info and info[0]]
                    if node_names: selector.initialize(node_names)
                    else: return jsonify({"error": "Cache server names still not resolved on re-init"}), 503
                 else: return jsonify({"error": "Cache servers not ready after re-init attempt"}), 503

            ordered_node_names = selector.select_arm()
            
            if ordered_node_names:
                last_steering_main_server_decision = ordered_node_names[0]
            else:
                last_steering_main_server_decision = "N/A_NO_NODES_FROM_RL"

            nodes_for_parser = []
            for node_name in ordered_node_names:
                nodes_for_parser.append((node_name, node_name)) 

            if not nodes_for_parser and ordered_node_names:
                 return jsonify({"error": "RL selection resulted in no valid cache servers for parser"}), 503
            elif not ordered_node_names:
                 return jsonify({"error": "No cache servers could be selected by RL"}), 503

            reload_base_uri = f"https://steering-service:{STEERING_PORT}"
            data = dash_parser.build(
                target=tar, nodes=nodes_for_parser, uri=reload_base_uri, request=request
            )
            return jsonify(data), 200

        @self.app.route("/coords", methods=["POST"])
        def coords_update():
            global selector_initialized, last_steering_main_server_decision
            if not request.json:
                return "Invalid request: No JSON body", 400

            data = request.json
            sim_time_client = data.get("time")
            client_lat = data.get("lat")
            client_lon = data.get("long") 
            experienced_latency_ms = data.get("rt")
            server_used_for_latency = data.get("server_used")

            if experienced_latency_ms is not None and server_used_for_latency is not None:
                log_entry = {
                    "timestamp_server": time.time(),
                    "sim_time_client": sim_time_client,
                    "client_lat": client_lat,
                    "client_lon": client_lon,
                    "server_used_for_latency": server_used_for_latency,
                    "experienced_latency_ms": experienced_latency_ms,
                    "steering_decision_main_server": last_steering_main_server_decision,
                    "rl_counts_json": json.dumps(selector.counts if selector else {}), 
                    "rl_values_json": json.dumps(selector.values if selector else {})
                }
                log_data_to_csv(log_entry)

                if not selector_initialized or not selector.nodes:
                    available_nodes_info = monitor.getNodes()
                    if available_nodes_info:
                        node_names = [info[0] for info in available_nodes_info if info and info[0]]
                        if node_names:
                            selector.initialize(node_names)
                            selector_initialized = True
                        else: return "Steering service not ready (selector node names unresolved on /coords)", 503
                    else: return "Steering service not ready (selector not initialized on /coords)", 503
                
                if server_used_for_latency not in selector.nodes:
                    current_monitor_node_names = [name for name, _ in monitor.getNodes() if name]
                    if server_used_for_latency in current_monitor_node_names:
                        selector.initialize(current_monitor_node_names) 
                        if server_used_for_latency in selector.nodes:
                             selector.update(server_used_for_latency, float(experienced_latency_ms))
                        else: return "Reported server still not recognized after re-init", 400
                    else: return "Reported server not in monitor list", 400 
                else:
                    selector.update(server_used_for_latency, float(experienced_latency_ms))
                return "RL updated and data logged", 200
            else:
                if client_lat is not None and client_lon is not None:
                    log_entry = {
                        "timestamp_server": time.time(),
                        "sim_time_client": sim_time_client,
                        "client_lat": client_lat,
                        "client_lon": client_lon,
                        "server_used_for_latency": None,
                        "experienced_latency_ms": None,
                        "steering_decision_main_server": last_steering_main_server_decision,
                        "rl_counts_json": json.dumps(selector.counts if selector else {}), 
                        "rl_values_json": json.dumps(selector.values if selector else {})
                    }
                    log_data_to_csv(log_entry)
                    return "Location data logged (no latency info for RL update)", 200
                else:
                    return "Invalid data: Missing location or latency info", 400

    def run(self):
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_base_dir_for_certs = os.path.dirname(current_script_dir) 
        ssl_context_cert = os.path.join(project_base_dir_for_certs, "certs", "steering-service.pem")
        ssl_context_key = os.path.join(project_base_dir_for_certs, "certs", "steering-service-key.pem")
        
        try:
            if not os.path.exists(ssl_context_cert) or not os.path.exists(ssl_context_key):
                raise FileNotFoundError(f"SSL certificate or key not found.")
            ssl_context = (ssl_context_cert, ssl_context_key)
            print(f"Steering service running on https://0.0.0.0:{STEERING_PORT}")
            self.app.run(host='0.0.0.0', port=STEERING_PORT, debug=False, ssl_context=ssl_context)
        except Exception as e:
            print(f"[STEERING_APP] ERROR starting Flask with SSL: {e}")
            print("[STEERING_APP] Falling back to HTTP.")
            print(f"Steering service running on http://0.0.0.0:{STEERING_PORT}")
            self.app.run(host='0.0.0.0', port=STEERING_PORT, debug=False)

if __name__ == "__main__":
    setup_csv_logging()
    monitor.start_collecting()
    
    main_app_instance = Main() 
    
    try:
        main_app_instance.run()
    except KeyboardInterrupt:
        print("\n[STEERING_APP] Steering service shutting down (Ctrl+C).")
    except Exception as e:
        print(f"[STEERING_APP] An unexpected error occurred during runtime: {e}")
    finally:
        print("[STEERING_APP] Service stopped.")