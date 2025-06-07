import time
import os
import csv
import json
import logging
import argparse

from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

from dash_parser import DashParser
from monitor import ContainerMonitor
from selector import EpsilonGreedy, RandomSelector, NoSteeringSelector, UCB1Selector

STEERING_PORT = 30500

selector_instance = None
selector_initialized = False
last_steering_main_server_decision = "N/A"
current_strategy_name = "N/A" 

PROJECT_ROOT_DIR = os.path.dirname( 
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
LOG_DIR = os.path.join(PROJECT_ROOT_DIR, "Graphics", "Logs") 
CSV_HEADERS = [
    "timestamp_server",
    "sim_time_client",
    "client_lat",
    "client_lon",
    "server_used_for_latency",
    "experienced_latency_ms",
    "steering_decision_main_server",
    "rl_strategy",
    "rl_counts_json",
    "rl_values_json",
]

active_log_filename = None 

def setup_csv_logging(filename):
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(filename, mode="w", newline="") as file: 
            writer = csv.writer(file)
            writer.writerow(CSV_HEADERS)
        print(f"[LOGGING_APP] CSV log an d headers set up for {filename}")
    except Exception as e:
        print(
            f"[LOGGING_APP] CRITICAL ERROR setting up CSV logging for {filename}: {e}"
        )


def log_data_to_csv(data_dict, filename): 
    row_to_write = [data_dict.get(header) for header in CSV_HEADERS]
    try:
        file_exists = os.path.exists(filename)
        if not file_exists or (file_exists and os.path.getsize(filename) == 0) :
             with open(filename, mode="a", newline="") as file: 
                writer = csv.writer(file)
                if not file_exists or os.path.getsize(filename) == 0:
                    writer.writerow(CSV_HEADERS)
                writer.writerow(row_to_write)
        else:
            with open(filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row_to_write)

    except Exception as e:
        print(f"[LOGGING_APP] Error writing to CSV {filename}: {e}")


class Main:
    def __init__(self, selected_selector_instance, strategy_name_arg, log_filename_to_use):
        global selector_instance, current_strategy_name, active_log_filename
        selector_instance = selected_selector_instance
        current_strategy_name = strategy_name_arg
        active_log_filename = log_filename_to_use

        self.app = Flask(__name__)
        CORS(self.app)

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        @self.app.route("/<path:name>", methods=["GET", "POST"])
        def do_remote_steering(name):
            global selector_initialized, last_steering_main_server_decision, selector_instance
            tar = request.args.get("_DASH_pathway", default="", type=str)

            if not selector_initialized:
                available_nodes_info = monitor.getNodes()
                if available_nodes_info:
                    node_names = [
                        info[0] for info in available_nodes_info if info and info[0]
                    ]
                    if node_names:
                        selector_instance.initialize(node_names)
                        selector_initialized = True
                    else:
                        return (
                            jsonify({"error": "Cache server names not resolved"}),
                            503,
                        )
                else:
                    return (
                        jsonify(
                            {"error": "No cache servers available or monitor not ready"}
                        ),
                        503,
                    )

            if not selector_instance.nodes:
                available_nodes_info = monitor.getNodes()
                if available_nodes_info:
                    node_names = [
                        info[0] for info in available_nodes_info if info and info[0]
                    ]
                    if node_names:
                        selector_instance.initialize(node_names)
                    else:
                        return (
                            jsonify(
                                {
                                    "error": "Cache server names still not resolved on re-init"
                                }
                            ),
                            503,
                        )
                else:
                    return (
                        jsonify(
                            {"error": "Cache servers not ready after re-init attempt"}
                        ),
                        503,
                    )

            ordered_node_names = selector_instance.select_arm()

            if ordered_node_names:
                last_steering_main_server_decision = ordered_node_names[0]
            else:
                last_steering_main_server_decision = "N/A_NO_NODES_FROM_SELECTION"

            nodes_for_parser = []
            for node_name in ordered_node_names:
                nodes_for_parser.append((node_name, node_name))

            if not nodes_for_parser and ordered_node_names:
                return (
                    jsonify(
                        {
                            "error": "RL selection resulted in no valid cache servers for parser"
                        }
                    ),
                    503,
                )
            elif not ordered_node_names:
                return (
                    jsonify({"error": "No cache servers could be selected by RL"}),
                    503,
                )

            reload_base_uri = f"https://steering-service:{STEERING_PORT}"
            data = dash_parser.build(
                target=tar, nodes=nodes_for_parser, uri=reload_base_uri, request=request
            )
            return jsonify(data), 200

        @self.app.route("/coords", methods=["POST"])
        def coords_update():
            global selector_initialized, last_steering_main_server_decision, selector_instance, current_strategy_name, active_log_filename
            if not request.json:
                return "Invalid request: No JSON body", 400

            data = request.json
            sim_time_client = data.get("time")
            client_lat = data.get("lat")
            client_lon = data.get("long")
            experienced_latency_ms = data.get("rt")
            server_used_for_latency = data.get("server_used")

            log_entry_base = {
                "timestamp_server": time.time(),
                "sim_time_client": sim_time_client,
                "client_lat": client_lat,
                "client_lon": client_lon,
                "steering_decision_main_server": last_steering_main_server_decision,
                "rl_strategy": current_strategy_name,
                "rl_counts_json": json.dumps(getattr(selector_instance, "counts", {})),
                "rl_values_json": json.dumps(getattr(selector_instance, "values", {})),
            }

            if (
                experienced_latency_ms is not None
                and server_used_for_latency is not None
            ):
                log_entry = {
                    **log_entry_base,
                    "server_used_for_latency": server_used_for_latency,
                    "experienced_latency_ms": experienced_latency_ms,
                }
                log_data_to_csv(log_entry, filename=active_log_filename)

                if not selector_initialized or not selector_instance.nodes:
                    available_nodes_info = monitor.getNodes()
                    if available_nodes_info:
                        node_names = [
                            info[0] for info in available_nodes_info if info and info[0]
                        ]
                        if node_names:
                            selector_instance.initialize(node_names)
                            selector_initialized = True
                        else:
                            return (
                                "Steering service not ready (selector node names unresolved on /coords)",
                                503,
                            )
                    else:
                        return (
                            "Steering service not ready (selector not initialized on /coords)",
                            503,
                        )

                if hasattr(selector_instance, "update"):
                    if server_used_for_latency not in selector_instance.nodes:
                        current_monitor_node_names = [
                            name for name, _ in monitor.getNodes() if name
                        ]
                        if server_used_for_latency in current_monitor_node_names:
                            selector_instance.initialize(current_monitor_node_names)
                            if server_used_for_latency in selector_instance.nodes:
                                selector_instance.update(
                                    server_used_for_latency,
                                    float(experienced_latency_ms),
                                )
                            else:
                                return (
                                    "Reported server still not recognized after re-init",
                                    400,
                                )
                        else:
                            return "Reported server not in monitor list", 400
                    else:
                        selector_instance.update(
                            server_used_for_latency, float(experienced_latency_ms)
                        )
                    return "RL/Selector updated and data logged", 200
                else:
                    return "Data logged (selector does not support update)", 200
            else:
                if client_lat is not None and client_lon is not None:
                    log_entry = {
                        **log_entry_base,
                        "server_used_for_latency": None,
                        "experienced_latency_ms": None,
                    }
                    log_data_to_csv(log_entry, filename=active_log_filename)
                    return (
                        "Location data logged (no latency info for RL/selector update)",
                        200,
                    )
                else:
                    return "Invalid data: Missing location or latency info", 400

    def run(self):
        global current_strategy_name 
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_base_dir_for_certs = os.path.dirname(current_script_dir)
        ssl_context_cert = os.path.join(
            project_base_dir_for_certs, "certs", "steering-service.pem"
        )
        ssl_context_key = os.path.join(
            project_base_dir_for_certs, "certs", "steering-service-key.pem"
        )

        try:
            if not os.path.exists(ssl_context_cert) or not os.path.exists(
                ssl_context_key
            ):
                raise FileNotFoundError(
                    f"SSL certificate or key not found at {ssl_context_cert} or {ssl_context_key}"
                )
            ssl_context = (ssl_context_cert, ssl_context_key)
            print(
                f"Steering service (Strategy: {current_strategy_name}) running on https://0.0.0.0:{STEERING_PORT}"
            )
            self.app.run(
                host="0.0.0.0", port=STEERING_PORT, debug=False, ssl_context=ssl_context
            )
        except Exception as e:
            print(f"[STEERING_APP] ERROR starting Flask with SSL: {e}")
            print("[STEERING_APP] Falling back to HTTP.")
            print(
                f"Steering service (Strategy: {current_strategy_name}) running on http://0.0.0.0:{STEERING_PORT}"
            )
            self.app.run(host="0.0.0.0", port=STEERING_PORT, debug=False)


dash_parser = DashParser()
monitor = ContainerMonitor()

def get_unique_log_filename(base_filename_without_ext, suffix_arg, directory=LOG_DIR):
    base_with_suffix = f"{base_filename_without_ext}{suffix_arg}"
    counter = 1
    while True:
        numbered_filename = f"{base_with_suffix}_{counter}.csv"
        candidate_filename = os.path.join(directory, numbered_filename)
        if not os.path.exists(candidate_filename):
            return candidate_filename
        counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Content Steering Service with selectable strategies."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="epsilon_greedy",
        choices=["epsilon_greedy", "no_steering", "random", "ucb1"],
        help="Steering strategy to use (epsilon_greedy, no_steering, random, ucb1).",
    )
    parser.add_argument(
        "--log_suffix",
        type=str,
        default="",
        help="Suffix to append to the log file name (e.g., _runA, _mobile_stress).",
    )
    args = parser.parse_args()

    current_strategy_name = args.strategy 
    print(f"[STEERING_APP] Using strategy: {current_strategy_name}")

    log_base_name = f"log_{current_strategy_name}"
    active_log_filename = get_unique_log_filename(log_base_name, args.log_suffix, directory=LOG_DIR)
    
    print(f"[LOGGING_APP] Active log file: {active_log_filename}")

    if args.strategy == "epsilon_greedy":
        current_selector_instance = EpsilonGreedy(
            epsilon=0.3, counts={}, values={}, monitor=monitor
        )
    elif args.strategy == "no_steering":
        current_selector_instance = NoSteeringSelector(monitor=monitor)
    elif args.strategy == "random":
        current_selector_instance = RandomSelector(monitor=monitor)
    elif args.strategy == "ucb1":
        current_selector_instance = UCB1Selector(monitor=monitor)
    else:
        print(
            f"CRITICAL: Unknown strategy '{args.strategy}'. Defaulting to EpsilonGreedy."
        )
        current_strategy_name = "epsilon_greedy" 
        log_base_name = f"log_{current_strategy_name}" 
        active_log_filename = get_unique_log_filename(log_base_name, args.log_suffix, directory=LOG_DIR) 
        current_selector_instance = EpsilonGreedy(
            epsilon=0.3, counts={}, values={}, monitor=monitor
        )

    setup_csv_logging(filename=active_log_filename) 
    monitor.start_collecting()

    main_app_instance = Main(
        current_selector_instance, current_strategy_name, active_log_filename 
    )

    try:
        main_app_instance.run()
    except KeyboardInterrupt:
        print(
            f"\n[STEERING_APP] Steering service (Strategy: {current_strategy_name}) shutting down (Ctrl+C)."
        )
    except Exception as e:
        print(
            f"[STEERING_APP] An unexpected error occurred during runtime (Strategy: {current_strategy_name}): {e}"
        )
    finally:
        print(f"[STEERING_APP] Service (Strategy: {current_strategy_name}) stopped.")