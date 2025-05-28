import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM_DATA_DIR = os.path.join(BASE_DIR, "Files", "Data")
DEFAULT_IMG_DIR = os.path.join(BASE_DIR, "Files", "Img")

CACHE_SERVER_LABELS = {
    "video-streaming-cache-1": "Cache 1 (BR)",
    "video-streaming-cache-2": "Cache 2 (CL)",
    "video-streaming-cache-3": "Cache 3 (CO)",
    "N/A_NO_NODES_FROM_SELECTION": "N/A - No Selection",
    "N/A_NO_NODES_FROM_RL": "N/A - No RL Nodes",
    "N/A": "N/A"
}

CACHE_COLORS = {
    "video-streaming-cache-1": "green",
    "video-streaming-cache-2": "darkorange",
    "video-streaming-cache-3": "dodgerblue",
}

def parse_json_column(series, key_prefix=""):
    parsed_data = []
    all_keys = set()
    temp_parsed_dicts = []
    for item in series:
        try:
            if pd.isna(item):
                temp_parsed_dicts.append({})
            else:
                data_dict = json.loads(item) if isinstance(item, str) else item
                if isinstance(data_dict, dict):
                    all_keys.update(data_dict.keys())
                    temp_parsed_dicts.append(data_dict)
                else:
                    temp_parsed_dicts.append({})
        except (json.JSONDecodeError, TypeError):
            temp_parsed_dicts.append({})
    for data_dict in temp_parsed_dicts:
        row = {f"{key_prefix}{k}": data_dict.get(k) for k in all_keys}
        parsed_data.append(row)
    return pd.DataFrame(parsed_data, index=series.index)

def generate_plots(csv_file_path):
    if not os.path.exists(csv_file_path):
        return

    csv_filename = os.path.basename(csv_file_path)
    simulation_name = os.path.splitext(csv_filename)[0]
    current_img_dir = os.path.join(DEFAULT_IMG_DIR, simulation_name)
    os.makedirs(current_img_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        return

    if df.empty:
        return

    df.sort_values(by="sim_time_client", inplace=True)
    df.reset_index(drop=True, inplace=True)

    strategy_name_from_df = df['rl_strategy'].iloc[0] if 'rl_strategy' in df.columns and not df.empty else "N/A"

    # Gráfico 1 - Latência Geral (Linear)
    plt.figure(figsize=(14, 7))
    if not df.empty and 'experienced_latency_ms' in df.columns:
        df_latency_points = df.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_latency_points.empty:
            plt.plot(df_latency_points['sim_time_client'], df_latency_points['experienced_latency_ms'],
                    marker='.', linestyle='', markersize=5, alpha=0.6,
                    label='Experienced Latency (per fragment)')

    window_size = 10
    if 'experienced_latency_ms' in df.columns and len(df['experienced_latency_ms'].dropna()) >= window_size:
        if 'latency_ma_calculated' not in df.columns or df['latency_ma_calculated'].isnull().all():
             df['latency_ma_calculated'] = df['experienced_latency_ms'].rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(df['sim_time_client'], df['latency_ma_calculated'],
                linestyle='--', color='red', linewidth=2,
                label=f'Moving Average ({window_size} points)')

    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Experienced Latency (ms)")
    plt.title(f"Client's Experienced Latency Over Time - Linear Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plot_path = os.path.join(current_img_dir, "1_latency_timeline_linear.png")
    plt.savefig(plot_path)
    plt.close()

    # Gráfico 2 - Latência Geral (Log)
    plt.figure(figsize=(14, 7))
    if not df.empty and 'experienced_latency_ms' in df.columns:
        df_latency_points = df.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_latency_points.empty:
            plt.plot(df_latency_points['sim_time_client'], df_latency_points['experienced_latency_ms'],
                    marker='.', linestyle='', markersize=5, alpha=0.6,
                    label='Experienced Latency (per fragment)')
    if 'latency_ma_calculated' in df.columns and not df['latency_ma_calculated'].isnull().all():
        plt.plot(df['sim_time_client'], df['latency_ma_calculated'],
                linestyle='--', color='red', linewidth=2,
                label=f'Moving Average ({window_size} points)')

    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Experienced Latency (ms) - Log Scale")
    plt.yscale('log')
    plt.title(f"Client's Experienced Latency Over Time - Log Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
    plt.legend(loc='best')
    plt.grid(True, which="both", linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(current_img_dir, "2_latency_timeline_logscale.png")
    plt.savefig(plot_path)
    plt.close()

    # Gráfico 3 - Latência por Servidor (Linear)
    plt.figure(figsize=(14, 7))
    unique_servers_used = df['server_used_for_latency'].dropna().unique()
    if len(unique_servers_used) > 0:
        for server_name in sorted(list(unique_servers_used), key=lambda x: CACHE_SERVER_LABELS.get(x, x)):
            server_df = df[df['server_used_for_latency'] == server_name]
            if not server_df.empty:
                color_to_use = CACHE_COLORS.get(server_name, 'black')
                plt.plot(server_df['sim_time_client'], server_df['experienced_latency_ms'],
                         marker='o', linestyle='-', markersize=4, alpha=0.8,
                         label=f"Latency from {CACHE_SERVER_LABELS.get(server_name, server_name)}", color=color_to_use)
        plt.legend(loc='best')
    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Experienced Latency (ms)")
    plt.title(f"Experienced Latency from Each Server (When Used) - Linear Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plot_path = os.path.join(current_img_dir, "3_latency_per_server_timeline_linear.png")
    plt.savefig(plot_path)
    plt.close()

    # Gráfico 4 - Latência por Servidor (Log)
    plt.figure(figsize=(14, 7))
    if len(unique_servers_used) > 0:
        for server_name in sorted(list(unique_servers_used), key=lambda x: CACHE_SERVER_LABELS.get(x, x)):
            server_df = df[df['server_used_for_latency'] == server_name]
            if not server_df.empty:
                color_to_use = CACHE_COLORS.get(server_name, 'black')
                plt.plot(server_df['sim_time_client'], server_df['experienced_latency_ms'],
                         marker='o', linestyle='-', markersize=4, alpha=0.8,
                         label=f"Latency from {CACHE_SERVER_LABELS.get(server_name, server_name)}", color=color_to_use)
        plt.legend(loc='best')
    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Experienced Latency (ms) - Log Scale")
    plt.yscale('log')
    plt.title(f"Experienced Latency from Each Server (When Used) - Log Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
    plt.grid(True, which="both", linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(current_img_dir, "4_latency_per_server_timeline_logscale.png")
    plt.savefig(plot_path)
    plt.close()

    # Gráfico 5 - Decisão de Steering
    plt.figure(figsize=(14, 7))
    df_steering_decisions = df.dropna(subset=['steering_decision_main_server', 'sim_time_client'])
    if not df_steering_decisions.empty:
        df_sd_unique = df_steering_decisions.drop_duplicates(subset=['sim_time_client'], keep='first').copy()
        actual_cache_server_names = ["video-streaming-cache-1", "video-streaming-cache-2", "video-streaming-cache-3"]
        relevant_servers_for_plot = sorted(actual_cache_server_names)
        if relevant_servers_for_plot:
            server_to_int = {server: i for i, server in enumerate(relevant_servers_for_plot)}
            int_to_server_label_map = {i: CACHE_SERVER_LABELS.get(server, server) for server, i in server_to_int.items()}
            df_sd_unique['decision_int'] = df_sd_unique['steering_decision_main_server'].map(server_to_int)
            df_to_plot = df_sd_unique.dropna(subset=['decision_int'])
            if not df_to_plot.empty:
                plt.plot(df_to_plot['sim_time_client'], df_to_plot['decision_int'],
                         drawstyle='steps-post', marker='.', markersize=5, label='Main Steered Server', color='tab:blue')
                if int_to_server_label_map:
                    valid_ticks = sorted(list(int_to_server_label_map.keys()))
                    valid_labels = [int_to_server_label_map[tick] for tick in valid_ticks]
                    if valid_ticks:
                        plt.yticks(ticks=valid_ticks, labels=valid_labels)
                        plt.ylim(min(valid_ticks) - 0.5, max(valid_ticks) + 0.5)
    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Steering Decision (Main Server)")
    plt.title(f"Steering Service Main Server Decision Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(current_img_dir, "5_steering_decision_timeline.png")
    plt.savefig(plot_path)
    plt.close()

    # Gráfico 6 - Performance Estimada do Servidor
    if strategy_name_from_df in ["epsilon_greedy", "ucb1"]:
        if 'rl_values_json' in df.columns and not df['rl_values_json'].dropna().empty:
            df_values_raw = parse_json_column(df['rl_values_json'].dropna())
            if not df_values_raw.empty:
                df_values_transformed = df_values_raw.copy()
                y_axis_label = "Estimated Average Reward (e.g., 1000/latency)"
                plot_title = f"Estimated Average Rewards per Server Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename}"
                legend_prefix = "Avg. Reward"


                if strategy_name_from_df == "epsilon_greedy":
                    y_axis_label = "Estimated Performance (e.g., 1000/Avg. Latency)"
                    plot_title = f"Estimated Server Performance (Lower Latency is Better)\nStrategy: {strategy_name_from_df} - File: {csv_filename}"
                    legend_prefix = "Perf. (1000/Avg.Lat)"
                    for col in df_values_transformed.columns:
                        df_values_transformed[col] = df_values_transformed[col].apply(
                            lambda x: 1000.0 / x if x > 1 else (1000.0 if x == 1 else 0)
                        ) 

                df_with_values = pd.concat([df['sim_time_client'][df_values_transformed.index], df_values_transformed], axis=1)
                df_wv_unique = df_with_values.drop_duplicates(subset=['sim_time_client'], keep='first')

                plt.figure(figsize=(14, 7))
                value_cols = [col for col in df_wv_unique.columns if col != 'sim_time_client']
                if value_cols:
                    for col_name in sorted(value_cols, key=lambda x: CACHE_SERVER_LABELS.get(x, x)):
                        color_to_use = CACHE_COLORS.get(col_name, 'grey')
                        plt.plot(df_wv_unique['sim_time_client'], df_wv_unique[col_name],
                                marker='.', linestyle='-', markersize=3, alpha=0.7,
                                label=f"{legend_prefix} {CACHE_SERVER_LABELS.get(col_name, col_name)}",
                                color=color_to_use)
                    plt.xlabel("Simulation Time (client, seconds)")
                    plt.ylabel(y_axis_label)
                    plt.title(plot_title)
                    plt.legend(loc='best')
                    plt.grid(True, linestyle=':', alpha=0.7)
                    plt.tight_layout()
                    plot_path = os.path.join(current_img_dir, "6_estimated_performance_timeline.png")
                    plt.savefig(plot_path)
                    plt.close()

    # Gráfico 7 - Contagens
    if 'rl_counts_json' in df.columns and not df['rl_counts_json'].dropna().empty:
        df_counts = parse_json_column(df['rl_counts_json'].dropna())
        if not df_counts.empty:
            df_with_counts = pd.concat([df['sim_time_client'][df_counts.index], df_counts], axis=1)
            df_wc_unique = df_with_counts.drop_duplicates(subset=['sim_time_client'], keep='first')
            plt.figure(figsize=(14, 7))
            count_cols = [col for col in df_wc_unique.columns if col != 'sim_time_client']
            if count_cols:
                for col_name in sorted(count_cols, key=lambda x: CACHE_SERVER_LABELS.get(x, x)):
                    color_to_use = CACHE_COLORS.get(col_name, 'grey')
                    plt.plot(df_wc_unique['sim_time_client'], df_wc_unique[col_name],
                            marker='.', linestyle='-', markersize=3, alpha=0.7,
                            label=f"Pulls for {CACHE_SERVER_LABELS.get(col_name, col_name)}",
                            color=color_to_use)
                plt.xlabel("Simulation Time (client, seconds)")
                plt.ylabel("Number of Times Server Selected (Pulls)")
                plt.title(f"Server Selection Counts (Exploration/Exploitation)\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
                plt.legend(loc='best')
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout()
                plot_path = os.path.join(current_img_dir, "7_selection_counts_timeline.png")
                plt.savefig(plot_path)
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from Content Steering simulation CSV logs.")
    parser.add_argument(
        "csv_file",
        type=str,
        nargs='?',
        default=None,
        help="Path to the simulation CSV log file. If not provided, attempts to process all CSVs in Files/Data."
    )
    args = parser.parse_args()

    if args.csv_file:
        if not os.path.isabs(args.csv_file):
            csv_to_process = os.path.join(DEFAULT_SIM_DATA_DIR, args.csv_file)
        else:
            csv_to_process = args.csv_file
        generate_plots(csv_to_process)
    else:
        processed_any = False
        for filename in os.listdir(DEFAULT_SIM_DATA_DIR):
            if filename.endswith(".csv"):
                generate_plots(os.path.join(DEFAULT_SIM_DATA_DIR, filename))
                processed_any = True
        if not processed_any:
            pass