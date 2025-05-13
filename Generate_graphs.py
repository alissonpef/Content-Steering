import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn ainda pode ser útil para estética de outros gráficos, mas não é estritamente necessário se removermos o boxplot
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
        print(f"Erro: Arquivo CSV não encontrado em {csv_file_path}")
        return

    csv_filename = os.path.basename(csv_file_path)
    simulation_name = os.path.splitext(csv_filename)[0]
    current_img_dir = os.path.join(DEFAULT_IMG_DIR, simulation_name)
    os.makedirs(current_img_dir, exist_ok=True)

    print(f"Lendo dados de: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        print(f"Arquivo CSV {csv_file_path} está vazio ou é inválido. Nenhum gráfico será gerado.")
        return

    if df.empty:
        print(f"Arquivo CSV {csv_file_path} está vazio. Nenhum gráfico será gerado.")
        return

    print(f"Dados carregados de {csv_filename}. {len(df)} linhas.")
    df.sort_values(by="sim_time_client", inplace=True)
    df.reset_index(drop=True, inplace=True)

    strategy_name_from_df = df['rl_strategy'].iloc[0] if 'rl_strategy' in df.columns and not df.empty else "N/A"

    # --- 1. Gráfico de Latência ---
    plt.figure(figsize=(14, 7))
    plt.plot(df['sim_time_client'], df['experienced_latency_ms'], marker='.', linestyle='-', markersize=5, alpha=0.7, label='Experienced Latency (per fragment)')
    window_size = 10
    if len(df['experienced_latency_ms'].dropna()) >= window_size:
        df['latency_ma'] = df['experienced_latency_ms'].rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(df['sim_time_client'], df['latency_ma'], linestyle='--', color='red', linewidth=2, label=f'Moving Average ({window_size} points)')
    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Experienced Latency (ms)")
    plt.title(f"Client's Experienced Latency Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plot_path = os.path.join(current_img_dir, "1_latency_timeline.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico salvo: {plot_path}")

    # --- 2. Gráfico de Latência por Servidor ---
    plt.figure(figsize=(14, 7))
    unique_servers_used = df['server_used_for_latency'].dropna().unique()
    if len(unique_servers_used) > 0:
        colors = plt.cm.get_cmap('viridis', len(unique_servers_used))
        for i, server_name in enumerate(unique_servers_used):
            server_df = df[df['server_used_for_latency'] == server_name]
            if not server_df.empty:
                plt.plot(server_df['sim_time_client'], server_df['experienced_latency_ms'],
                         marker='o', linestyle='-', markersize=4, alpha=0.8,
                         label=f"Latency from {CACHE_SERVER_LABELS.get(server_name, server_name)}", color=colors(i))
        plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Experienced Latency (ms)")
    plt.title(f"Experienced Latency from Each Server (When Used)\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plot_path = os.path.join(current_img_dir, "2_latency_per_server_timeline.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico salvo: {plot_path}")

    # --- 3. Distribuição de Latência (Boxplot) - REMOVIDO ---
    # print("Gráfico 3 (Boxplot de Distribuição de Latência) foi removido conforme solicitado.")


    # --- 4. Decisão de Steering ---
    plt.figure(figsize=(14, 7))
    df_steering_decisions = df.dropna(subset=['steering_decision_main_server', 'sim_time_client'])
    if not df_steering_decisions.empty:
        df_sd_unique = df_steering_decisions.drop_duplicates(subset=['sim_time_client'], keep='first').copy()
        
        all_possible_servers_labels = list(CACHE_SERVER_LABELS.keys())
        all_possible_servers_log = list(df_sd_unique['steering_decision_main_server'].unique())
        all_distinct_servers = sorted(list(set(all_possible_servers_labels + all_possible_servers_log)))

        server_to_int = {server: i for i, server in enumerate(all_distinct_servers)}
        int_to_server_label_map = {i: CACHE_SERVER_LABELS.get(server, server) for server, i in server_to_int.items()}

        df_sd_unique['decision_int'] = df_sd_unique['steering_decision_main_server'].map(server_to_int)

        plt.plot(df_sd_unique['sim_time_client'], df_sd_unique['decision_int'],
                 drawstyle='steps-post', marker='.', markersize=5, label='Main Steered Server')
        if int_to_server_label_map:
            plt.yticks(ticks=list(int_to_server_label_map.keys()), labels=list(int_to_server_label_map.values()))
            if int_to_server_label_map.keys(): # Adicionar verificação para evitar erro com min/max em lista vazia
                plt.ylim(min(int_to_server_label_map.keys()) - 0.5, max(int_to_server_label_map.keys()) + 0.5)

        plt.xlabel("Simulation Time (client, seconds)")
        plt.ylabel("Steering Decision (Main Server)")
        plt.title(f"Steering Service Main Server Decision Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(current_img_dir, "4_steering_decision_timeline.png") # Renomeado para 3 se o anterior foi removido (ou manter numeração)
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico salvo: {plot_path}")
    else:
        print("Não há dados suficientes para o gráfico de decisão de steering.")

    # --- 5. Gráfico de Recompensas Médias Estimadas (rl_values) ---
    # (O gráfico 5 agora será o 4º gráfico gerado se o boxplot foi removido)
    if strategy_name_from_df in ["epsilon_greedy", "ucb1"]:
        if 'rl_values_json' in df.columns and not df['rl_values_json'].dropna().empty:
            df_values = parse_json_column(df['rl_values_json'].dropna())
            if not df_values.empty:
                df_with_values = pd.concat([df['sim_time_client'][df_values.index], df_values], axis=1)
                df_wv_unique = df_with_values.drop_duplicates(subset=['sim_time_client'], keep='first')

                plt.figure(figsize=(14, 7))
                value_cols = [col for col in df_wv_unique.columns if col != 'sim_time_client']
                if value_cols:
                    for col_name in value_cols:
                        plt.plot(df_wv_unique['sim_time_client'], df_wv_unique[col_name],
                                marker='.', linestyle='-', markersize=3, alpha=0.7,
                                label=f"Avg. Reward {CACHE_SERVER_LABELS.get(col_name, col_name)}")
                    plt.xlabel("Simulation Time (client, seconds)")
                    plt.ylabel("Estimated Average Reward (e.g., 1000/latency)")
                    plt.title(f"Estimated Average Rewards per Server Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
                    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
                    plt.grid(True, linestyle=':', alpha=0.7)
                    plt.tight_layout(rect=[0, 0, 0.80, 1])
                    plot_path = os.path.join(current_img_dir, "5_estimated_rewards_timeline.png") # Ou "4_..." se renumerar
                    plt.savefig(plot_path)
                    plt.close()
                    print(f"Gráfico salvo: {plot_path}")
                else:
                    print("Nenhuma coluna de 'values' encontrada após o parse do JSON para o gráfico de recompensas.")
            else:
                print("Coluna 'rl_values_json' resultou em DataFrame vazio após parse.")
        else:
            print("Coluna 'rl_values_json' não encontrada ou vazia para o gráfico de recompensas.")
    else:
        print(f"Gráfico de recompensas não aplicável para a estratégia: {strategy_name_from_df}")

    # --- 6. Gráfico de Contagens de Seleção (rl_counts) ---
    # (O gráfico 6 agora será o 5º gráfico gerado se o boxplot foi removido)
    if 'rl_counts_json' in df.columns and not df['rl_counts_json'].dropna().empty:
        df_counts = parse_json_column(df['rl_counts_json'].dropna())
        if not df_counts.empty:
            df_with_counts = pd.concat([df['sim_time_client'][df_counts.index], df_counts], axis=1)
            df_wc_unique = df_with_counts.drop_duplicates(subset=['sim_time_client'], keep='first')

            plt.figure(figsize=(14, 7))
            count_cols = [col for col in df_wc_unique.columns if col != 'sim_time_client']
            if count_cols:
                for col_name in count_cols:
                    plt.plot(df_wc_unique['sim_time_client'], df_wc_unique[col_name],
                            marker='.', linestyle='-', markersize=3, alpha=0.7,
                            label=f"Pulls for {CACHE_SERVER_LABELS.get(col_name, col_name)}")
                plt.xlabel("Simulation Time (client, seconds)")
                plt.ylabel("Number of Times Server Selected (Pulls)")
                plt.title(f"Server Selection Counts (Exploration/Exploitation)\nStrategy: {strategy_name_from_df} - File: {csv_filename}")
                plt.legend(loc='upper left', bbox_to_anchor=(1,1))
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout(rect=[0, 0, 0.80, 1])
                plot_path = os.path.join(current_img_dir, "6_selection_counts_timeline.png") # Ou "5_..." se renumerar
                plt.savefig(plot_path)
                plt.close()
                print(f"Gráfico salvo: {plot_path}")
            else:
                print("Nenhuma coluna de 'counts' encontrada após o parse do JSON para o gráfico de contagens.")
        else:
            print("Coluna 'rl_counts_json' resultou em DataFrame vazio após parse.")
    else:
        print("Coluna 'rl_counts_json' não encontrada ou vazia para o gráfico de contagens.")

    print(f"\n--- Geração de gráficos para '{simulation_name}' concluída. Salvos em: {current_img_dir} ---")

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
        print(f"\n--- Processando arquivo especificado: {os.path.basename(csv_to_process)} ---")
        generate_plots(csv_to_process)
    else:
        print(f"\n--- Processando todos os arquivos CSV em {DEFAULT_SIM_DATA_DIR} ---")
        processed_any = False
        for filename in os.listdir(DEFAULT_SIM_DATA_DIR):
            if filename.endswith(".csv"):
                print(f"\n--- Processando {filename} ---")
                generate_plots(os.path.join(DEFAULT_SIM_DATA_DIR, filename))
                processed_any = True
        if not processed_any:
            print(f"Nenhum arquivo CSV encontrado em {DEFAULT_SIM_DATA_DIR} para processar.")