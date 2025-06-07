import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import json
import argparse
import numpy as np
import logging

logger = logging.getLogger("generate_graphs")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM_DATA_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs")
DEFAULT_IMG_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img")

SERVER_DISPLAY_NAMES = {
    "video-streaming-cache-1": "Cache Server 1 (BR)",
    "video-streaming-cache-2": "Cache Server 2 (CL)",
    "video-streaming-cache-3": "Cache Server 3 (CO)",
    "N/A_NO_NODES_FROM_SELECTION": "No Selection",
    "N/A_NO_NODES_FROM_RL": "No RL Nodes",
    "N/A": "N/A",
    "DynamicBest": "Optimal Server"
}
SERVER_COLORS = {
    "video-streaming-cache-1": "tab:green",
    "video-streaming-cache-2": "tab:orange",
    "video-streaming-cache-3": "tab:blue",
    "DynamicBest": "tab:red",
    "N/A_NO_NODES_FROM_SELECTION": "tab:grey",
    "N/A_NO_NODES_FROM_RL": "silver",
    "N/A": "lightgrey"
}
KNOWN_CACHE_SERVER_KEYS_UNDERSCORE = [
    "video_streaming_cache_1", "video_streaming_cache_2", "video_streaming_cache_3"
]
ACTUAL_CACHE_SERVER_NAMES_HYPHEN = [key for key in SERVER_DISPLAY_NAMES.keys() if "cache" in key.lower()]

# Converte uma Série de strings JSON em um DataFrame, normalizando chaves e tratando erros.
def parse_json_series_to_dataframe(series: pd.Series, prefix: str = "") -> pd.DataFrame:
    parsed_rows = []
    all_normalized_keys_in_series = set()
    temp_parsed_dicts = []
    valid_indices = series.dropna().index
    for json_str in series.dropna():
        try:
            data_dict = json.loads(json_str)
            if isinstance(data_dict, dict):
                normalized_dict = {str(k).replace('-', '_'): v for k, v in data_dict.items()}
                all_normalized_keys_in_series.update(normalized_dict.keys())
                temp_parsed_dicts.append(normalized_dict)
            else: temp_parsed_dicts.append({})
        except (json.JSONDecodeError, TypeError):
            logger.debug(f"Falha ao parsear JSON (parse_json_series_to_dataframe): '{str(json_str)[:70]}...'")
            temp_parsed_dicts.append({})
    final_column_keys = all_normalized_keys_in_series
    if not final_column_keys and (prefix.startswith("value_") or prefix.startswith("count_") or prefix == ""):
        final_column_keys = set(KNOWN_CACHE_SERVER_KEYS_UNDERSCORE)
    prefixed_final_column_keys = {f"{prefix}{key}" for key in final_column_keys} if prefix else final_column_keys
    for norm_dict in temp_parsed_dicts:
        row_data = {prefixed_key: norm_dict.get(prefixed_key.replace(prefix, "")) for prefixed_key in prefixed_final_column_keys}
        parsed_rows.append(row_data)
    if not parsed_rows:
        return pd.DataFrame(columns=list(prefixed_final_column_keys))
    df_result = pd.DataFrame(parsed_rows, index=valid_indices, columns=list(prefixed_final_column_keys))
    return df_result

# Extrai o nome e a latência do servidor otimizado dinamicamente de uma linha do DataFrame.
def find_dynamic_best_server_and_latency(row):
    if pd.isna(row['all_servers_oracle_latency_json']):
        return None, np.nan
    try:
        server_latencies = json.loads(row['all_servers_oracle_latency_json'])
        valid_server_latencies = {
            s_name: lat
            for s_name, lat in server_latencies.items()
            if s_name in ACTUAL_CACHE_SERVER_NAMES_HYPHEN and isinstance(lat, (int, float))
        }
        if not valid_server_latencies: return None, np.nan
        best_server_name = min(valid_server_latencies, key=valid_server_latencies.get)
        best_server_latency = valid_server_latencies[best_server_name]
        return best_server_name, best_server_latency
    except (json.JSONDecodeError, TypeError): return None, np.nan
    except Exception: return None, np.nan

# Aplica formatação padronizada aos gráficos de logs individuais.
def format_plot(ax, title, xlabel, ylabel, legend_loc='best', y_log_scale=False, custom_legend_handles=None, custom_legend_labels=None):
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if y_log_scale:
        ax.set_yscale('log')
        if ax.has_data(): ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    else:
        has_plotted_data = False
        if ax.has_data():
            for line in ax.get_lines():
                ydata = line.get_ydata()
                if isinstance(ydata, (pd.Series, np.ndarray)) and ydata.size > 0:
                    numeric_ydata = pd.to_numeric(ydata, errors='coerce')
                    if np.any(numeric_ydata[~np.isnan(numeric_ydata)] >= 0):
                        has_plotted_data = True
                        break
        if has_plotted_data:
            ax.set_ylim(bottom=0)
    if custom_legend_handles and custom_legend_labels:
        ax.legend(custom_legend_handles, custom_legend_labels, loc=legend_loc, fontsize=10)
    else:
        handles, labels = ax.get_legend_handles_labels()
        if handles: ax.legend(handles, labels, loc=legend_loc, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, which='major')
    if y_log_scale and ax.has_data(): ax.grid(True, linestyle=':', alpha=0.3, which='minor')
    plt.tight_layout(pad=1.2)

# Gera um conjunto de gráficos a partir de um arquivo CSV de log de simulação individual.
def generate_plots(csv_file_path: str):
    if not os.path.exists(csv_file_path):
        logger.error(f"Arquivo CSV não encontrado: {csv_file_path}")
        return
    csv_filename_with_ext = os.path.basename(csv_file_path)
    current_img_dir = os.path.join(DEFAULT_IMG_DIR, os.path.splitext(csv_filename_with_ext)[0])
    os.makedirs(current_img_dir, exist_ok=True)
    logger.info(f"Lendo dados de: {csv_filename_with_ext}")
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Arquivo CSV {csv_filename_with_ext} vazio. Nenhum gráfico será gerado.")
        return
    if df.empty:
        logger.warning(f"Arquivo CSV {csv_filename_with_ext} vazio. Nenhum gráfico será gerado.")
        return
    df.sort_values(by="sim_time_client", inplace=True)
    df.reset_index(drop=True, inplace=True)
    strategy_name_from_df = df['rl_strategy'].iloc[0] if 'rl_strategy' in df.columns and not df.empty and pd.notna(df['rl_strategy'].iloc[0]) else "N/A"
    strategy_display_name = strategy_name_from_df.replace('_', ' ').title()
    window_size = 10
    if 'all_servers_oracle_latency_json' in df.columns:
        dynamic_best_info = df.apply(find_dynamic_best_server_and_latency, axis=1, result_type='expand')
        df[['dynamic_best_server_name', 'dynamic_best_server_latency']] = dynamic_best_info
    else:
        logger.warning("Coluna 'all_servers_oracle_latency_json' não encontrada. Gráficos de 'melhor dinâmico' não serão totalmente gerados.")
        df['dynamic_best_server_name'] = None
        df['dynamic_best_server_latency'] = np.nan

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_made_g1 = False
    legend1_handles, legend1_labels = [], []
    if 'experienced_latency_ms' in df.columns and 'sim_time_client' in df.columns:
        df_chosen_latency = df.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_chosen_latency.empty and len(df_chosen_latency) >= window_size:
            ma_chosen = df_chosen_latency['experienced_latency_ms'].rolling(window=window_size, center=True, min_periods=1).mean()
            line_chosen, = ax1.plot(df_chosen_latency['sim_time_client'], ma_chosen,
                                    linestyle='-', color='navy', linewidth=1.5, alpha=0.9)
            legend1_handles.append(line_chosen)
            legend1_labels.append(f'MA ({window_size}s) - Chosen Server')
            plot_made_g1 = True
    if 'dynamic_best_server_latency' in df.columns and 'sim_time_client' in df.columns:
        df_dynamic_best_latency = df.dropna(subset=['sim_time_client', 'dynamic_best_server_latency'])
        if not df_dynamic_best_latency.empty and len(df_dynamic_best_latency) >= window_size:
            ma_dynamic_best = df_dynamic_best_latency['dynamic_best_server_latency'].rolling(window=window_size, center=True, min_periods=1).mean()
            line_optimal, = ax1.plot(df_dynamic_best_latency['sim_time_client'], ma_dynamic_best,
                                     linestyle='--', color=SERVER_COLORS.get("DynamicBest", "tab:red"), linewidth=1.5, alpha=0.9)
            legend1_handles.append(line_optimal)
            legend1_labels.append(f'MA ({window_size}s) - Optimal Server')
            plot_made_g1 = True
    if plot_made_g1:
        format_plot(ax1, f"Chosen Server Latency vs Optimal Latency\nStrategy: {strategy_display_name}",
                    "Simulation Time (s)", "Simulated Latency (ms)", legend_loc='upper right',
                    custom_legend_handles=legend1_handles, custom_legend_labels=legend1_labels)
        plt.savefig(os.path.join(current_img_dir, "1_latency_chosen_vs_optimal.png"))
    else: logger.info("Dados insuficientes para Gráfico 1: Latência Escolhida vs. Ótima.")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_made_g2 = False
    legend2_handles, legend2_labels = [], []
    if 'steering_decision_main_server' in df.columns and 'sim_time_client' in df.columns:
        df_steering = df.dropna(subset=['steering_decision_main_server', 'sim_time_client'])
        if not df_steering.empty:
            df_s_unique = df_steering.drop_duplicates(subset=['sim_time_client'], keep='first').copy()
            all_y_entities = ACTUAL_CACHE_SERVER_NAMES_HYPHEN + \
                             [val for val in df_s_unique['steering_decision_main_server'].unique() if "N/A" in str(val) or pd.isna(val)] + \
                             (['DynamicBest'] if 'dynamic_best_server_name' in df.columns and df['dynamic_best_server_name'].notna().any() else [])
            unique_y_entities = sorted(list(set(entity for entity in all_y_entities if pd.notna(entity))))
            entity_to_int_map = {entity: i for i, entity in enumerate(unique_y_entities)}
            df_s_unique.loc[:, 'decision_int'] = df_s_unique['steering_decision_main_server'].map(entity_to_int_map)
            df_plot_decision = df_s_unique.dropna(subset=['decision_int'])
            if not df_plot_decision.empty:
                line_algo, = ax2.plot(df_plot_decision['sim_time_client'], df_plot_decision['decision_int'],
                                      drawstyle='steps-post', marker='o', markersize=3, alpha=0.8, color='tab:cyan')
                if not any(label == "Algorithm's Server Choice" for label in legend2_labels):
                    legend2_handles.append(line_algo)
                    legend2_labels.append("Algorithm's Server Choice")
                plot_made_g2 = True
            if 'dynamic_best_server_name' in df_s_unique.columns:
                df_s_unique.loc[:, 'dynamic_best_int'] = df_s_unique['dynamic_best_server_name'].map(entity_to_int_map).fillna(-1)
                df_plot_dynamic_best = df_s_unique[df_s_unique['dynamic_best_int'] != -1].dropna(subset=['sim_time_client'])
                if not df_plot_dynamic_best.empty:
                    line_optimal_proxy, = plt.plot([], [], linestyle='None', marker='x', markersize=7,
                                                  color=SERVER_COLORS.get("DynamicBest", "tab:red"), label="Optimal Server")
                    ax2.plot(df_plot_dynamic_best['sim_time_client'], df_plot_dynamic_best['dynamic_best_int'],
                             linestyle='None', marker='x', markersize=7, alpha=0.7,
                             color=SERVER_COLORS.get("DynamicBest", "tab:red"))
                    if not any(label == "Optimal Server" for label in legend2_labels):
                        legend2_handles.append(line_optimal_proxy)
                        legend2_labels.append("Optimal Server")
                    plot_made_g2 = True
            if plot_made_g2 and entity_to_int_map and unique_y_entities:
                ax2.set_yticks(list(entity_to_int_map.values()))
                ax2.set_yticklabels([SERVER_DISPLAY_NAMES.get(entity, str(entity).replace("_"," ").title()) for entity in unique_y_entities])
                ax2.set_ylim(min(entity_to_int_map.values()) - 0.5, max(entity_to_int_map.values()) + 0.5)
    if plot_made_g2:
        format_plot(ax2, f"Steering Decisions and Optimal Server\nStrategy: {strategy_display_name}",
                    "Simulation Time (s)", "Server Entity", legend_loc='upper right',
                    custom_legend_handles=legend2_handles, custom_legend_labels=legend2_labels)
        plt.setp(ax2.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        plt.tight_layout(pad=1.5)
        plt.savefig(os.path.join(current_img_dir, "2_steering_decision_vs_optimal.png"))
    else: logger.info("Dados insuficientes para Gráfico 2: Decisão de Steering vs. Ótimo.")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    plot_made_g3 = False
    legend3_handles, legend3_labels = [], []
    y_label_g3 = "Estimated RL Value"
    if strategy_name_from_df.lower() in ["epsilon_greedy", "ucb1"] and 'rl_values_json' in df.columns and not df['rl_values_json'].dropna().empty:
        df_values_parsed = parse_json_series_to_dataframe(df['rl_values_json'].dropna(), prefix="value_")
        if not df_values_parsed.empty:
            df_values_with_time = pd.concat([df.loc[df_values_parsed.index, 'sim_time_client'], df_values_parsed], axis=1).reset_index(drop=True)
            df_values_unique_time = df_values_with_time.drop_duplicates(subset=['sim_time_client'], keep='last').copy()
            if strategy_name_from_df.lower() == "epsilon_greedy":
                y_label_g3 = "Estimated Reward (Higher is Better)"
            elif strategy_name_from_df.lower() == "ucb1":
                y_label_g3 = "Estimated Average Reward (UCB1)"
            value_cols_plot = sorted([col for col in df_values_unique_time.columns if col.startswith('value_') and col.replace('value_', '') in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE])
            for col_name in value_cols_plot:
                s_key_u = col_name.replace('value_', '')
                s_key_h = s_key_u.replace('_', '-')
                color = SERVER_COLORS.get(s_key_h, 'grey')
                label_text = SERVER_DISPLAY_NAMES.get(s_key_h, s_key_u)
                df_subset = df_values_unique_time.dropna(subset=['sim_time_client', col_name]).copy()
                if strategy_name_from_df.lower() == "epsilon_greedy":
                    df_subset.loc[:, col_name] = df_subset[col_name].apply(lambda x: 1000.0 / x if isinstance(x, (int,float)) and x > 0 else 0.0)
                if not df_subset.empty:
                    line, = ax3.plot(df_subset['sim_time_client'], df_subset[col_name], marker='.', linestyle='-', ms=3, alpha=0.7, label=label_text, color=color)
                    if not any(l == label_text for l in legend3_labels):
                        legend3_handles.append(line)
                        legend3_labels.append(label_text)
                    plot_made_g3 = True
            if plot_made_g3:
                format_plot(ax3, f"RL Algorithm's Estimated Server Values\nStrategy: {strategy_display_name}", "Simulation Time (s)", y_label_g3, legend_loc='upper right',
                            custom_legend_handles=legend3_handles, custom_legend_labels=legend3_labels)
                plt.savefig(os.path.join(current_img_dir, "3_rl_estimated_values.png"))
            else: logger.info("Nenhuma coluna 'value_*' válida para Gráfico 3.")
        else: logger.info("DataFrame de 'rl_values_json' vazio após parse para Gráfico 3.")
    else: logger.info(f"Gráfico 3 (RL Values) não aplicável para {strategy_display_name}.")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    plot_made_g4 = False
    legend4_handles, legend4_labels = [], []
    if 'rl_counts_json' in df.columns and not df['rl_counts_json'].dropna().empty:
        df_counts_parsed = parse_json_series_to_dataframe(df['rl_counts_json'].dropna(), prefix="count_")
        if not df_counts_parsed.empty:
            df_counts_with_time = pd.concat([df.loc[df_counts_parsed.index, 'sim_time_client'], df_counts_parsed], axis=1).reset_index(drop=True)
            df_counts_unique_time = df_counts_with_time.drop_duplicates(subset=['sim_time_client'], keep='last').copy()
            count_cols_plot = sorted([col for col in df_counts_unique_time.columns if col.startswith('count_') and col.replace('count_', '') in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE])
            for col_name in count_cols_plot:
                s_key_u = col_name.replace('count_', '')
                s_key_h = s_key_u.replace('_', '-')
                color = SERVER_COLORS.get(s_key_h, 'grey')
                label_text = SERVER_DISPLAY_NAMES.get(s_key_h, s_key_u)
                df_subset = df_counts_unique_time.dropna(subset=['sim_time_client', col_name])
                if not df_subset.empty:
                    line, = ax4.plot(df_subset['sim_time_client'], df_subset[col_name], marker='.', linestyle='-', ms=3, alpha=0.7, color=color)
                    if not any(l == label_text for l in legend4_labels):
                        legend4_handles.append(line)
                        legend4_labels.append(label_text)
                    plot_made_g4 = True
            if plot_made_g4:
                format_plot(ax4, f"RL Algorithm's Server Selection Counts (Pulls)\nStrategy: {strategy_display_name}", "Simulation Time (s)", "Number of Selections (Pulls)", legend_loc='upper left',
                            custom_legend_handles=legend4_handles, custom_legend_labels=legend4_labels)
                plt.savefig(os.path.join(current_img_dir, "4_rl_selection_counts.png"))
            else: logger.info("Nenhuma coluna 'count_*' válida para Gráfico 4.")
        else: logger.info("DataFrame de 'rl_counts_json' vazio após parse para Gráfico 4.")
    else: logger.info("Coluna 'rl_counts_json' não encontrada para Gráfico 4.")
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(12, 6))
    plot_made_g5 = False
    legend5_handles, legend5_labels = [], []
    if 'all_servers_oracle_latency_json' in df.columns and not df['all_servers_oracle_latency_json'].dropna().empty:
        df_all_lat_parsed = parse_json_series_to_dataframe(df['all_servers_oracle_latency_json'].dropna(), prefix="")
        if not df_all_lat_parsed.empty:
            df_all_lat_with_time = pd.concat([df.loc[df_all_lat_parsed.index, 'sim_time_client'], df_all_lat_parsed], axis=1).reset_index(drop=True)
            df_all_lat_unique_time = df_all_lat_with_time.drop_duplicates(subset=['sim_time_client'], keep='last').copy()
            if not df_all_lat_unique_time.empty and 'sim_time_client' in df_all_lat_unique_time.columns:
                all_oracle_cols = sorted([col for col in df_all_lat_unique_time.columns if col != 'sim_time_client' and col in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE])
                for server_col_u in all_oracle_cols:
                    s_key_h = server_col_u.replace('_', '-')
                    color = SERVER_COLORS.get(s_key_h, 'grey')
                    label_text = SERVER_DISPLAY_NAMES.get(s_key_h, server_col_u)
                    df_subset = df_all_lat_unique_time.dropna(subset=['sim_time_client', server_col_u])
                    if not df_subset.empty:
                        line, = ax5.plot(df_subset['sim_time_client'], df_subset[server_col_u], marker='.', linestyle='-', ms=2, alpha=0.6, color=color)
                        if not any(l == label_text for l in legend5_labels):
                            legend5_handles.append(line)
                            legend5_labels.append(label_text)
                        plot_made_g5 = True
                if plot_made_g5:
                    format_plot(ax5, f"Simulated Latency Landscape for All Servers\nStrategy: {strategy_display_name}", "Simulation Time (s)", "Simulated Latency (ms)", legend_loc='upper right',
                                custom_legend_handles=legend5_handles, custom_legend_labels=legend5_labels)
                    plt.savefig(os.path.join(current_img_dir, "5_all_servers_oracle_latency.png"))
                else: logger.info("Nenhuma coluna válida para Gráfico 5 (Latência Oráculo Todos).")
            else: logger.info("DataFrame vazio ou sem sim_time_client para Gráfico 5.")
        else: logger.info("DataFrame 'all_servers_oracle_latency_json' vazio após parse para Gráfico 5.")
    else: logger.info("Coluna 'all_servers_oracle_latency_json' não encontrada para Gráfico 5.")
    plt.close(fig5)

    logger.info(f"Geração de gráficos para '{os.path.splitext(csv_filename_with_ext)[0]}' concluída. Salvos em: {current_img_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from Content Steering simulation CSV logs.")
    parser.add_argument("csv_argument", type=str, nargs='?', default=None,
                        help="Filename/path to CSV log. Searched in standard dirs if not absolute.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()
    handler_main = logging.StreamHandler()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
    if not logger.handlers:
        handler_main.setFormatter(formatter)
        logger.addHandler(handler_main)
    else:
        logger.handlers[0].setFormatter(formatter)
        logger.handlers[0].setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.csv_argument:
        csv_to_process, resolved_path = args.csv_argument, None
        paths_to_check = []
        if os.path.isabs(csv_to_process): paths_to_check.append(csv_to_process)
        if not csv_to_process.lower().endswith(".csv"):
             if os.path.isabs(csv_to_process): paths_to_check.append(csv_to_process + ".csv")
        for d_dir in [os.getcwd(), DEFAULT_SIM_DATA_DIR, os.path.join(DEFAULT_SIM_DATA_DIR, "Average")]:
            paths_to_check.append(os.path.join(d_dir, os.path.basename(csv_to_process)))
            if not csv_to_process.lower().endswith(".csv"):
                paths_to_check.append(os.path.join(d_dir, os.path.basename(csv_to_process) + ".csv"))
        unique_paths_to_check = sorted(list(set(paths_to_check)), key=lambda p: (not os.path.isabs(p) or not p.startswith(os.getcwd()), p))
        for potential_path in unique_paths_to_check:
            if os.path.exists(potential_path) and os.path.isfile(potential_path):
                resolved_path = os.path.abspath(potential_path)
                break
        if resolved_path:
            logger.info(f"Processando arquivo: {os.path.basename(resolved_path)} (Resolvido de '{args.csv_argument}')")
            generate_plots(resolved_path)
        else:
            logger.error(f"Arquivo '{args.csv_argument}' não encontrado nos diretórios de busca: CWD, {DEFAULT_SIM_DATA_DIR}, {os.path.join(DEFAULT_SIM_DATA_DIR, 'Average')}.")
    else:
        logger.info(f"Nenhum arquivo CSV especificado. Processando todos os arquivos CSV nos diretórios padrão.")
        processed_any = False
        for dirname in [DEFAULT_SIM_DATA_DIR]:
            if os.path.isdir(dirname):
                logger.info(f"Procurando arquivos CSV em: {dirname}")
                for filename in sorted(os.listdir(dirname)):
                    if filename.startswith("log_") and filename.endswith(".csv") and "_average" not in filename:
                        full_path = os.path.join(dirname, filename)
                        logger.info(f"---> Processando {filename} de {os.path.basename(dirname)}/")
                        generate_plots(full_path)
                        processed_any = True
            else: logger.warning(f"Diretório não encontrado: {dirname}")
        if not processed_any: logger.warning("Nenhum arquivo CSV encontrado para processar.")