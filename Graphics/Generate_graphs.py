import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse
import numpy as np
import logging

logger = logging.getLogger("generate_graphs")

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM_DATA_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs")
DEFAULT_IMG_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img")

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

def parse_json_column(series, prefix=""):
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
                    processed_dict_for_df_cols = {k.replace('-', '_'): v for k, v in data_dict.items()}
                    all_keys.update(processed_dict_for_df_cols.keys())
                    temp_parsed_dicts.append(processed_dict_for_df_cols)
                else:
                    temp_parsed_dicts.append({})
        except (json.JSONDecodeError, TypeError):
            logger.debug(f"Falha ao parsear string JSON em parse_json_column: {str(item)[:100]}...")
            temp_parsed_dicts.append({})

    for data_dict in temp_parsed_dicts:
        row = {f"{prefix}{k}": data_dict.get(k) for k in all_keys}
        parsed_data.append(row)
    return pd.DataFrame(parsed_data, index=series.index)


def generate_plots(csv_file_path):
    if not os.path.exists(csv_file_path):
        logger.error(f"Arquivo CSV não encontrado: {csv_file_path}")
        return

    csv_filename_with_ext = os.path.basename(csv_file_path)
    simulation_name = os.path.splitext(csv_filename_with_ext)[0]
    current_img_dir = os.path.join(DEFAULT_IMG_DIR, simulation_name)
    os.makedirs(current_img_dir, exist_ok=True)

    logger.info(f"Lendo dados de: {csv_filename_with_ext}")
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Arquivo CSV {csv_filename_with_ext} está vazio ou é inválido. Nenhum gráfico será gerado.")
        return
    if df.empty:
        logger.warning(f"Arquivo CSV {csv_filename_with_ext} está vazio. Nenhum gráfico será gerado.")
        return

    df.sort_values(by="sim_time_client", inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info(f"Dados carregados de {csv_filename_with_ext}. {len(df)} linhas.")

    strategy_name_from_df = df['rl_strategy'].iloc[0] if 'rl_strategy' in df.columns and not df.empty else "N/A"

    # Gráfico 1 - Latência Geral (Linear)
    plt.figure(figsize=(14, 7))
    plot_made = False
    if not df.empty and 'experienced_latency_ms' in df.columns and 'sim_time_client' in df.columns:
        df_latency_points = df.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_latency_points.empty:
            plt.plot(df_latency_points['sim_time_client'], df_latency_points['experienced_latency_ms'],
                    marker='.', linestyle='', markersize=5, alpha=0.6,
                    label='Experienced Latency (per fragment)')
            plot_made = True
    window_size = 10
    if 'experienced_latency_ms' in df.columns and len(df['experienced_latency_ms'].dropna()) >= window_size:
        if 'latency_ma_calculated' not in df.columns or df['latency_ma_calculated'].isnull().all():
             df.loc[:, 'latency_ma_calculated'] = df['experienced_latency_ms'].rolling(window=window_size, center=True, min_periods=1).mean()
        plt.plot(df['sim_time_client'], df['latency_ma_calculated'],
                linestyle='--', color='red', linewidth=2,
                label=f'Moving Average ({window_size} points)')
        plot_made = True
    if plot_made:
        plt.xlabel("Simulation Time (client, seconds)")
        plt.ylabel("Experienced Latency (ms)")
        plt.title(f"Client's Experienced Latency Over Time - Linear Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plot_path = os.path.join(current_img_dir, "1_latency_timeline_linear.png")
        plt.savefig(plot_path)
        logger.debug(f"Gráfico 1 (Linear) salvo: {plot_path}")
    else:
        logger.info("Dados insuficientes para o Gráfico 1 (Latência Linear).")
    plt.close()

    # Gráfico 2 - Latência Geral (Log)
    plt.figure(figsize=(14, 7))
    plot_made = False
    if not df.empty and 'experienced_latency_ms' in df.columns and 'sim_time_client' in df.columns:
        df_latency_points = df.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_latency_points.empty:
            plt.plot(df_latency_points['sim_time_client'], df_latency_points['experienced_latency_ms'],
                    marker='.', linestyle='', markersize=5, alpha=0.6,
                    label='Experienced Latency (per fragment)')
            plot_made = True
    if 'latency_ma_calculated' in df.columns and not df['latency_ma_calculated'].isnull().all():
        plt.plot(df['sim_time_client'], df['latency_ma_calculated'],
                linestyle='--', color='red', linewidth=2,
                label=f'Moving Average ({window_size} points)')
        plot_made = True
    if plot_made:
        plt.xlabel("Simulation Time (client, seconds)")
        plt.ylabel("Experienced Latency (ms) - Log Scale")
        plt.yscale('log')
        plt.title(f"Client's Experienced Latency Over Time - Log Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
        plt.legend(loc='best')
        plt.grid(True, which="both", linestyle=':', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(current_img_dir, "2_latency_timeline_logscale.png")
        plt.savefig(plot_path)
        logger.debug(f"Gráfico 2 (Log) salvo: {plot_path}")
    else:
        logger.info("Dados insuficientes para o Gráfico 2 (Latência Log).")
    plt.close()

    # Gráfico 3 - Latência por Servidor (Linear)
    plt.figure(figsize=(14, 7))
    unique_servers_used = df['server_used_for_latency'].dropna().unique()
    plot_made = False
    if len(unique_servers_used) > 0:
        for server_name_with_hyphen in sorted(list(unique_servers_used), key=lambda x: CACHE_SERVER_LABELS.get(x, x)):
            server_df = df[df['server_used_for_latency'] == server_name_with_hyphen]
            if not server_df.empty:
                color_to_use = CACHE_COLORS.get(server_name_with_hyphen, 'black')
                plt.plot(server_df['sim_time_client'], server_df['experienced_latency_ms'],
                         marker='o', linestyle='-', markersize=4, alpha=0.8,
                         label=f"Latency from {CACHE_SERVER_LABELS.get(server_name_with_hyphen, server_name_with_hyphen)}", color=color_to_use)
                plot_made = True
    if plot_made:
        plt.legend(loc='best')
        plt.xlabel("Simulation Time (client, seconds)")
        plt.ylabel("Experienced Latency (ms)")
        plt.title(f"Experienced Latency from Each Server (When Used) - Linear Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plot_path = os.path.join(current_img_dir, "3_latency_per_server_timeline_linear.png")
        plt.savefig(plot_path)
        logger.debug(f"Gráfico 3 salvo: {plot_path}")
    else:
        logger.info("Dados insuficientes para o Gráfico 3.")
    plt.close()

    # Gráfico 4 - Latência por Servidor (Log)
    plt.figure(figsize=(14, 7))
    plot_made = False
    if len(unique_servers_used) > 0:
        for server_name_with_hyphen in sorted(list(unique_servers_used), key=lambda x: CACHE_SERVER_LABELS.get(x, x)):
            server_df = df[df['server_used_for_latency'] == server_name_with_hyphen]
            if not server_df.empty:
                color_to_use = CACHE_COLORS.get(server_name_with_hyphen, 'black')
                plt.plot(server_df['sim_time_client'], server_df['experienced_latency_ms'],
                         marker='o', linestyle='-', markersize=4, alpha=0.8,
                         label=f"Latency from {CACHE_SERVER_LABELS.get(server_name_with_hyphen, server_name_with_hyphen)}", color=color_to_use)
                plot_made = True
    if plot_made:
        plt.legend(loc='best')
        plt.xlabel("Simulation Time (client, seconds)")
        plt.ylabel("Experienced Latency (ms) - Log Scale")
        plt.yscale('log')
        plt.title(f"Experienced Latency from Each Server (When Used) - Log Scale\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
        plt.grid(True, which="both", linestyle=':', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(current_img_dir, "4_latency_per_server_timeline_logscale.png")
        plt.savefig(plot_path)
        logger.debug(f"Gráfico 4 salvo: {plot_path}")
    else:
        logger.info("Dados insuficientes para o Gráfico 4.")
    plt.close()

    # Gráfico 5 - Decisão de Steering
    plt.figure(figsize=(14, 7))
    df_steering_decisions = df.dropna(subset=['steering_decision_main_server', 'sim_time_client'])
    plot_made = False
    if not df_steering_decisions.empty:
        df_sd_unique = df_steering_decisions.drop_duplicates(subset=['sim_time_client'], keep='first').copy()
        actual_cache_server_names = ["video-streaming-cache-1", "video-streaming-cache-2", "video-streaming-cache-3"]
        relevant_servers_for_plot = sorted(actual_cache_server_names)
        if relevant_servers_for_plot:
            server_to_int = {server: i for i, server in enumerate(relevant_servers_for_plot)}
            int_to_server_label_map = {i: CACHE_SERVER_LABELS.get(server, server) for server, i in server_to_int.items()}
            df_sd_unique.loc[:, 'decision_int'] = df_sd_unique['steering_decision_main_server'].map(server_to_int)
            df_to_plot = df_sd_unique.dropna(subset=['decision_int'])
            if not df_to_plot.empty:
                plt.plot(df_to_plot['sim_time_client'], df_to_plot['decision_int'],
                         drawstyle='steps-post', marker='.', markersize=5, label='Main Steered Server', color='tab:blue')
                plot_made = True
                if int_to_server_label_map:
                    valid_ticks = sorted(list(int_to_server_label_map.keys()))
                    valid_labels = [int_to_server_label_map[tick] for tick in valid_ticks]
                    if valid_ticks:
                        plt.yticks(ticks=valid_ticks, labels=valid_labels)
                        plt.ylim(min(valid_ticks) - 0.5, max(valid_ticks) + 0.5)
    if plot_made:
        plt.xlabel("Simulation Time (client, seconds)")
        plt.ylabel("Steering Decision (Main Server)")
        plt.title(f"Steering Service Main Server Decision Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(current_img_dir, "5_steering_decision_timeline.png")
        plt.savefig(plot_path)
        logger.debug(f"Gráfico 5 salvo: {plot_path}")
    else:
        logger.info("Dados insuficientes ou nenhuma decisão válida para o Gráfico 5.")
    plt.close()

    # Gráfico 6 - Performance Estimada do Servidor
    plot_made_rl_values = False
    if strategy_name_from_df in ["epsilon_greedy", "ucb1"]:
        if 'rl_values_json' in df.columns and not df['rl_values_json'].dropna().empty:
            df_values_raw = parse_json_column(df['rl_values_json'].dropna(), prefix="value_")
            if not df_values_raw.empty:
                df_values_transformed = df_values_raw.copy()
                y_axis_label = "Estimated Average Reward (e.g., 1000/latency)"
                plot_title_base = "Estimated Average Rewards per Server Over Time"
                legend_prefix = "Avg. Reward"

                if strategy_name_from_df == "epsilon_greedy":
                    for col in df_values_transformed.columns:
                        if col.startswith("value_"):
                            df_values_transformed[col] = df_values_transformed[col].apply(
                                lambda x: 1000.0 / x if pd.notna(x) and x > 1 else (1000.0 if pd.notna(x) and x == 1 else 0)
                            )
                
                # Alinhar índices antes de concatenar
                df_sim_time_for_concat = df.loc[df_values_transformed.index, 'sim_time_client'].reset_index(drop=True)
                df_values_transformed_for_concat = df_values_transformed.reset_index(drop=True)
                df_with_values = pd.concat([df_sim_time_for_concat, df_values_transformed_for_concat], axis=1)
                
                df_wv_unique = df_with_values.drop_duplicates(subset=['sim_time_client'], keep='first')

                plt.figure(figsize=(14, 7))
                value_cols_to_plot = [col for col in df_wv_unique.columns if col.startswith('value_')]
                if value_cols_to_plot:
                    for col_name in sorted(value_cols_to_plot, 
                                           key=lambda x: CACHE_SERVER_LABELS.get(x.replace('value_','').replace('_', '-'), x.replace('value_',''))):
                        
                        cache_col_suffix_with_underscore = col_name.replace('value_', '')
                        cache_key_for_lookup = cache_col_suffix_with_underscore.replace('_', '-')
                        
                        color_to_use = CACHE_COLORS.get(cache_key_for_lookup, 'grey')
                        label_for_plot = CACHE_SERVER_LABELS.get(cache_key_for_lookup, cache_key_for_lookup)
                        
                        if not df_wv_unique[col_name].isnull().all():
                            plt.plot(df_wv_unique['sim_time_client'], df_wv_unique[col_name],
                                    marker='.', linestyle='-', markersize=3, alpha=0.7,
                                    label=f"{legend_prefix} {label_for_plot}",
                                    color=color_to_use)
                            plot_made_rl_values = True
                if plot_made_rl_values:
                    plt.xlabel("Simulation Time (client, seconds)")
                    plt.ylabel(y_axis_label)
                    plt.title(f"{plot_title_base}\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
                    plt.legend(loc='best')
                    plt.grid(True, linestyle=':', alpha=0.7)
                    plt.tight_layout()
                    plot_path = os.path.join(current_img_dir, "6_estimated_server_values_timeline.png")
                    plt.savefig(plot_path)
                    logger.debug(f"Gráfico 6 salvo: {plot_path}")
                else:
                    logger.info("Nenhuma coluna de valor de RL válida para plotar no Gráfico 6.")
                plt.close()
            else:
                logger.info("DataFrame de valores de RL vazio após parse para Gráfico 6.")
        else:
            logger.info("Coluna 'rl_values_json' não encontrada ou vazia para Gráfico 6.")
    else:
        logger.info(f"Gráfico 6 (Valores RL) não aplicável para a estratégia: {strategy_name_from_df}")

    # Gráfico 7 - Contagens
    plot_made_rl_counts = False
    if 'rl_counts_json' in df.columns and not df['rl_counts_json'].dropna().empty:
        df_counts_raw = parse_json_column(df['rl_counts_json'].dropna(), prefix="count_")
        if not df_counts_raw.empty:
            df_sim_time_for_concat_counts = df.loc[df_counts_raw.index, 'sim_time_client'].reset_index(drop=True)
            df_counts_raw_for_concat = df_counts_raw.reset_index(drop=True)
            df_with_counts = pd.concat([df_sim_time_for_concat_counts, df_counts_raw_for_concat], axis=1)
            
            df_wc_unique = df_with_counts.drop_duplicates(subset=['sim_time_client'], keep='first')
            plt.figure(figsize=(14, 7))
            count_cols_to_plot = [col for col in df_wc_unique.columns if col.startswith('count_')]
            if count_cols_to_plot:
                for col_name in sorted(count_cols_to_plot, 
                                       key=lambda x: CACHE_SERVER_LABELS.get(x.replace('count_','').replace('_', '-'), x.replace('count_',''))):

                    cache_col_suffix_with_underscore = col_name.replace('count_', '')
                    cache_key_for_lookup = cache_col_suffix_with_underscore.replace('_', '-')
                    
                    color_to_use = CACHE_COLORS.get(cache_key_for_lookup, 'grey')
                    label_for_plot = CACHE_SERVER_LABELS.get(cache_key_for_lookup, cache_key_for_lookup)
                    
                    if not df_wc_unique[col_name].isnull().all():
                        plt.plot(df_wc_unique['sim_time_client'], df_wc_unique[col_name],
                                marker='.', linestyle='-', markersize=3, alpha=0.7,
                                label=f"Pulls for {label_for_plot}",
                                color=color_to_use)
                        plot_made_rl_counts = True
            if plot_made_rl_counts:
                plt.xlabel("Simulation Time (client, seconds)")
                plt.ylabel("Number of Times Server Selected (Pulls)")
                plt.title(f"Server Selection Counts (Exploration/Exploitation)\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
                plt.legend(loc='best')
                plt.grid(True, linestyle=':', alpha=0.7)
                plt.tight_layout()
                plot_path = os.path.join(current_img_dir, "7_selection_counts_timeline.png")
                plt.savefig(plot_path)
                logger.debug(f"Gráfico 7 salvo: {plot_path}")
            else:
                logger.info("Nenhuma coluna de contagem de RL válida para plotar no Gráfico 7.")
            plt.close()
        else:
            logger.info("DataFrame de contagens de RL vazio após parse para Gráfico 7.")
    else:
        logger.info("Coluna 'rl_counts_json' não encontrada ou vazia para Gráfico 7.")

    logger.info(f"Geração de gráficos para '{simulation_name}' concluída. Salvos em: {current_img_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from Content Steering simulation CSV logs.")
    parser.add_argument(
        "csv_argument", type=str, nargs='?', default=None,
        help="Filename or path to the CSV log file. Searched in ./, ./Logs/, ./Logs/Average/ if not absolute.")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging (DEBUG level).")
    args = parser.parse_args()

    handler = logging.StreamHandler()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    if not logger.handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)
        logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    if args.csv_argument:
        csv_to_process = args.csv_argument
        resolved_path = None
        paths_to_check = [
            csv_to_process, 
            os.path.join(os.getcwd(), csv_to_process), 
            os.path.join(DEFAULT_SIM_DATA_DIR, csv_to_process), 
            os.path.join(DEFAULT_SIM_DATA_DIR, "Average", csv_to_process) 
        ]
        if csv_to_process and not csv_to_process.lower().endswith(".csv"): 
            filename_with_ext = csv_to_process + ".csv"
            paths_to_check.extend([
                os.path.join(os.getcwd(), filename_with_ext),
                os.path.join(DEFAULT_SIM_DATA_DIR, filename_with_ext),
                os.path.join(DEFAULT_SIM_DATA_DIR, "Average", filename_with_ext)
            ])

        for path_option in paths_to_check:
            if os.path.exists(path_option) and os.path.isfile(path_option):
                resolved_path = os.path.abspath(path_option)
                break
        
        if resolved_path:
            logger.info(f"Processando arquivo: {os.path.basename(resolved_path)}")
            logger.debug(f"Caminho completo: {resolved_path}")
            generate_plots(resolved_path)
        else:
            logger.error(f"Arquivo '{args.csv_argument}' não encontrado nos diretórios de busca.")
            
    else:
        logger.info(f"Processando todos os arquivos CSV em {DEFAULT_SIM_DATA_DIR} e {os.path.join(DEFAULT_SIM_DATA_DIR, 'Average')}")
        processed_any = False
        
        for dirname in [DEFAULT_SIM_DATA_DIR, os.path.join(DEFAULT_SIM_DATA_DIR, 'Average')]:
            if os.path.isdir(dirname):
                for filename in os.listdir(dirname):
                    if filename.endswith(".csv"):
                        full_path = os.path.join(dirname, filename)
                        logger.info(f"Processando {filename} de {os.path.basename(dirname)}/")
                        generate_plots(full_path)
                        processed_any = True
            else:
                logger.warning(f"Diretório não encontrado: {dirname}")
        
        if not processed_any:
            logger.warning(f"Nenhum arquivo CSV encontrado nos diretórios de log para processar.")