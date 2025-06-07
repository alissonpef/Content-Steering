import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import argparse
import re
import json
import logging
import numpy as np

logger = logging.getLogger("plot_aggregated_logs")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGG_DATA_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs", "Average")
DEFAULT_IMG_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img")

SERVER_DISPLAY_NAMES = {
    "video-streaming-cache-1": "Cache Server 1 (BR)",
    "video-streaming-cache-2": "Cache Server 2 (CL)",
    "video-streaming-cache-3": "Cache Server 3 (CO)",
}
SERVER_COLORS = {
    "video-streaming-cache-1": "tab:green",
    "video-streaming-cache-2": "tab:orange",
    "video-streaming-cache-3": "tab:blue",
}
KNOWN_CACHE_SERVER_KEYS_UNDERSCORE = [
    "video_streaming_cache_1", "video_streaming_cache_2", "video_streaming_cache_3"
]

# Converte uma Série de strings JSON em um DataFrame, aplicando prefixo às colunas se especificado.
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
            logger.debug(f"Falha ao parsear JSON em parse_json_series_to_dataframe (agg_graphs): '{str(json_str)[:70]}...'")
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

# Aplica formatação padronizada aos gráficos agregados.
def format_plot_aggregated(ax, title, xlabel, ylabel, legend_loc='best', y_log_scale=False, custom_legend_handles=None, custom_legend_labels=None):
    ax.set_title(title, fontsize=14)
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
    elif ax.has_data() and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6, which='major')
    if y_log_scale and ax.has_data(): ax.grid(True, linestyle=':', alpha=0.3, which='minor')
    plt.tight_layout()

# Gera um conjunto de gráficos a partir de um arquivo CSV de log agregado.
def generate_plots_for_aggregated(csv_file_path: str):
    if not os.path.exists(csv_file_path):
        logger.error(f"Arquivo CSV agregado não encontrado: {csv_file_path}")
        return
    csv_filename_no_ext = os.path.splitext(os.path.basename(csv_file_path))[0]
    current_img_dir = os.path.join(DEFAULT_IMG_DIR, csv_filename_no_ext)
    os.makedirs(current_img_dir, exist_ok=True)
    logger.info(f"Lendo dados agregados de: {csv_filename_no_ext}.csv")
    try:
        df_agg = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Arquivo CSV agregado {csv_filename_no_ext}.csv está vazio.")
        return
    if df_agg.empty:
        logger.warning(f"Arquivo CSV agregado {csv_filename_no_ext}.csv está vazio.")
        return
    strategy_name_from_df = "N/A (Aggregated)"
    if 'rl_strategy' in df_agg.columns and not df_agg['rl_strategy'].empty and pd.notna(df_agg['rl_strategy'].iloc[0]):
        strategy_name_from_df = df_agg['rl_strategy'].iloc[0]
    else:
        match = re.match(r"log_([a-zA-Z0-9_]+?)_average", csv_filename_no_ext)
        if match: strategy_name_from_df = match.group(1)
    strategy_display_name = strategy_name_from_df.replace("_", " ").title()

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    plot_made_g1 = False
    legend1_handles, legend1_labels = [], []
    if 'experienced_latency_ms' in df_agg.columns and 'sim_time_client' in df_agg.columns:
        df_plot_chosen_oracle = df_agg.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_plot_chosen_oracle.empty:
            line_chosen, = ax1.plot(df_plot_chosen_oracle['sim_time_client'], df_plot_chosen_oracle['experienced_latency_ms'],
                                    marker='.', linestyle='-', markersize=4, alpha=0.8, color='darkblue')
            legend1_handles.append(line_chosen)
            legend1_labels.append('Avg. Chosen Server Latency')
            plot_made_g1 = True
    if 'dynamic_best_server_latency' in df_agg.columns and 'sim_time_client' in df_agg.columns:
        df_plot_optimal_oracle = df_agg.dropna(subset=['sim_time_client', 'dynamic_best_server_latency'])
        if not df_plot_optimal_oracle.empty:
            line_optimal, = ax1.plot(df_plot_optimal_oracle['sim_time_client'], df_plot_optimal_oracle['dynamic_best_server_latency'],
                                     marker='.', linestyle='--', markersize=4, alpha=0.7, color='tab:red')
            legend1_handles.append(line_optimal)
            legend1_labels.append('Avg. Optimal Server Latency')
            plot_made_g1 = True
    elif plot_made_g1:
        logger.warning("Coluna 'dynamic_best_server_latency' não encontrada no arquivo agregado para o Gráfico 1.")
    if plot_made_g1:
        format_plot_aggregated(ax1, f"Average Chosen Server Latency vs Optimal Latency\nStrategy: {strategy_display_name}",
                               "Average Simulation Time (s)", "Average Latency (ms)", legend_loc='upper right',
                               custom_legend_handles=legend1_handles, custom_legend_labels=legend1_labels)
        plt.savefig(os.path.join(current_img_dir, "1_avg_latency_chosen_vs_optimal.png"))
    else:
        logger.warning("Dados insuficientes para Gráfico 1 (Latência Agregada Escolhida vs Ótima).")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    plot_made_g2 = False
    legend2_handles, legend2_labels = [], []
    y_label_g2 = "Average Estimated RL Value"
    value_cols_agg = sorted([col for col in df_agg.columns if col.startswith('value_') and any(k_u in col for k_u in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE)])
    if value_cols_agg and 'sim_time_client' in df_agg.columns:
        if "epsilon_greedy" in strategy_name_from_df.lower():
            y_label_g2 = "Average Estimated Reward (Higher is Better)"
        elif "ucb1" in strategy_name_from_df.lower():
            y_label_g2 = "Average Estimated Reward (UCB1)"
        for col_name in value_cols_agg:
            server_key_u = col_name.replace('value_', '')
            server_key_h = server_key_u.replace('_', '-')
            color = SERVER_COLORS.get(server_key_h, 'grey')
            label_text = SERVER_DISPLAY_NAMES.get(server_key_h, server_key_u)
            df_subset = df_agg.dropna(subset=['sim_time_client', col_name]).copy()
            if not df_subset.empty:
                if "epsilon_greedy" in strategy_name_from_df.lower():
                    df_subset.loc[:, col_name] = df_subset[col_name].apply(lambda x: 1000.0 / x if isinstance(x, (int,float)) and x > 0 else 0.0)
                line, = ax2.plot(df_subset['sim_time_client'], df_subset[col_name],
                                 marker='.', linestyle='-', markersize=3, alpha=0.8, color=color)
                if not any(l == label_text for l in legend2_labels):
                    legend2_handles.append(line)
                    legend2_labels.append(label_text)
                plot_made_g2 = True
        if plot_made_g2:
            format_plot_aggregated(ax2, f"Average RL Algorithm's Estimated Server Values\nStrategy: {strategy_display_name}",
                                   "Average Simulation Time (s)", y_label_g2, legend_loc='upper right',
                                   custom_legend_handles=legend2_handles, custom_legend_labels=legend2_labels)
            plt.savefig(os.path.join(current_img_dir, "2_avg_rl_estimated_values.png"))
        else: logger.warning("Nenhuma coluna 'value_*' válida para Gráfico 2 (Agregado).")
    elif any(s_rl in strategy_name_from_df.lower() for s_rl in ["epsilon_greedy", "ucb1"]):
        logger.info(f"Nenhuma coluna 'value_*' encontrada para Gráfico 2 (Agregado) para estratégia {strategy_display_name}.")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    plot_made_g3 = False
    legend3_handles, legend3_labels = [], []
    count_cols_agg = sorted([col for col in df_agg.columns if col.startswith('count_') and any(k_u in col for k_u in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE)])
    if count_cols_agg and 'sim_time_client' in df_agg.columns:
        for col_name in count_cols_agg:
            server_key_u = col_name.replace('count_', '')
            server_key_h = server_key_u.replace('_', '-')
            color = SERVER_COLORS.get(server_key_h, 'grey')
            label_text = SERVER_DISPLAY_NAMES.get(server_key_h, server_key_u)
            df_subset = df_agg.dropna(subset=['sim_time_client', col_name])
            if not df_subset.empty:
                line, = ax3.plot(df_subset['sim_time_client'], df_subset[col_name],
                                 marker='.', linestyle='-', markersize=3, alpha=0.8, color=color)
                if not any(l == label_text for l in legend3_labels):
                    legend3_handles.append(line)
                    legend3_labels.append(label_text)
                plot_made_g3 = True
        if plot_made_g3:
            format_plot_aggregated(ax3, f"Average RL Algorithm's Server Selection Counts\nStrategy: {strategy_display_name}",
                                   "Average Simulation Time (s)", "Average Number of Selections (Pulls)", legend_loc='upper left',
                                   custom_legend_handles=legend3_handles, custom_legend_labels=legend3_labels)
            plt.savefig(os.path.join(current_img_dir, "3_avg_rl_selection_counts.png"))
        else: logger.warning("Nenhuma coluna 'count_*' válida para Gráfico 3 (Agregado).")
    elif any(s_rl in strategy_name_from_df.lower() for s_rl in ["epsilon_greedy", "ucb1"]):
        logger.info(f"Nenhuma coluna 'count_*' encontrada para Gráfico 3 (Agregado) para estratégia {strategy_display_name}.")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    plot_made_g4 = False
    legend4_handles, legend4_labels = [], []
    if 'all_servers_oracle_latency_json' in df_agg.columns and not df_agg['all_servers_oracle_latency_json'].dropna().empty:
        df_all_lat_agg_parsed = parse_json_series_to_dataframe(df_agg['all_servers_oracle_latency_json'].dropna(), prefix="")
        if not df_all_lat_agg_parsed.empty:
            df_all_lat_agg_with_time = pd.concat([
                df_agg.loc[df_all_lat_agg_parsed.index, 'sim_time_client'], df_all_lat_agg_parsed
            ], axis=1).reset_index(drop=True)
            if not df_all_lat_agg_with_time.empty and 'sim_time_client' in df_all_lat_agg_with_time.columns:
                all_oracle_cols = sorted([
                    col for col in df_all_lat_agg_with_time.columns
                    if col != 'sim_time_client' and col in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE
                ])
                for server_col_u in all_oracle_cols:
                    server_col_h = server_col_u.replace('_', '-')
                    color = SERVER_COLORS.get(server_col_h, 'grey')
                    label_text = SERVER_DISPLAY_NAMES.get(server_col_h, server_col_u)
                    df_subset = df_all_lat_agg_with_time.dropna(subset=['sim_time_client', server_col_u])
                    if not df_subset.empty:
                        line, = ax4.plot(df_subset['sim_time_client'], df_subset[server_col_u],
                                         marker='.', linestyle='-', markersize=2, alpha=0.7, color=color)
                        if not any(l == label_text for l in legend4_labels):
                            legend4_handles.append(line)
                            legend4_labels.append(label_text)
                        plot_made_g4 = True
                if plot_made_g4:
                    format_plot_aggregated(ax4, f"Average Simulated Latency Landscape for All Servers\nStrategy: {strategy_display_name}",
                                           "Average Simulation Time (s)", "Average Simulated Latency (ms)", legend_loc='upper right',
                                           custom_legend_handles=legend4_handles, custom_legend_labels=legend4_labels)
                    plt.savefig(os.path.join(current_img_dir, "4_avg_all_servers_oracle_latency.png"))
                else: logger.info("Nenhuma coluna válida para Gráfico 4 (Agregado - Latência Oráculo Todos).")
            else: logger.info("DataFrame vazio ou sem sim_time_client para Gráfico 4 (Agregado - Latência Oráculo Todos).")
        else: logger.info("DataFrame 'all_servers_oracle_latency_json' (agregado) vazio após parse para Gráfico 4.")
    else: logger.info("Coluna 'all_servers_oracle_latency_json' não encontrada para Gráfico 4 (Agregado).")
    plt.close(fig4)

    logger.info(f"Geração de gráficos agregados para '{csv_filename_no_ext}' concluída. Salvos em: {current_img_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera gráficos de logs CSV AGREGADOS de simulação.")
    parser.add_argument("csv_filename", type=str, help="Nome do arquivo CSV agregado (e.g., log_ucb1_average.csv).")
    parser.add_argument("--output_dir",type=str,default=DEFAULT_IMG_DIR,help=f"Diretório base para salvar gráficos. Padrão: {DEFAULT_IMG_DIR}")
    parser.add_argument("--verbose","-v",action="store_true",help="Habilita logging DEBUG.")
    args = parser.parse_args()
    if args.verbose: logger.setLevel(logging.DEBUG)
    if logger.handlers:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if args.verbose else '%(levelname)s - %(message)s'
        logger.handlers[0].setFormatter(logging.Formatter(log_format))
    csv_path = args.csv_filename
    if not os.path.isabs(csv_path) and not os.path.exists(csv_path):
        potential_path = os.path.join(DEFAULT_AGG_DATA_DIR, csv_path)
        if os.path.exists(potential_path):
            csv_path = potential_path
        elif not csv_path.lower().endswith(".csv"):
            potential_path_with_ext = os.path.join(DEFAULT_AGG_DATA_DIR, csv_path + ".csv")
            if os.path.exists(potential_path_with_ext):
                csv_path = potential_path_with_ext
    logger.info(f"Processando arquivo agregado: {os.path.basename(csv_path)}")
    abs_path = os.path.abspath(csv_path)
    if not os.path.exists(abs_path):
        logger.error(f"Arquivo final não encontrado: {abs_path}")
    else:
        generate_plots_for_aggregated(abs_path)