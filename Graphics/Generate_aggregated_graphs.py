import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import logging

logger = logging.getLogger("plot_aggregated_logs")

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AGG_DATA_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs", "Average")
DEFAULT_IMG_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img")

CACHE_SERVER_LABELS = {
    "video-streaming-cache-1": "Cache 1 (BR)",
    "video-streaming-cache-2": "Cache 2 (CL)",
    "video-streaming-cache-3": "Cache 3 (CO)",
}

CACHE_COLORS = {
    "video-streaming-cache-1": "green",
    "video-streaming-cache-2": "darkorange",
    "video-streaming-cache-3": "dodgerblue",
}


def generate_plots_for_aggregated(csv_file_path):
    if not os.path.exists(csv_file_path):
        logger.error(f"Arquivo CSV agregado não encontrado em {csv_file_path}")
        return

    csv_filename_with_ext = os.path.basename(csv_file_path)
    simulation_name = os.path.splitext(csv_filename_with_ext)[0]

    current_img_dir = os.path.join(DEFAULT_IMG_DIR, simulation_name)
    os.makedirs(current_img_dir, exist_ok=True)

    logger.info(f"Lendo dados agregados de: {csv_filename_with_ext}")
    try:
        df = pd.read_csv(csv_file_path)
    except pd.errors.EmptyDataError:
        logger.warning(f"Arquivo CSV agregado {csv_filename_with_ext} está vazio ou é inválido.")
        return
    if df.empty:
        logger.warning(f"Arquivo CSV agregado {csv_filename_with_ext} está vazio.")
        return

    strategy_name_from_df = df['rl_strategy'].iloc[0] if 'rl_strategy' in df.columns and not df.empty else "N/A (Aggregated)"

    # --- Gráfico 1: Latência Média Experimentada ---
    plt.figure(figsize=(14, 7))
    if 'experienced_latency_ms' in df.columns and 'sim_time_client' in df.columns:
        df_plot = df.dropna(subset=['sim_time_client', 'experienced_latency_ms'])
        if not df_plot.empty:
            plt.plot(df_plot['sim_time_client'], df_plot['experienced_latency_ms'],
                     marker='.', linestyle='-', markersize=5, alpha=0.7, color='red',
                     label='Average Experienced Latency')
        else:
            logger.warning("Não há dados de latência para plotar no Gráfico 1.")
    else:
        logger.warning("Colunas 'experienced_latency_ms' ou 'sim_time_client' não encontradas para o Gráfico 1.")

    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Average Experienced Latency (ms)")
    plt.title(f"Average Client's Experienced Latency Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plot_path = os.path.join(current_img_dir, "1_avg_latency_timeline.png")
    plt.savefig(plot_path)
    plt.close()
    logger.debug(f"Gráfico 1 salvo: {plot_path}")

    # --- Gráfico 2: Média dos Valores Estimados de RL ---
    value_cols = sorted([col for col in df.columns if col.startswith('value_')])
    if value_cols and 'sim_time_client' in df.columns:
        plt.figure(figsize=(14, 7))
        y_label = "Average Estimated RL Value"
        legend_prefix = "Avg. Value"

        if "ucb1" in strategy_name_from_df.lower():
            y_label = "Average Estimated Reward (1000/Avg.Lat)"
            legend_prefix = "Avg. Reward"
        elif "epsilon_greedy" in strategy_name_from_df.lower():
            y_label = "Average Estimated Performance (1000/Avg.Lat)"
            legend_prefix = "Avg. Perf."

        plot_successful = False
        for col_name in value_cols: 
            cache_col_suffix_with_underscore = col_name.replace('value_', '')
            cache_key_for_lookup = cache_col_suffix_with_underscore.replace('_', '-') 

            color_to_use = CACHE_COLORS.get(cache_key_for_lookup, 'grey')
            label_for_plot = CACHE_SERVER_LABELS.get(cache_key_for_lookup, cache_key_for_lookup)

            df_plot = df.dropna(subset=['sim_time_client', col_name])
            if not df_plot.empty and not df_plot[col_name].isnull().all():
                plt.plot(df_plot['sim_time_client'], df_plot[col_name],
                         marker='.', linestyle='-', markersize=3, alpha=0.7,
                         label=f"{legend_prefix} {label_for_plot}",
                         color=color_to_use)
                plot_successful = True
            else:
                logger.debug(f"Sem dados válidos para a coluna de valor {col_name} no Gráfico 2.")
        
        if plot_successful:
            plt.xlabel("Simulation Time (client, seconds)")
            plt.ylabel(y_label)
            plt.title(f"Average Estimated RL Values/Performance per Server Over Time\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
            plt.legend(loc='best')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            plot_path = os.path.join(current_img_dir, "2_avg_rl_values_timeline.png")
            plt.savefig(plot_path)
            logger.debug(f"Gráfico 2 salvo: {plot_path}")
        else:
            logger.warning("Nenhum dado de valor de RL para plotar no Gráfico 2.")
        plt.close()
    elif not value_cols:
        logger.info("Nenhuma coluna 'value_*' encontrada para o Gráfico 2 (Valores de RL).")


    # --- Gráfico 3: Média das Contagens de RL ---
    count_cols = sorted([col for col in df.columns if col.startswith('count_')])
    if count_cols and 'sim_time_client' in df.columns:
        plt.figure(figsize=(14, 7))
        plot_successful = False
        for col_name in count_cols: 
            cache_col_suffix_with_underscore = col_name.replace('count_', '')
            cache_key_for_lookup = cache_col_suffix_with_underscore.replace('_', '-')

            color_to_use = CACHE_COLORS.get(cache_key_for_lookup, 'grey')
            label_for_plot = CACHE_SERVER_LABELS.get(cache_key_for_lookup, cache_key_for_lookup)

            df_plot = df.dropna(subset=['sim_time_client', col_name])
            if not df_plot.empty and not df_plot[col_name].isnull().all():
                plt.plot(df_plot['sim_time_client'], df_plot[col_name],
                         marker='.', linestyle='-', markersize=3, alpha=0.7,
                         label=f"Avg. Pulls for {label_for_plot}",
                         color=color_to_use)
                plot_successful = True
            else:
                logger.debug(f"Sem dados válidos para a coluna de contagem {col_name} no Gráfico 3.")

        if plot_successful:
            plt.xlabel("Simulation Time (client, seconds)")
            plt.ylabel("Average Number of Times Server Selected (Pulls)")
            plt.title(f"Average Server Selection Counts (Exploration/Exploitation)\nStrategy: {strategy_name_from_df} - File: {csv_filename_with_ext}")
            plt.legend(loc='best')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout()
            plot_path = os.path.join(current_img_dir, "3_avg_rl_counts_timeline.png")
            plt.savefig(plot_path)
            logger.debug(f"Gráfico 3 salvo: {plot_path}")
        else:
            logger.warning("Nenhum dado de contagem de RL para plotar no Gráfico 3.")
        plt.close()
    elif not count_cols:
        logger.info("Nenhuma coluna 'count_*' encontrada para o Gráfico 3 (Contagens de RL).")

    logger.info(f"Geração de gráficos para '{simulation_name}' concluída. Salvos em: {current_img_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs from AGGREGATED Content Steering simulation CSV logs.")
    parser.add_argument(
        "csv_filename",
        type=str,
        help="Filename of the AGGREGATED simulation CSV log file (e.g., log_ucb1_average.csv). Assumed to be in the default Logs/Average/ directory if not an absolute/relative path that exists."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_IMG_DIR,
        help=f"Base directory to save the graph image subfolders. Default: {DEFAULT_IMG_DIR}"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging (DEBUG level)."
    )
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

    csv_to_process = args.csv_filename
    if not os.path.isabs(csv_to_process) and not os.path.exists(csv_to_process):
        potential_path = os.path.join(DEFAULT_AGG_DATA_DIR, csv_to_process)
        if os.path.exists(potential_path):
            csv_to_process = potential_path
        elif not csv_to_process.lower().endswith(".csv"):
            filename_with_ext = csv_to_process + ".csv"
            potential_path_with_ext = os.path.join(DEFAULT_AGG_DATA_DIR, filename_with_ext)
            if os.path.exists(potential_path_with_ext):
                csv_to_process = potential_path_with_ext

    logger.info(f"Processando arquivo agregado: {os.path.basename(csv_to_process)}")
    try:
        abs_path_to_process = os.path.abspath(csv_to_process)
        logger.debug(f"Caminho absoluto usado: {abs_path_to_process}")
        if not os.path.exists(abs_path_to_process):
             logger.error(f"Arquivo final não encontrado em {abs_path_to_process}")
        else:
            generate_plots_for_aggregated(abs_path_to_process)
    except Exception as e:
        logger.error(f"Ocorreu um erro ao tentar processar o caminho do arquivo: {e}")