import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import argparse
import numpy as np
import logging

logger = logging.getLogger("compare_strategies")

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AVERAGE_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs", "Average")
DEFAULT_IMG_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img")

STRATEGY_STYLES = {
    "ucb1": {"color": "blue", "label": "UCB1"},
    "epsilon_greedy": {"color": "green", "label": "Epsilon-Greedy"},
    "random": {"color": "red", "label": "Random"},
    "no_steering": {"color": "purple", "label": "No Steering"},
    "default": {"color": "black", "label": "Unknown"}
}

def plot_average_latency_comparison(average_logs_dir, output_dir=DEFAULT_IMG_DIR):
    plt.figure(figsize=(16, 8))
    all_strategies_plotted = False
    window_size_comparison = 5

    logger.info(f"Procurando arquivos CSV agregados em: {average_logs_dir}")
    found_files_for_plot = []

    if not os.path.isdir(average_logs_dir):
        logger.error(f"Diretório de logs agregados não encontrado: {average_logs_dir}")
        return

    for filename in os.listdir(average_logs_dir):
        if filename.endswith("_average.csv") or \
           (filename.startswith("log_") and "_average_" in filename and filename.endswith(".csv")):
            found_files_for_plot.append(os.path.join(average_logs_dir, filename))
    
    if not found_files_for_plot:
        logger.warning(f"Nenhum arquivo CSV agregado encontrado em {average_logs_dir}")
        return
        
    logger.info(f"Arquivos agregados encontrados para comparação: {', '.join(map(os.path.basename, found_files_for_plot))}")

    for agg_file_path in found_files_for_plot:
        try:
            df_agg = pd.read_csv(agg_file_path)
            if df_agg.empty or 'sim_time_client' not in df_agg.columns or 'experienced_latency_ms' not in df_agg.columns:
                logger.warning(f"Arquivo {os.path.basename(agg_file_path)} está vazio ou faltando colunas essenciais. Ignorando.")
                continue

            df_agg = df_agg.sort_values(by='sim_time_client').copy()

            filename_no_ext = os.path.splitext(os.path.basename(agg_file_path))[0]
            strategy_name_from_file = "Unknown"
            
            known_strategies = ["ucb1", "epsilon_greedy", "random", "no_steering"]
            for s_name in known_strategies:
                pattern_to_match = f"log_{s_name}"
                if filename_no_ext.startswith(pattern_to_match):
                    rest_of_filename = filename_no_ext[len(pattern_to_match):]
                    if rest_of_filename.startswith("_average") or \
                       (rest_of_filename.startswith("_") and "_average" in rest_of_filename) or \
                       not rest_of_filename :
                        strategy_name_from_file = s_name
                        break
            
            if strategy_name_from_file == "Unknown" and 'rl_strategy' in df_agg.columns and not df_agg['rl_strategy'].empty:
                temp_strat_name = df_agg['rl_strategy'].iloc[0]
                if temp_strat_name and isinstance(temp_strat_name, str) and temp_strat_name.lower() != "n/a (aggregated)":
                    strategy_name_from_file = temp_strat_name

            if strategy_name_from_file == "Unknown":
                match = re.match(r"log_([a-zA-Z0-9_]+?)_average", filename_no_ext)
                if match:
                    strategy_name_from_file = match.group(1)
                elif filename_no_ext.startswith("log_") and filename_no_ext.endswith("_average"):
                     strategy_name_from_file = filename_no_ext[4:-8] 

            style = STRATEGY_STYLES.get(strategy_name_from_file, STRATEGY_STYLES["default"])
            label_for_legend = style['label']
            
            if style['label'] == "Unknown" and strategy_name_from_file != "Unknown":
                label_for_legend = strategy_name_from_file.replace('_', ' ').title()
            
            df_plot = df_agg.dropna(subset=['sim_time_client', 'experienced_latency_ms']).copy()
            
            if not df_plot.empty:
                plt.plot(df_plot['sim_time_client'], df_plot['experienced_latency_ms'],
                         marker='.', linestyle='None', markersize=5, alpha=0.4, color=style["color"])
                
                if len(df_plot['experienced_latency_ms']) >= window_size_comparison:
                    df_plot.loc[:, 'latency_ma'] = df_plot['experienced_latency_ms'].rolling(window=window_size_comparison, center=True, min_periods=1).mean()
                    plt.plot(df_plot['sim_time_client'], df_plot['latency_ma'],
                             linestyle='-', linewidth=1.5, alpha=0.9, color=style["color"],
                             label=label_for_legend)
                else: 
                     plt.plot(df_plot['sim_time_client'], df_plot['experienced_latency_ms'],
                             linestyle='-', linewidth=1.5, alpha=0.9, color=style["color"],
                             label=label_for_legend)
                all_strategies_plotted = True
            else:
                logger.warning(f"Não há dados de latência válidos no arquivo {os.path.basename(agg_file_path)}.")

        except Exception as e:
            logger.error(f"Erro ao processar o arquivo {os.path.basename(agg_file_path)}: {e}")

    if not all_strategies_plotted:
        logger.warning("Nenhuma estratégia foi plotada. Verifique os arquivos de log agregados.")
        plt.close()
        return

    plt.xlabel("Simulation Time (client, seconds)")
    plt.ylabel("Average Experienced Latency (ms)")
    plt.title("Comparison of Average Client's Experienced Latency Across Strategies")
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.ylim(bottom=0)
    plt.tight_layout()

    output_filename = "all_strategies_latency_comparison.png"
    plot_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Gráfico de comparação salvo em: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare average latencies from aggregated logs of different strategies.")
    parser.add_argument(
        "--agg_dir", type=str, default=DEFAULT_AVERAGE_LOGS_DIR,
        help=f"Directory containing the aggregated CSV files. Default: {DEFAULT_AVERAGE_LOGS_DIR}")
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_IMG_DIR,
        help=f"Directory to save the comparison graph. Default: {DEFAULT_IMG_DIR}")
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

    logger.info(f"Comparando latências dos arquivos em: {args.agg_dir}")
    plot_average_latency_comparison(args.agg_dir, args.output_dir)