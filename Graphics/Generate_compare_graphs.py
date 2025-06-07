import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import re
import argparse
import logging
import numpy as np

logger = logging.getLogger("compare_strategies")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AVERAGE_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs", "Average")
DEFAULT_IMG_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img")

STRATEGY_STYLES = {
    "ucb1": {"color": "tab:blue", "label": "UCB1"},
    "epsilon_greedy": {"color": "tab:green", "label": "Epsilon Greedy"},
    "random": {"color": "tab:red", "label": "Random"},
    "oracle_best_choice": {"color": "tab:purple", "label": "Optimal Strategy"},
    "no_steering": {"color": "tab:brown", "label": "No Steering"},
    "default": {"color": "tab:grey", "label": "Unknown"}
}
KNOWN_STRATEGY_KEYS = list(STRATEGY_STYLES.keys())

# Extrai o nome da estratégia do nome do arquivo de log ou de uma coluna do DataFrame.
def extract_strategy_name(filename_no_ext: str, df_column_data: pd.Series = None) -> str:
    if df_column_data is not None and not df_column_data.empty:
        first_valid_strategy = df_column_data.dropna().iloc[0] if not df_column_data.dropna().empty else None
        if first_valid_strategy and isinstance(first_valid_strategy, str):
            normalized_strategy = first_valid_strategy.lower().replace(" ", "_")
            for known_key in KNOWN_STRATEGY_KEYS:
                if normalized_strategy.startswith(known_key):
                    return known_key
            if normalized_strategy != "n/a_(aggregated)":
                 return normalized_strategy
    for known_key in KNOWN_STRATEGY_KEYS:
        if filename_no_ext.startswith(f"log_{known_key}"):
            match_key = re.match(rf"log_({known_key})", filename_no_ext)
            if match_key:
                return known_key
    match = re.match(r"log_([a-zA-Z0-9_]+?)_average", filename_no_ext)
    if match:
        return match.group(1)
    if filename_no_ext.startswith("log_") and "_average" in filename_no_ext:
        temp_name = filename_no_ext[4:]
        return temp_name.split("_average")[0]
    return "Unknown"

# Aplica formatação padronizada ao gráfico de comparação de estratégias.
def format_comparison_plot(ax, title, xlabel, ylabel, legend_loc='best', custom_legend_handles=None, custom_legend_labels=None):
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    if custom_legend_handles and custom_legend_labels:
        ax.legend(custom_legend_handles, custom_legend_labels, loc=legend_loc, fontsize=11)
    elif ax.has_data() and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)
    has_plotted_data_non_negative = False
    if ax.has_data():
        for line in ax.get_lines():
            ydata = line.get_ydata()
            if isinstance(ydata, (pd.Series, np.ndarray)) and ydata.size > 0:
                numeric_ydata = pd.to_numeric(ydata, errors='coerce')
                if np.any(numeric_ydata[~np.isnan(numeric_ydata)] >= 0):
                    has_plotted_data_non_negative = True
                    break
    if has_plotted_data_non_negative:
        ax.set_ylim(bottom=0)
    plt.tight_layout(pad=1.5)

# Gera um gráfico comparando uma métrica de latência média entre diferentes estratégias.
def plot_average_latency_comparison(average_logs_dir: str, output_dir: str = DEFAULT_IMG_DIR, metric_to_plot: str = 'experienced_latency_ms'):
    fig, ax = plt.subplots(figsize=(15, 7))
    strategies_plotted_count = 0
    window_size = 5
    legend_handles, legend_labels = [], []
    logger.info(f"Procurando arquivos CSV agregados em: {average_logs_dir} para plotar '{metric_to_plot}'")
    if not os.path.isdir(average_logs_dir):
        logger.error(f"Diretório de logs agregados não encontrado: {average_logs_dir}")
        plt.close(fig)
        return
    y_axis_label = "Average Latency (ms)"
    if metric_to_plot == 'experienced_latency_ms':
        plot_title_metric_part = "Average Chosen Server Latency"
    elif metric_to_plot == 'experienced_latency_ms_CLIENT':
        plot_title_metric_part = "Average Client Measured Latency"
    elif metric_to_plot == 'dynamic_best_server_latency':
        plot_title_metric_part = "Average Optimal Server Latency"
    else:
        plot_title_metric_part = f"Average {metric_to_plot.replace('_', ' ').title()}"
    main_plot_title = f"Comparison of {plot_title_metric_part}\nAcross Steering Strategies"
    for filename in sorted(os.listdir(average_logs_dir)):
        if not (filename.startswith("log_") and "_average" in filename and filename.endswith(".csv")):
            continue
        agg_file_path = os.path.join(average_logs_dir, filename)
        logger.debug(f"Processando arquivo agregado para comparação: {filename}")
        try:
            df_agg = pd.read_csv(agg_file_path)
            if df_agg.empty or 'sim_time_client' not in df_agg.columns or metric_to_plot not in df_agg.columns:
                logger.warning(f"Arquivo {filename} vazio ou faltando colunas '{metric_to_plot}' ou 'sim_time_client'. Ignorando.")
                continue
            df_agg = df_agg.sort_values(by='sim_time_client').copy()
            filename_no_ext = os.path.splitext(filename)[0]
            strategy_col_data = df_agg['rl_strategy'] if 'rl_strategy' in df_agg.columns else None
            base_strategy_name = extract_strategy_name(filename_no_ext, strategy_col_data)
            style = STRATEGY_STYLES.get(base_strategy_name, STRATEGY_STYLES["default"])
            current_legend_label = style['label']
            if style['label'] == "Unknown" and base_strategy_name != "Unknown":
                current_legend_label = base_strategy_name.replace('_', ' ').title()
            df_plot_data = df_agg.dropna(subset=['sim_time_client', metric_to_plot]).copy()
            if not df_plot_data.empty:
                if len(df_plot_data[metric_to_plot]) >= window_size:
                    df_plot_data.loc[:, 'metric_ma_plot'] = df_plot_data[metric_to_plot].rolling(
                        window=window_size, center=True, min_periods=1).mean()
                    line, = ax.plot(df_plot_data['sim_time_client'], df_plot_data['metric_ma_plot'],
                                     linestyle='-', linewidth=2, alpha=0.9, color=style["color"])
                else:
                     line, = ax.plot(df_plot_data['sim_time_client'], df_plot_data[metric_to_plot],
                                     linestyle='-', linewidth=2, alpha=0.9, color=style["color"])
                if not any(lbl == current_legend_label for lbl in legend_labels):
                    legend_handles.append(line)
                    legend_labels.append(current_legend_label)
                strategies_plotted_count += 1
            else:
                logger.warning(f"Nenhum dado de '{metric_to_plot}' válido em {filename} para plotar.")
        except Exception as e:
            logger.error(f"Erro ao processar arquivo {filename} para comparação: {e}", exc_info=True)
    if strategies_plotted_count == 0:
        logger.warning(f"Nenhuma estratégia foi plotada para a métrica '{metric_to_plot}'. Verifique os arquivos de log agregados.")
        plt.close(fig)
        return
    format_comparison_plot(ax,
                           main_plot_title,
                           "Average Simulation Time (s)",
                           y_axis_label,
                           legend_loc='upper right',
                           custom_legend_handles=legend_handles,
                           custom_legend_labels=legend_labels)
    output_filename_base = metric_to_plot.replace("experienced_latency_ms", "latency")
    if "CLIENT" in output_filename_base:
        output_filename_base = output_filename_base.replace("_CLIENT", "_client")
    if "dynamic_best_server_latency" in output_filename_base:
        output_filename_base = "optimal_latency"
    output_filename = f"all_strategies_{output_filename_base}_comparison.png"
    plot_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    logger.info(f"Gráfico de comparação ({metric_to_plot}) salvo em: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara métricas de latência média de logs agregados de diferentes estratégias.")
    parser.add_argument("--agg_dir", type=str, default=DEFAULT_AVERAGE_LOGS_DIR,
                        help=f"Diretório com arquivos CSV agregados. Padrão: {DEFAULT_AVERAGE_LOGS_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_IMG_DIR,
                        help=f"Diretório para salvar o gráfico. Padrão: {DEFAULT_IMG_DIR}")
    parser.add_argument("--metric", type=str, default="experienced_latency_ms",
                        choices=["experienced_latency_ms", "experienced_latency_ms_CLIENT", "dynamic_best_server_latency"],
                        help="Métrica de latência a ser plotada para comparação.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Habilita logging DEBUG.")
    args = parser.parse_args()
    if args.verbose: logger.setLevel(logging.DEBUG)
    if logger.handlers:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if args.verbose else '%(levelname)s - %(message)s'
        logger.handlers[0].setFormatter(logging.Formatter(log_format))
    logger.info(f"Iniciando comparação de estratégias (métrica: {args.metric}) a partir de logs em: {args.agg_dir}")
    plot_average_latency_comparison(args.agg_dir, args.output_dir, metric_to_plot=args.metric)