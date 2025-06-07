import pandas as pd
import os
import re
import argparse
import numpy as np
import json
import logging

logger = logging.getLogger("aggregate_logs")

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM_DATA_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_SIM_DATA_DIR, "Average")

CSV_HEADERS_EXPECTED = [
    "sim_time_client",
    "client_lat",
    "client_lon",
    "experienced_latency_ms",
    "steering_decision_main_server",
    "rl_strategy"
]


def parse_json_string_to_df(json_string, prefix=""):
    try:
        if pd.isna(json_string):
            return {}
        data_dict = json.loads(json_string) if isinstance(json_string, str) else json_string
        if isinstance(data_dict, dict):
            return {f"{prefix}{k.replace('-', '_')}": v for k, v in data_dict.items()}
        return {}
    except (json.JSONDecodeError, TypeError):
        logger.debug(f"Falha ao parsear string JSON: {str(json_string)[:100]}...")
        return {}

def aggregate_strategy_logs(strategy_name, suffix_pattern="", input_dir=DEFAULT_SIM_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    log_files = []
    base_pattern_str = f"log_{strategy_name}"
    if suffix_pattern:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str + suffix_pattern)}(_\d+)?\.csv$")
    else:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str)}(_[a-zA-Z][a-zA-Z0-9]*)?(_\d+)?\.csv$")

    logger.info(f"Iniciando agregação para estratégia: '{strategy_name}', padrão de sufixo: '{suffix_pattern}', diretório de entrada: '{input_dir}'")

    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)) and file_pattern.match(filename):
            log_files.append(os.path.join(input_dir, filename))

    if not log_files:
        logger.warning(f"Nenhum arquivo de log encontrado para '{strategy_name}{suffix_pattern}'. Nenhum arquivo agregado será gerado.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{len(log_files)} arquivo(s) encontrado(s) para agregação: {', '.join(map(os.path.basename, log_files))}")

    all_dfs_main_data = []
    all_dfs_rl_values = []
    all_dfs_rl_counts = []
    min_duration = float('inf')
    first_run_categorical_data_dict = {}

    for i, f_path in enumerate(log_files):
        try:
            df_run = pd.read_csv(f_path)
            if df_run.empty or 'sim_time_client' not in df_run.columns:
                logger.warning(f"Arquivo {os.path.basename(f_path)} está vazio ou não contém 'sim_time_client'. Ignorando.")
                continue

            df_run['sim_time_group'] = df_run['sim_time_client'].round().astype(int)
            current_max_time = df_run['sim_time_client'].max()
            if pd.notna(current_max_time):
                 min_duration = min(min_duration, current_max_time)

            relevant_main_cols = ['sim_time_group', 'sim_time_client', 'experienced_latency_ms']
            existing_main_cols = [col for col in relevant_main_cols if col in df_run.columns]
            all_dfs_main_data.append(df_run[existing_main_cols])

            if i == 0:
                cat_cols = ['client_lat', 'client_lon', 'steering_decision_main_server', 'rl_strategy']
                existing_cat_cols = [col for col in cat_cols if col in df_run.columns]
                if existing_cat_cols:
                    temp_cat_df = df_run.groupby('sim_time_group')[existing_cat_cols].first().reset_index()
                    for _, row in temp_cat_df.iterrows():
                        first_run_categorical_data_dict[row['sim_time_group']] = {
                            col: row[col] for col in existing_cat_cols
                        }

            if 'rl_values_json' in df_run.columns:
                temp_values_list = []
                for _, row_series in df_run.iterrows():
                    parsed_row = parse_json_string_to_df(row_series.get('rl_values_json'), prefix="value_")
                    if parsed_row:
                        parsed_row['sim_time_group'] = row_series['sim_time_group']
                        temp_values_list.append(parsed_row)
                if temp_values_list:
                    df_run_values = pd.DataFrame(temp_values_list)
                    if not df_run_values.empty:
                        df_run_values['sim_time_group'] = pd.to_numeric(df_run_values['sim_time_group'], errors='coerce')
                        df_run_values.dropna(subset=['sim_time_group'], inplace=True) 
                        df_run_values_grouped = df_run_values.groupby('sim_time_group').last().reset_index()
                        all_dfs_rl_values.append(df_run_values_grouped)

            if 'rl_counts_json' in df_run.columns:
                temp_counts_list = []
                for _, row_series in df_run.iterrows():
                    parsed_row = parse_json_string_to_df(row_series.get('rl_counts_json'), prefix="count_")
                    if parsed_row:
                        parsed_row['sim_time_group'] = row_series['sim_time_group']
                        temp_counts_list.append(parsed_row)
                if temp_counts_list:
                    df_run_counts = pd.DataFrame(temp_counts_list)
                    if not df_run_counts.empty:
                        df_run_counts['sim_time_group'] = pd.to_numeric(df_run_counts['sim_time_group'], errors='coerce')
                        df_run_counts.dropna(subset=['sim_time_group'], inplace=True)
                        df_run_counts_grouped = df_run_counts.groupby('sim_time_group').last().reset_index()
                        all_dfs_rl_counts.append(df_run_counts_grouped)
        except Exception as e:
            logger.error(f"Erro ao ler ou processar {os.path.basename(f_path)}: {e}. Ignorando arquivo.")

    if not all_dfs_main_data:
        logger.error("Nenhum dado válido encontrado nos arquivos para agregação.")
        return

    if min_duration == float('inf'):
        logger.warning("Não foi possível determinar uma duração mínima comum. O DataFrame agregado pode estar vazio ou incompleto.")
    else:
        logger.info(f"Processando dados até a duração mínima comum de {min_duration:.2f}s.")

    combined_main_df = pd.concat(all_dfs_main_data)
    if pd.notna(min_duration) and min_duration != float('inf'):
        combined_main_df = combined_main_df[combined_main_df['sim_time_client'] <= min_duration]

    if combined_main_df.empty:
        logger.error("DataFrame principal combinado está vazio após o filtro de duração. Não é possível agregar.")
        return

    agg_main_functions = {
        'sim_time_client': 'mean',
        'experienced_latency_ms': lambda x: np.nanmean(x) if not x.isnull().all() else np.nan
    }
    aggregated_df = combined_main_df.groupby('sim_time_group', as_index=False).agg(agg_main_functions)
    aggregated_df.rename(columns={'sim_time_group': 'sim_time_client_group'}, inplace=True)

    if first_run_categorical_data_dict:
        cat_df_from_dict = pd.DataFrame.from_dict(first_run_categorical_data_dict, orient='index').reset_index()
        cat_df_from_dict.rename(columns={'index': 'sim_time_client_group'}, inplace=True)
        aggregated_df = pd.merge(aggregated_df, cat_df_from_dict, on='sim_time_client_group', how='left')

    if all_dfs_rl_values:
        combined_rl_values_df = pd.concat(all_dfs_rl_values)
        if not combined_rl_values_df.empty:
            combined_rl_values_df['sim_time_group'] = pd.to_numeric(combined_rl_values_df['sim_time_group'], errors='coerce')
            combined_rl_values_df.dropna(subset=['sim_time_group'], inplace=True)
            
            combined_rl_values_df = combined_rl_values_df[combined_rl_values_df['sim_time_group'].isin(aggregated_df['sim_time_client_group'])]
            if not combined_rl_values_df.empty:
                avg_rl_values_df = combined_rl_values_df.groupby('sim_time_group').mean().reset_index()
                avg_rl_values_df.rename(columns={'sim_time_group': 'sim_time_client_group'}, inplace=True)
                aggregated_df = pd.merge(aggregated_df, avg_rl_values_df, on='sim_time_client_group', how='left')

    if all_dfs_rl_counts:
        combined_rl_counts_df = pd.concat(all_dfs_rl_counts)
        if not combined_rl_counts_df.empty:
            combined_rl_counts_df['sim_time_group'] = pd.to_numeric(combined_rl_counts_df['sim_time_group'], errors='coerce')
            combined_rl_counts_df.dropna(subset=['sim_time_group'], inplace=True)

            combined_rl_counts_df = combined_rl_counts_df[combined_rl_counts_df['sim_time_group'].isin(aggregated_df['sim_time_client_group'])]
            if not combined_rl_counts_df.empty:
                avg_rl_counts_df = combined_rl_counts_df.groupby('sim_time_group').mean().reset_index()
                avg_rl_counts_df.rename(columns={'sim_time_group': 'sim_time_client_group'}, inplace=True)
                aggregated_df = pd.merge(aggregated_df, avg_rl_counts_df, on='sim_time_client_group', how='left')

    if aggregated_df.empty:
        logger.error("DataFrame agregado final está vazio. Verifique os logs de entrada e o processo de agregação.")
        return

    final_cols_order = []
    present_cols = set(aggregated_df.columns)
    main_order = CSV_HEADERS_EXPECTED
    for col in main_order:
        if col in present_cols:
            final_cols_order.append(col)
            if col in present_cols: present_cols.remove(col)

    value_cols_in_agg = sorted([col for col in present_cols if col.startswith('value_')])
    final_cols_order.extend(value_cols_in_agg)
    for col in value_cols_in_agg:
        if col in present_cols: present_cols.remove(col)

    count_cols_in_agg = sorted([col for col in present_cols if col.startswith('count_')])
    final_cols_order.extend(count_cols_in_agg)
    for col in count_cols_in_agg:
        if col in present_cols: present_cols.remove(col)
    
    remaining_cols = sorted(list(present_cols - {'sim_time_client_group'}))
    final_cols_order.extend(remaining_cols)
    
    final_cols_order = [col for col in final_cols_order if col in aggregated_df.columns]
    aggregated_df = aggregated_df[final_cols_order]
    
    if 'sim_time_client_group' in aggregated_df.columns and 'sim_time_client_group' not in final_cols_order:
        aggregated_df.drop(columns=['sim_time_client_group'], inplace=True, errors='ignore')

    output_filename_base = f"log_{strategy_name}{suffix_pattern if suffix_pattern else ''}_average"
    counter = 1
    output_filename = os.path.join(output_dir, f"{output_filename_base}.csv")
    if os.path.exists(output_filename):
        while True:
            output_filename = os.path.join(output_dir, f"{output_filename_base}_{counter}.csv")
            if not os.path.exists(output_filename):
                break
            counter += 1

    aggregated_df.to_csv(output_filename, index=False, float_format='%.3f')
    logger.info(f"Arquivo CSV agregado salvo em: {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate multiple simulation logs for a given strategy.")
    parser.add_argument(
        "strategy_name", type=str,
        help="The base name of the strategy to aggregate logs for (e.g., ucb1, epsilon_greedy).")
    parser.add_argument(
        "--suffix_pattern", type=str, default="",
        help="An optional specific suffix pattern that log files must contain (e.g., _runA, _mobile_test). ")
    parser.add_argument(
        "--input_dir", type=str, default=DEFAULT_SIM_DATA_DIR,
        help=f"Directory containing the log files to be aggregated. Default: {DEFAULT_SIM_DATA_DIR}")
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the aggregated CSV file. Default: {DEFAULT_OUTPUT_DIR}")
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

    aggregate_strategy_logs(args.strategy_name, args.suffix_pattern, args.input_dir, args.output_dir)