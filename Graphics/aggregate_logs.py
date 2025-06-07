import pandas as pd
import os
import re
import argparse
import numpy as np
import json
import logging

logger = logging.getLogger("aggregate_logs")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM_DATA_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs")
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_SIM_DATA_DIR, "Average")

EXPECTED_MAIN_NUMERIC_COLS = [
    'sim_time_client', 'experienced_latency_ms',
    'experienced_latency_ms_CLIENT', 'experienced_latency_ms_ORACLE',
    'dynamic_best_server_latency'
]
EXPECTED_CATEGORICAL_COLS_FROM_FIRST_RUN = [
    'client_lat', 'client_lon', 'steering_decision_main_server', 'rl_strategy'
]
KNOWN_CACHE_SERVER_KEYS_UNDERSCORE = [
    "video_streaming_cache_1", "video_streaming_cache_2", "video_streaming_cache_3"
]
ACTUAL_CACHE_SERVER_NAMES_HYPHEN_AGG = [
    "video-streaming-cache-1", "video-streaming-cache-2", "video-streaming-cache-3"
]

# Extrai o nome e a latência do servidor otimizado dinamicamente de uma linha do DataFrame.
def find_dynamic_best_server_and_latency_for_agg(row_series):
    if pd.isna(row_series['all_servers_oracle_latency_json']):
        return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
    try:
        raw_latencies = json.loads(row_series['all_servers_oracle_latency_json'])
        server_latencies = {str(k).replace('_', '-'): v for k,v in raw_latencies.items()}
        valid_server_latencies = {
            s_name: lat
            for s_name, lat in server_latencies.items()
            if s_name in ACTUAL_CACHE_SERVER_NAMES_HYPHEN_AGG and isinstance(lat, (int, float))
        }
        if not valid_server_latencies:
            return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
        best_server_name = min(valid_server_latencies, key=valid_server_latencies.get)
        best_server_latency = valid_server_latencies[best_server_name]
        return pd.Series([best_server_name, best_server_latency], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
    except (json.JSONDecodeError, TypeError):
        return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
    except Exception:
        return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])

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
            else:
                temp_parsed_dicts.append({})
        except (json.JSONDecodeError, TypeError):
            logger.debug(f"Falha ao parsear JSON em parse_json_series_to_dataframe: '{str(json_str)[:70]}...'")
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

# Agrega múltiplos arquivos de log CSV de uma estratégia em um único arquivo CSV médio.
def aggregate_strategy_logs(strategy_name: str, suffix_pattern: str = "",
                            input_dir: str = DEFAULT_SIM_DATA_DIR,
                            output_dir: str = DEFAULT_OUTPUT_DIR):
    log_files = []
    base_pattern_str = f"log_{strategy_name}"
    if suffix_pattern:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str + suffix_pattern)}(_\d+)?\.csv$")
    else:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str)}(.*?)(_\d+)?\.csv$")
    logger.info(f"Agregando logs para estratégia: '{strategy_name}', padrão de sufixo de usuário: '{suffix_pattern}', entrada: '{input_dir}'")
    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)) and file_pattern.match(filename):
            if not suffix_pattern:
                match = file_pattern.match(filename)
                user_suffix_part = match.group(1) if match else ""
                if user_suffix_part and not user_suffix_part.replace('_','').isdigit():
                    logger.debug(f"Ignorando {filename} na agregação padrão devido ao sufixo de usuário '{user_suffix_part}'. Use --suffix_pattern='{user_suffix_part}' para agregá-lo.")
                    continue
            log_files.append(os.path.join(input_dir, filename))
    if not log_files:
        logger.warning(f"Nenhum log encontrado para '{strategy_name}{suffix_pattern}'. Arquivo agregado não será gerado.")
        return
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{len(log_files)} arquivo(s) para agregação: {', '.join(map(os.path.basename, log_files))}")
    all_main_dfs, all_rl_values_dfs, all_rl_counts_dfs, all_server_latencies_dfs = [], [], [], []
    min_duration = float('inf')
    first_run_categorical_data = {}
    for i, f_path in enumerate(log_files):
        try:
            df_run = pd.read_csv(f_path, na_filter=True)
            if df_run.empty or 'sim_time_client' not in df_run.columns:
                logger.warning(f"Arquivo {os.path.basename(f_path)} vazio ou sem 'sim_time_client'. Ignorando.")
                continue
            if df_run['sim_time_client'].isnull().any():
                logger.warning(f"Arquivo {os.path.basename(f_path)} contém valores NaN em 'sim_time_client'. Removendo essas linhas.")
                df_run.dropna(subset=['sim_time_client'], inplace=True)
                if df_run.empty:
                    logger.warning(f"Arquivo {os.path.basename(f_path)} ficou vazio após remover NaNs de 'sim_time_client'. Ignorando.")
                    continue
            if 'all_servers_oracle_latency_json' in df_run.columns:
                best_info = df_run.apply(find_dynamic_best_server_and_latency_for_agg, axis=1)
                df_run['dynamic_best_server_latency'] = best_info['dynamic_best_server_latency']
            else:
                df_run['dynamic_best_server_latency'] = np.nan
            df_run['sim_time_group'] = df_run['sim_time_client'].round().astype(int)
            max_time_this_run = df_run['sim_time_client'].max()
            if pd.notna(max_time_this_run):
                min_duration = min(min_duration, max_time_this_run)
            cols_to_avg = [col for col in EXPECTED_MAIN_NUMERIC_COLS if col in df_run.columns]
            all_main_dfs.append(df_run[['sim_time_group'] + cols_to_avg].copy())
            if i == 0:
                cols_cat = [col for col in EXPECTED_CATEGORICAL_COLS_FROM_FIRST_RUN if col in df_run.columns]
                if cols_cat:
                    temp_cat_df = df_run.groupby('sim_time_group')[cols_cat].first().reset_index()
                    for _, row in temp_cat_df.iterrows():
                        first_run_categorical_data[row['sim_time_group']] = {
                            col: row[col] for col in cols_cat if pd.notna(row[col])
                        }
            for json_col_name, target_list, prefix in [
                ('rl_values_json', all_rl_values_dfs, 'value_'),
                ('rl_counts_json', all_rl_counts_dfs, 'count_'),
                ('all_servers_oracle_latency_json', all_server_latencies_dfs, '')
            ]:
                if json_col_name in df_run.columns and not df_run[json_col_name].dropna().empty:
                    parsed_df = parse_json_series_to_dataframe(df_run[json_col_name], prefix=prefix)
                    if not parsed_df.empty:
                        parsed_df.loc[:, 'sim_time_group'] = df_run.loc[parsed_df.index, 'sim_time_group'].values
                        if prefix.startswith("value_") or prefix.startswith("count_"):
                             target_list.append(parsed_df.groupby('sim_time_group').last().reset_index())
                        else:
                             target_list.append(parsed_df)
        except Exception as e:
            logger.error(f"Erro ao processar {os.path.basename(f_path)}: {e}. Ignorando.", exc_info=True)

    if not all_main_dfs:
        logger.error("Nenhum dado principal válido encontrado para agregação.")
        return
    if min_duration == float('inf') and all_main_dfs:
         min_duration = pd.concat(all_main_dfs)['sim_time_client'].max() if all_main_dfs else 0
    elif not all_main_dfs:
        min_duration = 0
    logger.info(f"Agregando dados até a duração mínima comum de {min_duration:.2f}s.")
    combined_main_df = pd.concat(all_main_dfs)
    combined_main_df = combined_main_df[combined_main_df['sim_time_client'] <= min_duration]
    if combined_main_df.empty:
        logger.error("DataFrame principal combinado vazio após filtro de duração.")
        return
    agg_funcs_main = {col: (lambda x: np.nanmean(x.astype(float)) if pd.to_numeric(x, errors='coerce').notnull().any() else np.nan)
                      for col in EXPECTED_MAIN_NUMERIC_COLS if col in combined_main_df.columns and col != 'sim_time_client'}
    agg_funcs_main['sim_time_client'] = 'mean'
    aggregated_df = combined_main_df.groupby('sim_time_group', as_index=False).agg(agg_funcs_main)
    aggregated_df.rename(columns={'sim_time_group': 'sim_time_client_group'}, inplace=True)
    if first_run_categorical_data:
        cat_df = pd.DataFrame.from_dict(first_run_categorical_data, orient='index').reset_index()
        cat_df.rename(columns={'index': 'sim_time_client_group'}, inplace=True)
        aggregated_df = pd.merge(aggregated_df, cat_df, on='sim_time_client_group', how='left')
    for data_list, is_rl_data in [
        (all_server_latencies_dfs, False),
        (all_rl_values_dfs, True),
        (all_rl_counts_dfs, True)
    ]:
        if data_list:
            combined_json_df = pd.concat(data_list)
            if not combined_json_df.empty and 'sim_time_group' in combined_json_df.columns:
                combined_json_df['sim_time_group'] = pd.to_numeric(combined_json_df['sim_time_group'], errors='coerce')
                combined_json_df.dropna(subset=['sim_time_group'], inplace=True)
                combined_json_df['sim_time_group'] = combined_json_df['sim_time_group'].astype(int)
                combined_json_df = combined_json_df[combined_json_df['sim_time_group'].isin(aggregated_df['sim_time_client_group'])]
                if not combined_json_df.empty:
                    numeric_cols_to_avg = [col for col in combined_json_df.columns if col != 'sim_time_group' and pd.api.types.is_numeric_dtype(combined_json_df[col])]
                    if numeric_cols_to_avg:
                        avg_json_df = combined_json_df.groupby('sim_time_group')[numeric_cols_to_avg].mean().reset_index()
                        avg_json_df.rename(columns={'sim_time_group': 'sim_time_client_group'}, inplace=True)
                        if not is_rl_data:
                            server_latency_cols = [col for col in avg_json_df.columns if col in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE or col.replace("value_","").replace("count_","") in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE]
                            if server_latency_cols:
                                avg_json_df['all_servers_oracle_latency_json_AGG'] = avg_json_df.apply(
                                    lambda row: json.dumps({col: row[col] for col in server_latency_cols if pd.notna(row[col])}),axis=1)
                                df_to_merge = avg_json_df[['sim_time_client_group', 'all_servers_oracle_latency_json_AGG']].copy()
                                df_to_merge.rename(columns={'all_servers_oracle_latency_json_AGG': 'all_servers_oracle_latency_json'}, inplace=True)
                                if 'all_servers_oracle_latency_json' in aggregated_df.columns:
                                    aggregated_df = aggregated_df.drop(columns=['all_servers_oracle_latency_json'])
                                aggregated_df = pd.merge(aggregated_df, df_to_merge, on='sim_time_client_group', how='left')
                        else:
                            aggregated_df = pd.merge(aggregated_df, avg_json_df, on='sim_time_client_group', how='left')
                    elif not combined_json_df.drop(columns=['sim_time_group'], errors='ignore').empty:
                         logger.debug(f"Nenhuma coluna numérica para agregar em dados JSON (is_rl_data={is_rl_data}). Colunas presentes: {combined_json_df.columns.tolist()}")
    if aggregated_df.empty:
        logger.error("DataFrame agregado final está vazio.")
        return
    final_cols_order_base = [
        "sim_time_client", "client_lat", "client_lon",
        "experienced_latency_ms_CLIENT", "experienced_latency_ms_ORACLE", "experienced_latency_ms",
        "dynamic_best_server_latency",
        "all_servers_oracle_latency_json",
        "steering_decision_main_server", "rl_strategy"
    ]
    value_cols = sorted([col for col in aggregated_df.columns if col.startswith('value_')])
    count_cols = sorted([col for col in aggregated_df.columns if col.startswith('count_')])
    final_cols_order = final_cols_order_base + value_cols + count_cols
    current_cols_set = set(aggregated_df.columns)
    ordered_cols_set = set(final_cols_order)
    cols_to_exclude_from_remaining = {'sim_time_client_group', 'server_used_for_latency'}
    remaining_cols = sorted(list(current_cols_set - ordered_cols_set - cols_to_exclude_from_remaining))
    final_cols_order.extend(remaining_cols)
    final_cols_order = [col for col in final_cols_order if col in aggregated_df.columns]
    aggregated_df_final = aggregated_df[final_cols_order].copy()
    aggregated_df_final.sort_values(by='sim_time_client', inplace=True)
    output_base = f"log_{strategy_name}{suffix_pattern}_average"
    output_file, counter = os.path.join(output_dir, f"{output_base}.csv"), 1
    while os.path.exists(output_file):
        output_file = os.path.join(output_dir, f"{output_base}_{counter}.csv")
        counter += 1
    aggregated_df_final.to_csv(output_file, index=False, float_format='%.3f')
    logger.info(f"Arquivo CSV agregado salvo em: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrega múltiplos logs de simulação para uma estratégia.")
    parser.add_argument("strategy_name", type=str, help="Nome base da estratégia (e.g., ucb1, oracle_best_choice).")
    parser.add_argument("--suffix_pattern", type=str, default="", help="Padrão de sufixo opcional nos nomes dos arquivos (e.g., _runA). Se não fornecido, agrega arquivos como log_strategyname.csv, log_strategyname_1.csv, etc.")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_SIM_DATA_DIR, help=f"Diretório dos logs. Padrão: {DEFAULT_SIM_DATA_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Diretório para salvar agregado. Padrão: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Habilita logging DEBUG.")
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' if args.verbose else '%(levelname)s - %(message)s'
    if logger.handlers:
        logger.handlers[0].setFormatter(logging.Formatter(log_format))
    aggregate_strategy_logs(args.strategy_name, args.suffix_pattern, args.input_dir, args.output_dir)