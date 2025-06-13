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
MAX_AGGREGATION_TIME_SECONDS = 150

EXPECTED_MAIN_NUMERIC_COLS = [
    'sim_time_client', 'experienced_latency_ms',
    'experienced_latency_ms_CLIENT', 'experienced_latency_ms_ORACLE',
    'dynamic_best_server_latency', 'gamma_value'
]
EXPECTED_CATEGORICAL_COLS_FROM_FIRST_RUN = [
    'client_lat', 'client_lon', 'steering_decision_main_server', 'rl_strategy'
]
KNOWN_CACHE_SERVER_KEYS_UNDERSCORE = [
    "video_streaming_cache_1", "video_streaming_cache_2", "video_streaming_cache_3"
]

def find_dynamic_best_server_and_latency_for_agg(row_series):
    if pd.isna(row_series['all_servers_oracle_latency_json']):
        return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
    try:
        raw_latencies = json.loads(row_series['all_servers_oracle_latency_json'])
        valid_server_latencies = {
            s_key_underscore: lat
            for s_key_underscore, lat in raw_latencies.items()
            if s_key_underscore in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE and isinstance(lat, (int, float))
        }
        if not valid_server_latencies:
            return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
        best_server_key_underscore = min(valid_server_latencies, key=valid_server_latencies.get)
        best_server_latency = valid_server_latencies[best_server_key_underscore]
        best_server_name_hyphen = best_server_key_underscore.replace('_', '-')
        return pd.Series([best_server_name_hyphen, best_server_latency], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
    except (json.JSONDecodeError, TypeError, AttributeError):
        logger.debug(f"JSON error in find_dynamic_best_server_and_latency_for_agg: {str(row_series['all_servers_oracle_latency_json'])[:70]}")
        return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])
    except Exception as e:
        logger.error(f"Unexpected exception in find_dynamic_best_server_and_latency_for_agg: {e}", exc_info=True)
        return pd.Series([None, np.nan], index=['dynamic_best_server_name_temp', 'dynamic_best_server_latency'])

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
            logger.debug(f"Failed to parse JSON in parse_json_series_to_dataframe: '{str(json_str)[:70]}...'")
            temp_parsed_dicts.append({})
    final_column_keys = all_normalized_keys_in_series
    if not final_column_keys:
        if prefix.startswith("value_") or prefix.startswith("count_") or prefix.startswith("actual_count_") or prefix == "":
            final_column_keys = set(KNOWN_CACHE_SERVER_KEYS_UNDERSCORE)
        else:
             if not parsed_rows and not temp_parsed_dicts :
                 return pd.DataFrame(columns=list(f"{prefix}{key}" for key in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE))
    prefixed_final_column_keys = {f"{prefix}{key}" for key in final_column_keys}
    for norm_dict in temp_parsed_dicts:
        row_data = {prefixed_key: norm_dict.get(prefixed_key.replace(prefix, "", 1)) for prefixed_key in prefixed_final_column_keys}
        parsed_rows.append(row_data)
    if not parsed_rows:
        return pd.DataFrame(columns=list(prefixed_final_column_keys))
    df_result = pd.DataFrame(parsed_rows, index=valid_indices, columns=list(prefixed_final_column_keys))
    return df_result

def aggregate_strategy_logs(strategy_name: str, suffix_pattern: str = "",
                            input_dir: str = DEFAULT_SIM_DATA_DIR,
                            output_dir: str = DEFAULT_OUTPUT_DIR):
    log_files = []
    base_pattern_str = f"log_{strategy_name}"
    if suffix_pattern:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str + suffix_pattern)}(_\d+)?\.csv$")
    else:
        file_pattern = re.compile(rf"^{re.escape(base_pattern_str)}(.*?)(_\d+)?\.csv$")
    logger.info(f"Aggregating logs for strategy: '{strategy_name}', suffix pattern: '{suffix_pattern}', input: '{input_dir}'")
    for filename in os.listdir(input_dir):
        if os.path.isfile(os.path.join(input_dir, filename)):
            match = file_pattern.match(filename)
            if match:
                if not suffix_pattern:
                    user_suffix_part_if_any = match.group(1)
                    if user_suffix_part_if_any and not user_suffix_part_if_any.replace('_','').isdigit() and user_suffix_part_if_any != "":
                        logger.debug(f"Ignoring {filename} due to unspecified suffix '{user_suffix_part_if_any}'. Use --suffix_pattern='{user_suffix_part_if_any}'.")
                        continue
                log_files.append(os.path.join(input_dir, filename))
    if not log_files:
        logger.warning(f"No logs found for '{strategy_name}{suffix_pattern}'.")
        return
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"{len(log_files)} file(s) for aggregation: {', '.join(map(os.path.basename, log_files))}")
    all_main_dfs, all_rl_values_dfs, all_rl_counts_dfs, all_rl_actual_counts_dfs, all_server_latencies_dfs = [], [], [], [], []

    actual_min_common_duration = float('inf')

    first_run_categorical_data = {}
    for i, f_path in enumerate(log_files):
        try:
            df_run = pd.read_csv(f_path, na_filter=True)
            if df_run.empty or 'sim_time_client' not in df_run.columns:
                logger.warning(f"File {os.path.basename(f_path)} empty or missing 'sim_time_client'. Skipping.")
                continue
            df_run.dropna(subset=['sim_time_client'], inplace=True)
            if df_run.empty:
                logger.warning(f"File {os.path.basename(f_path)} empty after removing NaNs from 'sim_time_client'. Skipping.")
                continue

            df_run = df_run[df_run['sim_time_client'] <= MAX_AGGREGATION_TIME_SECONDS].copy()
            if df_run.empty:
                logger.info(f"File {os.path.basename(f_path)} has no data up to {MAX_AGGREGATION_TIME_SECONDS}s. Skipping.")
                continue

            max_time_this_run_after_cap = df_run['sim_time_client'].max()
            if pd.notna(max_time_this_run_after_cap):
                actual_min_common_duration = min(actual_min_common_duration, max_time_this_run_after_cap)

            if 'all_servers_oracle_latency_json' in df_run.columns:
                best_info = df_run.apply(find_dynamic_best_server_and_latency_for_agg, axis=1)
                df_run['dynamic_best_server_latency'] = best_info['dynamic_best_server_latency']
            else:
                df_run['dynamic_best_server_latency'] = np.nan
            df_run['sim_time_group'] = df_run['sim_time_client'].round().astype(int)

            cols_to_avg = [col for col in EXPECTED_MAIN_NUMERIC_COLS if col in df_run.columns]
            all_main_dfs.append(df_run[['sim_time_group'] + cols_to_avg].copy())
            if i == 0:
                cols_cat = [col for col in EXPECTED_CATEGORICAL_COLS_FROM_FIRST_RUN if col in df_run.columns]
                if cols_cat:
                    temp_cat_df = df_run.groupby('sim_time_group')[cols_cat].first().reset_index()
                    temp_cat_df = temp_cat_df[temp_cat_df['sim_time_group'] <= MAX_AGGREGATION_TIME_SECONDS]
                    for _, row in temp_cat_df.iterrows():
                        first_run_categorical_data[row['sim_time_group']] = {
                            col: row[col] for col in cols_cat if pd.notna(row[col])
                        }
            json_processing_map = [
                ('rl_values_json', all_rl_values_dfs, 'value_'),
                ('rl_counts_json', all_rl_counts_dfs, 'count_'),
                ('all_servers_oracle_latency_json', all_server_latencies_dfs, '')
            ]
            if 'rl_actual_counts_json' in df_run.columns:
                logger.debug(f"Processing 'rl_actual_counts_json' from {os.path.basename(f_path)}")
                json_processing_map.append(('rl_actual_counts_json', all_rl_actual_counts_dfs, 'actual_count_'))

            for json_col_name, target_list, prefix in json_processing_map:
                if json_col_name in df_run.columns and not df_run[json_col_name].dropna().empty:
                    parsed_df = parse_json_series_to_dataframe(df_run[json_col_name], prefix=prefix)
                    if not parsed_df.empty:
                        valid_indices_for_time_group = parsed_df.index.intersection(df_run.index)
                        if not valid_indices_for_time_group.empty:
                            parsed_df.loc[valid_indices_for_time_group, 'sim_time_group'] = df_run.loc[valid_indices_for_time_group, 'sim_time_group'].values
                            parsed_df.dropna(subset=['sim_time_group'], inplace=True)
                            parsed_df['sim_time_group'] = parsed_df['sim_time_group'].astype(int)
                            parsed_df = parsed_df[parsed_df['sim_time_group'] <= MAX_AGGREGATION_TIME_SECONDS]
                            if not parsed_df.empty:
                                if prefix.startswith("value_") or prefix.startswith("count_") or prefix.startswith("actual_count_"):
                                    target_list.append(parsed_df.groupby('sim_time_group').last().reset_index())
                                else:
                                    target_list.append(parsed_df)
        except Exception as e:
            logger.error(f"Error processing {os.path.basename(f_path)}: {e}. Skipping.", exc_info=True)

    if not all_main_dfs:
        logger.error("No valid main data found for aggregation.")
        return

    effective_aggregation_duration = min(actual_min_common_duration if actual_min_common_duration != float('inf') else MAX_AGGREGATION_TIME_SECONDS,
                                         MAX_AGGREGATION_TIME_SECONDS)
    logger.info(f"Aggregating data up to effective simulation time of {effective_aggregation_duration:.2f}s (limit: {MAX_AGGREGATION_TIME_SECONDS}s).")

    combined_main_df = pd.concat(all_main_dfs)
    combined_main_df = combined_main_df[combined_main_df['sim_time_group'] <= effective_aggregation_duration]
    if combined_main_df.empty:
        logger.error("Combined main DataFrame empty after final duration filter.")
        return
    agg_funcs_main = {col: (lambda x: np.nanmean(x.astype(float)) if pd.to_numeric(x, errors='coerce').notnull().any() else np.nan)
                      for col in EXPECTED_MAIN_NUMERIC_COLS if col in combined_main_df.columns and col != 'sim_time_client'}
    aggregated_df = combined_main_df.groupby('sim_time_group', as_index=False).agg(agg_funcs_main)
    aggregated_df.rename(columns={'sim_time_group': 'sim_time_client'}, inplace=True)

    if first_run_categorical_data:
        cat_df = pd.DataFrame.from_dict(first_run_categorical_data, orient='index').reset_index()
        cat_df.rename(columns={'index': 'sim_time_client'}, inplace=True)
        cat_df = cat_df[cat_df['sim_time_client'] <= effective_aggregation_duration]
        if aggregated_df['sim_time_client'].dtype != cat_df['sim_time_client'].dtype:
            try:
                aggregated_df['sim_time_client'] = aggregated_df['sim_time_client'].astype(int)
                cat_df['sim_time_client'] = cat_df['sim_time_client'].astype(int)
            except ValueError:
                logger.error("Could not convert 'sim_time_client' to int for merge.")
        aggregated_df = pd.merge(aggregated_df, cat_df, on='sim_time_client', how='left')

    json_data_to_merge_final = [
        (all_server_latencies_dfs, False, ""),
        (all_rl_values_dfs, True, "value_"),
        (all_rl_counts_dfs, True, "count_")
    ]
    if all_rl_actual_counts_dfs:
        json_data_to_merge_final.append((all_rl_actual_counts_dfs, True, "actual_count_"))

    for data_list, is_rl_data_type, original_prefix in json_data_to_merge_final:
        if data_list:
            combined_json_df = pd.concat(data_list)
            if not combined_json_df.empty and 'sim_time_group' in combined_json_df.columns:
                combined_json_df = combined_json_df[combined_json_df['sim_time_group'] <= effective_aggregation_duration]
                if combined_json_df.empty: continue

                combined_json_df.dropna(subset=['sim_time_group'], inplace=True)
                combined_json_df['sim_time_group'] = combined_json_df['sim_time_group'].astype(int)
                combined_json_df = combined_json_df[combined_json_df['sim_time_group'].isin(aggregated_df['sim_time_client'])]
                if not combined_json_df.empty:
                    numeric_cols_to_avg_json = [col for col in combined_json_df.columns if col != 'sim_time_group' and pd.api.types.is_numeric_dtype(combined_json_df[col])]
                    if numeric_cols_to_avg_json:
                        avg_json_df = combined_json_df.groupby('sim_time_group')[numeric_cols_to_avg_json].mean().reset_index()
                        avg_json_df.rename(columns={'sim_time_group': 'sim_time_client'}, inplace=True)
                        if original_prefix == "":
                            server_latency_cols = [col for col in avg_json_df.columns if col in KNOWN_CACHE_SERVER_KEYS_UNDERSCORE]
                            if server_latency_cols:
                                avg_json_df['all_servers_oracle_latency_json_AGG'] = avg_json_df.apply(
                                    lambda row: json.dumps({col: row[col] for col in server_latency_cols if pd.notna(row[col])}),axis=1)
                                df_to_merge = avg_json_df[['sim_time_client', 'all_servers_oracle_latency_json_AGG']].copy()
                                df_to_merge.rename(columns={'all_servers_oracle_latency_json_AGG': 'all_servers_oracle_latency_json'}, inplace=True)
                                if 'all_servers_oracle_latency_json' in aggregated_df.columns:
                                    aggregated_df = aggregated_df.drop(columns=['all_servers_oracle_latency_json'])
                                aggregated_df = pd.merge(aggregated_df, df_to_merge, on='sim_time_client', how='left')
                        else:
                            aggregated_df = pd.merge(aggregated_df, avg_json_df, on='sim_time_client', how='left')
                    elif not combined_json_df.drop(columns=['sim_time_group'], errors='ignore').empty:
                         logger.debug(f"No numeric columns to aggregate in JSON (prefix: {original_prefix}). Columns: {combined_json_df.columns.tolist()}")
    if aggregated_df.empty:
        logger.error("Final aggregated DataFrame is empty.")
        return

    final_cols_order_base = [
        "sim_time_client", "client_lat", "client_lon",
        "experienced_latency_ms_CLIENT", "experienced_latency_ms_ORACLE", "experienced_latency_ms",
        "dynamic_best_server_latency",
        "all_servers_oracle_latency_json",
        "steering_decision_main_server", "rl_strategy", "gamma_value"
    ]
    value_cols = sorted([col for col in aggregated_df.columns if col.startswith('value_')])
    count_cols = sorted([col for col in aggregated_df.columns if col.startswith('count_')])
    actual_count_cols = sorted([col for col in aggregated_df.columns if col.startswith('actual_count_')])

    final_ordered_cols = []
    for col in final_cols_order_base:
        if col in aggregated_df.columns:
            final_ordered_cols.append(col)
    for col_group in [value_cols, count_cols, actual_count_cols]:
        for col in col_group:
            if col not in final_ordered_cols and col in aggregated_df.columns:
                final_ordered_cols.append(col)

    for json_col in ['rl_counts_json', 'rl_actual_counts_json', 'rl_values_json']:
        if json_col in aggregated_df.columns and json_col not in final_ordered_cols:
            final_ordered_cols.append(json_col)

    remaining_cols = sorted(list(set(aggregated_df.columns) - set(final_ordered_cols)))
    final_ordered_cols.extend(remaining_cols)
    final_ordered_cols = [col for col in final_ordered_cols if col in aggregated_df.columns]
    final_ordered_cols = list(dict.fromkeys(final_ordered_cols))

    aggregated_df_final = aggregated_df[final_ordered_cols].copy()
    aggregated_df_final.sort_values(by='sim_time_client', inplace=True)

    output_base_name = f"log_{strategy_name}{suffix_pattern}_average"
    output_file, counter = os.path.join(output_dir, f"{output_base_name}.csv"), 1
    while os.path.exists(output_file):
        output_file = os.path.join(output_dir, f"{output_base_name}_{counter}.csv")
        counter += 1

    aggregated_df_final.to_csv(output_file, index=False, float_format='%.3f')
    logger.info(f"Aggregated CSV file saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregates multiple simulation logs for a strategy.")
    parser.add_argument("strategy_name", type=str, help="Base name of the strategy (e.g., ucb1, d_ucb).")
    parser.add_argument("--suffix_pattern", type=str, default="",
                        help="Optional suffix pattern in filenames (e.g., _runA).")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_SIM_DATA_DIR, help=f"Directory of logs. Default: {DEFAULT_SIM_DATA_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save aggregated log. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    _handler_agg = logging.StreamHandler()
    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    _formatter_agg = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    _handler_agg.setFormatter(_formatter_agg)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(_handler_agg)
    logger.setLevel(log_level_to_set)

    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")
    aggregate_strategy_logs(args.strategy_name, args.suffix_pattern, args.input_dir, args.output_dir)