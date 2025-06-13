import pandas as pd
import os
import re
import argparse
import json
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager

logger = logging.getLogger("analyze_server_choices")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

BASE_GRAPHICS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AVERAGE_LOGS_DIR = os.path.join(BASE_GRAPHICS_DIR, "Logs", "Average")
DEFAULT_IMG_OUTPUT_DIR = os.path.join(BASE_GRAPHICS_DIR, "Img", "analysis_tables")

SERVER_DISPLAY_NAMES = {
    "video-streaming-cache-1": "Cache 1 (BR)",
    "video-streaming-cache-2": "Cache 2 (CL)",
    "video-streaming-cache-3": "Cache 3 (CO)",
    "N/A_NO_NODES_FROM_SELECTION": "No Selection",
}
ACTUAL_CACHE_SERVER_NAMES_HYPHEN = [
    "video-streaming-cache-1", "video-streaming-cache-2", "video-streaming-cache-3"
]

def dataframe_to_image(df: pd.DataFrame, output_image_path: str, title: str ="Server Choice Analysis"):
    try:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Verdana', 'Bitstream Vera Sans']
        plt.rcParams['font.family'] = 'sans-serif'
    except Exception: pass

    num_cols, num_rows = len(df.columns), len(df)
    fig_width = max(7, num_cols * 2.5)
    fig_height = max(2.5, num_rows * 0.4 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('tight'); ax.axis('off')

    table_data = df.values.tolist()
    column_labels_orig = df.columns.tolist()
    column_labels_formatted = []
    for col_name in column_labels_orig:
        if "Dynamic Best Server Choices" in col_name: column_labels_formatted.append("Dynamic Best\nServer Choices (#)")
        elif "Total Decisions" in col_name: column_labels_formatted.append("Total\nDecisions (#)")
        elif "Dynamic Best Server Accuracy" in col_name: column_labels_formatted.append("Dynamic Best\nServer Accuracy (%)")
        else: column_labels_formatted.append(col_name)

    the_table = ax.table(cellText=table_data, colLabels=column_labels_formatted,
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    the_table.auto_set_font_size(False)
    font_size = 10 if num_cols <= 4 else 8
    if num_rows > 10: font_size = max(6, font_size -1)
    the_table.set_fontsize(font_size)

    for (i, j), cell in the_table.get_celld().items():
        cell.set_edgecolor('black')
        if i == 0:
            cell.set_text_props(weight='bold', color='white', wrap=True)
            cell.set_facecolor('#2C3E50') 
            cell.set_height(0.12 if '\n' in cell.get_text().get_text() else 0.08)
        else:
            cell.set_height(0.07)
            cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
        cell.set_alpha(0.95)
    title_y = 1.0 - ( (0.08 * (1 + (1 if any("\n" in str(c) for c in column_labels_formatted) else 0) ) ) / fig_height if fig_height > 0 else 0.05)
    plt.title(title, fontsize=14, y=title_y, pad=15 if any("\n" in str(c) for c in column_labels_formatted) else 10)

    try:
        output_dir = os.path.dirname(output_image_path)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_image_path, dpi=200, bbox_inches='tight', pad_inches=0.3)
        logger.info(f"Table image saved: {output_image_path}")
    except Exception as e: logger.error(f"Error saving table image: {e}", exc_info=False)
    finally: plt.close(fig)

def analyze_server_choices(logs_dir: str, output_csv_path: str = None, output_img_path: str = None):
    results = {}
    logger.info(f"Analyzing server choices in: {logs_dir}")
    if not os.path.isdir(logs_dir):
        logger.error(f"Logs directory not found: {logs_dir}")
        return

    log_files_processed_count = 0
    for filename in sorted(os.listdir(logs_dir)):
        if not (filename.startswith("log_") and filename.endswith("_average.csv")):
            if os.path.basename(os.path.normpath(logs_dir)).lower() == "average":
                logger.debug(f"Skipping non-average file {filename} in Average directory.")
            continue

        file_path = os.path.join(logs_dir, filename)
        logger.debug(f"Processing aggregated file: {filename}")
        try:
            df_log = pd.read_csv(file_path)
            required_cols = ['steering_decision_main_server', 'all_servers_oracle_latency_json', 'sim_time_client']
            if df_log.empty or not all(col in df_log.columns for col in required_cols):
                logger.warning(f"Skipping {filename}: empty or missing essential columns ({required_cols}).")
                continue

            strategy_name = "Unknown"
            if 'rl_strategy' in df_log.columns and not df_log['rl_strategy'].empty and pd.notna(df_log['rl_strategy'].iloc[0]):
                strategy_name = str(df_log['rl_strategy'].iloc[0])
                if strategy_name == "oracle_best_choice":
                    strategy_name = "Optimal Strategy"
            else:
                match = re.match(r"log_([a-zA-Z0-9_]+?)_average", filename)
                if match:
                    strategy_name = match.group(1)
                    if strategy_name == "oracle_best_choice":
                        strategy_name = "Optimal Strategy"

            if strategy_name == "Unknown":
                logger.warning(f"Could not determine strategy for {filename}. Skipping.")
                continue

            current_file_total_decisions = 0
            current_file_dynamic_best_choices = 0
            for index, row in df_log.iterrows():
                decision = row['steering_decision_main_server']
                all_latencies_json_str = row['all_servers_oracle_latency_json']
                if pd.isna(decision) or pd.isna(all_latencies_json_str) or "N/A" in str(decision):
                    continue
                current_file_total_decisions += 1
                try:
                    server_latencies_from_json = json.loads(all_latencies_json_str)
                    server_latencies_hyphen_keys = { k.replace('_', '-'): v for k, v in server_latencies_from_json.items() }
                    valid_server_latencies = {
                        s_name: lat for s_name, lat in server_latencies_hyphen_keys.items()
                        if s_name in ACTUAL_CACHE_SERVER_NAMES_HYPHEN and isinstance(lat, (int, float))
                    }
                    if not valid_server_latencies:
                        logger.debug(f"No valid server latencies in JSON for {filename} line {index}")
                        continue
                    dynamic_best_server = min(valid_server_latencies, key=valid_server_latencies.get)
                    if decision == dynamic_best_server:
                        current_file_dynamic_best_choices += 1
                except json.JSONDecodeError:
                    logger.warning(f"JSON decode error in {filename} line {index}. JSON: {all_latencies_json_str[:100]}")
                except Exception as e_row:
                    logger.error(f"Error processing row {index} of {filename}: {e_row}")

            if current_file_total_decisions > 0:
                if strategy_name not in results:
                    results[strategy_name] = {'total_decisions': 0, 'dynamic_best_server_choices': 0}
                results[strategy_name]['total_decisions'] += current_file_total_decisions
                results[strategy_name]['dynamic_best_server_choices'] += current_file_dynamic_best_choices
            log_files_processed_count +=1
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}", exc_info=False)

    if not results:
        logger.warning("No results processed. No table generated.")
        return

    logger.info(f"Processed {log_files_processed_count} aggregated log files.")
    table_data_for_df = []
    for strategy, data in results.items():
        total_decisions = data['total_decisions']
        dynamic_best_choices = data['dynamic_best_server_choices']
        accuracy = (dynamic_best_choices / total_decisions * 100) if total_decisions > 0 else 0
        display_strategy_name = strategy.replace('_', ' ').title()
        if strategy == "Optimal Strategy":
             display_strategy_name = "Optimal Strategy"

        row = {
            'Strategy': display_strategy_name,
            'Dynamic Best Server Choices (#)': dynamic_best_choices,
            'Total Decisions (#)': total_decisions,
            'Dynamic Best Server Accuracy (%)': f"{accuracy:.1f}%"
        }
        table_data_for_df.append(row)

    df_results = pd.DataFrame(table_data_for_df)
    if not df_results.empty:
        cols_order = ['Strategy', 'Dynamic Best Server Choices (#)', 'Total Decisions (#)', 'Dynamic Best Server Accuracy (%)']
        df_results_ordered = df_results[[col for col in cols_order if col in df_results.columns]].copy()
        df_results_ordered.sort_values(by='Strategy', inplace=True)
        logger.info("\nDynamic Best Server Choice Analysis:\n" + df_results_ordered.to_string(index=False))
        if output_csv_path:
            try:
                output_dir_csv = os.path.dirname(output_csv_path)
                if output_dir_csv and not os.path.exists(output_dir_csv): os.makedirs(output_dir_csv, exist_ok=True)
                df_results_ordered.to_csv(output_csv_path, index=False)
                logger.info(f"Analysis table CSV saved: {output_csv_path}")
            except Exception as e: logger.error(f"Error saving analysis CSV: {e}", exc_info=False)
        if output_img_path:
            dataframe_to_image(df_results_ordered, output_img_path, title="Dynamic Best Server Choice Accuracy by Strategy")
    else:
        logger.warning("Resulting DataFrame is empty, no table to show or save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzes logs for dynamic best server choices.")
    parser.add_argument("--output_csv", type=str,
                        default=os.path.join(DEFAULT_AVERAGE_LOGS_DIR, "dynamic_best_choice_accuracy.csv"),
                        help="Optional: Path to save the CSV table. Default: <AverageLogsDir>/dynamic_best_choice_accuracy.csv")
    parser.add_argument("--output_img", type=str,
                        default=os.path.join(DEFAULT_IMG_OUTPUT_DIR, "dynamic_best_choice_accuracy_table.png"),
                        help="Optional: Path to save the image of the table. Default: <ImgAnalysisTablesDir>/dynamic_best_choice_accuracy_table.png")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args()

    log_level_to_set = logging.DEBUG if args.verbose else logging.INFO
    if logger.handlers:
        logger.handlers[0].setFormatter(logging.Formatter('%(levelname)s - %(message)s')) 
        logger.handlers[0].setLevel(log_level_to_set)
    logger.setLevel(log_level_to_set)
    logger.info(f"Logging level set to {logging.getLevelName(logger.getEffectiveLevel())}.")


    output_image_file = args.output_img
    if output_image_file and not os.path.isabs(output_image_file):
        output_image_file = os.path.join(DEFAULT_IMG_OUTPUT_DIR, os.path.basename(output_image_file))
    if output_image_file:
        os.makedirs(os.path.dirname(output_image_file), exist_ok=True)

    output_csv_file = args.output_csv
    if output_csv_file and not os.path.isabs(output_csv_file):
        output_csv_file = os.path.join(DEFAULT_AVERAGE_LOGS_DIR, os.path.basename(output_csv_file))
    if output_csv_file:
         os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)

    analyze_server_choices(DEFAULT_AVERAGE_LOGS_DIR, output_csv_file, output_image_file)