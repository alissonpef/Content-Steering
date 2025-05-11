import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(BASE_DIR, "Files", "Data", "simulation_log.csv")
IMG_DIR = os.path.join(BASE_DIR, "Files", "Img")

def generate_plots():
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Erro: Arquivo CSV não encontrado em {CSV_FILE_PATH}")
        return

    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"Lendo dados de: {CSV_FILE_PATH}")
    df = pd.read_csv(CSV_FILE_PATH)

    if df.empty:
        print("Arquivo CSV está vazio. Nenhum gráfico será gerado.")
        return

    print(f"Dados carregados. {len(df)} linhas.")
    df.sort_values(by="sim_time_client", inplace=True) 
    
    plt.figure(figsize=(14, 7))
    plt.plot(df['sim_time_client'], df['experienced_latency_ms'], marker='.', linestyle='-', markersize=5, alpha=0.7, label='Experienced Latency (per fragment)')
    
    window_size = 10 
    if len(df['experienced_latency_ms']) >= window_size:
        df['latency_ma'] = df['experienced_latency_ms'].rolling(window=window_size, center=True).mean()
        plt.plot(df['sim_time_client'], df['latency_ma'], linestyle='--', color='red', linewidth=2, label=f'Moving Average ({window_size} points)')

    plt.xlabel("Simulation Time (from client, seconds)")
    plt.ylabel("Experienced Latency (ms)")
    plt.title("Client's Experienced Latency Over Simulation Time")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(IMG_DIR, "1_chosen_server_latency_timeline.png")
    plt.savefig(plot_path)
    print(f"Gráfico salvo em: {plot_path}")

    plt.figure(figsize=(14, 7))
    unique_servers = df['server_used_for_latency'].unique()
    
    colors = plt.cm.get_cmap('viridis', len(unique_servers)) 

    for i, server_name in enumerate(unique_servers):
        if pd.isna(server_name): continue 
        server_df = df[df['server_used_for_latency'] == server_name]
        if not server_df.empty:
            plt.plot(server_df['sim_time_client'], server_df['experienced_latency_ms'],
                     marker='o', linestyle='-', markersize=4, alpha=0.8,
                     label=f"Latency from {server_name}", color=colors(i))

    plt.xlabel("Simulation Time (from client, seconds)")
    plt.ylabel("Experienced Latency (ms)")
    plt.title("Experienced Latency from Each Server (When Used)")
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plot_path = os.path.join(IMG_DIR, "2_all_servers_latency_timeline.png")
    plt.savefig(plot_path)
    print(f"Gráfico salvo em: {plot_path}")

    plt.figure(figsize=(10, 6))
    df_for_boxplot = df.dropna(subset=['server_used_for_latency', 'experienced_latency_ms'])
    if not df_for_boxplot.empty:
        sns.boxplot(x='server_used_for_latency', y='experienced_latency_ms', data=df_for_boxplot, palette="pastel")
        sns.stripplot(x='server_used_for_latency', y='experienced_latency_ms', data=df_for_boxplot, color=".25", size=3, alpha=0.5) 
        plt.xlabel("Cache Server Used")
        plt.ylabel("Experienced Latency (ms)")
        plt.title("Distribution of Experienced Latency per Server")
        plt.xticks(rotation=15, ha='right')
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plot_path = os.path.join(IMG_DIR, "3_latency_distribution_per_server.png")
        plt.savefig(plot_path)
        print(f"Gráfico salvo em: {plot_path}")
    else:
        print("Não há dados suficientes para o boxplot de latência por servidor.")

    plt.figure(figsize=(14, 7))

    df_steering_decisions = df.dropna(subset=['steering_decision_main_server', 'sim_time_client'])
    if not df_steering_decisions.empty:
        df_steering_decisions = df_steering_decisions.drop_duplicates(subset=['sim_time_client'], keep='first')
        
        all_possible_servers = sorted(list(CACHE_COORDS.keys()) + ["N/A_NO_NODES_SELECTED"]) 
        server_to_int = {server: i for i, server in enumerate(all_possible_servers)}
        int_to_server_label = {i: CACHE_COORDS.get(server, {}).get('label', server) for server, i in server_to_int.items()}

        y_values_decision = df_steering_decisions['steering_decision_main_server'].map(server_to_int).fillna(-1)

        plt.plot(df_steering_decisions['sim_time_client'], y_values_decision,
                 drawstyle='steps-post', marker='.', markersize=5, label='Main Steered Server')
        
        plt.yticks(ticks=list(int_to_server_label.keys()), labels=list(int_to_server_label.values()))
        plt.xlabel("Simulation Time (from client, seconds)")
        plt.ylabel("Steering Decision (Main Server)")
        plt.title("Steering Service Main Server Decision Over Time")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.ylim(min(int_to_server_label.keys()) - 0.5, max(int_to_server_label.keys()) + 0.5)
        plt.tight_layout()
        plot_path = os.path.join(IMG_DIR, "4_steering_decision_timeline.png")
        plt.savefig(plot_path)
        print(f"Gráfico salvo em: {plot_path}")
    else:
        print("Não há dados suficientes para o gráfico de decisão de steering.")
        
    print("Geração de gráficos concluída.")

if __name__ == "__main__":
    CACHE_COORDS = {
        "video-streaming-cache-1": { "label": "Cache 1 (BR)" },
        "video-streaming-cache-2": { "label": "Cache 2 (CL)" },
        "video-streaming-cache-3": { "label": "Cache 3 (CO)" }
    }
    generate_plots()