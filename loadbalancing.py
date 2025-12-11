import random
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from matplotlib.patches import Patch

# Konfigurasi untuk algoritma Ant Colony Optimization
ALPHA = 1.0  # Bobot jejak request
BETA = 2.0   # Bobot Heuristik (1/Load)
EVAPORATION = 0.1
INIT_PHEROMONE = 1.0


class Server:
    def __init__(self, server_id, capacity):
        self.id = server_id
        self.capacity = capacity      # Kapasitas pemrosesan (unit/sec)
        self.current_load = 0.0       # Total size tugas yang sedang mengantre
        self.processed_time = 0.0     # Total waktu simulasi server saat bekerja
        self.pheromone = INIT_PHEROMONE  # variable jejak request untuk Ant Colony Optimization
        self.request_history = []     # Menyimpan riwayat beban untuk visualisasi
        
    def reset(self):
        self.current_load = 0.0
        self.processed_time = 0.0
        self.pheromone = INIT_PHEROMONE
        self.request_history = []


class Request:
    def __init__(self, request_id, size):
        self.id = request_id
        self.size = size  # Ukuran tugas (unit)


def reset_servers(servers):
    for server in servers:
        server.reset()


def save_results_to_csv(filename, rr_results, aco_results):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "ResponseTime", "Variance", "Throughput", "Step"])
        
        # Simpan data RR
        for i, (rt, var, thr) in enumerate(zip(rr_results['response_times'], 
                                               rr_results['variances'], 
                                               rr_results['throughputs'])):
            writer.writerow(["RoundRobin", rt, var, thr, i])
        
        # Simpan data ACO
        for i, (rt, var, thr) in enumerate(zip(aco_results['response_times'], 
                                               aco_results['variances'], 
                                               aco_results['throughputs'])):
            writer.writerow(["ACO", rt, var, thr, i])


def plot_comparison_results(rr_results, aco_results):
    # Buat figure dengan 4 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Data untuk plotting
    steps_rr = list(range(len(rr_results['response_times'])))
    steps_aco = list(range(len(aco_results['response_times'])))
    
    # Plot 1: Response Time Comparison
    ax1 = axes[0, 0]
    if len(steps_rr) > 1:
        ax1.plot(steps_rr, rr_results['response_times'], 'r-', label='Round Robin', linewidth=2)
    else:
        ax1.plot(steps_rr, rr_results['response_times'], 'ro', label='Round Robin', markersize=8)
    
    if len(steps_aco) > 1:
        ax1.plot(steps_aco, aco_results['response_times'], 'b-', label='Ant Colony', linewidth=2)
    else:
        ax1.plot(steps_aco, aco_results['response_times'], 'bs', label='Ant Colony', markersize=8)
    
    ax1.set_title("Response Time Comparison", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Simulation Step")
    ax1.set_ylabel("Time (ms/unit)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Load Variance Comparison
    ax2 = axes[0, 1]
    if len(steps_rr) > 1:
        ax2.plot(steps_rr, rr_results['variances'], 'g-', label='Round Robin', linewidth=2)
    else:
        ax2.plot(steps_rr, rr_results['variances'], 'go', label='Round Robin', markersize=8)
    
    if len(steps_aco) > 1:
        ax2.plot(steps_aco, aco_results['variances'], 'm-', label='Ant Colony', linewidth=2)
    else:
        ax2.plot(steps_aco, aco_results['variances'], 'ms', label='Ant Colony', markersize=8)
    
    ax2.set_title("Load Variance Comparison", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Variance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Throughput Comparison
    ax3 = axes[1, 0]
    if len(steps_rr) > 1:
        ax3.plot(steps_rr, rr_results['throughputs'], 'c-', label='Round Robin', linewidth=2)
    else:
        ax3.plot(steps_rr, rr_results['throughputs'], 'co', label='Round Robin', markersize=8)
    
    if len(steps_aco) > 1:
        ax3.plot(steps_aco, aco_results['throughputs'], 'y-', label='Ant Colony', linewidth=2)
    else:
        ax3.plot(steps_aco, aco_results['throughputs'], 'ys', label='Ant Colony', markersize=8)
    
    ax3.set_title("Throughput Comparison", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Simulation Step")
    ax3.set_ylabel("Requests/sec")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Server Load Distribution (Bar Chart)
    ax4 = axes[1, 1]
    
    # Ambil data beban server terakhir
    if rr_results['server_loads'] and aco_results['server_loads']:
        server_ids = list(range(len(rr_results['server_loads'])))
        rr_loads = rr_results['server_loads']
        aco_loads = aco_results['server_loads']
        
        bar_width = 0.35
        x = np.arange(len(server_ids))
        
        ax4.bar(x - bar_width/2, rr_loads, bar_width, label='Round Robin', color='red', alpha=0.7)
        ax4.bar(x + bar_width/2, aco_loads, bar_width, label='Ant Colony', color='blue', alpha=0.7)
        
        ax4.set_title("Final Server Load Distribution", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Server ID")
        ax4.set_ylabel("Load (units)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(server_ids, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle("Load Balancing Algorithms Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def calculate_variance(servers):
    sum_load = sum(server.current_load for server in servers)
    mean_load = sum_load / len(servers)
    
    temp = 0
    for server in servers:
        temp += (server.current_load - mean_load) ** 2
    
    return temp / len(servers)


def run_round_robin(servers, requests, batch_size=1000):
    print("--- Menjalankan Round Robin ---")
    
    # Buat salinan server
    servers_copy = []
    for server in servers:
        new_server = Server(server.id, server.capacity)
        new_server.current_load = 0
        servers_copy.append(new_server)
    
    total_response_time = 0.0
    server_count = len(servers_copy)
    
    response_times = []
    variances = []
    throughputs = []
    
    # Proses request dalam batch untuk mendapatkan data bertahap
    for batch_start in range(0, len(requests), batch_size):
        batch_end = min(batch_start + batch_size, len(requests))
        batch_requests = requests[batch_start:batch_end]
        
        batch_response_time = 0.0
        
        for i, request in enumerate(batch_requests):
            server_index = i % server_count
            processing_time = request.size / servers_copy[server_index].capacity
            servers_copy[server_index].current_load += request.size
            batch_response_time += processing_time
        
        total_response_time += batch_response_time
        
        # Hitung metrik untuk batch ini
        avg_response_time = total_response_time / batch_end
        variance = calculate_variance(servers_copy)
        
        # Hitung throughput
        max_load_time = 0.0
        for server in servers_copy:
            time_needed = server.current_load / server.capacity
            if time_needed > max_load_time:
                max_load_time = time_needed
        
        throughput = batch_end / max_load_time if max_load_time > 0 else 0
        
        response_times.append(avg_response_time)
        variances.append(variance)
        throughputs.append(throughput)
    
    # Ambil beban server akhir untuk visualisasi
    final_loads = [server.current_load for server in servers_copy]
    
    print(f"Final Results:")
    print(f"Avg Response Time: {response_times[-1]:.4f} ms/unit")
    print(f"Load Variance    : {variances[-1]:.4f}")
    print(f"Est. Throughput  : {throughputs[-1]:.4f} req/sec")
    print("-------------------------------")
    
    return {
        'response_times': response_times,
        'variances': variances,
        'throughputs': throughputs,
        'server_loads': final_loads
    }


def run_aco(servers, requests, batch_size=1000):
    print("--- Menjalankan Ant Colony Optimization ---")
    
    # Buat salinan server
    servers_copy = []
    for server in servers:
        new_server = Server(server.id, server.capacity)
        new_server.current_load = 0
        new_server.pheromone = INIT_PHEROMONE
        servers_copy.append(new_server)
    
    total_response_time = 0.0
    
    response_times = []
    variances = []
    throughputs = []
    
    # Proses request dalam batch
    for batch_start in range(0, len(requests), batch_size):
        batch_end = min(batch_start + batch_size, len(requests))
        batch_requests = requests[batch_start:batch_end]
        
        batch_response_time = 0.0
        
        for request in batch_requests:
            probabilities = []
            sum_prob = 0.0
            
            for server in servers_copy:
                visibility = 1.0 / (server.current_load + 0.1)
                prob = (server.pheromone ** ALPHA) * (visibility ** BETA)
                probabilities.append(prob)
                sum_prob += prob
            
            r = random.random() * sum_prob
            selected_server_index = -1
            cumulative = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected_server_index = i
                    break
            
            if selected_server_index == -1:
                selected_server_index = len(servers_copy) - 1
            
            selected_server = servers_copy[selected_server_index]
            processing_time = request.size / selected_server.capacity
            selected_server.current_load += request.size
            batch_response_time += processing_time
            
            deposit = 1.0 / (selected_server.current_load + 1.0)
            
            for server in servers_copy:
                server.pheromone = (1.0 - EVAPORATION) * server.pheromone
            
            selected_server.pheromone += deposit
        
        total_response_time += batch_response_time
        
        # Hitung metrik untuk batch ini
        avg_response_time = total_response_time / batch_end
        variance = calculate_variance(servers_copy)
        
        # Hitung throughput
        max_load_time = 0.0
        for server in servers_copy:
            time_needed = server.current_load / server.capacity
            if time_needed > max_load_time:
                max_load_time = time_needed
        
        throughput = batch_end / max_load_time if max_load_time > 0 else 0
        
        response_times.append(avg_response_time)
        variances.append(variance)
        throughputs.append(throughput)
    
    # Ambil beban server akhir untuk visualisasi
    final_loads = [server.current_load for server in servers_copy]
    
    print(f"Final Results:")
    print(f"Avg Response Time: {response_times[-1]:.4f} ms/unit")
    print(f"Load Variance    : {variances[-1]:.4f}")
    print(f"Est. Throughput  : {throughputs[-1]:.4f} req/sec")
    print("-------------------------------")
    
    return {
        'response_times': response_times,
        'variances': variances,
        'throughputs': throughputs,
        'server_loads': final_loads
    }


def main():
    # --- 1. Setup Parameter ---
    num_servers = 30
    num_requests = 10000
    
    # Inisialisasi server heterogen
    servers = []
    for i in range(num_servers):
        capacity = 20.0 if i < 2 else 10.0  # 2 server dengan kapasitas tinggi
        servers.append(Server(i, capacity))
    
    # Inisialisasi requests
    random.seed(time.time())
    requests = []
    for i in range(num_requests):
        size = 10 + random.randint(0, 90)  # Ukuran antara 10-100
        requests.append(Request(i, size))
    
    print(f"Simulasi Load Balancing: {num_requests} requests ke {num_servers} servers.")
    print("Kondisi: Server Heterogen (Kapasitas berbeda).\n")
    
    # Jalankan kedua algoritma
    print("\n" + "="*50)
    rr_results = run_round_robin(servers, requests, batch_size=1000)
    
    print("\n" + "="*50)
    reset_servers(servers)
    aco_results = run_aco(servers, requests, batch_size=1000)
    
    # Simpan dan tampilkan hasil
    save_results_to_csv("results_comparison.csv", rr_results, aco_results)
    plot_comparison_results(rr_results, aco_results)
    
    # Tampilkan tabel perbandingan akhir
    print("\n" + "="*60)
    print("=== PERBANDINGAN AKHIR HASIL ===")
    print("="*60)
    print(f"{'METRIC':<20} {'ROUND ROBIN':<15} {'ANT COLONY':<15} {'DIFFERENCE':<15}")
    print("-"*65)
    
    rr_rt = rr_results['response_times'][-1]
    aco_rt = aco_results['response_times'][-1]
    rt_diff = aco_rt - rr_rt
    rt_better = "ACO" if aco_rt < rr_rt else "RR"
    
    rr_var = rr_results['variances'][-1]
    aco_var = aco_results['variances'][-1]
    var_diff = aco_var - rr_var
    var_better = "ACO" if aco_var < rr_var else "RR"
    
    rr_thr = rr_results['throughputs'][-1]
    aco_thr = aco_results['throughputs'][-1]
    thr_diff = aco_thr - rr_thr
    thr_better = "ACO" if aco_thr > rr_thr else "RR"
    
    print(f"{'Response Time':<20} {rr_rt:<15.4f} {aco_rt:<15.4f} {rt_diff:<15.4f} ({rt_better} better)")
    print(f"{'Load Variance':<20} {rr_var:<15.4f} {aco_var:<15.4f} {var_diff:<15.4f} ({var_better} better)")
    print(f"{'Throughput':<20} {rr_thr:<15.4f} {aco_thr:<15.4f} {thr_diff:<15.4f} ({thr_better} better)")
    print("="*60)


if __name__ == "__main__":
    main()