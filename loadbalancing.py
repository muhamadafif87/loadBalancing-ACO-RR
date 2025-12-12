import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# KONFIGURASI SIMULASI
NUM_REQUESTS = 2000      # Jumlah request
NUM_SERVERS = 10         # Jumlah server
# Server Heterogen: 2 Server Kuat (Cap 50), 3 Sedang (Cap 20), 5 Lemah (Cap 5)
SERVER_CAPACITIES = [50, 50, 20, 20, 20, 5, 5, 5, 5, 5] 

# Parameter ACO
ACO_ALPHA = 1.0          # Pengaruh Pheromone (Jejak)
ACO_BETA = 2.5           # Pengaruh Heuristik (Kapasitas & Load saat ini) - Ditingkatkan
ACO_RHO = 0.1            # Evaporation rate (Penguapan)
ACO_Q = 100              # Konstanta update pheromone
INIT_PHEROMONE = 1.0

class Server:
    def __init__(self, server_id, capacity):
        self.id = server_id
        self.capacity = capacity
        self.current_load_size = 0.0  # Total size task dalam antrean
        self.finish_time = 0.0        # Waktu kapan server ini akan bebas
        self.pheromone = INIT_PHEROMONE
        self.task_count = 0

    def process_request(self, request_size):
        # Hitung waktu proses untuk request ini
        processing_time = request_size / self.capacity
        
        # Response time = Waktu antri (finish_time saat ini) + Waktu proses
        # Jika finish_time < waktu sekarang (0 untuk batch ini), maka mulai dari 0.
        # Dalam simulasi batch, waiting time adalah akumulasi finish_time sebelumnya.
        response_time = self.finish_time + processing_time
        
        # Update state server
        self.finish_time += processing_time
        self.current_load_size += request_size
        self.task_count += 1
        
        return response_time

    def reset(self):
        self.current_load_size = 0.0
        self.finish_time = 0.0
        self.pheromone = INIT_PHEROMONE
        self.task_count = 0

class LoadBalancerSim:
    def __init__(self, requests):
        self.requests = requests # List of request sizes
        self.servers = [Server(i, cap) for i, cap in enumerate(SERVER_CAPACITIES)]

    def reset_servers(self):
        for s in self.servers:
            s.reset()

    # -------------------------------------------------------
    # 1. ALGORITMA ROUND ROBIN
    # -------------------------------------------------------
    def run_round_robin(self):
        self.reset_servers()
        response_times = []
        
        print(f"[*] Menjalankan Round Robin pada {len(self.requests)} requests...")
        
        for i, req_size in enumerate(self.requests):
            # Pilih server secara berurutan (Blind Allocation)
            server_idx = i % NUM_SERVERS
            selected_server = self.servers[server_idx]
            
            rt = selected_server.process_request(req_size)
            response_times.append(rt)
            
        return self.calculate_metrics("Round Robin", response_times)

    # -------------------------------------------------------
    # 2. ALGORITMA ANT COLONY OPTIMIZATION (REFACTORED)
    # -------------------------------------------------------
    def run_aco(self):
        self.reset_servers()
        response_times = []
        
        print(f"[*] Menjalankan ACO pada {len(self.requests)} requests...")
        
        for req_size in self.requests:
            probabilities = []
            sum_prob = 0.0
            
            # --- STEP 1: Hitung Probabilitas Pindah ke Tiap Server ---
            for server in self.servers:
                # [FIX LOGIC] Heuristik:
                # Code lama: 1 / load. 
                # Code baru: Capacity / (CurrentLoad + 1).
                # Ini memprioritaskan server kapasitas besar yg load-nya masih rendah.
                heuristic = server.capacity / (server.finish_time + 1.0)
                
                # Rumus ACO standar
                prob = (server.pheromone ** ACO_ALPHA) * (heuristic ** ACO_BETA)
                probabilities.append(prob)
                sum_prob += prob
            
            # Normalisasi probabilitas
            probabilities = [p / sum_prob for p in probabilities]
            
            # --- STEP 2: Roulette Wheel Selection ---
            # Semut memilih server berdasarkan probabilitas
            selected_server = random.choices(self.servers, weights=probabilities, k=1)[0]
            
            # --- STEP 3: Eksekusi Request ---
            rt = selected_server.process_request(req_size)
            response_times.append(rt)
            
            # --- STEP 4: Local Pheromone Update ---
            # Mengurangi pheromone di jalur yang baru dilewati (agar semut lain explore)
            # Atau menambah jika hasilnya bagus. Disini kita pakai Global Update saja di akhir batch
            # tapi kita bisa update sedikit agar adaptif thd load yg baru masuk.
            selected_server.pheromone = (1 - ACO_RHO) * selected_server.pheromone + \
                                        (ACO_RHO * (ACO_Q / (selected_server.finish_time + 1.0)))

        return self.calculate_metrics("ACO", response_times)

    def calculate_metrics(self, algo_name, response_times):
        # 1. Average Response Time (Lower is Better)
        avg_rt = np.mean(response_times)
        
        # 2. Makespan (Waktu total penyelesaian batch, Lower is Better)
        makespan = max(s.finish_time for s in self.servers)
        
        # 3. Throughput (Requests per Second based on Makespan)
        # Higher is Better
        throughput = len(self.requests) / makespan if makespan > 0 else 0
        
        # 4. Imbalance (Standard Deviation of Server Finish Times)
        # Menunjukkan seberapa timpang beban antar server
        server_loads = [s.finish_time for s in self.servers]
        load_variance = np.var(server_loads)
        
        return {
            "Algorithm": algo_name,
            "Avg Response Time": avg_rt,
            "Throughput": throughput,
            "Load Variance": load_variance,
            "Makespan": makespan,
            "Server Loads": [s.finish_time for s in self.servers],
            "Task Counts": [s.task_count for s in self.servers]
        }

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Generate random requests (Size 10 - 100)
    print("Generating requests...")
    np.random.seed(42) # Agar hasil reproducible
    requests_data = np.random.randint(10, 100, size=NUM_REQUESTS)

    sim = LoadBalancerSim(requests_data)
    
    # Run Simulations
    rr_metrics = sim.run_round_robin()
    aco_metrics = sim.run_aco()

    # ==========================================
    # PRINT RESULTS TABLE
    # ==========================================
    results_df = pd.DataFrame([rr_metrics, aco_metrics])
    cols = ["Algorithm", "Avg Response Time", "Throughput", "Load Variance", "Makespan"]
    print("\n" + "="*60)
    print("HASIL KOMPARASI PERFORMA (Lower RT/Var & Higher TP is Better)")
    print("="*60)
    print(results_df[cols].to_string(index=False))
    print("="*60)

    # Analisis Singkat Otomatis
    if aco_metrics["Avg Response Time"] < rr_metrics["Avg Response Time"]:
        print("\n[KESIMPULAN] ACO Terbukti Lebih Unggul ✅")
        print(f"ACO {rr_metrics['Avg Response Time']/aco_metrics['Avg Response Time']:.2f}x lebih cepat daripada Round Robin.")
    else:
        print("\n[KESIMPULAN] ACO Masih Kalah ❌ (Cek parameter beta/alpha)")

    # ==========================================
    # VISUALISASI
    # ==========================================
    
    # 1. Bar Chart: Server Load Distribution (Waktu Selesai tiap Server)
    # Ini membuktikan apakah beban terbagi sesuai kapasitas
    fig, ax = plt.subplots(figsize=(12, 6))
    
    indices = np.arange(NUM_SERVERS)
    width = 0.35
    
    # Urutkan server berdasarkan ID agar terlihat mana yg High Cap dan Low Cap
    # Ingat: Server 0-1 (High Cap), 2-4 (Med), 5-9 (Low)
    
    rects1 = ax.bar(indices - width/2, rr_metrics["Server Loads"], width, label='Round Robin', color='salmon')
    rects2 = ax.bar(indices + width/2, aco_metrics["Server Loads"], width, label='ACO', color='skyblue')
    
    ax.set_xlabel('Server ID (0-1: High Cap, 2-4: Med, 5-9: Low)')
    ax.set_ylabel('Total Finish Time (Load)')
    ax.set_title('Distribusi Beban Akhir per Server (Semakin Rata Semakin Baik)')
    ax.set_xticks(indices)
    ax.legend()
    
    # Tambahkan text kapasitas di atas bar
    def autolabel(rects):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            cap = SERVER_CAPACITIES[i]
            # ax.annotate(f'Cap:{cap}',
            #             xy=(rect.get_x() + rect.get_width() / 2, height),
            #             xytext=(0, 3),  # 3 points vertical offset
            #             textcoords="offset points",
            #             ha='center', va='bottom', fontsize=8, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.show()

    # 2. Visualisasi Performance Metrics Comparison
    metrics_to_plot = ["Avg Response Time", "Load Variance"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, metric in enumerate(metrics_to_plot):
        vals = [rr_metrics[metric], aco_metrics[metric]]
        axes[i].bar(["Round Robin", "ACO"], vals, color=['salmon', 'skyblue'])
        axes[i].set_title(metric)
        axes[i].set_ylabel("Value (Lower is Better)")
        
        # Add labels
        for j, v in enumerate(vals):
            axes[i].text(j, v, f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()