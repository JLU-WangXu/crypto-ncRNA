# Utility Functions
import os
import time
import json
import random
import string
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tracemalloc
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Util.Padding import pad, unpad
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import math
from multiprocessing import Process, Queue


# Global variable for single-threaded operation
SINGLE_THREAD_MODE = True

def ensure_directory_exists(directory):
    """Ensure the given directory exists, create if not."""
    os.makedirs(directory, exist_ok=True)


def generate_random_string(length=32, charset=string.ascii_letters + string.digits):
    if charset is not None:
        return ''.join(random.choices(charset, k=length))
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def save_binary_file(data, file_path):
    """Save binary data to a file."""
    with open(file_path, 'wb') as bin_file:
        bin_file.write(data)


def save_results_to_json(results, folder_path, filename, iteration):
    """Save the given results to a JSON file."""
    ensure_directory_exists(folder_path)
    file_path = os.path.join(folder_path, f"iteration_{iteration}_{filename}")
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4)

# Encryption and Decryption in an Isolated Process
def run_in_isolated_process(target_func, queue, *args):
    p = Process(target=target_func, args=(queue, *args))
    p.start()
    p.join()
    return queue.get()


# Function to Benchmark Encryption and Decryption
def encryption_benchmark_task(queue, algorithm_name, encrypt_function, decrypt_function, plaintext, *args):
    # Start measuring memory allocation
    tracemalloc.start()

    # Encrypt
    start_time = time.perf_counter()
    if algorithm_name == "AES":
        seed, salt = args[:2]
        encrypted_data = encrypt_function(plaintext, seed, salt)
    elif algorithm_name == "RSA":
        encrypted_data, public_key, private_key = encrypt_function(plaintext)
    elif algorithm_name == "ncRNA":
        seed, seed_sequence, salt = args[:3]
        encrypted_data_tuple = encrypt_function(plaintext, seed, seed_sequence, salt)
        encrypted_data, substitution_matrix, indices_order = encrypted_data_tuple
    else:
        encrypted_data = encrypt_function(plaintext, *args)
    encryption_time = time.perf_counter() - start_time

    # Stop measuring memory allocation
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_usage = peak

    # Decrypt
    start_time = time.perf_counter()
    if algorithm_name == "AES":
        decrypted_data = decrypt_function(encrypted_data, seed, salt)
    elif algorithm_name == "RSA":
        decrypted_data = decrypt_function(encrypted_data, private_key)
    elif algorithm_name == "ncRNA":
        decrypted_data = decrypt_function(encrypted_data, seed, seed_sequence, salt, substitution_matrix, indices_order)
    else:
        decrypted_data = decrypt_function(encrypted_data, *args)
    decryption_time = time.perf_counter() - start_time

    # Collect results
    run_result = {
        "Encryption Time (s)": round(encryption_time, 8),
        "Decryption Time (s)": round(decryption_time, 8),
        "Memory Usage (bytes)": memory_usage,
        "Decrypted Correctly": decrypted_data == plaintext
    }
    queue.put(run_result)


# Benchmarking Encryption and Decryption
def benchmark_encryption(algorithm_name, encrypt_function, decrypt_function, data_lengths, run_times, iteration=1, *args, keys=None):
    results = {}
    for data_length in data_lengths:
        plaintext = generate_random_string(data_length)
        runs = []
        for run in range(1, run_times + 1):
            queue = Queue()
            run_result = run_in_isolated_process(encryption_benchmark_task, queue, algorithm_name, encrypt_function, decrypt_function, plaintext, *args)
            if keys:
                run_result["Keys"] = keys
            runs.append(run_result)
        results[data_length] = runs
    return results  

# Plotting Results with Dynamic Canvas Adjustment
def plot_results(results, output_folder, iteration):
    ensure_directory_exists(output_folder)
    metrics = ["Encryption Time (s)", "Decryption Time (s)", "Memory Usage (bytes)"]
    num_metrics = len(metrics)

    # Self-Adaptive Plot Layout for Each Data Length
    data_lengths = list(results[list(results.keys())[0]].keys())
    num_data_lengths = len(data_lengths)

    # Dynamically determine figure size based on number of plots and length of titles
    total_plots = num_data_lengths * num_metrics * 2
    figure_width = max(20, total_plots * 1.5)  # Dynamically set width, ensuring sufficient space
    figure_height = 4 * num_data_lengths  # Keep height proportional to the number of data lengths

    plt.figure(figsize=(figure_width, figure_height))
    plot_index = 1
    plt.subplots_adjust(hspace=0.6, wspace=0.5)  # Adjust spacing between subplots to prevent overlap

    for data_length in data_lengths:
        # Generate all runtime line charts first
        for metric in metrics:
            ax = plt.subplot(num_data_lengths, num_metrics * 2, plot_index)
            for algorithm_name, data in results.items():
                run_values = [run[metric] for run in data[data_length]]
                ax.plot(range(1, len(run_values) + 1), run_values, label=algorithm_name, marker='o')
            ax.set_xlabel('Run Number', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)

            # Calculate title length and adjust if needed
            title = f'{metric} - Data Length: {data_length} Bytes'
            estimated_title_width = len(title) * 12  # Assume each character is about 12 pixels wide
            current_figure_width = plt.gcf().get_size_inches()[0] * plt.gcf().dpi
            if estimated_title_width > current_figure_width:
                # Wrap title by adding newline after '-'
                title = title.replace(' - ', ' -\n')

            ax.set_title(title, fontsize=12)  # Adjusted font size for clarity
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize='small')
            ax.grid(True)
            plot_index += 1

        # Generate all average bar charts after runtime charts
        for metric in metrics:
            ax = plt.subplot(num_data_lengths, num_metrics * 2, plot_index)
            avg_values = []
            for algorithm_name, data in results.items():
                avg_value = np.mean([run[metric] for run in data[data_length]])
                avg_values.append(avg_value)
                ax.bar(algorithm_name, avg_value, alpha=0.7, width=0.4)
                ax.text(algorithm_name, avg_value, f'{avg_value:.4f}', fontsize=8, ha='center')
            ax.set_xlabel('Algorithm', fontsize=10)
            ax.set_ylabel(f'Average {metric}', fontsize=10)

            # Calculate title length and adjust if needed
            title = f'Average {metric} - Data Length: {data_length} Bytes'
            estimated_title_width = len(title) * 12  # Assume each character is about 12 pixels wide
            if estimated_title_width > current_figure_width:
                # Wrap title by adding newline after '-'
                title = title.replace(' - ', ' -\n')

            ax.set_title(title, fontsize=12)  # Adjusted font size for clarity
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.grid(True)
            plot_index += 1

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'iteration_{iteration}_combined_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Decryption Accuracy
    plt.figure(figsize=(max(12, total_plots * 0.5), 6))  # Adjusted width dynamically for better clarity
    ax = plt.gca()
    for algorithm_name, data in results.items():
        accuracies = [sum(1 for run in data[data_length] if run["Decrypted Correctly"]) / len(data[data_length]) for data_length in data.keys()]
        ax.plot(data.keys(), accuracies, label=algorithm_name, marker='o')
    ax.set_xlabel('Data Length (bytes)', fontsize=12)
    ax.set_ylabel('Decryption Success Rate (%)', fontsize=12)
    ax.set_title('Decryption Success Rate Across Algorithms and Data Lengths', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'iteration_{iteration}_decryption_success_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Main Execution
if __name__ == "__main__":
    from algorithm.ncRNA3_4 import encrypt as ncRNA_encrypt, decrypt as ncRNA_decrypt
    from algorithm.AES import aes_encrypt, aes_decrypt
    from algorithm.RSA import rsa_encrypt, rsa_decrypt

    data_lengths = [100, 500, 1000, 5000, 10000]
    run_times = 10
    results_folder = './results'
    img_folder = './results/images'
    json_folder = './results/json'

    # Ensure folders exist
    ensure_directory_exists(results_folder)
    ensure_directory_exists(img_folder)
    ensure_directory_exists(json_folder)

    iteration = 1 if not os.listdir(json_folder) else len([name for name in os.listdir(json_folder) if name.startswith('iteration_')]) + 1

    results = {}

    # AES Benchmark
    seed = generate_random_string(32, string.digits)
    salt = generate_random_string(16)
    aes_results = benchmark_encryption("AES", aes_encrypt, aes_decrypt, data_lengths, run_times, iteration, seed, salt, keys={"Seed": seed, "Salt": salt})
    results["AES"] = aes_results

    # RSA Benchmark
    rsa_results = benchmark_encryption("RSA", rsa_encrypt, rsa_decrypt, data_lengths, run_times, iteration)
    results["RSA"] = rsa_results

    # ncRNA Benchmark
    seed_sequence = generate_random_string(32, charset='ACGU')
    nc_rna_results = benchmark_encryption("ncRNA", ncRNA_encrypt, ncRNA_decrypt, data_lengths, run_times, iteration, seed.encode(), seed_sequence, salt, keys={"Seed": seed, "Seed Sequence": seed_sequence, "Salt": salt})
    results["ncRNA"] = nc_rna_results

    # Save Results to JSON
    save_results_to_json(results, json_folder, "encryption_results.json", iteration)

    plot_results(results, img_folder, iteration)
