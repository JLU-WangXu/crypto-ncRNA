import sys
import os
import time
import random
import string
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.append(os.path.abspath('./Test/__pycache__'))

# -------------------------------
# 工具函数部分
# -------------------------------

def generate_random_string(length=32, charset=string.ascii_letters + string.digits):
    return ''.join(random.choices(charset, k=length))

def calculate_entropy(data):
    byte_data = data if isinstance(data, bytes) else data.encode()
    data_length = len(byte_data)
    if data_length == 0:
        return 0.0
    frequencies = Counter(byte_data)
    probs = np.array(list(frequencies.values())) / data_length
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def bytes_to_bin(byte_data):
    if isinstance(byte_data, list):
        return ''.join(bytes_to_bin(item) for item in byte_data)
    return ''.join(format(byte, '08b') for byte in byte_data)

# -------------------------------
# 图表绘制辅助函数
# -------------------------------

def plot_time(ax, x_values, y_values, label, marker=None, title=None, xlabel=None, ylabel=None):
    ax.plot(x_values, y_values, label=label, marker=marker)
    ax.set_title(title, pad=5, fontsize=10)
    ax.set_xlabel(xlabel, labelpad=5, fontsize=8)
    ax.set_ylabel(ylabel, labelpad=5, fontsize=8)
    ax.legend(fontsize=8)

def plot_bar(ax, categories, values, color, title, xlabel, ylabel):
    ax.bar(categories, values, color=color)
    ax.set_title(title, pad=5, fontsize=10)
    ax.set_xlabel(xlabel, labelpad=5, fontsize=8)
    ax.set_ylabel(ylabel, labelpad=5, fontsize=8)

# -------------------------------
# 图表绘制部分
# -------------------------------

def plot_results(results_db, data_lengths):
    max_per_figure = 3

    for start_idx in range(0, len(data_lengths), max_per_figure):
        current_lengths = data_lengths[start_idx:start_idx + max_per_figure]
        
        fig, axes = plt.subplots(len(current_lengths), 4, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        for idx, char_data in enumerate(current_lengths):
            ax_encrypt, ax_decrypt, ax_bar_encrypt, ax_bar_decrypt = axes[idx]

            avg_encrypt_times = []
            avg_decrypt_times = []
            algorithms = []

            for algorithm_name, run_data_by_length in results_db.items():
                run_data = run_data_by_length[char_data]
                encrypt_times = [entry["Encryption Time (s)"] for entry in run_data]
                decrypt_times = [entry["Decryption Time (s)"] for entry in run_data]
                x_values = list(range(1, len(encrypt_times) + 1))

                ax_encrypt.set_yscale('log')
                ax_decrypt.set_yscale('log')
                
                formatter = ticker.ScalarFormatter()
                formatter.set_scientific(False)
                
                ax_encrypt.yaxis.set_major_formatter(formatter)
                ax_decrypt.yaxis.set_major_formatter(formatter)

                plot_time(ax_encrypt, x_values, encrypt_times, f'{algorithm_name}', marker=None, title=f'Encryption Time - Data Length: {char_data}', xlabel='Run Number', ylabel='Time (s)')
                plot_time(ax_decrypt, x_values, decrypt_times, f'{algorithm_name}', marker=None, title=f'Decryption Time - Data Length: {char_data}', xlabel='Run Number', ylabel='Time (s)')

                avg_encrypt_times.append(np.mean(encrypt_times))
                avg_decrypt_times.append(np.mean(decrypt_times))
                algorithms.append(algorithm_name)

            plot_bar(ax_bar_encrypt, algorithms, avg_encrypt_times, 'skyblue', f'Avg Encryption Time\n(Data Length: {char_data})', 'Algorithm', 'Avg Time (s)')
            for i, v in enumerate(avg_encrypt_times):
                ax_bar_encrypt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

            plot_bar(ax_bar_decrypt, algorithms, avg_decrypt_times, 'lightgreen', f'Avg Decryption Time\n(Data Length: {char_data})', 'Algorithm', 'Avg Time (s)')
            for i, v in enumerate(avg_decrypt_times):
                ax_bar_decrypt.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

# -------------------------------
# 主程序执行部分
# -------------------------------

def import_functions_from_pyc(modules):
    functions = {}
    for algorithm_name, module_name in modules.items():
        module = __import__(module_name)
        encrypt_function = getattr(module, 'encrypt')
        decrypt_function = getattr(module, 'decrypt')
        functions[algorithm_name] = (encrypt_function, decrypt_function)
    return functions

if __name__ == "__main__":
    war_and_peace_txt = open('./Test/war_and_peace.txt', 'r', encoding="UTF-8").read()
    nist_test_data_txt = open('./Test/nist_test_data.txt', 'r').read()

    algorithm_modules = {
        "ncRNA3_1_optimization": "ncRNA3_1_optimization",
        "AES_crypto": "AES_crypto",
        "RSA_crypto": "RSA_crypto",
    }

    dataset = {
        "text data": war_and_peace_txt,
        "html data": nist_test_data_txt
    }

    run_times = 10
    data_type = "text data"
    seed = generate_random_string(32, string.digits)
    seed_sequence = generate_random_string(32, "ACGU")
    salt = generate_random_string(32)

    data_lengths = [50, 100, 200]
    results_db = {}

    algorithm_functions = import_functions_from_pyc(algorithm_modules)

    total_tasks = len(data_lengths) * len(algorithm_modules)
    completed_tasks = 0

    for char_data in data_lengths:
        plaintext = dataset[data_type][:char_data]

        for algorithm_name, (encrypt_function, decrypt_function) in algorithm_functions.items():
            algorithm_results = []
            for _ in range(run_times):
                encrypt_start_time = time.time()
                encrypt_args = (plaintext, seed, seed_sequence, salt)
                encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt_function(*encrypt_args)
                encrypt_time = time.time() - encrypt_start_time

                decrypt_args = (encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)
                decryption_start_time = time.time()
                decrypted_data = decrypt_function(*decrypt_args)
                decryption_time = time.time() - decryption_start_time

                encrypted_bytes = encrypted_data_with_checksum[0] if encrypted_data_with_checksum else b""
                try:
                    entropy_value = calculate_entropy(encrypted_bytes)
                    entropy_value_formatted = float("{:.4f}".format(entropy_value))
                except Exception as e:
                    entropy_value_formatted = "NA"

                encrypted_data_bin = "NA"
                for i in range(len(encrypted_data_with_checksum)):
                    try:
                        encrypted_data_bin = encrypted_bytes + bytes_to_bin(encrypted_data_with_checksum[i])
                    except Exception as e:
                        encrypted_data_bin = "NA"

                result_info = {
                    "Encryption Time (s)": encrypt_time,
                    "Decryption Time (s)": decryption_time,
                    "Encrypted Data": encrypted_data_with_checksum,
                    "Decrypted Data": decrypted_data,
                    "Entropy (bits/byte)": entropy_value_formatted,
                    "Encrypted Data (bin)": encrypted_data_bin,
                    "Encrypted Data Length": len(encrypted_data_bin)
                }
                algorithm_results.append(result_info)

            results_db.setdefault(algorithm_name, {})[char_data] = algorithm_results

            completed_tasks += 1
            print(f"Completed {completed_tasks}/{total_tasks} tasks (Data Length: {char_data}, Algorithm: {algorithm_name})")

    plot_results(results_db, data_lengths)
