# -------------------------------
# 标准库导入
# -------------------------------
import sys
import os
import re
import time
import json
import random
import string
import base64

# -------------------------------
# 第三方库导入
# -------------------------------
import psutil
import numpy as np
import matplotlib.pyplot as plt


# 修改相对路径为绝对路径
sys.path.append(os.path.abspath('D:/大学/2024/RNA密码算法/crypto-ncRNA-main/Test/__pycache__'))

# -------------------------------
# 环境设置与文件操作工具函数部分
# -------------------------------

def bytes_to_base64(d):
    """递归遍历字典，将 bytes 转换为 base64 字符串"""
    if isinstance(d, bytes):
        return base64.b64encode(d).decode('utf-8')
    elif isinstance(d, dict):
        return {k: bytes_to_base64(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [bytes_to_base64(v) for v in d]
    return d

def save_results_to_json(results_db, json_folder_path):
    """将结果保存到 JSON 文件夹中，保存时进行 Base64 编码"""
    ensure_directory_exists(json_folder_path)
    
    # 查找现有的文件夹并提取序号
    existing_folders = [f for f in os.listdir(json_folder_path) if os.path.isdir(os.path.join(json_folder_path, f))]
    folder_nums = [int(re.search(r'(\d+)', folder).group(1)) for folder in existing_folders if re.search(r'(\d+)', folder)]
    
    next_num = max(folder_nums, default=0) + 1
    new_folder_name = f"{next_num}_results_db"
    new_folder_path = os.path.join(json_folder_path, new_folder_name)
    
    ensure_directory_exists(new_folder_path)

    # 遍历 results_db 并分别存储每个算法的结果到不同的 JSON 文件中
    for algorithm_name, result_by_data_length in results_db.items():
        json_file_path = os.path.join(new_folder_path, f"{algorithm_name}.json")
        
        # 在存储前对结果中的 bytes 类型数据进行 Base64 编码
        encoded_result = bytes_to_base64(result_by_data_length)
        
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(encoded_result, json_file, ensure_ascii=False, indent=4)

def import_functions_from_pyc(modules):
    """导入加密和解密函数，供后续调用"""
    functions = {}
    for algorithm_name, module_name in modules.items():
        module = __import__(module_name)
        encrypt_function = getattr(module, 'encrypt')
        decrypt_function = getattr(module, 'decrypt')
        functions[algorithm_name] = (encrypt_function, decrypt_function)
    return functions

def generate_random_string(length=32, charset=string.ascii_letters + string.digits):
    """生成指定长度的随机字符串"""
    return ''.join(random.choices(charset, k=length))

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    os.makedirs(directory, exist_ok=True)

def get_folder_name(img_dir, min_data_length, max_data_length):
    """根据文件夹内的已有文件夹，生成一个递增的新文件夹名称"""
    ensure_directory_exists(img_dir)
    
    # 查找现有的文件夹并提取序号
    existing_folders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]
    folder_nums = [int(re.search(r'(\d+)', folder).group(1)) for folder in existing_folders if re.search(r'(\d+)', folder)]
    
    next_num = max(folder_nums, default=0) + 1
    new_folder_name = f"{next_num}_results_{min_data_length}-{max_data_length}"
    new_folder_path = os.path.join(img_dir, new_folder_name)
    
    ensure_directory_exists(new_folder_path)
    return new_folder_path

# -------------------------------
# 图表绘制函数部分
# -------------------------------

def set_common_properties(ax, title, xlabel, ylabel):
    """统一设置图表的通用属性，避免重复代码"""
    ax.set_title(title, pad=5)
    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.grid(True, linestyle='--', linewidth=0.5)

def plot_and_save(results_db, data_lengths, img_dir):
    """绘制图表并保存到指定目录"""
    min_data_length = min(data_lengths)
    max_data_length = max(data_lengths)
    
    folder_path = get_folder_name(img_dir, min_data_length, max_data_length)
    
    fig, axes = plt.subplots(len(data_lengths), 4, figsize=(16, 4 * len(data_lengths)))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.rcParams.update({'font.size': 7})
    
    for idx, data_length in enumerate(data_lengths):
        ax_encrypt, ax_decrypt, ax_bar_encrypt, ax_bar_decrypt = axes[idx]
        
        avg_encrypt_times = []
        avg_decrypt_times = []
        algorithms = []
        
        for algorithm_name, run_data_by_length in results_db.items():
            run_data = run_data_by_length[data_length]
            encrypt_times = [entry["Encryption Time (s)"] for entry in run_data]
            decrypt_times = [entry["Decryption Time (s)"] for entry in run_data]
            
            avg_encrypt_times.append(np.mean(encrypt_times))
            avg_decrypt_times.append(np.mean(decrypt_times))
            algorithms.append(algorithm_name)
            
            x_values = list(range(1, len(encrypt_times) + 1))
            ax_encrypt.plot(x_values, encrypt_times, label=algorithm_name)
            ax_decrypt.plot(x_values, decrypt_times, label=algorithm_name)
        
        set_common_properties(ax_encrypt, f'Encryption Time - Data Length: {data_length}', 'Run Number', 'Time (s)')
        set_common_properties(ax_decrypt, f'Decryption Time - Data Length: {data_length}', 'Run Number', 'Time (s)')
        ax_encrypt.legend(fontsize=6)
        ax_decrypt.legend(fontsize=6)
        
        ax_bar_encrypt.bar(algorithms, avg_encrypt_times, color='skyblue')
        ax_bar_decrypt.bar(algorithms, avg_decrypt_times, color='salmon')
        
        for i, v in enumerate(avg_encrypt_times):
            ax_bar_encrypt.text(i, v, f'{v:.2f}', ha='center', fontsize=6)
        for i, v in enumerate(avg_decrypt_times):
            ax_bar_decrypt.text(i, v, f'{v:.2f}', ha='center', fontsize=6)
        
        set_common_properties(ax_bar_encrypt, f'Average Encryption Time - Data Length: {data_length}', 'Algorithm', 'Time (s)')
        set_common_properties(ax_bar_decrypt, f'Average Decryption Time - Data Length: {data_length}', 'Algorithm', 'Time (s)')

    fig.tight_layout()

    filename = f"{folder_path}/results_{min_data_length}-{max_data_length}.png"
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)


# -------------------------------
# 加密和解密功能实现部分
# -------------------------------

def run_encryption(plaintext, encrypt_function, seed, seed_sequence, salt):
    """执行加密并返回加密数据和时间"""
    encrypt_start_time = time.perf_counter()
    encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt_function(plaintext, seed, seed_sequence, salt)
    encrypt_time = time.perf_counter() - encrypt_start_time
    return encrypted_data_with_checksum, substitution_matrix, indices_order, encrypt_time

def run_decryption(encrypted_data_with_checksum, decrypt_function, seed, seed_sequence, salt, substitution_matrix, indices_order):
    """执行解密并返回解密数据和时间"""
    decryption_start_time = time.perf_counter()
    decrypted_data = decrypt_function(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)
    decryption_time = time.perf_counter() - decryption_start_time
    return decrypted_data, decryption_time

# -------------------------------
# 主程序执行部分
# -------------------------------

if __name__ == "__main__":
    """主程序入口，加载数据并运行加密解密算法"""
    # 修改文件路径
    war_and_peace_txt = open('D:/大学/2024/RNA密码算法/crypto-ncRNA-main/Test/war_and_peace.txt', 'r', encoding="UTF-8").read()
    nist_test_data_txt = open('D:/大学/2024/RNA密码算法/crypto-ncRNA-main/Test/nist_test_data.txt', 'r').read()

    algorithm_modules = {
        "ncRNA3_1_op": "ncRNA3_1_op",
        "AES": "AES_crypto",
        "RSA": "RSA_crypto",
    }

    dataset = {
        "text data": war_and_peace_txt,
        "html data": nist_test_data_txt
    }

    data_lengths = [50, 100, 200, 500]
    run_times = 5
    results_db = {}

    # 修改输出目录路径
    json_folder_path = "D:/大学/2024/RNA密码算法/crypto-ncRNA-main/Test/dbjson"
    img_dir = "D:/大学/2024/RNA密码算法/crypto-ncRNA-main/Test/images"
    data_type = "text data"
    seed = generate_random_string(32, string.digits)
    seed_sequence = generate_random_string(32, "ACGU")
    salt = generate_random_string(32)

    algorithm_functions = import_functions_from_pyc(algorithm_modules)

    for char_data in data_lengths:
        plaintext = dataset[data_type][:char_data]

        for algorithm_name, (encrypt_function, decrypt_function) in algorithm_functions.items():
            algorithm_results = []
            for _ in range(run_times):
                encrypted_data_with_checksum, substitution_matrix, indices_order, encrypt_time = run_encryption(
                    plaintext, encrypt_function, seed, seed_sequence, salt)

                decrypted_data, decryption_time = run_decryption(
                    encrypted_data_with_checksum, decrypt_function, seed, seed_sequence, salt, substitution_matrix, indices_order)

                result_info = {
                    "Encryption Time (s)": encrypt_time,
                    "Decryption Time (s)": decryption_time,
                    "Encrypted Data": encrypted_data_with_checksum,
                    "Decrypted Data": decrypted_data,
                }
                algorithm_results.append(result_info)
            results_db.setdefault(algorithm_name, {})[char_data] = algorithm_results
            
    save_results_to_json(results_db, json_folder_path)
    plot_and_save(results_db, data_lengths, img_dir)
