import random
import string
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
import os
import csv

@dataclass
class EncryptionArgs:
    plaintext: str
    data_length: int
    params: tuple

def generate_random_string(length=32, charset=string.ascii_letters + string.digits, secure=False):
    if secure:
        return ''.join(random.SystemRandom().choices(charset, k=length))
    return ''.join(random.choices(charset, k=length))

def calculate_entropy(data):
    """计算密钥或密文的熵量"""
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()
    data_length = len(byte_data)
    if data_length == 0:
        return 0.0
    frequencies = Counter(byte_data)
    freqs = np.array(list(frequencies.values()), dtype=float)
    probabilities = freqs / data_length
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def benchmark_entropy(algorithm_name, encrypt_function, data_lengths, runs=10, *args):
    entropies = defaultdict(list)

    for data_length in data_lengths:
        run_entropies = []
        for _ in range(runs):
            plaintext = generate_random_string(data_length)

            if algorithm_name == "AES":
                seed = generate_random_string(32, string.digits, secure=True)
                salt = generate_random_string(16, secure=True)
                encrypted_data = encrypt_function(plaintext, seed, salt)
            elif algorithm_name == "RSA":
                encrypted_data, public_key, private_key = encrypt_function(plaintext)
            elif algorithm_name == "ncRNA":
                seed = generate_random_string(32, string.digits, secure=True).encode()
                seed_sequence = generate_random_string(32, charset='ACGU', secure=True)
                salt = generate_random_string(16, secure=True)
                encrypted_data_tuple = encrypt_function(plaintext, seed, seed_sequence, salt)
                encrypted_data = encrypted_data_tuple[0]

            entropy = calculate_entropy(encrypted_data)
            run_entropies.append(entropy)

        average_entropy = np.mean(run_entropies)
        entropies[algorithm_name].append((data_length, average_entropy))

    return entropies

def setup_algorithms():
    from algorithm.ncRNA3_5 import encrypt as ncRNA_encrypt, decrypt as ncRNA_decrypt
    from algorithm.AES import aes_encrypt, aes_decrypt
    from algorithm.RSA import rsa_encrypt, rsa_decrypt

    algorithms = [
        ("ncRNA", ncRNA_encrypt),
        ("AES", aes_encrypt),
        ("RSA", rsa_encrypt)
    ]

    return algorithms

def find_next_file_number(directory="./results/csv/shang"):
    """返回下一个可用的文件编号"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    existing_files = os.listdir(directory)
    existing_numbers = []
    
    for file_name in existing_files:
        if file_name.endswith(".csv"):
            try:
                num = int(file_name.split("_")[1].split(".")[0])  # 以文件名中的数字部分来判断编号
                existing_numbers.append(num)
            except ValueError:
                continue
    
    next_number = max(existing_numbers, default=0) + 1
    return next_number

def save_to_csv(entropies, file_number, directory="./results/csv/shang"):
    """将所有结果汇总到一个CSV文件"""
    file_path = f"{directory}/shang_{file_number}.csv"
    
    # 如果文件不存在，写入表头
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # 写入表头
            writer.writerow(["Algorithm", "Data Length", "Average Entropy"])

        # 遍历所有算法及其结果
        for algo_name, data in entropies.items():
            for data_length, average_entropy in data:
                writer.writerow([algo_name, data_length, f"{average_entropy:.16f}"])

    print(f"Results saved to {file_path}")


def main():
    data_lengths = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    runs = 20

    algorithms = setup_algorithms()

    all_entropies = defaultdict(list)  # 用来收集所有算法的熵值数据

    # 运行并获取熵值数据
    for algo_name, encrypt_fn in algorithms:
        entropies = benchmark_entropy(algo_name, encrypt_fn, data_lengths, runs)
        
        # 将每个算法的结果合并到 all_entropies
        for data_length, average_entropy in entropies[algo_name]:
            all_entropies[algo_name].append((data_length, average_entropy))

        # 查找下一个文件编号并保存数据
    file_number = find_next_file_number()
    save_to_csv(all_entropies, file_number)

if __name__ == "__main__":
    main()
