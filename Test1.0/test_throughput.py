import time 
import random 
import string 
import os 
import csv 
from dataclasses import dataclass, field 
from collections import defaultdict 
 
@dataclass 
class EncryptionArgs:
    plaintext: str 
    data_length: int 
    params: tuple 
 
def generate_random_string(length=32, charset=string.ascii_letters + string.digits, secure=False):
    if secure:
        return ''.join(random.SystemRandom().choices(charset, k=length))
    return ''.join(random.choices(charset, k=length))
 
def benchmark_throughput(algorithm_name, encrypt_function, decrypt_function, data_lengths, run_times, *args):
    times = defaultdict(lambda: {
        "encryption_throughput": [],
        "decryption_throughput": [],
        "data_lengths": []
    })
 
    loaded_args = {}
    for data_length in data_lengths:
        plaintext = generate_random_string(data_length)
        loaded_args[data_length] = EncryptionArgs(plaintext, data_length, ())
 
    for data_length in data_lengths:
        preloaded_data = loaded_args[data_length]
        plaintext = preloaded_data.plaintext 
 
        for _ in range(run_times):
            if algorithm_name == "AES":
                seed = generate_random_string(32, string.digits, secure=True)
                salt = generate_random_string(16, secure=True)
                start = time.perf_counter()
                encrypted_data = encrypt_function(plaintext, seed, salt)
                encryption_time = time.perf_counter() - start 
            elif algorithm_name == "RSA":
                start = time.perf_counter()
                encrypted_data, public_key, private_key = encrypt_function(plaintext)
                encryption_time = time.perf_counter() - start 
            elif algorithm_name == "ncRNA":
                seed = generate_random_string(32, string.digits, secure=True).encode()
                seed_sequence = generate_random_string(32, charset='ACGU', secure=True)
                salt = generate_random_string(16, secure=True)
                start = time.perf_counter()
                encrypted_data_tuple = encrypt_function(plaintext, seed, seed_sequence, salt)
                encryption_time = time.perf_counter() - start 
                encrypted_data = encrypted_data_tuple[0]
 
            # 計算加密吞吐量 (Bytes/Second)
            encryption_throughput = data_length / encryption_time 
            times[algorithm_name]["encryption_throughput"].append(encryption_throughput)
            times[algorithm_name]["data_lengths"].append(data_length)
 
            if algorithm_name == "AES":
                start = time.perf_counter()
                decrypt_function(encrypted_data, seed, salt)
                decryption_time = time.perf_counter() - start 
            elif algorithm_name == "RSA":
                start = time.perf_counter()
                decrypt_function(encrypted_data, private_key)
                decryption_time = time.perf_counter() - start 
            elif algorithm_name == "ncRNA":
                start = time.perf_counter()
                decrypt_function(encrypted_data, seed, seed_sequence, salt, *encrypted_data_tuple[1:])
                decryption_time = time.perf_counter() - start 
 
            # 計算解密吞吐量 (Bytes/Second)
            decryption_throughput = data_length / decryption_time 
            times[algorithm_name]["decryption_throughput"].append(decryption_throughput)
 
    return times 
 
def setup_algorithms():
    from algorithm.ncRNA3_5 import encrypt as ncRNA_encrypt, decrypt as ncRNA_decrypt 
    from algorithm.AES import aes_encrypt, aes_decrypt 
    from algorithm.RSA import rsa_encrypt, rsa_decrypt 
 
    algorithms = [
        ("ncRNA", ncRNA_encrypt, ncRNA_decrypt),
        ("AES", aes_encrypt, aes_decrypt),
        ("RSA", rsa_encrypt, rsa_decrypt)
    ]
 
    return algorithms 
 
def find_next_file_number(directory="./results/csv/throughput"):
    """返回下一个可用的文件编号"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    existing_files = os.listdir(directory)
    existing_numbers = []
    
    for file_name in existing_files:
        if file_name.endswith(".csv"):
            try:
                num = int(file_name.split("_")[1].split(".")[0])
                existing_numbers.append(num)
            except ValueError:
                continue 
    
    next_number = max(existing_numbers, default=0) + 1 
    return next_number 
 
def save_to_csv(all_throughputs, file_number):
    """將所有結果匯總到一個CSV文件"""
    file_path = f"./results/csv/throughput/throughput_{file_number}.csv"
    
    # 如果文件不存在，寫入表頭 
    file_exists = os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            # 寫入表頭 
            writer.writerow([
                "Algorithm",
                "Data Length",
                "Encryption Throughput (Bytes/Second)",
                "Decryption Throughput (Bytes/Second)"
            ])
 
        # 遍歷所有算法及其結果 
        for algo_name, throughput_data in all_throughputs.items():
            for i in range(len(throughput_data["encryption_throughput"])):
                data_length = throughput_data["data_lengths"][i]
                enc_throughput = throughput_data["encryption_throughput"][i]
                dec_throughput = throughput_data["decryption_throughput"][i]
                writer.writerow([
                    algo_name,
                    data_length,
                    enc_throughput,
                    dec_throughput 
                ])
 
    print(f"Results saved to {file_path}")
 
def main():
    data_lengths = [50, 100, 200, 400, 800, 1600, 3200, 6400]
    run_times = 100
 
    algorithms = setup_algorithms()
 
    # 生成下一个文件编号 
    file_number = find_next_file_number()
 
    # 存储所有算法的结果 
    all_throughputs = defaultdict(lambda: {
        "encryption_throughput": [],
        "decryption_throughput": [],
        "data_lengths": []
    })
 
    # 循環算法並進行加密解密吞吐量測試 
    for algo_name, encrypt_fn, decrypt_fn in algorithms:
        throughputs = benchmark_throughput(algo_name, encrypt_fn, decrypt_fn, data_lengths, run_times)
        
        # 合並每個算法的結果 
        for key in throughputs:
            all_throughputs[key]["encryption_throughput"].extend(throughputs[key]["encryption_throughput"])
            all_throughputs[key]["decryption_throughput"].extend(throughputs[key]["decryption_throughput"])
            all_throughputs[key]["data_lengths"].extend(throughputs[key]["data_lengths"])
 
    # 將所有結果保存到同一个CSV文件 
    save_to_csv(all_throughputs, file_number)
 
if __name__ == "__main__":
    main()