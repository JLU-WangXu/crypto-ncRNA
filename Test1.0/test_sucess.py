import random
import string
import csv
import os
from collections import defaultdict

def generate_random_string(length=32, charset=string.ascii_letters + string.digits, secure=False):
    """
    生成随机字符串，secure=True时使用系统随机生成器
    """
    if secure:
        return ''.join(random.SystemRandom().choices(charset, k=length))
    return ''.join(random.choices(charset, k=length))

def benchmark_ncRNA_success_rate(ncRNA_encrypt, ncRNA_decrypt, data_lengths, run_times):
    """
    针对ncRNA算法，统计不同数据长度下加密与解密的成功率
    """
    # 记录每个数据长度下的加密/解密成功与失败次数
    results = defaultdict(lambda: {"enc_success": 0, "enc_fail": 0,
                                   "dec_success": 0, "dec_fail": 0})
    
    for data_length in data_lengths:
        # 生成测试明文
        plaintext = generate_random_string(data_length)
        
        for _ in range(run_times):
            try:
                # 加密前准备必要的参数
                seed = generate_random_string(32, string.digits, secure=True).encode()
                seed_sequence = generate_random_string(32, charset='ACGU', secure=True)
                salt = generate_random_string(16, secure=True)
                
                # 调用加密函数，假定返回一个元组，首项为加密数据，后续为解密所需的附加参数
                encrypted_data_tuple = ncRNA_encrypt(plaintext, seed, seed_sequence, salt)
                encrypted_data = encrypted_data_tuple[0]
                results[data_length]["enc_success"] += 1
            except Exception as e:
                # 如果加密过程中发生异常，则计为加密失败，同时解密也视为失败
                results[data_length]["enc_fail"] += 1
                results[data_length]["dec_fail"] += 1
                continue

            try:
                # 调用解密函数进行解密，并判断解密后的明文是否与原明文一致
                decrypted_text = ncRNA_decrypt(encrypted_data, seed, seed_sequence, salt, *encrypted_data_tuple[1:])
                if decrypted_text == plaintext:
                    results[data_length]["dec_success"] += 1
                else:
                    results[data_length]["dec_fail"] += 1
            except Exception as e:
                results[data_length]["dec_fail"] += 1

    return results

def save_success_rate_csv(results, file_path):
    """
    将测试结果保存到 CSV 文件中
    """
    # 如果目录不存在则创建
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(["Data Length", "Total Runs",
                         "Encryption Success", "Encryption Failure",
                         "Decryption Success", "Decryption Failure",
                         "Encryption Success Rate (%)", "Decryption Success Rate (%)"])
        
        # 对每个数据长度下的结果计算成功率并写入CSV
        for data_length, counts in results.items():
            total_runs = counts["enc_success"] + counts["enc_fail"]
            enc_success_rate = (counts["enc_success"] / total_runs * 100) if total_runs > 0 else 0
            dec_success_rate = (counts["dec_success"] / total_runs * 100) if total_runs > 0 else 0
            
            writer.writerow([data_length, total_runs,
                             counts["enc_success"], counts["enc_fail"],
                             counts["dec_success"], counts["dec_fail"],
                             f"{enc_success_rate:.2f}", f"{dec_success_rate:.2f}"])
    
    print(f"Success rate CSV saved to {file_path}")

def find_next_file_number(directory):
    """
    返回指定目录下下一个可用的文件编号
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    existing_files = os.listdir(directory)
    existing_numbers = []
    
    for file_name in existing_files:
        if file_name.endswith(".csv"):
            try:
                # 假设文件名格式为 ncRNA_success_数字.csv
                num = int(file_name.split("_")[2].split(".")[0])
                existing_numbers.append(num)
            except ValueError:
                continue
    
    next_number = max(existing_numbers, default=0) + 1
    return next_number

def main():
    # 测试参数：不同数据长度以及每个长度下运行次数
    data_lengths = [50, 100, 1000, 100000, 1000000]
    run_times = 10

    # 导入ncRNA算法的加密与解密函数（请确保此模块路径正确）
    from algorithm.ncRNA3_5 import encrypt as ncRNA_encrypt, decrypt as ncRNA_decrypt

    # 进行成功率测试
    results = benchmark_ncRNA_success_rate(ncRNA_encrypt, ncRNA_decrypt, data_lengths, run_times)

    # 指定保存结果的目录和文件编号
    directory = "./results/csv/success_rate"
    next_number = find_next_file_number(directory)
    file_path = f"{directory}/ncRNA_success_{next_number}.csv"

    # 保存结果到CSV
    save_success_rate_csv(results, file_path)

if __name__ == "__main__":
    main()