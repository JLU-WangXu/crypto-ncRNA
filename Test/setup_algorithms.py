import sys
import os
import time
import random
import string
import numpy as np
from collections import Counter

# 添加 __pycache__ 目录路径
sys.path.append(os.path.abspath('./Test/__pycache__'))

# -------------------------------
# 工具函数部分
# -------------------------------

def import_function_from_pyc(module_name, function_name):
    """导入指定的模块函数"""
    module = __import__(module_name)
    return getattr(module, function_name)

def generate_random_seed(length=32):
    """生成指定长度的随机数字字符串"""
    return ''.join(random.choices(string.digits, k=length))

def generate_random_seed_sequence(length=32):
    """生成指定长度的碱基序列 (A, C, G, U)"""
    return ''.join(random.choices(['A', 'C', 'G', 'U'], k=length))

def generate_random_salt(length=32):
    """生成指定长度的随机盐值"""
    charset = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choices(charset, k=length))

def calculate_entropy(data):
    """计算给定数据的熵（以 bits/byte 为单位）"""
    byte_data = data if isinstance(data, bytes) else data.encode()
    data_length = len(byte_data)
    if data_length == 0:
        return 0.0
    frequencies = Counter(byte_data)
    probs = np.array(list(frequencies.values())) / data_length
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def bytes_to_bin(byte_data):
    """将字节数据或字节列表转换为二进制字符串"""
    if isinstance(byte_data, list):
        return ''.join(bytes_to_bin(item) for item in byte_data)
    return ''.join(format(byte, '08b') for byte in byte_data)


def truncated_dict_print(data, length=100):
    """对字典中的每个字段进行截断"""
    truncated_data = {}
    for key, value in data.items():
        value_str = str(value)
        if isinstance(value, (bytes, list)):
            value_str = repr(value)
        if len(value_str) > length:
            truncated_data[key] = value_str[:length] + '... (truncated)'
        else:
            truncated_data[key] = value_str
    return truncated_data
# -------------------------------
# 核心算法执行部分
# -------------------------------

def run_algorithm(algorithm_name, encrypt_args, decrypt_args, encrypt_function, decrypt_function):
    """运行加密和解密算法，并记录时间和结果"""
    # 加密
    start_time = time.time()
    encrypted_data = encrypt_function(*encrypt_args)
    encryption_time = time.time() - start_time
    
    # 解密
    start_time = time.time()
    decrypted_data = decrypt_function(*decrypt_args)
    decryption_time = time.time() - start_time
    
    # 计算加密数据的熵值
    try:
        encrypted_bytes = encrypted_data[0]
        entropy_value = calculate_entropy(encrypted_bytes)
        entropy_value_formatted = float("{:.4f}".format(entropy_value))
    except Exception as e:
        entropy_value_formatted = "NA"
    
    try:
        encrypted_data_bin = bytes_to_bin(encrypted_data[0])
    except Exception as e:
        encrypted_data_bin = "NA"

    return {
        "Algorithm": algorithm_name,
        "Encryption Time (s)": encryption_time,
        "Decryption Time (s)": decryption_time,
        "Encrypted Data": encrypted_data,
        "Decrypted Data": decrypted_data,
        "Entropy (bits/byte)": entropy_value_formatted,
        "Encrypted Data (bin)": encrypted_data_bin
    }

# -------------------------------
# 主程序入口
# -------------------------------

# 定义模块字典，只包含每个算法的模块名
algorithm_modules = {
    "Algorithm_1": "ncRNA3_1_optimization",
    "Algorithm_2": "AES_crypto",
    "Algorithm_3": "RSA_crypto",
}

# 随机生成加密参数
plaintext = "Sample text to encrypt"
seed = generate_random_seed()
seed_sequence = generate_random_seed_sequence()
salt = generate_random_salt()
encrypt_args = (plaintext, seed, seed_sequence, salt)

# 遍历模块字典并运行所有算法
for algorithm_name, module_name in algorithm_modules.items():
    # 动态导入 encrypt 和 decrypt 函数
    encrypt_function = import_function_from_pyc(module_name, 'encrypt')
    decrypt_function = import_function_from_pyc(module_name, 'decrypt')

    # 调用加密函数，获取返回值
    encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt_function(*encrypt_args)

    # 解密参数
    decrypt_args = (encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)

    # 运行算法并获取结果
    result_info = run_algorithm(
        module_name,
        encrypt_args,
        decrypt_args,
        encrypt_function,
        decrypt_function
    )
    
    # 打印结果
    print(truncated_dict_print(result_info, length=200))
