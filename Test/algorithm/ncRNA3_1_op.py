import hashlib  # 导入hashlib模块用于生成哈希值
import random   # 导入random模块用于生成随机数
import time     # 导入time模块用于计算时间
from collections import Counter  # 导入Counter用于计数对象
from Crypto.Cipher import AES  # 导入AES加密模块
from Crypto.Util.Padding import pad, unpad  # 导入填充和去填充函数
import base64   # 导入base64模块用于编码
import math     # 导入math模块用于数学运算
import matplotlib.pyplot as plt  # 导入matplotlib.pyplot用于绘图
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256  # 导入正确的哈希模块
import numpy as np
from functools import lru_cache

# 定义64个密码子
codons = [a + b + c for a in 'ACGU' for b in 'ACGU' for c in 'ACGU']

# 定义Base64字符集（不包括'='）
base64_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

# 1. 生成密码子替换矩阵
def generate_codon_substitution_matrix(seed):
    random.seed(seed)  # 设置随机种子
    shuffled_codons = codons.copy()  # 复制密码子列表
    random.shuffle(shuffled_codons)  # 随机打乱密码子顺序
    substitution_matrix = dict(zip(codons, shuffled_codons))  # 创建替换矩阵
    # 类型检查
    for key, value in substitution_matrix.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("替换矩阵的键和值必须是字符串。")
    return substitution_matrix

# 2. 将明文编码为密码子
def encode_plaintext_to_codons(plaintext):
    plaintext_bytes = plaintext.encode('utf-8')  # 将明文编码为字节
    base64_bytes = base64.b64encode(plaintext_bytes)  # 将字节编码为Base64
    base64_str = base64_bytes.decode('ascii')  # 将Base64字节解码为字符串
    codon_sequence = []
    for char in base64_str:
        if char == '=':
            continue  # 跳过填充字符
        codon_index = base64_chars.index(char)  # 获符在Base64字符集中的索引
        codon_sequence.append(codons[codon_index])  # 根据索引获取对应的密码子
    return codon_sequence

# 3. 密码子替换
def substitute_codons(codon_sequence, substitution_matrix):
    return [substitution_matrix[codon] for codon in codon_sequence]  # 替换密码子

# 4. RNA二级结构折叠（使用Nussinov算法进行迭代追踪）
@lru_cache(maxsize=None)
def can_pair(b1, b2):
    pairs = frozenset([('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C')])
    return (b1, b2) in pairs

def nussinov_algorithm(sequence):
    n = len(sequence)
    dp = np.zeros((n, n), dtype=int)
    # 填充动态规划表
    for k in range(1, n):
        for i in range(n - k):
            j = i + k
            if can_pair(sequence[i], sequence[j]):
                dp[i][j] = max(dp[i+1][j], dp[i][j-1], dp[i+1][j-1] + 1)
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    # 执行迭代追踪
    structure = ['.'] * n
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if i >= j:
            continue
        elif dp[i][j] == dp[i+1][j]:
            stack.append((i+1, j))
        elif dp[i][j] == dp[i][j-1]:
            stack.append((i, j-1))
        elif can_pair(sequence[i], sequence[j]) and dp[i][j] == dp[i+1][j-1] + 1:
            structure[i] = '('
            structure[j] = ')'
            stack.append((i+1, j-1))
        else:
            stack.append((i+1, j-1))
    return ''.join(structure)

def apply_rna_secondary_structure(codon_sequence):
    # 将密码子序列展平为碱基序列
    base_sequence = ''.join(codon_sequence)
    # 获取二级结构
    structure = nussinov_algorithm(base_sequence)
    # 根据结构重新排列序列
    paired_indices = []
    unpaired_indices = []
    for i, s in enumerate(structure):
        if s == '(' or s == ')':
            paired_indices.append(i)
        else:
            unpaired_indices.append(i)
    # 记录索引顺序
    indices_order = paired_indices + unpaired_indices
    # 重新排列序列
    new_sequence = ''.join([base_sequence[i] for i in indices_order])
    # 将序列转换回密码子序列
    new_codon_sequence = [new_sequence[i:i+3] for i in range(0, len(new_sequence), 3)]
    return new_codon_sequence, indices_order

# 5. 从生物数据生成动态密钥
def generate_dynamic_key_from_biological_data(seed_sequence, salt, iterations=100000):
    # 检查有效的碱基
    valid_bases = set('ACGU')
    if not set(seed_sequence.upper()).issubset(valid_bases):
        raise ValueError("Seed sequence contains invalid bases.")
    # 使用PBKDF2派生密钥
    dynamic_key = PBKDF2(seed_sequence, salt, dkLen=32, count=iterations, hmac_hash_module=SHA256)
    return dynamic_key

# 6. AES加密
def aes_encrypt(data_sequence, key):
    data_str = ''.join(data_sequence)
    data_bytes = data_str.encode()
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(pad(data_bytes, AES.block_size))
    return cipher.nonce + tag + ciphertext

# 7. 添加校验和
def add_checksum(encrypted_data):
    checksum = hashlib.sha256(encrypted_data).digest()  # 计算数据的SHA-256哈希值
    return encrypted_data + checksum  # 将校验和附加到加密数据后

# 加密函数
def encrypt(plaintext, seed, seed_sequence, salt):
    substitution_matrix = generate_codon_substitution_matrix(seed)  # 生成替换矩阵
    codon_sequence = encode_plaintext_to_codons(plaintext)  # 将明文编码为密码子序列
    substituted_sequence = substitute_codons(codon_sequence, substitution_matrix)  # 替换密码子
    folded_sequence, indices_order = apply_rna_secondary_structure(substituted_sequence)  # 应用RNA二级结构
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)  # 从生物数据生成动态密钥
    encrypted_data = aes_encrypt(folded_sequence, dynamic_key)  # 使用动态密钥进行AES加密
    encrypted_data_with_checksum = add_checksum(encrypted_data)  # 添加校验和
    return encrypted_data_with_checksum, substitution_matrix, indices_order  # 返回加密数据和相关信息

# 校验和验证
def verify_and_remove_checksum(encrypted_data_with_checksum):
    encrypted_data = encrypted_data_with_checksum[:-32]  # 移除校验和
    checksum = encrypted_data_with_checksum[-32:]  # 提取校验和
    computed_checksum = hashlib.sha256(encrypted_data).digest()  # 计算校验和
    if checksum != computed_checksum:  # 校验和不匹配时抛出异常
        raise ValueError("Checksum does not match. Data may be corrupted.")
    return encrypted_data  # 返回没有校验和的加密数据

# AES解密
def aes_decrypt(encrypted_data, key):
    nonce = encrypted_data[:16]
    tag = encrypted_data[16:32]
    ciphertext = encrypted_data[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    decrypted_padded = cipher.decrypt_and_verify(ciphertext, tag)
    decrypted_data = unpad(decrypted_padded, AES.block_size)
    data_str = decrypted_data.decode()
    codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
    return codon_sequence

# 修正 inverse_rna_secondary_structure 函数，确保返回的 original_codon_sequence 是字符串列表
def inverse_rna_secondary_structure(codon_sequence, indices_order):
    base_sequence = ''.join(codon_sequence)  # 将密码子序列展平为碱基序列
    original_sequence = [''] * len(base_sequence)  # 创建一个空序列用于存储原始序列
    for i, idx in enumerate(indices_order):
        if i < len(base_sequence):
            original_sequence[idx] = base_sequence[i]
    # 填充未被赋值的位置
    for i in range(len(original_sequence)):
        if original_sequence[i] == '':
            original_sequence[i] = 'A'  # 使用'A'作为默认填充值，可以根据需求调整
    # 确保每个密码子都是字符串
    original_codon_sequence = []
    for i in range(0, len(original_sequence), 3):
        codon = ''.join(original_sequence[i:i+3])
        if len(codon) == 3:
            original_codon_sequence.append(codon)
    return original_codon_sequence  # 返回原始密码子序列

# 修正 inverse_substitute_codons 函数，确保 substitution_matrix 的值是字符串
def inverse_substitute_codons(codon_sequence, substitution_matrix):
    inverse_substitution_matrix = {}
    for k, v in substitution_matrix.items():
        if isinstance(v, list):
            raise TypeError(f"替换矩阵中的值不应为列表，但键 {k} 的值是列表。")
        if v in inverse_substitution_matrix:
            raise ValueError(f"密码子替换矩阵中存在重复的替换密码子: {v}")
        inverse_substitution_matrix[v] = k
    original_codon_sequence = []
    for codon in codon_sequence:
        if codon in inverse_substitution_matrix:
            original_codon_sequence.append(inverse_substitution_matrix[codon])
        else:
            raise ValueError(f"密码子 {codon} 未在逆替换矩阵中找到")
    return original_codon_sequence  # 返回原始密码子序列

# 将密码子序列解码为明文
def decode_codons_to_plaintext(codon_sequence):
    base64_chars_list = []
    for codon in codon_sequence:
        base64_char = base64_chars[codons.index(codon)]  # 根据密码子找到对应的Base64字符
        base64_chars_list.append(base64_char)  # 将Base64字符添加到列表中
    base64_str = ''.join(base64_chars_list)  # Base64字符列表连接成字符串
    # 如果需要，添加Base64填充
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    try:
        plaintext_bytes = base64.b64decode(base64_str)  # 将Base64字符串解码为字节
        plaintext = plaintext_bytes.decode('utf-8')  # 将字节解码为明文
    except Exception as e:
        plaintext = ""  # 设置明文为空字符串
    return plaintext  # 返回明文

# 解密函
def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    encrypted_data = verify_and_remove_checksum(encrypted_data_with_checksum)  # 验证并移除校验和
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence, salt)  # 从生物数据生成动态密钥
    decrypted_sequence = aes_decrypt(encrypted_data, dynamic_key)  # 使用动态密钥进行AES解密
    unfolded_sequence = inverse_rna_secondary_structure(decrypted_sequence, indices_order)  # 逆RNA二级结构
    original_codon_sequence = inverse_substitute_codons(unfolded_sequence, substitution_matrix)  # 逆密码子替换
    plaintext = decode_codons_to_plaintext(original_codon_sequence)  # 将密码子序列解码为明文
    return plaintext  # 返回明文

# 计算熵
def calculate_entropy(data):
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()  # 如果数据是字符串，先将其编码为字节
    data_length = len(byte_data)  # 获取数据长度
    if data_length == 0:
        return 0.0  # 如果数据长度为0，熵为0
    frequencies = Counter(byte_data)  # 计算每个字节出现的频率
    entropy = -sum((freq / data_length) * math.log2(freq / data_length) for freq in frequencies.values())  # 计算熵
    return entropy  # 返回熵

# 绘制熵直方图
def plot_entropy_histogram(data):
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()  # 如果数据是字符串，先将其编码为字节
    frequencies = Counter(byte_data)  # 计算每个字节出现的频率
    bytes_list = list(frequencies.keys())  # 获取所有字节值
    counts = list(frequencies.values())  # 获取每字节值的计数
    plt.bar(bytes_list, counts)  # 绘制条形图
    plt.xlabel('Byte Value')  # 设置x轴标签
    plt.ylabel('Frequency')  # 设置y轴标签
    plt.title('Byte Frequency Distribution of Encrypted Data')  # 设置图表标题
    plt.show()  # 显示图表

# 性能测试函数
def test_performance():
    plaintext_lengths = [50, 100, 200, 400, 800, 1600]  # 定义测试的明文长度列表
    encryption_times = []  # 用于存储加密时间
    decryption_times = []  # 用于存储解密时间
    seed = "123456789"  # 定义随机种子
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"  # 定义种子序列
    salt = b'salt_123'  # 定义盐值
    for length in plaintext_lengths:
        plaintext = 'A' * length  # 创建指定长度的明文
        # 测试加密时间
        start_time = time.time()  # 开始计时
        encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt(plaintext, seed, seed_sequence, salt)  # 执行加密
        encryption_time = time.time() - start_time  # 计算加密时间
        encryption_times.append(encryption_time)  # 将加密时间添加到列表
        # 测试解密时间
        start_time = time.time()  # 开始计时
        decrypted_plaintext = decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)  # 执行解密
        decryption_time = time.time() - start_time  # 计算解密时间
        decryption_times.append(decryption_time)  # 将解密时间添加到列表
    # 绘制性能测试图表
    plt.plot(plaintext_lengths, encryption_times, label='Encryption Time')  # 绘制加密时间曲线
    plt.plot(plaintext_lengths, decryption_times, label='Decryption Time')  # 绘制解密时间曲线
    plt.xlabel('Plaintext Length (characters)')  # 设置x轴标签
    plt.ylabel('Time (seconds)')  # 设置y轴标签
    plt.title('Encryption and Decryption Time vs. Plaintext Length')  # 设置图表标题
    plt.legend()  # 显示图例
    plt.show()  # 显示图表

# 主测试代码
if __name__ == "__main__":
    with open("./Test/war_and_peace.txt", "r", encoding="utf-8") as file:
        data = file.read()
    
    plaintext = data[:50]  # 定义明文
    seed = "123456789"  # 定义随机种子
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"  # 定义种子序列
    salt = b'salt_123'  # 定义盐值
    print("Original Plaintext:", plaintext)  # 打印原始明文
    # 执行加密
    start_time = time.time()  # 开始计时
    encrypted_data_with_checksum, substitution_matrix, indices_order = encrypt(plaintext, seed, seed_sequence, salt)  # 调用加密函数
    encryption_time = time.time() - start_time  # 计算加密时间
    print("Encryption completed in {:.6f} seconds".format(encryption_time))  # 打印加密时间
    # 计算加密据的熵
    encrypted_data = encrypted_data_with_checksum[:-32]  # 移除校验和
    entropy = calculate_entropy(encrypted_data)  # 调用计算熵的函数
    print("Entropy of encrypted data: {:.4f} bits/byte".format(entropy))  # 打印熵
    # 绘制熵直方图
    plot_entropy_histogram(encrypted_data)  # 调用绘制熵直方图的函数
    # 执行解密
    start_time = time.time()  # 开始计时
    decrypted_plaintext = decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order)  # 调用解密函数
    decryption_time = time.time() - start_time  # 计算解密时间
    print("Decryption completed in {:.6f} seconds".format(decryption_time))  # 打印解密时间
    print("Decrypted Plaintext:", decrypted_plaintext)  # 打印解密后的明文
    # 准确性检查
    if plaintext == decrypted_plaintext:
        accuracy = 100.0
        print("Encryption and decryption successful. Accuracy: 100%")  # 如果明文和解密后的明文相同，打印成功信息
    else:
        accuracy = 0.0
        print("Encryption and decryption failed. Accuracy: 0%")  # 如果明文和解密后的明文不同，打印失败信息
    # 测试总结
    print("\nTest Summary:")  # 打印测试总结标题
    print("Plaintext Length: {} characters".format(len(plaintext)))  # 打印明文长度
    print("Encrypted Data Length: {} bytes".format(len(encrypted_data_with_checksum)))  # 打印加密数据长度
    print("Encryption Time: {:.6f} seconds".format(encryption_time))  # 打印加密时间
    print("Decryption Time: {:.6f} seconds".format(decryption_time))  # 打印解密时间
    print("Entropy of Encrypted Data: {:.4f} bits/byte".format(entropy))  # 打印加密数据的熵
    print("Accuracy: {}%".format(accuracy))  # 打印准确率