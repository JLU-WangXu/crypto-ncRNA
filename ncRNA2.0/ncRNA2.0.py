

"""
这个是ncRNA2.0版本
主要改动的创新点是：

1. 增加替换矩阵的复杂度：原始替换矩阵的弱点： 仅使用4个碱基'A', 'C', 'G', 'T'进行替换,可能的替换方式只有4! = 24种,容易被穷举攻击破解。
使用密码子替换：密码子数量： 有4^3 = 64种可能的密码子,三个核苷酸组成一个密码子）。

2. 引入生物学中的非线性过程
问题分析：异或操作的线性性： 异或操作是线性的,缺乏密码学中的混淆和扩散特性。模拟RNA二级结构折叠: RNA分子可以形成复杂的二级结构,如发卡结构、茎环结构等。这些结构形成的过程是非线性的。
函数定义： 定义一个基于RNA二级结构预测算法的非线性函数f,将序列映射到其二级结构表示。加密过程在转录和剪接之后,对序列应用函数f,引入非线性变换。实现方式,使用RNA二级结构预测算法: 如Nussinov算法或能量最小化方法,计算序列的二级结构。将结构信息用于加密:将二级结构的表示（如配对信息、能量值）引入到加密过程中，增加复杂性。

3. 动态密钥的改进：基于生物序列的种子： 使用难以预测的生物序列（如基因序列、蛋白质序列）作为种子。增加种子的熵： 将种子扩展为高熵的生物数据，并通过哈希函数生成动态密钥。

4. 加入生物学中的冗余和纠错机制
"""

import hashlib
import random
import time
from collections import Counter
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import base64
import math
import matplotlib.pyplot as plt

# 定义密码子列表（64种）
codons = [a + b + c for a in 'ACGU' for b in 'ACGU' for c in 'ACGU']

# 定义Base64字符集
base64_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'

# 1. 生成密码子替换矩阵
def generate_codon_substitution_matrix(seed):
    random.seed(seed)
    shuffled_codons = codons.copy()
    random.shuffle(shuffled_codons)
    substitution_matrix = dict(zip(codons, shuffled_codons))
    return substitution_matrix

# 2. 明文编码为密码子序列
def encode_plaintext_to_codons(plaintext):
    # 将明文编码为UTF-8字节串
    plaintext_bytes = plaintext.encode('utf-8')
    # 使用Base64编码
    base64_bytes = base64.b64encode(plaintext_bytes)
    base64_str = base64_bytes.decode('ascii')
    # 将每个Base64字符映射到密码子
    codon_sequence = []
    for char in base64_str:
        codon_index = base64_chars.index(char)
        codon_sequence.append(codons[codon_index])
    return codon_sequence

# 3. 密码子替换
def substitute_codons(codon_sequence, substitution_matrix):
    return [substitution_matrix[codon] for codon in codon_sequence]

# 4. RNA二级结构折叠（简单模拟，实际过程复杂）
def apply_rna_secondary_structure(codon_sequence):
    # 简单模拟，将序列反转
    return codon_sequence[::-1]

# 5. 生成动态密钥
def generate_dynamic_key_from_biological_data(seed_sequence):
    # 使用生物序列的哈希值作为密钥
    hash_object = hashlib.sha256(seed_sequence.encode())
    dynamic_key = hash_object.digest()  # 256位密钥
    return dynamic_key

# 6. AES加密序列
def aes_encrypt(data_sequence, key):
    # 将密码子序列转换为字符串
    data_str = ''.join(data_sequence)
    # 转换为字节
    data_bytes = data_str.encode()
    # 使用AES加密
    cipher = AES.new(key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(data_bytes, AES.block_size))
    return cipher.iv + ciphertext  # 返回IV和密文

# 7. 添加校验和
def add_checksum(encrypted_data):
    checksum = hashlib.sha256(encrypted_data).digest()
    return encrypted_data + checksum

# 加密函数
def encrypt(plaintext, seed, seed_sequence):
    # 步骤1-3
    substitution_matrix = generate_codon_substitution_matrix(seed)
    codon_sequence = encode_plaintext_to_codons(plaintext)
    substituted_sequence = substitute_codons(codon_sequence, substitution_matrix)

    # 步骤4
    folded_sequence = apply_rna_secondary_structure(substituted_sequence)

    # 步骤5
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence)

    # 步骤6
    encrypted_data = aes_encrypt(folded_sequence, dynamic_key)

    # 步骤7
    encrypted_data_with_checksum = add_checksum(encrypted_data)

    return encrypted_data_with_checksum, substitution_matrix

# 1. 校验和验证
def verify_and_remove_checksum(encrypted_data_with_checksum):
    encrypted_data = encrypted_data_with_checksum[:-32]  # SHA-256校验和为32字节
    checksum = encrypted_data_with_checksum[-32:]
    computed_checksum = hashlib.sha256(encrypted_data).digest()
    if checksum != computed_checksum:
        raise ValueError("数据校验和不匹配，数据可能已被篡改。")
    return encrypted_data

# 2. AES解密
def aes_decrypt(encrypted_data, key):
    iv = encrypted_data[:16]
    ciphertext = encrypted_data[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_padded = cipher.decrypt(ciphertext)
    decrypted_data = unpad(decrypted_padded, AES.block_size)
    data_str = decrypted_data.decode()
    # 将字符串转换为密码子序列
    codon_sequence = [data_str[i:i+3] for i in range(0, len(data_str), 3)]
    return codon_sequence

# 3. RNA二级结构展开
def inverse_rna_secondary_structure(codon_sequence):
    # 逆转序列
    return codon_sequence[::-1]

# 4. 逆密码子替换
def inverse_substitute_codons(codon_sequence, substitution_matrix):
    inverse_substitution_matrix = {v: k for k, v in substitution_matrix.items()}
    return [inverse_substitution_matrix[codon] for codon in codon_sequence]

# 5. 密码子解码为明文
def decode_codons_to_plaintext(codon_sequence):
    base64_chars_list = []
    for codon in codon_sequence:
        codon_index = codons.index(codon)
        base64_char = base64_chars[codon_index]
        base64_chars_list.append(base64_char)
    base64_str = ''.join(base64_chars_list)
    # 使用Base64解码
    plaintext_bytes = base64.b64decode(base64_str)
    plaintext = plaintext_bytes.decode('utf-8')
    return plaintext

# 解密函数
def decrypt(encrypted_data_with_checksum, seed, seed_sequence, substitution_matrix):
    # 步骤1
    encrypted_data = verify_and_remove_checksum(encrypted_data_with_checksum)

    # 步骤2
    dynamic_key = generate_dynamic_key_from_biological_data(seed_sequence)

    # 步骤3
    decrypted_sequence = aes_decrypt(encrypted_data, dynamic_key)

    # 步骤4
    unfolded_sequence = inverse_rna_secondary_structure(decrypted_sequence)

    # 步骤5
    original_codon_sequence = inverse_substitute_codons(unfolded_sequence, substitution_matrix)

    # 步骤6
    plaintext = decode_codons_to_plaintext(original_codon_sequence)

    return plaintext

# 计算熵值
def calculate_entropy(data):
    # 将数据视为字节序列
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()
    elif isinstance(data, bytes):
        byte_data = data
    else:
        raise TypeError("数据类型必须为字符串或字节串。")

    data_length = len(byte_data)
    if data_length == 0:
        return 0.0

    # 统计每个字节的出现次数
    frequencies = Counter(byte_data)
    # 计算熵值
    entropy = -sum((freq / data_length) * math.log2(freq / data_length) for freq in frequencies.values())

    return entropy

# 绘制熵值直方图
def plot_entropy_histogram(data):
    byte_data = data
    if isinstance(data, str):
        byte_data = data.encode()
    elif isinstance(data, bytes):
        byte_data = data
    else:
        raise TypeError("数据类型必须为字符串或字节串。")

    frequencies = Counter(byte_data)
    bytes_list = list(frequencies.keys())
    counts = list(frequencies.values())

    plt.bar(bytes_list, counts)
    plt.xlabel('Byte value')
    plt.ylabel('Frequency')
    plt.title('Byte frequency distribution of encrypted data')
    plt.show()

# 测试代码
if __name__ == "__main__":
    # 输入明文
    plaintext = "Hello, World! This is a test of the encryption algorithm based on ncRNA."
    # 使用的种子和生物序列
    seed = "123456789"
    seed_sequence = "ACGUACGUACGUACGUACGUACGUACGUACGU"

    print("原始明文：", plaintext)

    # 加密
    start_time = time.time()
    encrypted_data_with_checksum, substitution_matrix = encrypt(plaintext, seed, seed_sequence)
    encryption_time = time.time() - start_time
    print("加密完成，耗时：{:.6f} 秒".format(encryption_time))

    # 计算加密数据的熵值
    encrypted_data = encrypted_data_with_checksum[:-32]  # 移除校验和部分
    entropy = calculate_entropy(encrypted_data)
    print("加密数据的熵值：{:.4f} bits/byte".format(entropy))

    # 绘制熵值直方图
    plot_entropy_histogram(encrypted_data)

    # 解密
    start_time = time.time()
    decrypted_plaintext = decrypt(encrypted_data_with_checksum, seed, seed_sequence, substitution_matrix)
    decryption_time = time.time() - start_time
    print("解密完成，耗时：{:.6f} 秒".format(decryption_time))

    print("解密后的明文：", decrypted_plaintext)

    # 准确性验证
    if plaintext == decrypted_plaintext:
        accuracy = 100.0
        print("加密解密成功，准确率：100%")
    else:
        accuracy = 0.0
        print("加密解密失败，准确率：0%")

    # 输出测试结果汇总
    print("\n测试结果汇总：")
    print("原始明文长度：{} 字符".format(len(plaintext)))
    print("加密数据长度：{} 字节".format(len(encrypted_data_with_checksum)))
    print("加密时间：{:.6f} 秒".format(encryption_time))
    print("解密时间：{:.6f} 秒".format(decryption_time))
    print("加密数据的熵值：{:.4f} bits/byte".format(entropy))
    print("准确率：{}%".format(accuracy))
