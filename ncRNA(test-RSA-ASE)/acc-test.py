import random
import hashlib
import time

# Custom Encryption Functions
def generate_substitution_matrix(seed):
    random.seed(seed)
    bases = ['A', 'C', 'G', 'T']
    substitution = dict()
    shuffled_bases = random.sample(bases, len(bases))
    for i, base in enumerate(bases):
        substitution[base] = shuffled_bases[i]
    return substitution

def transcribe_dna_to_rna(plaintext, substitution_matrix):
    transcribed = ''.join([substitution_matrix.get(char, char) for char in plaintext])
    return transcribed

def split_and_splice(data, block_size=3):
    blocks = [data[i:i+block_size] for i in range(0, len(data), block_size)]
    original_order = blocks.copy()
    random.shuffle(blocks)  # 模拟随机剪接
    spliced_data = ''.join(blocks)
    return spliced_data, original_order

def inverse_splice(data, original_order):
    return ''.join(original_order)

def generate_dynamic_key(seed=None):
    if seed is None:
        seed = int(time.time())
    seed_str = str(seed)
    hash_object = hashlib.sha256(seed_str.encode())
    dynamic_key = int(hash_object.hexdigest(), 16) % (2**128)
    return dynamic_key

def apply_dynamic_key(data, key):
    key_bin = format(key, '0128b')
    data_bin = ''.join(format(ord(char), '08b') for char in data)
    encrypted_data = ''.join('1' if data_bin[i] != key_bin[i % len(key_bin)] else '0' for i in range(len(data_bin)))
    chars = [chr(int(encrypted_data[i:i+8], 2)) for i in range(0, len(encrypted_data), 8)]
    return ''.join(chars)

def reverse_dynamic_key(data, key):
    return apply_dynamic_key(data, key)

def insert_redundancy(encrypted_data):
    redundancy = ''.join(random.choice('01') for _ in range(8))
    return encrypted_data + redundancy

# 自定义加密函数
def custom_encrypt(plaintext, seed=None):
    start_time = time.time()
    
    # 确保seed和密钥生成一致性
    if seed is None:
        seed = "default_seed_value"

    # Step 1: 转录和置换矩阵
    substitution_matrix = generate_substitution_matrix(seed)
    transcribed_data = transcribe_dna_to_rna(plaintext, substitution_matrix)
    
    # Step 2: 剪接操作
    spliced_data, original_order = split_and_splice(transcribed_data)
    
    # Step 3: 动态密钥生成
    dynamic_key = generate_dynamic_key(seed)  # 确保加密解密密钥一致
    
    # Step 4: 动态密钥加密
    encrypted_data = apply_dynamic_key(spliced_data, dynamic_key)
    
    # Step 5: 插入冗余信息
    encrypted_with_redundancy = insert_redundancy(encrypted_data)
    
    encryption_time = time.time() - start_time
    return encrypted_with_redundancy, original_order, encryption_time

# 自定义解密函数
def custom_decrypt(encrypted_with_redundancy, seed, original_order):
    start_time = time.time()

    # 确保seed和密钥生成一致性
    if seed is None:
        seed = "default_seed_value"

    # Step 1: 去除冗余
    encrypted_data = encrypted_with_redundancy[:-8]  # 移除冗余位
    
    # Step 2: 动态密钥生成
    dynamic_key = generate_dynamic_key(seed)
    
    # Step 3: 逆动态密钥解密
    decrypted_spliced_data = reverse_dynamic_key(encrypted_data, dynamic_key)
    
    # Step 4: 剪接逆操作，恢复原始顺序
    decrypted_data = inverse_splice(decrypted_spliced_data, original_order)
    
    # Step 5: 逆转录，恢复明文
    inverse_substitution_matrix = {v: k for k, v in generate_substitution_matrix(seed).items()}
    recovered_plaintext = ''.join([inverse_substitution_matrix.get(char, char) for char in decrypted_data])
    
    decryption_time = time.time() - start_time
    return recovered_plaintext, decryption_time

# Accuracy Calculation
def calculate_accuracy(original, decrypted):
    matches = sum([1 if o == d else 0 for o, d in zip(original, decrypted)])
    accuracy = matches / len(original)
    return accuracy

# 测试函数
def test_custom_encryption():
    data = "ACGTACGTACGT" * 10  # 测试数据
    seed = "123456789"  # 保证加密解密时密钥一致
    encrypted_data, original_order, encryption_time = custom_encrypt(data, seed)
    decrypted_data, decryption_time = custom_decrypt(encrypted_data, seed, original_order)
    
    accuracy = calculate_accuracy(data, decrypted_data)
    print(f"加密时间: {encryption_time:.6f} 秒")
    print(f"解密时间: {decryption_time:.6f} 秒")
    print(f"解密准确率: {accuracy * 100:.2f}%")

# 运行测试
test_custom_encryption()
