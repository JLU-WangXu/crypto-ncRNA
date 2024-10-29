from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Util.Padding import pad, unpad

def encrypt(plaintext, seed, seed_sequence, salt):
    key = scrypt(seed.encode(), salt.encode(), 32, N=2**14, r=8, p=1)
    cipher = AES.new(key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    
    # 将 substitution_matrix 和 indices_order 设为 NA
    substitution_matrix = "NA"
    indices_order = "NA"
    
    return cipher.iv + ciphertext, substitution_matrix, indices_order  # 返回完整加密数据和占位符


def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    # 从加密数据中提取前 16 字节（IV）
    iv = encrypted_data_with_checksum[:AES.block_size]
    key = scrypt(seed.encode(), salt.encode(), 32, N=2**14, r=8, p=1)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # 解密剩余部分数据
    decrypted_data = unpad(cipher.decrypt(encrypted_data_with_checksum[AES.block_size:]), AES.block_size).decode()
    return decrypted_data
