from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt
from Crypto.Util.Padding import pad, unpad

def encrypt(plaintext, seed, seed_sequence, salt):
    """
    使用AES加密数据，返回加密后的数据和占位符的substitution_matrix和indices_order。
    """
    key = scrypt(seed.encode(), salt.encode(), 32, N=2**14, r=8, p=1)
    cipher = AES.new(key, AES.MODE_CBC)
    ciphertext = cipher.encrypt(pad(plaintext.encode(), AES.block_size))
    
    # 使用占位符，保持接口一致性
    substitution_matrix = "NA"
    indices_order = "NA"
    
    return cipher.iv + ciphertext, substitution_matrix, indices_order

def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, substitution_matrix, indices_order):
    """
    使用AES解密数据，从加密数据中提取IV，并解密剩余的数据块。
    """
    iv = encrypted_data_with_checksum[:AES.block_size]
    key = scrypt(seed.encode(), salt.encode(), 32, N=2**14, r=8, p=1)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    decrypted_data = unpad(cipher.decrypt(encrypted_data_with_checksum[AES.block_size:]), AES.block_size).decode()
    return decrypted_data
