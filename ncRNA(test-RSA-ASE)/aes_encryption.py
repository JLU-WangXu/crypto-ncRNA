from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
import time
import matplotlib.pyplot as plt

# AES加密
def aes_encrypt(data):
    key = get_random_bytes(16)  # 生成随机的16字节密钥
    cipher_aes = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher_aes.encrypt(pad(data.encode(), AES.block_size))
    return cipher_aes.iv + ct_bytes, key

# AES解密
def aes_decrypt(key, data):
    iv = data[:AES.block_size]
    cipher_aes = AES.new(key, AES.MODE_CBC, iv)
    decrypted = unpad(cipher_aes.decrypt(data[AES.block_size:]), AES.block_size).decode()
    return decrypted

# 测试AES算法
def test_aes_encryption(num_tests=10):
    key = get_random_bytes(16)  # 128位密钥
    data = "ACGTACGTACGT" * 10  # 测试数据
    encryption_times = []
    decryption_times = []

    for _ in range(num_tests):
        # 加密
        start_time = time.time()
        cipher = AES.new(key, AES.MODE_ECB)
        encrypted = cipher.encrypt(pad(data.encode(), AES.block_size))
        encryption_times.append(time.time() - start_time)

        # 解密
        start_time = time.time()
        cipher = AES.new(key, AES.MODE_ECB)
        decrypted = unpad(cipher.decrypt(encrypted), AES.block_size).decode()
        decryption_times.append(time.time() - start_time)

    print(f"AES Algorithm - 平均加密时间：{sum(encryption_times)/num_tests:.6f} 秒")
    print(f"AES Algorithm - 平均解密时间：{sum(decryption_times)/num_tests:.6f} 秒")

    return encryption_times, decryption_times  # 添加这行来返回结果

if __name__ == "__main__":
    test_aes_encryption()
