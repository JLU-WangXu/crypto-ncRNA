from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import time

# 生成RSA密钥对
def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

# RSA加密
def rsa_encrypt(public_key, data):
    rsa_key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    encrypted = cipher_rsa.encrypt(data.encode())
    return encrypted

# RSA解密
def rsa_decrypt(private_key, data):
    rsa_key = RSA.import_key(private_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    decrypted = cipher_rsa.decrypt(data).decode()
    return decrypted

# 测试RSA算法
def test_rsa_encryption(num_tests=10):
    # 生成RSA密钥对
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    
    data = "ACGTACGTACGT" * 10  # 测试数据
    encryption_times = []
    decryption_times = []

    for _ in range(num_tests):
        # 加密
        start_time = time.time()
        cipher_rsa = PKCS1_OAEP.new(RSA.import_key(public_key))
        encrypted = cipher_rsa.encrypt(data.encode())
        encryption_times.append(time.time() - start_time)

        # 解密
        start_time = time.time()
        cipher_rsa = PKCS1_OAEP.new(RSA.import_key(private_key))
        decrypted = cipher_rsa.decrypt(encrypted).decode()
        decryption_times.append(time.time() - start_time)

    print(f"RSA Algorithm - 平均加密时间：{sum(encryption_times)/num_tests:.6f} 秒")
    print(f"RSA Algorithm - 平均解密时间：{sum(decryption_times)/num_tests:.6f} 秒")

    return encryption_times, decryption_times

if __name__ == "__main__":
    test_rsa_encryption()
