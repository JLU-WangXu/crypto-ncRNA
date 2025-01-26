from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

# RSA加密函数，内部拼接密文
def rsa_encrypt(plaintext, block_size=190):
    """
    使用动态生成的公钥对数据进行分块加密，并拼接成完整密文。
    
    参数：
        plaintext (str): 需要加密的明文。
        block_size (int): 每次加密的块大小，默认值为190。
    返回：
        Tuple[bytes, bytes, bytes]: 拼接后的密文、公钥和私钥。
    """
    key = RSA.generate(2048)
    public_key = key.publickey()
    cipher_rsa = PKCS1_OAEP.new(public_key)

    # 分块加密并拼接
    encrypted_blocks = [
        cipher_rsa.encrypt(plaintext[i:i + block_size].encode())
        for i in range(0, len(plaintext), block_size)
    ]
    encrypted_data = b"".join(encrypted_blocks)  # 拼接密文
    return encrypted_data, public_key.export_key(), key.export_key()

# RSA解密函数，处理拼接密文
def rsa_decrypt(encrypted_data, private_key_data, block_size=256):
    """
    使用私钥对拼接后的密文进行解密。
    
    参数：
        encrypted_data (bytes): 拼接后的加密密文。
        private_key_data (bytes): 私钥数据。
        block_size (int): 每块密文的大小，默认值为256字节。
    返回：
        str: 解密后的明文。
    """
    private_key = RSA.import_key(private_key_data)
    cipher_rsa = PKCS1_OAEP.new(private_key)

    # 分块解密
    decrypted_blocks = [
        cipher_rsa.decrypt(encrypted_data[i:i + block_size]).decode()
        for i in range(0, len(encrypted_data), block_size)
    ]
    return ''.join(decrypted_blocks)
