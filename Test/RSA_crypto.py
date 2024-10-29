from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Random import get_random_bytes

def encrypt(plaintext, seed, seed_sequence, salt, block_size=190):
    # RSA 加密每个块的数据大小应小于密钥长度（例如 2048位密钥 -> 256字节）
    key = RSA.generate(2048)
    public_key = key.publickey()
    cipher_rsa = PKCS1_OAEP.new(public_key)

    encrypted_blocks = []
    for i in range(0, len(plaintext), block_size):
        block = plaintext[i:i+block_size].encode()
        encrypted_block = cipher_rsa.encrypt(block)
        encrypted_blocks.append(encrypted_block)

    return encrypted_blocks, public_key.export_key(), key.export_key()

def decrypt(encrypted_data_with_checksum, seed, seed_sequence, salt, public_key_data, private_key_data):
    private_key = RSA.import_key(private_key_data)
    cipher_rsa = PKCS1_OAEP.new(private_key)

    decrypted_blocks = []
    for block in encrypted_data_with_checksum:
        decrypted_block = cipher_rsa.decrypt(block).decode()
        decrypted_blocks.append(decrypted_block)

    return ''.join(decrypted_blocks)
