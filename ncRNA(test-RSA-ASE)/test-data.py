import urllib.request
import os

# -------------------------------------------
# Standard Cryptography Test Dataset Loader
# -------------------------------------------

def load_nist_dataset(url="https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program/cavp-testing-block-cipher-modes"):
    """
    Load a standard dataset for cryptography testing, such as the NIST dataset.
    NIST dataset for AES and other block ciphers can be downloaded from the NIST site.
    """
    dataset_path = "nist_test_data.txt"
    if not os.path.exists(dataset_path):
        print("Downloading NIST test data...")
        # You can replace this with actual NIST test dataset URL
        urllib.request.urlretrieve(url, dataset_path)
    else:
        print("NIST test data already exists.")
    
    # Load and return the dataset
    with open(dataset_path, 'r') as file:
        data = file.read()
    return data

def load_classic_text_dataset(text="war_and_peace", file_name="war_and_peace.txt"):
    """
    Load classic text datasets like 'War and Peace' or 'Moby Dick' for cryptography testing.
    """
    url_dict = {
        "war_and_peace": "https://www.gutenberg.org/files/2600/2600-0.txt",  # War and Peace by Leo Tolstoy
        "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt"       # Moby Dick by Herman Melville
    }
    
    if not os.path.exists(file_name):
        print(f"Downloading {text} dataset...")
        urllib.request.urlretrieve(url_dict[text], file_name)
    else:
        print(f"{text} dataset already exists.")
    
    # Load and return the dataset
    with open(file_name, 'r') as file:
        data = file.read()
    return data

def load_shattered_sha1_data():
    """
    Load the SHA-1 collision dataset provided by Google researchers.
    """
    collision_url = "https://shattered.io/static/shattered-1.pdf"
    collision_file = "shattered-1.pdf"
    
    if not os.path.exists(collision_file):
        print("Downloading SHA-1 collision test data...")
        urllib.request.urlretrieve(collision_url, collision_file)
    else:
        print("SHA-1 collision test data already exists.")
    
    return collision_file  # You can process the file further as needed


def load_standard_dataset(name):
    """
    Load a standard dataset based on the name provided.
    Options include 'nist', 'war_and_peace', 'moby_dick', and 'sha1_collision'.
    """
    if name == "nist":
        return load_nist_dataset()
    elif name == "war_and_peace":
        return load_classic_text_dataset("war_and_peace")
    elif name == "moby_dick":
        return load_classic_text_dataset("moby_dick")
    elif name == "sha1_collision":
        return load_shattered_sha1_data()
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Test loading a dataset
if __name__ == "__main__":
    # Example: Load NIST dataset
    nist_data = load_standard_dataset("nist")
    print(f"NIST Dataset (First 100 chars): {nist_data[:100]}")
    
    # Example: Load War and Peace
    war_and_peace_data = load_standard_dataset("war_and_peace")
    print(f"War and Peace Dataset (First 100 chars): {war_and_peace_data[:100]}")
    
    # Example: Load SHA-1 collision data
    sha1_collision_file = load_standard_dataset("sha1_collision")
    print(f"SHA-1 Collision file downloaded: {sha1_collision_file}")
dataset = load_standard_dataset("war_and_peace")  # 加载《战争与和平》数据集

# 然后使用自定义加密算法、RSA、AES等进行加密性能测试
custom_enc_times, custom_dec_times, custom_accuracies, custom_entropies = test_custom_encryption([dataset[:1000]])
rsa_enc_times, rsa_dec_times, rsa_accuracies, rsa_entropies = test_rsa_encryption([dataset[:128]])
aes_enc_times, aes_dec_times, aes_accuracies, aes_entropies = test_aes_encryption([dataset[:1000]])
