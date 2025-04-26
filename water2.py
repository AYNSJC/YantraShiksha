import os
import hashlib

# Sample suspicious keywords (for scripts)
SUSPICIOUS_KEYWORDS = ["eval(", "exec(", "base64", "import os", "powershell", "subprocess"]

# Add known malicious file hashes (MD5 for demo)
KNOWN_MALICIOUS_HASHES = {
    "5d41402abc4b2a76b9719d911017c592",  # fake example hash
}

def md5sum(filename):
    h = hashlib.md5()
    with open(filename, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def scan_file(path):
    try:
        # Check hash
        file_hash = md5sum(path)
        if file_hash in KNOWN_MALICIOUS_HASHES:
            return f"[!] MALICIOUS HASH DETECTED in {path}"

        # Check for suspicious keywords in text files
        if path.endswith(('.py', '.js', '.vbs', '.sh', '.bat', '.ps1')):
            with open(path, 'r', errors='ignore') as f:
                content = f.read()
                for word in SUSPICIOUS_KEYWORDS:
                    if word in content:
                        return f"[!] Suspicious keyword '{word}' found in {path}"

    except Exception as e:
        return f"[!] Error scanning {path}: {e}"

    return None  # clean file

def scan_directory(root):
    print(f" Scanning directory: {root}")
    threats_found = []
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            full_path = os.path.join(dirpath, file)
            result = scan_file(full_path)
            if result:
                threats_found.append(result)

    if threats_found:
        print(" Threats Detected:")
        for threat in threats_found:
            print(threat)
    else:
        print(" No threats found. All clear!")

#  Change this path to scan a specific folder
scan_directory("C:\\Users\\YourUsername\\Downloads")