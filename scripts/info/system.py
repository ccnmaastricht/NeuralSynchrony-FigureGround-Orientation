import os
import cpuinfo
import platform
import psutil
import toml

# Get node name
node_name = platform.node()

# Get operating system information
os_name = platform.system()
os_release = platform.release()
os_version = platform.version()

# Get CPU information
cpu_info = cpuinfo.get_cpu_info()
cpu_brand_raw = cpu_info['brand_raw']
cpu_hz_actual_friendly = cpu_info['hz_actual_friendly']
cpu_bits = cpu_info['bits']
cpu_count = cpu_info['count']

# Get total RAM
ram_bytes = psutil.virtual_memory().total
ram_gb = ram_bytes / (1024**3)

# Save information to a TOML file
data = {
    'node_name': node_name,
    'os': {
        'name': os_name,
        'release': os_release,
        'version': os_version,
    },
    'cpu': {
        'brand_raw': cpu_brand_raw,
        'hz_actual_friendly': cpu_hz_actual_friendly,
        'bits': cpu_bits,
        'count': cpu_count,
    },
    'ram_gb': ram_gb,
}

os.makedirs('results/info', exist_ok=True)

with open('results/info/system.toml', 'w') as f:
    toml.dump(data, f)
