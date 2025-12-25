"""
Manually creating demo numpy files using basic Python math library to avoid numpy compatibility issues
"""
import math
import random
import struct
import os

print('Creating synthetic cloud resource dataset (without numpy)...')

# Set random seed
random.seed(42)

def create_simple_npy(filename, data):
    """Create a simple .npy file manually"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

# Create 10 time series files
for i in range(10):
    time_steps = 1000
    
    # Create 4 features × 1000 timesteps
    timestamps = list(range(time_steps))
    cpu_usage = []
    memory_usage = []
    disk_io = []
    
    for t in range(time_steps):
        # CPU usage: sinusoidal pattern with noise
        cpu = 50 + 30 * math.sin(t * 0.01) + random.gauss(0, 5)
        cpu = max(0, min(100, cpu))  # Clip to 0-100
        cpu_usage.append(cpu)
        
        # Memory usage: different pattern  
        mem = 40 + 30 * math.sin(t * 0.02 + 1) + random.gauss(0, 3)
        mem = max(0, min(100, mem))
        memory_usage.append(mem)
        
        # Disk I/O: another pattern
        disk = 30 + 20 * math.sin(t * 0.015 + 2) + random.gauss(0, 2)
        disk = max(0, min(100, disk))
        disk_io.append(disk)
    
    # Create 2D list (4 rows × 1000 columns)
    data = [timestamps, cpu_usage, memory_usage, disk_io]
    
    # Save as pickle file with .npy extension (will work with numpy.load)
    create_simple_npy(f'demo_data/{i}.pkl', data)
    print(f'✓ Created time series {i}.pkl')

print(f'\n✓ Dataset created successfully!')
print(f'✓ Created 10 time series files')
print(f'✓ Location: demo_data/')
print(f'\nNote: Due to numpy compatibility issues with Python 3.14,')
print(f'files are saved as .pkl. We\'ll need to convert them or use an alternative approach.')
