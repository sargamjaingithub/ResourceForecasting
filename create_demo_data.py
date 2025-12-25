import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
import os
import sys

print('Creating synthetic cloud resource dataset...')

# Create demo data directory
os.makedirs('demo_data', exist_ok=True)

# Create header file
header = 'time_stamp,cpu_usage,memory_usage,disk_io'
with open('demo_data/header.csv', 'w') as f:
    f.write(header)
print('✓ Header file created')

try:
    import numpy as np
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create 10 time series (representing different cloud workloads)
    for i in range(10):
        try:
            # Each time series has 1000 time steps and 4 features
            time_steps = 1000
            
            # Time stamps
            timestamps = np.arange(time_steps, dtype=np.float32)
            
            # CPU usage: sinusoidal pattern with noise (0-100%)
            cpu_usage = (50 + 30 * np.sin(timestamps * 0.01) + np.random.randn(time_steps) * 5).astype(np.float32)
            cpu_usage = np.clip(cpu_usage, 0, 100)
            
            # Memory usage: different frequency sinusoidal pattern (0-100%)
            memory_usage = (40 + 30 * np.sin(timestamps * 0.02 + 1) + np.random.randn(time_steps) * 3).astype(np.float32)
            memory_usage = np.clip(memory_usage, 0, 100)
            
            # Disk I/O: another pattern (0-100%)
            disk_io = (30 + 20 * np.sin(timestamps * 0.015 + 2) + np.random.randn(time_steps) * 2).astype(np.float32)
            disk_io = np.clip(disk_io, 0, 100)
            
            # Stack all features (shape: 4 x 1000)
            time_series = np.vstack([timestamps, cpu_usage, memory_usage, disk_io]).astype(np.float32)
            
            # Save as .npy file
            np.save(f'demo_data/{i}.npy', time_series)
            print(f'✓ Created time series {i}.npy')
        except Exception as e:
            print(f'✗ Failed to create time series {i}: {e}')
            continue
    
    # Check how many files were created
    npy_files = [f for f in os.listdir('demo_data') if f.endswith('.npy')]
    print(f'\n✓ Dataset created successfully!')
    print(f'✓ Created {len(npy_files)} time series files')
    print(f'✓ Location: demo_data/')
    
except Exception as e:
    print(f'\n✗ Error: {e}')
    print(f'Error type: {type(e).__name__}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
