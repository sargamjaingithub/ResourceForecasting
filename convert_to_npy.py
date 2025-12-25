"""
Convert pickle files to proper numpy .npy format
"""
import sys
import os
import pickle

# Suppress all numpy warnings
import warnings
warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Now import numpy
try:
    import numpy as np
    print("Converting pickle files to numpy format...")
    
    success_count = 0
    for i in range(10):
        try:
            pkl_file = f'demo_data/{i}.pkl'
            npy_file = f'demo_data/{i}.npy'
            
            if os.path.exists(pkl_file):
                with open(pkl_file, 'rb') as f:
                    data_list = pickle.load(f)
                
                # Convert list to numpy array
                data_array = np.array(data_list, dtype=np.float32)
                
                # Save as .npy
                np.save(npy_file, data_array)
                
                # Remove pkl file
                os.remove(pkl_file)
                
                print(f'✓ Converted {i}.pkl -> {i}.npy (shape: {data_array.shape})')
                success_count += 1
        except Exception as e:
            print(f'✗ Failed to convert {i}.pkl: {e}')
            
    print(f'\n✓ Successfully converted {success_count} files!')
    
    # Verify
    npy_files = [f for f in os.listdir('demo_data') if f.endswith('.npy')]
    print(f'✓ Total .npy files in demo_data: {len(npy_files)}')
    
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
