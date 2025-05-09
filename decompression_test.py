
#!/usr/bin/env python
# Test script to verify logits decompression works
import json
import torch
import numpy as np
import base64
import zlib
import io
import os

def decompress_tensor_optimized(compressed_data):
    # Handle sparse format
    if compressed_data.get('format') == 'sparse':
        # Decode and decompress
        compressed_bytes = base64.b64decode(compressed_data['data'])
        decompressed_json = zlib.decompress(compressed_bytes)
        sparse_data = json.loads(decompressed_json)
        
        # Reconstruct sparse tensor
        shape = sparse_data['shape']
        indices = sparse_data['indices']
        values = sparse_data['values']
        min_val = sparse_data['min']
        max_val = sparse_data['max']
        
        # Create empty array
        array = np.zeros(shape, dtype=np.float32)
        
        # Fill non-zero values
        for idx, val in zip(zip(*indices), values):
            array[idx] = val
        
        return torch.tensor(array)
    
    # Handle quantized format
    if 'quantized' in compressed_data.get('format', ''):
        # Extract parameters
        shape = compressed_data['shape']
        min_val = compressed_data['min_val']
        max_val = compressed_data['max_val']
        precision_bits = int(compressed_data['format'].split('_')[1].replace('bit', ''))
        
        # Decode and decompress
        compressed_bytes = base64.b64decode(compressed_data['data'])
        decompressed_bytes = zlib.decompress(compressed_bytes)
        
        # Convert to numpy array
        if precision_bits == 8:
            # Direct 8-bit quantization
            quantized = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            normalized = quantized.astype(np.float32) / 255.0
            
        elif precision_bits == 4:
            # Unpack 4-bit values
            packed = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            total_values = np.prod(shape)
            
            # Create array for unpacked values
            quantized = np.zeros(total_values, dtype=np.uint8)
            
            # Unpack values
            even_indices = np.arange(0, total_values, 2)
            odd_indices = np.minimum(even_indices + 1, total_values - 1)
            
            # Extract 4-bit values
            quantized[even_indices] = (packed >> 4) & 0xF
            if odd_indices[-1] < total_values:
                quantized[odd_indices] = packed & 0xF
            
            normalized = quantized.astype(np.float32) / 15.0
            
        elif precision_bits == 2:
            # Unpack 2-bit values
            packed = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            total_values = np.prod(shape)
            
            # Create array for unpacked values
            quantized = np.zeros(total_values, dtype=np.uint8)
            
            # Calculate number of complete bytes
            num_complete_bytes = total_values // 4
            
            # Unpack each byte into 4 values
            for i in range(num_complete_bytes):
                byte = packed[i]
                base_idx = i * 4
                quantized[base_idx] = (byte >> 6) & 0x3
                quantized[base_idx + 1] = (byte >> 4) & 0x3
                quantized[base_idx + 2] = (byte >> 2) & 0x3
                quantized[base_idx + 3] = byte & 0x3
            
            # Handle remaining values
            remaining = total_values % 4
            if remaining > 0:
                byte = packed[num_complete_bytes]
                base_idx = num_complete_bytes * 4
                for j in range(remaining):
                    shift = 6 - j * 2
                    quantized[base_idx + j] = (byte >> shift) & 0x3
            
            normalized = quantized.astype(np.float32) / 3.0
        
        # Denormalize
        array = normalized * (max_val - min_val) + min_val
        
        # Reshape to original shape
        array = array.reshape(shape)
        
        # Convert to tensor
        return torch.tensor(array, dtype=torch.float32)
    
    # Fallback to original method for other formats
    if 'data' in compressed_data:
        try:
            binary_data = base64.b64decode(compressed_data['data'])
            stream = io.BytesIO(binary_data)
            loaded = np.load(stream, allow_pickle=True)
            array = loaded['data']
            
            # Restore the original dtype
            if compressed_data.get('dtype') == 'float16':
                array = array.astype(np.float16)
            elif compressed_data.get('dtype') == 'float32':
                array = array.astype(np.float32)
            
            # Convert to tensor
            return torch.tensor(array)
        except Exception as e:
            print(f"Error decompressing data: {e}")
            return None
    
    return None

print("Testing decompression of logits...")

# Find the dataset file
dataset_file = "dataset_with_predictions.jsonl"
if not os.path.exists(dataset_file):
    dataset_file = os.path.join(os.path.dirname(__file__), dataset_file)

# Load the first entry
with open(dataset_file, 'r') as f:
    first_entry = json.loads(f.readline())

print("Entry loaded successfully.")

# Print teacher assertion
print("\nTeacher prediction:")
print(first_entry["teacher_prediction"])

# Compare with reference
print("\nReference assertions:")
for assertion in first_entry["assertions"]:
    print(assertion)

# Decompress the logits
try:
    logits = decompress_tensor_optimized(first_entry["teacher_logits"])
    
    # Print info
    print(f"\nSuccessfully decompressed logits!")
    print(f"Shape: {logits.shape}")
    print(f"Data type: {logits.dtype}")
    print(f"Min value: {logits.min().item():.4f}")
    print(f"Max value: {logits.max().item():.4f}")
    print(f"Sample values (first 5): {logits.flatten()[:5].tolist()}")
    
    compression_format = first_entry["teacher_logits"].get("format", "unknown")
    print(f"Compression format: {compression_format}")
    if 'compressed_size' in first_entry["teacher_logits"] and 'original_size' in first_entry["teacher_logits"]:
        original_size = first_entry["teacher_logits"]["original_size"]
        compressed_size = first_entry["teacher_logits"]["compressed_size"]
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        print(f"Compression ratio: {ratio:.2f}x")
        print(f"Original size: {original_size/1024/1024:.2f} MB")
        print(f"Compressed size: {compressed_size/1024/1024:.2f} MB")
except Exception as e:
    print(f"Error decompressing logits: {e}")
