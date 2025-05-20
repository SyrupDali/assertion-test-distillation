import numpy as np
import torch


def entropy_encode(data_bytes):
    """
    Apply LZ4 compression to binary data.
    Simple, fast, and effective entropy coding.

    Args:
        data_bytes: Binary data as bytes

    Returns:
        Dictionary with compressed data and metadata
    """
    try:
        import lz4.frame

        # Apply LZ4 compression with maximum compression level
        compressed = lz4.frame.compress(data_bytes, compression_level=9)

        # Return compressed data and metadata
        return {
            'encoding': 'lz4',
            'data': compressed.hex(),
            'original_size': len(data_bytes),
            'compressed_size': len(compressed)
        }
    except ImportError:
        print("Warning: LZ4 not installed. Please install with 'pip install lz4'.")
        # Return uncompressed data if LZ4 is not available
        return {
            'encoding': 'none',
            'data': data_bytes.hex(),
            'original_size': len(data_bytes),
            'compressed_size': len(data_bytes)
        }
    except Exception as e:
        print(f"Error during LZ4 compression: {e}")
        import traceback
        traceback.print_exc()
        # Return uncompressed data if compression fails
        return {
            'encoding': 'none',
            'data': data_bytes.hex(),
            'original_size': len(data_bytes),
            'compressed_size': len(data_bytes)
        }


def entropy_decode(encoded_data):
    """
    Decompress data that was compressed with LZ4.

    Args:
        encoded_data: Dictionary with compressed data and metadata

    Returns:
        Original binary data
    """
    encoding = encoded_data.get('encoding', 'none')

    try:
        if encoding == 'lz4':
            import lz4.frame

            # Get compressed data
            compressed = bytes.fromhex(encoded_data.get('data'))

            # Decompress with LZ4
            decompressed = lz4.frame.decompress(compressed)
            return decompressed
        elif encoding == 'none':
            # No compression was applied
            return bytes.fromhex(encoded_data.get('data'))
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
    except ImportError:
        print("Error: LZ4 not installed. Please install with 'pip install lz4'.")
        raise
    except Exception as e:
        print(f"Error during LZ4 decompression: {e}")
        import traceback
        traceback.print_exc()
        raise


def compress_logits(logits, bits=16):
    """
    Compress logits using bit-depth reduction and LZ4 compression.
    Simple, fast, and effective compression for neural network logits.

    Args:
        logits: Tensor of logits
        bits: Bit depth for quantization (4, 8, 16, 32)

    Returns:
        Compressed representation with statistics
    """
    try:
        # Convert to numpy for processing
        logits_np = logits.cpu().numpy()
        original_shape = logits_np.shape
        original_size = logits_np.size * 4  # 32-bit float = 4 bytes per value

        # Step 1: Apply bit-depth reduction
        if bits == 16:
            # 16-bit compression (FP16)
            compressed_np = logits_np.astype(np.float16)
            bit_compressed_size = compressed_np.size * 2  # 2 bytes per value
            quant_params = {}

        elif bits == 8:
            # 8-bit quantization
            logits_min = logits_np.min()
            logits_max = logits_np.max()
            scale = (logits_max - logits_min) / 255 if logits_max > logits_min else 1
            zero_point = -logits_min / scale if scale > 0 else 0
            compressed_np = np.clip(np.round(logits_np / scale + zero_point), 0, 255).astype(np.uint8)
            bit_compressed_size = compressed_np.size  # 1 byte per value

            quant_params = {
                'scale': float(scale),
                'zero_point': float(zero_point)
            }

        elif bits == 4:
            # 4-bit quantization
            logits_min = logits_np.min()
            logits_max = logits_np.max()
            scale = (logits_max - logits_min) / 15 if logits_max > logits_min else 1
            zero_point = -logits_min / scale if scale > 0 else 0
            compressed_np = np.clip(np.round(logits_np / scale + zero_point), 0, 15).astype(np.uint8)

            # Pack 2 4-bit values per byte
            flat = compressed_np.flatten()
            if len(flat) % 2 == 1:
                flat = np.append(flat, 0)  # Pad if odd length

            # Pack 2 4-bit values per byte
            even_indices = np.arange(0, len(flat), 2)
            odd_indices = np.arange(1, len(flat), 2)

            # Handle the case where there are more even indices than odd indices
            if len(odd_indices) < len(even_indices):
                packed = np.zeros(len(even_indices), dtype=np.uint8)
                packed[:len(even_indices)] = flat[even_indices] & 0x0F
                packed[:len(odd_indices)] |= (flat[odd_indices] << 4) & 0xF0
            else:
                packed = (flat[even_indices] | (flat[odd_indices] << 4)).astype(np.uint8)

            compressed_np = packed
            bit_compressed_size = len(packed)  # 0.5 bytes per value, packed

            quant_params = {
                'scale': float(scale),
                'zero_point': float(zero_point),
                'packed': True,
                'values_per_byte': 2,
                'padded': bool(len(flat) != np.prod(logits_np.shape))
            }

        else:  # bits == 32 or other
            # 32-bit (unchanged)
            compressed_np = logits_np
            bit_compressed_size = compressed_np.size * 4  # 4 bytes per value
            quant_params = {}

        # Step 2: Get binary data
        data_bytes = compressed_np.tobytes()

        # Step 3: Apply LZ4 compression
        encoded_data = entropy_encode(data_bytes)
        final_size = encoded_data['compressed_size']

        # Create result with compression statistics
        result = {
            'format': 'lz4',
            'compression': {
                'bits': bits,
                'original_size_bytes': int(original_size),
                'bit_compressed_size_bytes': int(bit_compressed_size),
                'final_size_bytes': int(final_size),
                'compression_ratio': float(original_size / final_size)
            },
            'shape': list(original_shape),
            'data_encoded': encoded_data,
            'bits': bits
        }

        # Add quantization parameters if needed
        if quant_params:
            result.update(quant_params)

        return result

    except Exception as e:
        print(f"Error during compression: {e}")
        import traceback
        traceback.print_exc()
        return None


def decompress_logits(compressed_logits):
    """
    Decompress logits that were compressed with bit-depth reduction and LZ4.

    Args:
        compressed_logits: The compressed representation

    Returns:
        PyTorch tensor with decompressed logits
    """
    if compressed_logits is None:
        return None

    try:
        # Extract basic information
        bits = compressed_logits.get('bits')
        shape = compressed_logits.get('shape')

        # Convert shape from list to tuple if necessary
        if isinstance(shape, list):
            shape = tuple(shape)

        # Step 1: Decompress LZ4 data
        encoded_data = compressed_logits.get('data_encoded')
        data_bytes = entropy_decode(encoded_data)

        # Step 2: Process according to bit depth
        if bits == 16:
            # 16-bit decompression (FP16)
            logits_np = np.frombuffer(data_bytes, dtype=np.float16).reshape(shape)
            # Convert to float32 for compatibility with PyTorch operations
            logits_np = logits_np.astype(np.float32)
            return torch.tensor(logits_np)

        elif bits == 8:
            # 8-bit dequantization
            logits_int8 = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)

            # Dequantize
            scale = compressed_logits.get('scale', 1.0)
            zero_point = compressed_logits.get('zero_point', 0.0)
            logits_np = (logits_int8.astype(np.float32) - zero_point) * scale
            return torch.tensor(logits_np)

        elif bits == 4:
            # Check if values were packed
            is_packed = compressed_logits.get('packed', False)

            if is_packed:
                # Unpack 4-bit values (2 values per byte)
                packed = np.frombuffer(data_bytes, dtype=np.uint8)

                # Calculate total values in original tensor
                total_values = np.prod(shape)
                unpacked = np.zeros(total_values, dtype=np.uint8)

                # Handle even indices (lower 4 bits of each byte)
                even_indices = np.arange(0, total_values, 2)
                even_indices = even_indices[even_indices < total_values]
                unpacked[even_indices] = packed[:len(even_indices)] & 0x0F

                # Handle odd indices (upper 4 bits of each byte)
                odd_indices = np.arange(1, total_values, 2)
                odd_indices = odd_indices[odd_indices < total_values]
                unpacked[odd_indices] = (packed[:len(odd_indices)] >> 4) & 0x0F

                # Reshape to original shape
                logits_int4 = unpacked.reshape(shape)
            else:
                # Direct interpretation (rarely used for 4-bit)
                logits_int4 = np.frombuffer(data_bytes, dtype=np.uint8).reshape(shape)

            # Dequantize
            scale = compressed_logits.get('scale', 1.0)
            zero_point = compressed_logits.get('zero_point', 0.0)
            logits_np = (logits_int4.astype(np.float32) - zero_point) * scale
            return torch.tensor(logits_np)

        elif bits == 32:
            # 32-bit (float32) decompression
            logits_np = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape)
            return torch.tensor(logits_np)

        else:
            raise ValueError(f"Unsupported bit depth: {bits}")

    except Exception as e:
        print(f"Error during decompression: {e}")
        import traceback
        traceback.print_exc()
        return None
