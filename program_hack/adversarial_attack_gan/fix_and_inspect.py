import h5py
import json
import tensorflow as tf
import os

ORIGINAL_MODEL = 'program_hack/adversarial_attack_gan/model.h5'
FIXED_MODEL = 'program_hack/adversarial_attack_gan/model_fixed.h5'

def fix_and_load():
    # 1. Copy file to avoid corrupting original
    with open(ORIGINAL_MODEL, 'rb') as f_src:
        with open(FIXED_MODEL, 'wb') as f_dst:
            f_dst.write(f_src.read())

    # 2. Modify the config in the new file
    with h5py.File(FIXED_MODEL, 'r+') as f:
        if 'model_config' not in f.attrs:
            print("[-] 'model_config' not found in h5 attributes.")
            return

        config_str = f.attrs['model_config']
        if isinstance(config_str, bytes):
            config_str = config_str.decode('utf-8')
        
        config = json.loads(config_str)
        
        # Recursive function to clean config
        def clean_layer_config(layer):
            if 'class_name' in layer and layer['class_name'] == 'DepthwiseConv2D':
                if 'config' in layer and 'groups' in layer['config']:
                    print(f"[*] Removing 'groups' from layer {layer['config'].get('name', 'unknown')}")
                    del layer['config']['groups']
            
            # Recurse into nested layers (e.g. Model, Sequential)
            if 'layers' in layer.get('config', {}):
                for sub_layer in layer['config']['layers']:
                    clean_layer_config(sub_layer)
        
        # Handle Sequential/Functional list of layers
        if 'config' in config and 'layers' in config['config']:
            for layer in config['config']['layers']:
                clean_layer_config(layer)
        
        # Write back
        f.attrs['model_config'] = json.dumps(config).encode('utf-8')
        print("[*] Config patched.")

    # 3. Try loading
    try:
        model = tf.keras.models.load_model(FIXED_MODEL)
        print("\n[+] Model Loaded Successfully!")
        model.summary()
        print(f"[+] Input Shape: {model.input_shape}")
        
        # Save input shape for next steps
        input_shape = model.input_shape
        if input_shape[0] is None:
            input_shape = input_shape[1:] # Remove batch dim
        
        print(f"[+] Target Input Shape for Attack: {input_shape}")
        
    except Exception as e:
        print(f"[-] Loading failed even after patch: {e}")

if __name__ == "__main__":
    fix_and_load()

