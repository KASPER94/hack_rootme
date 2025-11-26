import tensorflow as tf
import numpy as np
from PIL import Image
import os
import io

MODEL_PATH = 'program_hack/adversarial_attack_gan/model_fixed.h5'
# On repart de l'image de base qui a une bonne "transférabilité"
SOURCE_IMAGE = 'program_hack/adversarial_attack_gan/user.jpg' 
OUTPUT_DIR = 'program_hack/adversarial_attack_gan/solutions/robust'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    return img_np.reshape(1, 224, 224, 3)

@tf.function
def robust_loss(img_var, model, target_class):
    # 1. Standard prediction
    pred_standard = model(img_var)
    
    # 2. Simulate Min-Max scaling variations
    # Server might do: (img - min) / (max - min)
    # If we force min=0, max=1, it's identity.
    # But maybe they do [-1, 1]?
    img_scaled = (img_var * 2.0) - 1.0
    pred_scaled = model(img_scaled)
    
    # 3. Average logits
    logits = (pred_standard[0] + pred_scaled[0]) / 2.0
    
    target_logit = logits[target_class]
    other_logits = tf.concat([logits[:target_class], logits[target_class+1:]], axis=0)
    max_other = tf.reduce_max(other_logits)
    
    # Maximize margin
    margin = target_logit - max_other
    loss = -margin
    
    return loss, logits[target_class], margin

def attack_class(model, source_img, target_class, steps=500):
    print(f"[*] Attacking Class {target_class} with Robust FGSM...")
    
    img_var = tf.Variable(source_img, dtype=tf.float32)
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    
    best_margin = -999
    best_img = source_img[0]
    
    for step in range(steps):
        with tf.GradientTape() as tape:
            loss, prob, margin = robust_loss(img_var, model, target_class)
            
        gradients = tape.gradient(loss, img_var)
        gradients = tf.sign(gradients) # FGSM style: use sign of gradient
        
        # Apply update
        img_var.assign_add(-gradients * 0.0005) # Tiny steps
        img_var.assign(tf.clip_by_value(img_var, 0, 1))
        
        m = margin.numpy()
        if m > best_margin:
            best_margin = m
            best_img = img_var.numpy()[0]
            
        if step % 100 == 0:
            print(f"    Step {step}: Margin={m:.2f}")
            
    return best_img

def main():
    print(f"[*] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    source_img = load_image(SOURCE_IMAGE)
    
    # Target only the promising classes
    for cls in [4, 5]:
        final_img = attack_class(model, source_img, cls, steps=1000)
        
        # Save as High Quality JPEG to survive compression
        save_path = os.path.join(OUTPUT_DIR, f'robust_class_{cls}.jpg')
        img_uint8 = (final_img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(save_path, format='JPEG', quality=100, subsampling=0)
        
        print(f"[+] Saved {save_path}")
        print(f"[*] Test: curl -k -F 'image=@{save_path}' https://programmation-ch5.challenge01.root-me.org/authenticate")

if __name__ == "__main__":
    main()