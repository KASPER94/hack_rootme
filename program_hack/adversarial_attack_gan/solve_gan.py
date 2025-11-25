import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Use the fixed model
MODEL_PATH = 'program_hack/adversarial_attack_gan/model_fixed.h5'
OUTPUT_DIR = 'program_hack/adversarial_attack_gan/solutions'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

@tf.function
def train_step(img_var, model, target_class):
    with tf.GradientTape() as tape:
        tape.watch(img_var)
        
        # Hypothesis: The model might expect inputs in [-1, 1] range (common for MobileNet/Inception)
        # But since Prob goes to 0 immediately with large LR, maybe it's extremely sensitive.
        # Let's try NO preprocessing first (keeping [0, 1]), but with much smaller LR.
        
        # Actually, if Min-Max scaling is mentioned, the server does:
        # (img - min) / (max - min)
        # If we provide an image with min=0 and max=255, it becomes [0, 1] float.
        # So the model likely expects [0, 1].
        
        predictions = model(img_var)
        
        loss = -predictions[0][target_class]
    
    gradients = tape.gradient(loss, img_var)
    gradients /= tf.math.reduce_std(gradients) + 1e-8
    return gradients, predictions[0][target_class]

def optimize_image(model, target_class, steps=200, lr=0.001):
    # Start with gray 0.5
    img_np = np.ones((1, 224, 224, 3), dtype=np.float32) * 0.5
    # Add tiny noise
    img_np += np.random.uniform(-0.01, 0.01, (1, 224, 224, 3)).astype(np.float32)
    img_np = np.clip(img_np, 0, 1)
    
    img_var = tf.Variable(img_np, dtype=tf.float32)

    print(f"[*] Optimizing for Class {target_class} (LR={lr})...")

    best_prob = 0.0
    best_img = None

    for step in range(steps):
        gradients, prob = train_step(img_var, model, target_class)
        
        img_var.assign_add(gradients * lr)
        img_var.assign(tf.clip_by_value(img_var, 0, 1))

        current_prob = prob.numpy()
        if current_prob > best_prob:
            best_prob = current_prob
            best_img = img_var.numpy()

        if step % 50 == 0:
            print(f"    Step {step}: Prob = {current_prob:.4f}")
            if current_prob > 0.999:
                break
    
    return best_img[0], best_prob

def main():
    print(f"[*] Loading model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"[-] Failed to load model: {e}")
        return

    # Class 2 and 0 seemed to have non-zero starts in previous attempts
    target_classes = [2, 0, 8] 
    
    for class_idx in target_classes:
        img, prob = optimize_image(model, class_idx, steps=300, lr=0.005)
        print(f"[*] Class {class_idx} Final Prob: {prob:.4f}")
        
        if prob > 0.5: # Save anything decent
            save_path = os.path.join(OUTPUT_DIR, f'sol_class_{class_idx}.png')
            
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Enforce Min-Max to be 0 and 255 exactly
            img_uint8[0, 0, 0] = 0
            img_uint8[0, 0, 1] = 255
            
            Image.fromarray(img_uint8).save(save_path)
            print(f"[+] Saved {save_path}")

if __name__ == "__main__":
    main()
