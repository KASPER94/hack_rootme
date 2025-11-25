import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# ============== GPU SETUP WITH FALLBACK ==============
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[*] GPU detected: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple Silicon
    print("[*] Apple MPS (Metal) detected")
else:
    DEVICE = torch.device("cpu")
    print("[*] No GPU detected, using CPU (slower)")

print(f"[*] Using device: {DEVICE}")

# Configuration
MODEL_PATH = 'ch6/model2.pth'
IMAGE_PATH = 'ch6/user.png'
OUTPUT_PATH = 'ch6/user_hacked.png'
PATCH_SIZE = (8, 8) # 64 pixels - Optimal size for L0 limit
TARGET_CLASS = 4 # The Flag Class
ITERATIONS = 12000 # Reduced slightly, GPU makes it fast
NUM_CANDIDATES = 5 # Generate multiple images to test

# Positions that showed promise
CANDIDATE_POSITIONS = [
    (165, 45),  # Best server result: 0.94
    (165, 75),  # Gave 0.91 server
    (162, 45),
    (168, 45),
    (165, 42),
]

def load_model_robust(path):
    print(f"[*] Loading model from {path}...")
    try:
        model = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(model, dict):
             print("[*] Loaded state_dict. Initializing Inception v3...")
             num_classes = 1000
             if 'fc.weight' in model:
                 num_classes = model['fc.weight'].shape[0]
             
             print(f"[*] Using num_classes={num_classes} for model init.")
             net = models.inception_v3(pretrained=False, aux_logits=True, num_classes=num_classes)
             net.load_state_dict(model)
             model = net
        
        model.eval()
        model = model.to(DEVICE)  # Move model to GPU/MPS if available
        print(f"[+] Model loaded and moved to {DEVICE}")
        return model
    except Exception as e:
        print(f"[-] Failed to load: {e}")
        exit(1)

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def inverse_transform(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def apply_patch(image_tensor, patch_tensor, x, y):
    """
    Applies the patch to the image tensor at position (x, y).
    patch_tensor: (3, H_p, W_p)
    """
    patched_img = image_tensor.clone()
    h, w = patch_tensor.shape[1], patch_tensor.shape[2]
    patched_img[:, y:y+h, x:x+w] = patch_tensor
    return patched_img

def attack_patch_at_position(model, image_tensor, target_label, patch_h, patch_w, position, seed):
    """Attack at a specific position with a specific seed."""
    torch.manual_seed(seed)
    x, y = position
    
    # Initialize patch
    patch = torch.rand((3, patch_h, patch_w), requires_grad=True)
    
    # Optimizer
    optimizer = torch.optim.Adam([patch], lr=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=0.01)
    kappa = 5000
    
    best_prob_so_far = 0.0
    best_patch_saved = None
    stagnation_counter = 0
    last_best = 0.0
    
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        
        # Inject noise if stagnating
        if stagnation_counter > 4:
            with torch.no_grad():
                noise = torch.randn_like(patch) * 0.15
                patch.add_(noise)
                stagnation_counter = 0
        
        patched_img = apply_patch(image_tensor, patch, x, y)
        output = model(patched_img.unsqueeze(0))
        
        # CW Loss
        logits = output[0]
        target_logit = logits[target_label]
        other_logits = logits.clone()
        other_logits[target_label] = -1e9
        max_other = torch.max(other_logits)
        
        loss = torch.max(max_other - target_logit, torch.tensor(-kappa).float())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track best
        with torch.no_grad():
            prob = torch.softmax(output, dim=1)[0, target_label].item()
            if prob > best_prob_so_far:
                best_prob_so_far = prob
                best_patch_saved = patch.detach().clone()
        
        if i % 500 == 0:
            if best_prob_so_far - last_best < 0.005:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            last_best = best_prob_so_far
            
            # Early stop
            if best_prob_so_far >= 0.80:
                break

    if best_patch_saved is not None:
        final_image = apply_patch(image_tensor, best_patch_saved, x, y)
    else:
        final_image = apply_patch(image_tensor, patch, x, y)
    return final_image, best_prob_so_far

def save_image(tensor, output_path):
    denorm = inverse_transform(tensor.cpu())
    denorm = torch.clamp(denorm, 0, 1) * 255.0
    img_np = denorm.permute(1, 2, 0).detach().numpy().astype(np.uint8)
    
    # CRITICAL: The server counts difference even on unchanged pixels if compression adds artifacts.
    # We must forcefully RESTORE the original pixels outside the patch area.
    # Load original again to be sure
    orig = Image.open(IMAGE_PATH).convert('RGB')
    orig_np = np.array(orig)
    
    # Create final array starting as a copy of original
    final_np = orig_np.copy()
    
    # Ideally we should know the patch coordinates (x, y, h, w) here.
    # But since we don't pass them, let's do a diff and only keep top 56 pixels change.
    # Or better, pass coordinates to this function.
    
    # Let's rely on the fact that attack_patch returns a tensor where only patch is modified.
    # But float -> uint8 conversion introduces rounding errors on the WHOLE image.
    # So we must use the mask trick again.
    
    diff = np.abs(img_np.astype(np.int16) - orig_np.astype(np.int16))
    diff_sum = np.sum(diff, axis=2)
    
    # Keep only top K changes (where K is patch size = 64, but save 63 to be safe)
    k = 63
    flat = diff_sum.flatten()
    if k < len(flat):
        idx = np.argpartition(flat, -k)[-k:]
        mask = np.zeros_like(diff_sum, dtype=bool)
        r, c = np.unravel_index(idx, diff_sum.shape)
        mask[r, c] = True
        
        # Apply changes only where mask is True
        final_np[mask] = img_np[mask]
    else:
        final_np = img_np # Should not happen given image size
        
    img = Image.fromarray(final_np)
    img.save(output_path, compress_level=0)
    print(f"[+] Image saved to {output_path}")

def main():
    model = load_model_robust(MODEL_PATH)
    img = Image.open(IMAGE_PATH).convert('RGB')
    preprocess = get_transforms()
    img_tensor = preprocess(img)
    
    print(f"[*] Generating {NUM_CANDIDATES} candidate images...")
    print(f"[*] Positions to test: {CANDIDATE_POSITIONS}")
    
    results = []
    
    for idx, pos in enumerate(CANDIDATE_POSITIONS):
        seed = 42 + idx * 1000  # Different seed for each
        print(f"\n{'='*60}")
        print(f"[*] Candidate {idx+1}/{NUM_CANDIDATES}: Position {pos}, Seed {seed}")
        print(f"{'='*60}")
        
        adv_tensor, local_prob = attack_patch_at_position(
            model, img_tensor, TARGET_CLASS, 
            PATCH_SIZE[0], PATCH_SIZE[1], 
            pos, seed
        )
        
        output_file = f"ch6/candidate_{idx+1}_pos{pos[0]}_{pos[1]}.png"
        save_image(adv_tensor, output_file)
        
        results.append((output_file, pos, local_prob))
        print(f"[+] Candidate {idx+1} saved: {output_file} (Local prob: {local_prob:.4f})")
    
    print(f"\n{'='*60}")
    print("[*] ALL CANDIDATES GENERATED!")
    print(f"{'='*60}")
    print("\nTest each one with:")
    for output_file, pos, local_prob in results:
        print(f"  curl -k -F 'image=@{output_file}' https://programmation-ch6.challenge01.root-me.org/authenticate")
        print(f"    -> Position {pos}, Local prob: {local_prob:.4f}")
    
    # Also save the best local prob as the main output
    best_result = max(results, key=lambda x: x[2])
    print(f"\n[*] Best local result: {best_result[0]} with prob {best_result[2]:.4f}")
    
    # Copy best to main output
    import shutil
    shutil.copy(best_result[0], OUTPUT_PATH)
    print(f"[+] Copied best to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

