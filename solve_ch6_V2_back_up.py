import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# Configuration
MODEL_PATH = 'ch6/model2.pth'
IMAGE_PATH = 'ch6/user.png'
OUTPUT_PATH = 'ch6/user_hacked.png'
PATCH_SIZE = (8, 8) # 64 pixels - Optimal size for L0 limit
TARGET_CLASS = 4 # The Flag Class
ITERATIONS = 20000 # Enough to converge with restarts
SCAN_FOCUS = (165, 75) # The "Hotspot" found previously
SCAN_RADIUS = 30 # Search area around hotspot
SCAN_STRIDE = 3 # High precision scan

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

def attack_patch(model, image_tensor, target_label, patch_h, patch_w):
    print(f"[*] Starting Patch Attack (Size {patch_h}x{patch_w}) on Class {target_label}...")
    
    _, img_h, img_w = image_tensor.shape
    
    best_prob = -1.0
    best_patch = None
    best_pos = (0, 0)
    
    # 1. Coarse Search: Slide the patch over the image to find best location
    print(f"[*] Step 1: Micro-Scanning around {SCAN_FOCUS} with stride {SCAN_STRIDE}...")
    
    start_x = max(0, SCAN_FOCUS[0] - SCAN_RADIUS)
    end_x = min(img_w - patch_w, SCAN_FOCUS[0] + SCAN_RADIUS)
    start_y = max(0, SCAN_FOCUS[1] - SCAN_RADIUS)
    end_y = min(img_h - patch_h, SCAN_FOCUS[1] + SCAN_RADIUS)
    
    for y in range(start_y, end_y, SCAN_STRIDE):
        for x in range(start_x, end_x, SCAN_STRIDE):
            # Initialize patch with random noise or mean color
            patch = torch.rand((3, patch_h, patch_w), requires_grad=True)
            opt = torch.optim.Adam([patch], lr=0.1)
            
            # Quick optimization for this location
            for _ in range(40): # Little bit more thorough check
                opt.zero_grad()
                patched_img = apply_patch(image_tensor, patch, x, y)
                out = model(patched_img.unsqueeze(0))
                loss = nn.CrossEntropyLoss()(out, torch.tensor([target_label]))
                loss.backward()
                opt.step()
                
            # Evaluate
            with torch.no_grad():
                patched_img = apply_patch(image_tensor, patch, x, y)
                out = model(patched_img.unsqueeze(0))
                prob = torch.softmax(out, dim=1)[0, target_label].item()
                
                if prob > best_prob:
                    best_prob = prob
                    best_pos = (x, y)
                    best_patch = patch.detach().clone()
                    print(f"    [+] Found better pos ({x}, {y}) -> Prob: {prob:.4f}")

    print(f"[*] Best location found: {best_pos} with initial prob {best_prob:.4f}")
    
    # 2. Fine Tuning: Optimize the patch at the best location
    print("[*] Step 2: Fine-tuning patch colors (AGGRESSIVE)...")
    x, y = best_pos
    patch = best_patch.clone().detach() # Start from best coarse patch
    patch.requires_grad = True
    
    # AGGRESSIVE optimizer with warm restarts and noise injection
    optimizer = torch.optim.Adam([patch], lr=0.25) # High Start LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2, eta_min=0.01)
    kappa = 5000 # Maximize margin
    
    best_prob_so_far = 0.0
    best_patch_saved = None
    stagnation_counter = 0
    last_best = 0.0
    
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        
        # Inject noise if stagnating
        if stagnation_counter > 5:
            with torch.no_grad():
                noise = torch.randn_like(patch) * 0.1
                patch.add_(noise)
                stagnation_counter = 0
                print(f"[!] Stagnation detected. Injected noise.")
        
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
        
        if i % 100 == 0:
             # Check for stagnation logic (check less frequently)
             if i % 500 == 0:
                 print(f"[Step {i}] Loss: {loss.item():.4f} Prob: {prob:.4f} Best: {best_prob_so_far:.4f}")
                 if best_prob_so_far - last_best < 0.005:
                     stagnation_counter += 1
                 else:
                     stagnation_counter = 0
                 last_best = best_prob_so_far
             
             # Early stop if we reach target (Local 0.70 is likely > 0.99 server)
             if best_prob_so_far >= 0.75:
                 print(f"[*] Target 0.75 reached! Stopping early.")
                 break

    # Use the BEST patch found during optimization
    print(f"[*] Final best probability achieved: {best_prob_so_far:.4f}")
    if best_patch_saved is not None:
        final_image = apply_patch(image_tensor, best_patch_saved, x, y)
    else:
        final_image = apply_patch(image_tensor, patch, x, y)
    return final_image

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
    
    # --- PATCH ATTACK ---
    adv_tensor = attack_patch(model, img_tensor, TARGET_CLASS, PATCH_SIZE[0], PATCH_SIZE[1])
    
    save_image(adv_tensor, OUTPUT_PATH)
    
    # Verify
    print("[*] Sending with curl...")
    os.system(f"curl -k -F 'image=@{OUTPUT_PATH}' https://programmation-ch6.challenge01.root-me.org/authenticate")

if __name__ == "__main__":
    main()

