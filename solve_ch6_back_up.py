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
PIXEL_BUDGET = 200  # Max number of pixels to modify
# EPSILON is effectively 1.0 (unbounded L_inf) because we don't clip
ITERATIONS = 100000 # MAX ITERATIONS for Extreme Confidence

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

def attack(model, image_tensor, original_label, target_label=None):
    print(f"[*] Starting Attack. Original Class: {original_label}")
    
    perturbed_image = image_tensor.clone().detach()
    perturbed_image.requires_grad = True
    
    # High Learning Rate for aggressive changes on few pixels
    optimizer = torch.optim.SGD([perturbed_image], lr=0.5)
    loss_fn = nn.CrossEntropyLoss()
    
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        output = model(perturbed_image.unsqueeze(0))
        
        if target_label is not None:
            loss = loss_fn(output, torch.tensor([target_label]))
        else:
            loss = -loss_fn(output, torch.tensor([original_label]))
        
        loss.backward()
        
        with torch.no_grad():
            grad = perturbed_image.grad
            if target_label is not None:
                 perturbed_image -= 0.1 * torch.sign(grad)
            else:
                 perturbed_image += 0.1 * torch.sign(grad)
            
            # Project onto L0 (Strict budget during optimization)
            perturbed_image.copy_(project_l0(image_tensor, perturbed_image, k=PIXEL_BUDGET))
            
            perturbed_image.grad.zero_()
            
        if i % 50 == 0:
            with torch.no_grad():
                output = model(perturbed_image.unsqueeze(0))
                pred = output.argmax(dim=1).item()
                probs = torch.softmax(output, dim=1)
                prob = probs[0, pred].item()
                
                print(f"[Step {i}] Pred: {pred} Prob: {prob:.4f}")
                # Only stop if EXTREMELY confident
                if target_label is not None and pred == target_label and prob > 0.999:
                     print(f"[+] Extreme confidence reached! Class {pred} (Prob: {prob:.4f})")
                     return perturbed_image

    return perturbed_image

def save_exact_l0_pixel(original_path, adv_tensor, output_path, k=200):
    """
    Ensures exactly k pixels are modified in the final PNG.
    """
    orig_img = Image.open(original_path).convert('RGB')
    orig_np = np.array(orig_img).astype(np.int16)
    
    adv_denorm = inverse_transform(adv_tensor.cpu())
    adv_denorm = torch.clamp(adv_denorm, 0, 1) * 255.0
    adv_np = adv_denorm.permute(1, 2, 0).detach().numpy().astype(np.int16)
    
    diff = np.abs(adv_np - orig_np)
    diff_sum = np.sum(diff, axis=2) # (H, W)
    
    flat_diff = diff_sum.flatten()
    if k >= len(flat_diff):
        top_k_indices = np.arange(len(flat_diff))
    else:
        top_k_indices = np.argpartition(flat_diff, -k)[-k:]
        
    mask = np.zeros_like(diff_sum, dtype=bool)
    rows, cols = np.unravel_index(top_k_indices, diff_sum.shape)
    mask[rows, cols] = True
    
    final_np = orig_np.copy()
    final_np[mask] = adv_np[mask]
    
    final_diff = np.sum(np.any(final_np != orig_np, axis=2))
    print(f"[*] Verifying PIXEL-WISE L0 count: {final_diff} pixels modified.")
    
    final_img = Image.fromarray(final_np.astype(np.uint8))
    # Disable compression to avoid artifacts creating extra "modified" pixels
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
    # Strategy:
    # k=53 -> Diff 207 (Failed, limit 200).
    # k=50 -> Diff < 200 (OK), Prob 0.94 (Failed, need 0.95).
    
    # We need to find the GOLDILOCKS ZONE.
    # Diff 207 / 53 pixels = 3.9 diff per pixel.
    # We need diff < 200.
    # Let's try k=51.
    # 51 might pass. Or 52.
    # Let's try k=51 pixels.
    # AND increase iterations even more (5000) to compensate for the 2 missing pixels compared to k=53.
    
    target_class = 11
    print(f"[*] Targeting Class: {target_class} (Admin?) with MAX ITERATIONS (5000) & k=51")
    
    adv_tensor = attack(model, img_tensor, original_class, target_label=target_class)
    
    save_exact_l0_pixel(IMAGE_PATH, adv_tensor, OUTPUT_PATH, k=50)

    print(f"[+] Adversarial image saved to {OUTPUT_PATH}")
    print("[*] Send this file with curl:")
    print(f"    curl -k -F 'image=@{OUTPUT_PATH}' https://programmation-ch6.challenge01.root-me.org/authenticate")

if __name__ == "__main__":
    main()
