import requests
import re
import base64
import time
import zxingcpp
from PIL import Image, ImageDraw
from io import BytesIO

URL = "http://challenge01.root-me.org/programmation/ch7/"

# Configuration des couleurs
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def solve():
    s = requests.Session()
    
    try:
        start_time = time.time()
        
        # 1. Récupération de la page
        print(f"[*] Récupération de {URL}...")
        r = s.get(URL)
        if r.status_code != 200:
            print(f"[-] Erreur HTTP: {r.status_code}")
            return

        # 2. Extraction de l'image Base64
        match = re.search(r'src="data:image/png;base64,([^"]+)"', r.text)
        if not match:
            print("[-] Image Base64 non trouvée.")
            return
        
        b64_data = match.group(1)
        img_bytes = base64.b64decode(b64_data)
        
        # 3. Reconstruction du QR Code (Fix des coins manquants)
        image = Image.open(BytesIO(img_bytes))
        draw = ImageDraw.Draw(image)
        
        # Paramètres du script original (adaptés à ce challenge spécifique)
        # Chaque module (pixel du QR) fait apparemment 9 pixels de large
        w = 9 
        w2 = w * 2
        w5 = w * 5
        w6 = w * 6
        w7 = w * 7
        
        # Les positions des 3 Finder Patterns (Haut-Gauche, Bas-Gauche, Haut-Droit)
        # Coordonnées (x, y) du coin supérieur gauche de chaque pattern
        patterns_pos = [
            (18, 18),   # Haut-Gauche
            (18, 216),  # Bas-Gauche
            (216, 18)   # Haut-Droit
        ]
        
        print("[*] Réparation des patterns de position...")
        for x, y in patterns_pos:
            # 1. Grand carré noir extérieur (7x7 modules)
            draw.rectangle([(x, y), (x + w7 - 1, y + w7 - 1)], fill=BLACK)
            # 2. Carré blanc intérieur (5x5 modules) -> crée l'anneau
            draw.rectangle([(x + w, y + w), (x + w6 - 1, y + w6 - 1)], fill=WHITE)
            # 3. Petit carré noir central (3x3 modules)
            draw.rectangle([(x + w2, y + w2), (x + w5 - 1, y + w5 - 1)], fill=BLACK)
            
        # Sauvegarde pour debug (optionnel, à commenter pour la vitesse pure)
        # image.save("debug_fixed_qr.png")
        
        # 4. Décodage avec ZXing-CPP
        # zxing-cpp prend directement l'image PIL ou des bytes/numpy
        # Convertissons en niveau de gris pour être sûr
        gray_img = image.convert('L')
        
        # Lecture via zxing-cpp
        # Note: zxing-cpp attend souvent un array numpy ou bytes avec dimensions
        # Utilisons l'interface qui accepte les objets PIL si dispo, sinon conversion
        try:
            # Tentative directe (dépend de la version du binding python)
            results = zxingcpp.read_barcodes(gray_img)
        except:
            # Fallback: conversion explicite
            import numpy as np
            np_img = np.array(gray_img)
            results = zxingcpp.read_barcodes(np_img)
            
        if not results:
            print("[-] Échec du décodage après réparation.")
            image.save("debug_failed_fix.png")
            return
            
        raw_text = results[0].text
        print(f"[*] Contenu brut : {raw_text}")
        
        # Le challenge demande souvent juste la clé, parfois préfixée
        # Exemple: "The key is XXXXX" -> on extrait XXXXX
        key = raw_text
        match_key = re.search(r'The key is (.+)', raw_text)
        if match_key:
            key = match_key.group(1)
            
        print(f"[*] Clé extraite : {key}")

        # 5. Envoi de la réponse
        payload = {'metu': key}
        print(f"[*] Envoi...")
        r_post = s.post(URL, data=payload)
        
        print(f"[*] Temps total : {time.time() - start_time:.2f}s")

        # 6. Vérification (Affiche plus de détails pour debug)
        print(f"[*] Réponse serveur ({len(r_post.text)} chars):")
        
        if "congratulations" in r_post.text.lower() or "password" in r_post.text.lower() or "flag" in r_post.text.lower():
            clean_response = re.sub(r'<[^>]+>', '', r_post.text)
            for line in clean_response.split('\n'):
                line = line.strip()
                if line and ("flag" in line.lower() or "congratz" in line.lower() or "password" in line.lower()):
                    print(f"\n[+] VICTOIRE : {line}")
                    return # Stop si trouvé
            
            # Si pas trouvé dans les lignes filtrées, affiche tout le body propre
            print(clean_response)
        else:
            # Affiche le body nettoyé pour comprendre l'erreur
            clean_response = re.sub(r'<[^>]+>', '', r_post.text)
            # On affiche les lignes non vides
            for line in clean_response.split('\n'):
                if line.strip(): print(f"    {line.strip()}")

    except Exception as e:
        print(f"[-] Exception : {e}")

if __name__ == "__main__":
    solve()

