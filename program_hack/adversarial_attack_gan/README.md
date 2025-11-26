# Analyse Technique : Adversarial Attack sur CNN (Root-Me)

Ce document détaille les concepts théoriques et mathématiques derrière la solution du challenge "Adversarial Attack - GAN".

## 1. Le Problème : Attaque Ciblée (Targeted Evasion)

L'objectif est de tromper un classificateur d'images (CNN basé sur MobileNetV2) pour qu'il identifie une image donnée comme appartenant à une classe cible spécifique (ici, l'Administrateur, Classe 0), avec une confiance extrême ($P(y_{target}|x) > 0.999$).

Mathématiquement, nous cherchons une perturbation $\delta$ telle que :
$$ \text{argmax } f(x + \delta) = y_{target} $$
$$ P(y_{target} | x + \delta) > 0.999 $$
$$ ||\delta||_p < \epsilon \quad (\text{contrainte de discrétion, optionnelle ici}) $$

## 2. Le Piège : "Min-Max Scaling" et Transférabilité

Le challenge donnait un indice crucial : **"Min-Max Scaling"**.

En apprentissage profond, les entrées sont normalisées. La normalisation standard est souvent :
$$ x_{norm} = \frac{x - \mu}{\sigma} \quad (\text{ex: ImageNet}) $$
Ou le scaling fixe :
$$ x_{norm} = \frac{x}{255.0} \in [0, 1] $$

Le **Min-Max Scaling dynamique** est défini par :
$$ x_{norm} = \frac{x - \min(x)}{\max(x) - \min(x)} $$

**L'impact sur l'attaque :**
Si l'attaquant génère une image $x_{adv}$ en supposant une normalisation fixe, mais que le serveur applique un Min-Max dynamique, les gradients calculés localement ($\nabla_x J$) ne correspondent plus exactement à la surface de perte du serveur.
*   Une perturbation $\delta$ ajoutée pour augmenter la confiance peut modifier $\min(x)$ ou $\max(x)$, ce qui change toute la distribution des pixels après normalisation.
*   Cela explique pourquoi nos attaques générées à partir de "bruit" (from scratch) plafonnaient à $0.997$ : elles étaient fragiles aux variations de prétraitement.

## 3. La Solution : Morphing Robuste (I-FGSM)

Pour contourner ce problème de transférabilité, nous avons utilisé une approche de **Morphing** plutôt que de génération ex-nihilo.

### A. Point de départ stratégique ($x_0$)
Au lieu de partir d'un bruit aléatoire $x_0 \sim \mathcal{N}(0, I)$, nous avons utilisé l'image `user.jpg` fournie.
*   **Pourquoi ?** Cette image obtenait déjà un score de $0.942$ sur le serveur (même si la classe était mauvaise). Cela prouve qu'elle réside dans une région "stable" de l'espace latent du modèle serveur, peu affectée par le Min-Max scaling.

### B. L'Algorithme : Iterative Fast Gradient Sign Method (I-FGSM)
Nous avons appliqué une variante itérative de FGSM pour déplacer progressivement l'image de la classe source (Scarlett Johansson) vers la classe cible (Admin).

À chaque étape $t$, nous mettons à jour l'image :
$$ x_{t+1} = \text{Clip}_{[0,1]} \left( x_t - \alpha \cdot \text{sign}(\nabla_{x} J(\theta, x_t, y_{target})) \right) $$

Où :
*   $J(\theta, x, y)$ est la fonction de perte (ici, on maximise la probabilité de la cible, donc on minimise $-J$).
*   $\alpha$ est le pas d'apprentissage (Learning Rate), gardé très faible ($0.0005$) pour ne pas détruire la structure de l'image originale.
*   $\text{sign}(\cdot)$ permet de s'affranchir de l'amplitude du gradient (souvent très faible ou explosive dans les réseaux profonds) et de ne garder que la direction optimale.

### C. Fonction de Perte "Logit Margin" (Carlini-Wagner)
Pour atteindre une confiance de $0.999$, la simple Cross-Entropy ($\log P(y)$) ne suffit pas car elle sature (gradients proches de 0 quand $P \approx 1$).
Nous avons utilisé une perte basée sur la marge des logits $Z(x)$ (avant Softmax) :

$$ \mathcal{L} = \max( \max_{i \neq t} Z(x)_i - Z(x)_t, -\kappa ) $$

Cela force le réseau non seulement à choisir la bonne classe, mais à créer un écart (marge) énorme entre la classe cible et la seconde meilleure classe, garantissant une robustesse face à la compression JPEG ou au scaling du serveur.

## 4. Résultat

En appliquant cette méthode sur toutes les classes possibles, nous avons découvert que :
1.  La classe 9 (Source) et la classe 2 (Target facile) étaient des leurres ("Welcome... but no flag").
2.  La **Classe 0 (Administrator)** était la vraie cible.
3.  L'image générée par morphing robuste a résisté au prétraitement du serveur et a passé le seuil de $0.999$.

---
*Généré par l'Assistant IA pour l'analyse post-mortem du challenge.*
