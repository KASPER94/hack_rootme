🔍 ANALYSE COMPLÈTE - ON Y EST PRESQUE !
Résultat Actuel
Local Prob	Server Prob	Gap restant
0.5622	0.91	0.04
Le ratio serveur/local est de ~1.62x (0.91 / 0.5622)
Pour atteindre 0.95 serveur, il faut : 0.95 / 1.62 ≈ 0.586 local
On est à 0.5622, il manque ~0.024 de probabilité locale !
Historique Complet des Tests
Classe	Méthode	Local Prob	Server Prob	Message
0,1,3,6,7,9,11	Sparse	>0.95	N/A	"Morgan Freeman"
8	Sparse	0.94	0.94	"Morgan Freeman"
5	Sparse	0.94	0.94	"Morgan Freeman"
2	Sparse	~0.90	<0.95	Failed
10	Sparse	~0.88	<0.95	Failed
4	Patch	0.5622	0.91	"need 0.95"
La Classe 4 est la BONNE cible ! C'est la seule qui :
N'est PAS "Morgan Freeman"
A un ratio d'amplification serveur de 1.6x
Est proche du seuil (0.91 vs 0.95 requis)
Pourquoi la Probabilité Stagne à ~0.56 ?
Le problème est que l'optimisation converge vers un minimum local. Après 15000 itérations, la prob n'augmente que de 0.001 par tranche de 500 steps.
Solutions possibles :
Augmenter le nombre de pixels : 64 pixels (8x8) → 81 pixels (9x9)
Plus de surface d'attaque = plus de capacité à manipuler le modèle
Risque : dépasser la limite serveur (~66 pixels)
Changer la position du patch : (165, 75) donne 0.11 initial. Peut-être qu'une autre zone est plus sensible.
Utiliser un patch NON-CARRÉ : Un patch plus large horizontalement peut couvrir des features plus importantes du visage.
Combiner Patch + Sparse : Patch de 49 pixels (7x7) + 14 pixels sparse ailleurs = 63 pixels total.
🎯 STRATÉGIE FINALE
Option A : Patch Plus Grand (9x9 = 81 pixels, sauvé en k=65)
Risqué si le serveur compte strictement.
Option B : Optimisation Plus Agressive du Patch Actuel
Learning rate plus haut au début
Perturbation aléatoire périodique pour sortir du minimum local
Plus d'itérations (30k+)
Option C : Recherche de Position Plus Fine
Scanner avec stride=5 au lieu de 15
Tester plusieurs positions prometteuses en parallèle
Je recommande Option B car on est à 0.04 du but. On va :
Ajouter une perturbation aléatoire quand la loss stagne
Augmenter le learning rate initial à 0.3
Pousser à 30000 itérations