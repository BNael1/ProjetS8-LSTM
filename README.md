# lstm_sgd_canevas

> Projet **UFR MIM — Université de Lorraine — 2026**
> **Auteurs :** Hugo Morlet, Naël Bensaadi

Implémentation **from scratch en NumPy** d'un réseau récurrent **LSTM** entraîné
par descente de gradient stochastique avec :

- décroissance du *learning rate* (LR decay),
- rognage du gradient (*gradient clipping*),
- propagation arrière dans le temps (**BPTT**).

L'objectif est de **générer du texte** à partir d'un petit corpus.

---

## Structure du dépôt

| Fichier | Rôle |
|---|---|
| `lstm_sgd_canevas.ipynb` | Le **livrable principal**. Notebook qui reprend la structure du sujet (sections 1.1 à 1.3.6) avec code et explications. |
| `requirements.txt` | Dépendances Python (`numpy`). |

---

## Comment lancer

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter
jupyter notebook lstm_sgd_canevas.ipynb
```

Puis "Run All" depuis le menu Jupyter. L'apprentissage complet (100 epochs)
prend environ **2 à 3 minutes** sur un laptop standard. La perte tombe de
~4.15 (aléatoire) à ~0.05 et le modèle est capable de reproduire le corpus.


---

## Résultat attendu

```
Vocabulaire: 90 mots
Données   : 5750 tokens
Epoch 0:  Loss = 4.15, LR = 0.099
Epoch 10: Loss = 0.07, LR = 0.090
...
Epoch 90: Loss = 0.05, LR = 0.052

Génération de texte:
the new model that the company announced yesterday was better than
expected. investors were pleased with the results and the stock price ...
```

---

## Architecture du modèle

```
Mots (entiers)
     |
     v
[ E : matrice englobante  |V| x |E| ]   <- embedding
     |
     v
[ LSTMCell ]                            <- 1 couche récurrente
     |
     v
[ Wout, Bout : |H| x |V| ]              <- couche de sortie linéaire
     |
     v
[ Softmax ] -> probabilités sur le vocabulaire
```

Hyperparamètres par défaut :

- `embed_size = 50`
- `hidden_size = 100`
- `seq_length = 10`
- `epochs = 100`
- `lr = 0.1`, `decay_rate = 0.01`
- `clip_norm = 5.0`

