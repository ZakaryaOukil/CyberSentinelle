# Guide de Test d'Attaque DoS - CyberSentinelle
## Pour démonstration éducative uniquement

### ⚠️ IMPORTANT - AVERTISSEMENT LÉGAL
Ce guide est **UNIQUEMENT** pour des tests sur votre propre infrastructure dans un cadre éducatif.
Effectuer une attaque DoS sur un système sans autorisation est **ILLÉGAL** et passible de poursuites.

---

## Méthode 1 : Simulation intégrée (Recommandée)

La méthode la plus simple est d'utiliser le bouton **"Lancer l'attaque"** dans la page Live Monitor.
Cette méthode est sûre et contrôlée.

---

## Méthode 2 : Test de charge avec curl (Terminal)

### Installation (si nécessaire)
```bash
# Sur Ubuntu/Debian
sudo apt-get install curl

# Sur MacOS (déjà installé)
```

### Commande de test basique
Ouvrez un terminal et exécutez :

```bash
# Remplacez URL_DE_VOTRE_SITE par l'URL de votre site déployé
URL="https://votre-site.onrender.com"

# Envoyer 100 requêtes rapidement
for i in {1..100}; do
  curl -s "$URL/api/monitor/ping" -X POST &
done
wait
echo "100 requêtes envoyées"
```

### Script d'attaque continue (arrêtez avec Ctrl+C)
```bash
URL="https://votre-site.onrender.com"

while true; do
  for i in {1..50}; do
    curl -s "$URL/api/monitor/ping" -X POST &
  done
  sleep 1
done
```

---

## Méthode 3 : Apache Bench (ab) - Plus professionnel

### Installation
```bash
# Ubuntu/Debian
sudo apt-get install apache2-utils

# MacOS (déjà inclus)
```

### Commandes de test
```bash
URL="https://votre-site.onrender.com/api/monitor/ping"

# Test basique : 1000 requêtes, 100 en parallèle
ab -n 1000 -c 100 -p /dev/null -T 'application/json' "$URL"

# Test intensif : 5000 requêtes, 200 en parallèle
ab -n 5000 -c 200 -p /dev/null -T 'application/json' "$URL"
```

---

## Méthode 4 : Python (si vous préférez)

Créez un fichier `attack_test.py` :

```python
import requests
import threading
import time

# Configuration
TARGET_URL = "https://votre-site.onrender.com/api/monitor/ping"
THREADS = 50
DURATION_SECONDS = 30

def send_requests():
    while running:
        try:
            requests.post(TARGET_URL, timeout=5)
        except:
            pass

print(f"Démarrage de l'attaque test sur {TARGET_URL}")
print(f"Threads: {THREADS}, Durée: {DURATION_SECONDS}s")
print("Appuyez sur Ctrl+C pour arrêter\n")

running = True
threads = []

for i in range(THREADS):
    t = threading.Thread(target=send_requests)
    t.start()
    threads.append(t)

try:
    time.sleep(DURATION_SECONDS)
except KeyboardInterrupt:
    print("\nArrêt demandé...")

running = False
for t in threads:
    t.join()

print("Test terminé!")
```

Exécutez avec : `python attack_test.py`

---

## Ce que vous devriez observer

1. **Avant l'attaque** : 
   - Status: "SYSTÈME NORMAL" (vert)
   - Requêtes/sec: proche de 0

2. **Pendant l'attaque** :
   - Status: "ATTAQUE EN COURS" (rouge, clignotant)
   - Requêtes/sec: > 50 (seuil)
   - Graphique: pics rouges
   - Logs: alertes "CRITICAL"
   - Son d'alerte (si activé)

3. **Après l'arrêt** :
   - Retour progressif à "SYSTÈME NORMAL"
   - Message "Attaque terminée" dans les logs

---

## Patterns d'attaque détectés par le système

- **SINGLE_SOURCE_FLOOD** : > 50% du trafic vient d'une seule IP
- **RAPID_FIRE** : Requêtes espacées de < 50ms
- **IDENTICAL_REQUESTS** : Même endpoint bombardé

---

## Pour votre présentation

1. Montrez l'état normal du système
2. Expliquez les métriques (seuil, req/s, etc.)
3. Lancez l'attaque (bouton ou terminal)
4. Montrez la détection en temps réel
5. Expliquez les patterns détectés
6. Arrêtez l'attaque et montrez le retour à la normale

Bonne présentation ! 🎓
