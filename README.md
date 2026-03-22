# VisioFind

Projet de recherche visuelle inspire de ton cahier des charges:
- upload image ou video
- extraction de frames video
- selection manuelle d'une frame (important)
- recherche de similarite
- deux modes: `animaux` et `objets_lieux`
- index vectoriel persistant (cache disque)
- lien Google Maps pour classes de lieux connus

## 1) Installation

```bash
cd C:\Users\hp\visiofind
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si tu avais l’ancienne lib `duckduckgo-search`, migre vers `ddgs` :

```bash
pip uninstall duckduckgo-search -y
pip install ddgs
```

## 2) Generer le dataset de reference

Le script telecharge des images avec `ddgs` (DuckDuckGo), puis **repli automatique Bing** (`icrawler`) si ton reseau ne peut pas joindre duckduckgo.com.

- `data/reference/animaux/*`
- `data/reference/objets_lieux/*`

```bash
python scripts/build_dataset.py --preset extended --per-class 120
```

Tu peux aussi utiliser:

```bash
python scripts/build_dataset.py --preset basic --per-class 80
```

- `basic`: petit dataset, plus rapide
- `extended`: dataset riche (recommande)

### Option fiable: Kaggle (recommande si timeout reseau)

1) **Token API** (obligatoire pour que `kaggle` te reconnaisse)  
- Va sur [kaggle.com](https://www.kaggle.com/) → **Settings** (icône compte) → section **API**  
- Clique **Create New API Token** → un fichier `kaggle.json` est téléchargé

2) **Emplacement du fichier** (Windows)  
- Crée le dossier : `%USERPROFILE%\.kaggle\` (ex. `C:\Users\TonNom\.kaggle\`)  
- Copie **`kaggle.json`** dedans (attention Windows : pas `kaggle.json.txt` — affiche les extensions de fichiers).  
- Le fichier doit ressembler à : `{"username":"ton_pseudo_kaggle","key":"xxxxxxxx"}`  
  (encodage UTF-8.)

3) **Vérifier avant de télécharger les datasets**

```bash
python scripts/check_kaggle_auth.py
```

Si ça échoue : mauvais chemin, JSON corrompu, ou clé révoquée → régénère le token sur Kaggle.

**Alternative :** variables d’environnement (même effet que `kaggle.json`) :

```text
KAGGLE_USERNAME=ton_pseudo
KAGGLE_KEY=ta_cle
```

Ou authentification OAuth récente : `kaggle auth login` (voir doc officielle Kaggle CLI).

4) Lance la génération :

```bash
python scripts/build_dataset_kaggle.py --per-class 120
```

Ce script utilise:
- `alessiocorrado99/animals10` (animaux)
- `puneet6060/intel-image-classification` (lieux)

Apres Kaggle, certaines classes (ex: `oiseau`, objets comme `chaise`) peuvent rester vides.
Pour les remplir sans retélécharger Kaggle:

```bash
python -u scripts/fill_empty_classes.py --per-class 60
```

## 3) Lancer l'app

```bash
streamlit run app.py
```

## Notes

- Le moteur de similarite utilise CLIP (`openai/clip-vit-base-patch32`), compatible image et frames extraites de video.
- Plus ton dataset de reference est varie, plus la similarite est pertinente.
- Tu peux ajouter manuellement tes propres images dans les dossiers de classes, puis relancer l'app.
- Le cache d'index est stocke dans `data/index_cache/*.npz` pour accelerer les lancements.
- Dans la sidebar de l'app, active `Forcer reindexation` apres un gros ajout d'images.
