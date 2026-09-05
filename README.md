# Urban Flood Intelligence — Phase 0, 1, 2 (Data Collection & Preprocessing)

Ye tera part hai: Rainfall + Elevation data collect karna aur clean karke
ek merged dataset banana jo baaki team Phase 3 (ML model) mein use karegi.

## Kaunsa platform use kare (jo bina "jyada IDE knowledge" ke kaam kare)

**Best option tere liye: Google Colab**
- Browser mein khulta hai, koi install nahi chahiye
- https://colab.research.google.com par jaake "New Notebook"
- Har script ka code copy-paste karke ek cell mein run kar sakta hai
- `!pip install osmnx geopandas` jaise commands cell ke start mein `!` lagakar chalti hain

Agar VS Code use karna hai to bhi chalega, bas Python (3.10+) install hona chahiye.

## Setup (VS Code / local machine ke liye)

```bash
# 1. Project folder mein jao
cd urban-flood-intelligence

# 2. Virtual environment banao (optional but recommended)
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# 3. Packages install karo
pip install -r requirements.txt
```

## Run karne ka order (yehi Phase 0 -> 1 -> 2 ka flow hai)

```bash
# Phase 1 - Data collection (teeno alag alag chalayenge)
python src/data_pipeline/collect_rainfall.py
python src/data_pipeline/collect_osm.py
python src/data_pipeline/collect_dem.py

# Phase 2 - Cleaning & merging
python src/data_pipeline/preprocess.py
```

Har script apna data `data/raw/` folder mein save karega, aur final
merged file `data/processed/merged_dataset.csv` mein aayegi — yehi file
tumhare teammate ko doge jo Phase 3 (model training) kar raha hai.

## config.yaml zaroor check kar

Ye file sab settings ka control panel hai — city ka naam, area (bounding box),
date range, risk thresholds. Agar Mumbai ke alawa koi aur city use karni hai
to bas ye file edit karni hai, code nahi.

## Apne dosto ke sath kaise share karein (GitHub)

1. https://github.com par jaake free account banao (agar nahi hai)
2. Ek naya **Private Repository** banao, naam do: `urban-flood-intelligence`
3. Apne local folder mein ye commands chalao:
   ```bash
   git init
   git add .
   git commit -m "Phase 0-2: data collection and preprocessing"
   git remote add origin https://github.com/<your-username>/urban-flood-intelligence.git
   git push -u origin main
   ```
4. GitHub repo settings mein jaake teammates ko **Collaborator** add kar do
   (Settings -> Collaborators -> unka GitHub username daalo)
5. Wo log `git clone <repo-link>` karke apne system pe poora code le sakte hain

**Simple alternative (agar Git seekhne ka time nahi hai):** Poora
`urban-flood-intelligence` folder ko ZIP karo aur Google Drive/WhatsApp
pe share kar do. Kaam ho jayega lekin GitHub better hai kyunki sabka kaam
ek jagah update hota rehta hai bina overwrite hue.

## Important notes

- `data/raw/` aur `data/processed/` folders bade ho sakte hain — GitHub
  pe push karne se pehle ek `.gitignore` file bana lena jisme `data/` likh do,
  taaki sirf code push ho, heavy data files nahi.
- Rainfall data Open-Meteo se aa raha hai (free, no key). IMD ka data agar
  professor specifically maangte hain to unka open data portal alag se
  check karna padega — ye ek reliable substitute hai jab tak.
- `collect_osm.py` chalane mein 1-2 minute lag sakta hai bade city ke liye,
  ye normal hai, internet pe depend karta hai.
