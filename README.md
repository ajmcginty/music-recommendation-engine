# Music Recommender CLI

A **command-line music recommendation system** built with Python and scikit-learn.  
Given a song, the CLI outputs **genre-specific recommendations** based on song features.

---

## Features
- KNN-based recommendations using **danceability, energy, valence, acousticness, instrumentalness, speechiness, tempo**
- Lightweight CLI — no huge CSV needed after training
- Supports custom number of recommendations

---

## Project Structure
```

music-recommender/
├── data/               # Original CSVs (not committed)
├── models/             # Trained model, metadata, scaled vectors
├── train.py            # Train model and save artifacts
├── recommend.py        # CLI for song recommendations
├── requirements.txt
└── README.md

````

---

## Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
````

2. **Train the model** (if models/ is not provided)

```bash
python train.py
```

3. **Get recommendations**

```bash
python recommend.py

Enter a song: {Enter name of song here}
```


---

## Example Output

```
Recommended songs:
1044. The Lonely Mountains by Kim Janssen (Similarity: 99.76%)
9135. You Turned To Me by Elvis Costello (Similarity: 99.68%)
5044. Northern Italy by Margaux Beylier (Similarity: 99.66%)
9729. I've Never Been In Love Before by Linda Ronstadt (Similarity: 99.62%)
2311. Across the Borderline by Willie Nelson (Similarity: 99.59%)
```

---

## Notes

* The original Spotify dataset (`data/`) is **not included** due to size.
* After training, only `models/` is needed for the CLI.

```