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
python recommend.py "Shape of You" --num 5
```

or run interactively:

```bash
python recommend.py
```

---

## Example Output

```
Getting recommendations for 'Shape of You'...

Recommended songs:
1. Perfect by Ed Sheeran (Similarity: 92.45%)
2. Thinking Out Loud by Ed Sheeran (Similarity: 90.12%)
3. ...
```

---

## Notes

* The original Spotify dataset (`data/`) is **not included** due to size.
* After training, only `models/` is needed for the CLI.

```