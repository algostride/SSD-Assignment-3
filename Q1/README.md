# VidyaVichar — Code Similarity Analysis

This repository provides a full pipeline to analyze similarity between MERN stack projects using a combination of textual, structural, and semantic similarity metrics. It includes preprocessing, feature extraction, visualization, and report generation in a Jupyter notebook format.

---

## Features

* Automatic preprocessing of JS, JSX, JSON, and CSS files
* Comment stripping, whitespace normalization, minified-file detection
* Extraction of React components, Express routes, and Mongoose models
* Textual similarity: TF-IDF + cosine
* Structural similarity: function/classes/imports/exports patterns
* Semantic similarity: hashed embeddings
* Combined similarity matrix
* Heatmaps, network graphs, similarity distributions
* Automatic CSV summary + result exports

---

## Requirements

Before running the code, ensure you have the following installed:

### **Python Version**

* Python **3.8+** recommended

### **Required Python Packages**

Install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn networkx
```

Packages used:

* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `scikit-learn`
* `networkx`

> The notebook uses standard libraries only; no GPU or ML frameworks required.

---

## Project Structure

Expected project layout:

```
projects/
├── team1_vidyavichar/
├── team2_vidyavichar/
├── team3_vidyavichar/
└── ...
```

Each folder should contain a MERN project with:

* Backend (`.js`) files
* Frontend files (`.js`, `.jsx`, `.css`)
* `.json` config files

The system automatically ignores:

* `node_modules/`
* `dist/`, `build/`
* `.git/`

---

## Running the Notebook

Simply open the notebook in Jupyter and run:

```python
results = main('./projects')
```

Ensure that:

* Your `projects/` directory exists
* Each subfolder is a valid project

All results will be saved in:

```
./results/
```

Including:

* Preprocessing summary CSV
* Textual/structural/semantic/combined similarity matrices
* Heatmaps and network graphs

---

## Outputs

The notebook produces:

* **Preprocessing Summary** (`preprocessing_summary.csv`)
* **Similarity Matrices** (4 CSV files)
* **Visualizations** (5 PNG images)
* **Console Insights** (most/least similar projects)

---

## Assumptions

The system assumes:

* Codebases follow typical MERN patterns (React components, Express routes, Mongoose models)
* Projects have readable (non-minified) source code
* Comments can be safely removed for similarity purposes
* Structural similarity is approximated via regex (suitable for student projects or mid-sized apps)
* Semantic similarity uses simple hashed embeddings — upgradeable to CodeBERT/OpenAI if needed

---

## Performance Notes

For best performance:

* Keep project counts reasonable (5–50 projects recommended)
* Very large repositories may increase runtime

---

