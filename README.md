# CineNexus

**CineNexus** is a transformer-powered movie explorer that uses Sentence-BERT (SBERT) to discover movies with semantically similar descriptions. It combines natural-language movie search, content-based recommendations, and a knowledge graph to help users explore relationships between movies, genres, directors, and actors.

This web application was made to fulfill the partial requirements for the course **CSC173 - Intelligent Systems** in the program *Bachelor of Science in Computer Science* of **Mindanao State University - Iligan Institute of Technology**.
## Features

- **Semantic Movie Search** – Search for movies using a title, phrase, or natural-language description. CineNexus uses Sentence-BERT to find movies with semantically relevant plot descriptions.
- **Similar Movie Recommendations** – Select a movie to discover other movies with similar descriptions, ranked by cosine similarity.
- **Movie Knowledge Graph** – Explore connections between movies and their genres, directors, and actors through an interactive graph.
- **Dataset Explorer** – Browse the loaded movie catalog and download the current dataset as a CSV file.
- **Embedding Caching** – Precomputed movie embeddings are stored locally and reused on subsequent runs, avoiding unnecessary model inference.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application and user interface |
| Sentence-Transformers | Generating semantic sentence embeddings |
| all-MiniLM-L6-v2 | Pretrained Sentence-BERT model |
| NumPy | Embedding storage and similarity calculations |
| Pandas | Movie dataset loading and processing |
| NetworkX | Knowledge graph construction |
| Plotly | Knowledge graph visualization |

## Installation

Create and activate a virtual environment, then install dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Dataset Setup

CineNexus requires a movie dataset in CSV format. The dataset is not included in this repository, so users must provide their own.

### 1. Prepare the Dataset

Create a `data` directory in the project's root directory and place your movie dataset inside it using the following filename:

```text
CineNexus/
└── data/
    └── movies.csv
```

### 2. Required CSV Fields

The CSV file must contain the following columns:

| Column | Description |
|---|---|
| `id` | Unique identifier for each movie |
| `title` | Movie title |
| `year` | Movie release year |
| `overview` | Movie plot description or overview |
| `genres` | Movie genres |
| `director` | Movie director |
| `actors` | Movie cast members |

**Important:**
- Column names must match the required names exactly.
- The `genres` and `actors` fields must use the pipe character (`|`) to separate multiple entries.
- Missing values are handled by the application.
- Each movie should have a unique ID.

Example:

```csv
id,title,year,overview,genres,director,actors
1,Example Movie,2020,A story about an unexpected adventure,Adventure|Drama,John Smith,Jane Doe|Richard Roe
2,Another Movie,2022,A detective investigates a mysterious disappearance,Crime|Mystery,Alex Johnson,Michael Brown|Sarah White
```

### 3. Embedding Generation

On the first run, CineNexus generates sentence embeddings for the movie descriptions using Sentence-BERT. The generated embeddings are saved locally in the `data` directory and reused on subsequent runs when the dataset and model remain unchanged.

The dataset and generated embedding files are excluded from Git tracking. Users must provide their own dataset when setting up the project.

## Machine Learning Model

CineNexus uses the pretrained model:

**[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**

The model produces 384-dimensional sentence embeddings and is used for semantic comparison between movie descriptions and user queries.

CineNexus currently uses the pretrained model for embedding generation. It does not train or fine-tune the model on the movie dataset.

## Current Scope

CineNexus currently focuses on semantic search, plot-based movie recommendations, and graph-based exploration of movie relationships.

Recommendations are primarily based on plot-description similarity. Although genres, directors, and actors are represented in the knowledge graph, they are not currently incorporated into the recommendation scoring process.

## Members

1. Leonard John T. Corpuz
2. Rene, Jr. C. Estrella
3. Bryle Jared R. Fantilanan
4. Jazrel Xandrei R. Quinlob