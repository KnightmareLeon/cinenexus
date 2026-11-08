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

CineNexus uses the [Letterboxd Movie Dataset](https://www.kaggle.com/datasets/gsimonx37/letterboxd/data) from Kaggle. The dataset is not included in this repository, so users must download and provide the required CSV files themselves.

CineNexus uses SQLite for persistent movie data storage. Before running the application for the first time, users must prepare the CSV files, create the database, and import the dataset.

Only the following six CSV files from the original dataset are required:

- `movies.csv`
- `actors.csv`
- `crew.csv`
- `genres.csv`
- `studios.csv`
- `themes.csv`

### 1. Download and Prepare the Dataset

Download the dataset from [Kaggle – Letterboxd Movie Dataset](https://www.kaggle.com/datasets/gsimonx37/letterboxd/data).

Create a `data` directory in the project's root directory and place the six CSV files inside it.

The directory structure should look like this:

```text
CineNexus/
├── data/
│   ├── movies.csv
│   ├── actors.csv
│   ├── crew.csv
│   ├── genres.csv
│   ├── studios.csv
│   └── themes.csv
├── app.py
└── ...
```

### 2. Required CSV Fields

Each CSV file must contain the following columns:

**`movies.csv`**

| Column | Description |
|---|---|
| `id` | Unique identifier for each movie |
| `name` | Movie title |
| `date` | Movie release date |
| `tagline` | Movie tagline |
| `description` | Movie plot description or overview |
| `minute` | Movie duration in minutes |
| `rating` | Movie rating |

**`actors.csv`**

| Column | Description |
|---|---|
| `id` | Movie identifier |
| `name` | Actor's name |
| `role` | Character or role associated with the actor |

**`crew.csv`**

| Column | Description |
|---|---|
| `id` | Movie identifier |
| `role` | Crew member's production role |
| `name` | Crew member's name |

**`genres.csv`**

| Column | Description |
|---|---|
| `id` | Movie identifier |
| `genre` | Movie genre |

**`studios.csv`**

| Column | Description |
|---|---|
| `id` | Movie identifier |
| `studio` | Production studio |

**`themes.csv`**

| Column | Description |
|---|---|
| `id` | Movie identifier |
| `theme` | Movie theme |

**Important:**

- Column names must match the required names exactly.
- The `id` column associates actors, crew members, genres, studios, and themes with their corresponding movies.
- The `id` values must be consistent across all six CSV files.
- The application filters `crew.csv` to obtain directors using the `director` role.
- Multiple actors, genres, studios, and themes associated with a movie are grouped together during dataset processing.
- Missing values are handled by the application.

### 3. Create the SQLite Database

After preparing the CSV files, create the SQLite database.

From the project's root directory, run:

```powershell
python -m scripts.create_database
```

This initializes the SQLite database and creates the required tables for storing movie information, relationships, and embedding metadata.

The generated database is stored at:

```text
data/cinenexus.db
```

### 4. Import the Dataset

Once the database has been created, import the CSV files:

```powershell
python -m scripts.import_dataset
```

The importer processes the six CSV files and transfers the resulting records into the SQLite database.

During this process, CineNexus:

1. Reads the six CSV files.
2. Associates related records using their movie IDs.
3. Extracts directors from the crew data.
4. Groups related actors, genres, studios, and themes by movie.
5. Constructs searchable text representations using each movie's title, tagline, and description.
6. Generates a dataset fingerprint and stores the corresponding embedding metadata in the database.

The resulting SQLite database contains the processed movie records and their associated relationships.

### 5. Embedding Generation and Caching

CineNexus uses the pretrained Sentence-BERT model `sentence-transformers/all-MiniLM-L6-v2` to generate 384-dimensional embeddings for movie search and recommendations.

After importing the dataset, run the application:

```powershell
streamlit run app.py
```

During application initialization, CineNexus checks whether valid precomputed embeddings are available.

If embeddings do not exist or do not match the current dataset fingerprint, the application generates new embeddings from the searchable movie text and saves them locally.

The generated embeddings are stored in:

```text
data/embeddings.npz
```

On subsequent runs, the application reuses the stored embeddings when their dataset fingerprint and model configuration remain valid. This avoids unnecessary embedding generation and model inference.

CineNexus also uses Streamlit caching to avoid repeating frequently requested database queries, recommendation calculations, and graph construction.

### 6. Generated Data Files

After completing the database setup and running the application for the first time, the `data` directory should contain:

```text
data/
├── movies.csv
├── actors.csv
├── crew.csv
├── genres.csv
├── studios.csv
├── themes.csv
├── cinenexus.db
└── embeddings.npz
```

The original CSV files are retained as the source dataset. The SQLite database stores the processed movie records, while the NumPy file stores the precomputed embeddings.

**Important:**

- The CSV files, SQLite database, and generated embeddings are excluded from Git tracking.
- Users must download the required CSV files and perform the database setup themselves.
- The database and embeddings should be regenerated when importing a different dataset.
- The application retrieves movie information from SQLite during normal operation instead of processing the original CSV files on every startup.

## Machine Learning Model

CineNexus uses the pretrained model:

**[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)**

The model produces 384-dimensional sentence embeddings and is used for semantic comparison between movie descriptions and user queries.

CineNexus currently uses the pretrained model for embedding generation. It does not train or fine-tune the model on the movie dataset.

## Current Scope

CineNexus currently focuses on semantic search, plot-based movie recommendations, and graph-based exploration of movie relationships.

Recommendations are primarily based on plot-description similarity. Although genres, directors, and actors are represented in the knowledge graph, they are not currently incorporated into the recommendation scoring process.

## References

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 4171–4186. https://doi.org/10.18653/v1/N19-1423

Patil, S., Bijapur, R. R., Bidargaddi, A. P., Kothari, S. D., & Dabade, S. H. (2024). Enhancing movie recommendation systems with BERT: A deep learning approach. *2024 5th International Conference for Emerging Technology (INCET)*, 1–8. https://doi.org/10.1109/INCET61516.2024.10593284

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, 3982–3992. https://doi.org/10.18653/v1/D19-1410

Simon, G. (n.d.). *Letterboxd movie dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/gsimonx37/letterboxd/data

## Members

1. Leonard John T. Corpuz
2. Rene, Jr. C. Estrella
3. Bryle Jared R. Fantilanan
4. Jazrel Xandrei R. Quinlob