# DynamicData

DynamicData is a modular framework for transforming clinical case reports into structured dynamic graphs and generating synthetic case narratives using large language models.

---

## 🔧 Installation

### Clone the Repository

```
git clone https://github.com/DaneshjouLab/DynamicData.git <project_directory>
cd <project_directory>
```

### Create and Activate a Virtual Environment

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file to store API keys and model names.

**`.config/.env`:**
```
DSPY_MODEL="gemini/gemini-2.0-flash"
GEMINI_APIKEY="your-gemini-api-key"
GPTKEY="your-openai-api-key"
ncbi_api_key="your-ncbi-key"
Entrez_email="your-email@example.com"
```


**Recommended `.gitignore`:**
```
.config/.env
```

---

## 📄 Graph Generation Pipeline

Converts full PMC HTML case reports into dynamic DAGs with node and edge information using DSPy.

### Input

- Place your PMC HTML files in:
  ```
  ./pmc_htmls/
  ```

### Run the pipeline

```
python src/pipeline/generate_graphs.py
```

### Output

- JSON graphs stored in:
  ```
  webapp/static/graphs/
  ```
- Metadata stored in:
  ```
  webapp/static/graphs/graph_metadata.csv
  ```

---

## 🧬 Synthetic Case Generation

Generates synthetic case narratives from dynamic graphs using LLMs.

### Input Requirements

- Graphs must be saved in `webapp/static/graphs/`
- Metadata must be indexed in `graph_metadata.csv`

### Run the generation script

```
python src/pipeline/generate_synthetic_cases.py
```

### Output

- Synthetic narratives and metadata stored in:
  ```
  synthetic_outputs/
  ```

Each output consists of:
- `.txt` files for each generated case (control or sample)
- `index.jsonl` recording metadata (graph ID, model used, etc.)

---

## ⚙ Project Structure

```
DynamicData/
├── .config/                      # API keys and model config
├── pmc_htmls/                    # Input HTML articles
├── webapp/static/graphs/        # Output graph JSONs and metadata
├── synthetic_outputs/           # Output synthetic case reports
├── src/
│   ├── pipeline/                # Main execution scripts
│   ├── agent/                   # DSPy programs
│   ├── data/                    # Preprocessing logic
│   └── benchmark/modules/       # Graph reconstruction + utils
├── requirements.txt
└── README.md
```

---

## 🧪 Citation

```
@software{dynamicdata2025,
  author = {Daneshjou Lab},
  title = {DynamicData: A Framework for Patient Timeline Graph Construction and Simulation},
  year = {2025},
  url = {https://github.com/DaneshjouLab/DynamicData}
}
```

---

## 📜 License

This project is licensed under the Apache License 2.0. See the `LICENSE` file for details.
