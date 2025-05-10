# 📊 Benchmarking Pipeline for Graph-Based Biomedical Case Reports

This repository contains a benchmarking framework for analyzing and visualizing the performance of NLP models (e.g., `BERTScore`) on biomedical case reports represented as graphs. It includes tools for:

- Computing similarity scores (e.g., `BERTScore`, `ROUGE`, `BLEU`)
- Visualizing results (`F1` plots, `t-SNE`, topology distributions)
- Summarizing metrics in tables
- Preparing data for downstream evaluations

---

## 🔧 Setup Instructions

To set up the environment and run the benchmark scripts, use the provided shell script:

```bash
bash setup_and_run.sh
```

This script:
- Creates a Python 3.11 virtual environment
- Installs dependencies from `requirements.txt`
- Runs the benchmark modules as well as the visualization module to generate summary plots

---

## 🗂 Directory Structure

```
src/
└── benchmark/
    │
    ├── batch_run.py                 # Main script to execute benchmarking
    ├── generate_visuals.py          # Generates plots and summary tables
    ├── requirements.txt             # Python dependencies
    ├── setup_and_run.sh             # Script to setup the environment and run visuals
    ├── output/
    │   ├── results/                 # Directory where results are saved
    │   └── plots/                   # Directory where plots are saved
    └── modules/
        ├── __init__.py              # Marks the directory as a Python package
        ├── config.py                # Contains configuration variables and constants for the benchmarking pipeline
        ├── embedding.py             # Functions for computing and managing graph or text embeddings
        ├── evalution.py             # Evaluation metrics (e.g., BERTScore, ROUGE, BLEU) and utility functions
        ├── io_utils.py              # File I/O helper functions (e.g., loading graphs, saving outputs)
        ├── logging_utils.py         # Configures logging for tracking experiment progress and debugging
        ├── reconstruciton.py        # Graph or text reconstruction methods from processed data
        ├── regex_utils.py           # Utility functions for pattern matching and extraction using regular expressions
        ├── run_benchmark.py         # Orchestrates the full benchmarking pipeline
        └── visualization.py         # Plotting functions for visual summaries and metric reporting
```

---

## 📉 Outputs

- `output/plots/bertscore_f1_barplot.png` – Bar chart of BERTScore F1 scores
- `output/plots/trajectory_tsne.png` – 2D t-SNE embedding of graph trajectories
- `output/plots/topology_distributions.png` – Histograms of node and edge counts
- `bertscore_stats.csv` – Summary statistics of BERTScore F1 (optional export)

---

## 📊 Example Metrics

If you're using string similarity tools, results may include:

- `ROUGE-1`, `ROUGE-2`, `ROUGE-L`
- `BLEU` score
- Mean, median, std deviation, and percentiles of F1 scores

---

## 📬 Contact

For questions or contributions, please contact the maintainers of the [Daneshjou Lab](https://daneshjoulab.stanford.edu).
