# Veridion Logo Clustering

This project extracts company logos from a list of domains and groups them by visual similarity — no machine learning involved. It's rule-based, explainable, and flexible.

---

## Setup

1. Make sure you place the input `.parquet` file in the `input/` folder.
2. Install dependencies:

```bash
pip install -r requirements.tx

Run the pipeline:
    python main.py

Or run the scripts one by one if needed:

 - logo_extraction.py

 - propagate_logos.py

 - feature_extraction.py

 - clustering.py

How it works

 - Read domain list from a Parquet file

 - Extract logos from websites using common patterns or fallback to favicons

 - Propagate logos to similar domains from the same brand

 - Extract features like color, shape, sharpness, etc.

 - Cluster logos based on a similarity threshold

 - Save results to JSON

Configurable Settings
All options (like which features to use, weights, and file paths) are in config.py.

Output

Logos -> output/logos/

Features -> output/features/logo_features.json

Clusters -> output/clusters/clusters_filtered.json

Logs -> output/logs/

Notes
Broken or missing logos are skipped and logged

Alpha transparency is handled

Threshold and weights can be tuned in config.py