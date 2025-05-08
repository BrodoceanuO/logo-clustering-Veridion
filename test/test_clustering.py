import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from clustering import load_features_from_json, cluster_logos, save_cluster_results

logo_entries = load_features_from_json()
print(logo_entries)
clusters = cluster_logos(logo_entries)
save_cluster_results(clusters, logo_entries)

