import json
import os
import shutil
from collections import defaultdict
import tldextract
from utils import print_stats
import config

def propagate_logos():
    with open(config.LOGO_LOG_RESULTS_PATH) as f:
        results = json.load(f)

    STATS_BEFORE_PROPAGATION_PATH = os.path.join(config.LOGS_DIR, "stats_before_propagation.json")
    print_stats(results, title="Stats Before Propagation", output_file = STATS_BEFORE_PROPAGATION_PATH)

    grouped = defaultdict(list)
    for domain in results:
        ext = tldextract.extract(domain)
        grouped[ext.domain].append(domain)

    propagations = []

    for brand, domains in grouped.items():
        # Find a domain in the group that succeeded
        source_domain = None
        source_filename = None

        for d in domains:
            entry = results[d]
            if entry["status"] == "success" and entry["source"] != "propagated":
                source_domain = d
                source_filename = entry["filename"]
                break

        if not source_filename:
            continue  # No usable logo in this group

        # Propagate to failed domains in the group
        for d in domains:
            if results[d]["status"] != "success":
                new_filename = d.replace(".", "_") + os.path.splitext(source_filename)[1]
                src_path = os.path.join(config.LOGO_DIR, source_filename)
                dst_path = os.path.join(config.LOGO_DIR, new_filename)

                if not os.path.exists(src_path):
                    print(f"Missing source logo for {source_domain}: {src_path}")
                    continue

                shutil.copy(src_path, dst_path)

                results[d] = {
                    "status": "success",
                    "source": "propagated",
                    "origin": source_domain,
                    "filename": new_filename
                }

                propagations.append({
                    "from": source_domain,
                    "to": d,
                    "filename": new_filename
                })

    # Save updated results
    with open(config.LOGO_LOG_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    PROPAGATION_LOG_PATH = os.path.join(config.LOGS_DIR, "propagation_log.json")

    # Save propagation log
    with open(PROPAGATION_LOG_PATH, "w") as f:
        json.dump(propagations, f, indent=2)

    print(f"\nPropagated {len(propagations)} logos")
    print(f"Details saved to {PROPAGATION_LOG_PATH}")

    STATS_AFTER_PROPAGATION_PATH = os.path.join(config.LOGS_DIR, "stats_after_propagation.json")
    print_stats(results, title="Stats After Propagation", output_file=STATS_AFTER_PROPAGATION_PATH)

def main():
    propagate_logos()
    
if __name__ == "__main__":
    main()