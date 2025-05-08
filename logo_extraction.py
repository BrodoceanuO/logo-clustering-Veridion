import os
import json
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import print_stats
from io import BytesIO
from PIL import Image
import config
import pandas as pd

results = {}

def extract_logos(domains, output_dir=config.LOGO_OUTPUT_PATH, max_workers=20):
    os.makedirs(output_dir, exist_ok=True)

    def process_domain(domain):
        record = {"status": None, "source": None, "filename": None}
        try:
            url = f"https://{domain}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            candidates = find_logo_candidates(soup, url, domain)

            logo_url = None
            source = None
            image_data = None

            # Try each candidate until one works
            for candidate_url, candidate_source in candidates:
                try:
                    r = requests.get(candidate_url, timeout=10)
                    if not r.ok:
                        continue

                    content_type = r.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        continue

                    # Try to load with PIL
                    Image.open(BytesIO(r.content)).verify()

                    # Valid image found
                    logo_url = candidate_url
                    source = candidate_source
                    image_data = r.content
                    break

                except Exception:
                    continue  # Try next candidate

            if not logo_url:
                raise ValueError("No valid logo image found")

            ext = os.path.splitext(logo_url)[-1].split("?")[0] or ".png"
            filename = domain.replace(".", "_") + ext
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(image_data)

            record.update({
                "status": "success",
                "source": source,
                "filename": filename
            })

        except requests.exceptions.ConnectTimeout:
            record.update({"status": "failed", "reason": "Timeout"})
        except requests.exceptions.ConnectionError:
            record.update({"status": "failed", "reason": "ConnectionError"})
        except Exception as e:
            record.update({"status": "failed", "reason": str(e)})

        results[domain] = record
        return f"{domain}: {record['status']} ({record.get('source') or record.get('reason')})"

    print(f"Running logo extraction with {max_workers} threads...\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_domain, d) for d in domains]
        for f in as_completed(futures):
            print(f.result())

    with open(config.LOGO_LOG_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\nExtraction complete. Results saved to output/logs/logo_results.json")
    print_stats(results, title="Stats AFTER Logo Extraction")

def find_logo_candidates(soup, base_url, domain):
    candidates = []

    for rel in ['icon', 'shortcut icon']:
        link = soup.find("link", rel=lambda x: x and rel in x.lower())
        if link and link.get("href"):
            candidates.append((urljoin(base_url, link["href"]), "link"))

    og = soup.find("meta", property="og:image")
    if og and og.get("content"):
        candidates.append((urljoin(base_url, og["content"]), "og:image"))

    for img in soup.find_all("img"):
        src = img.get("src", "")
        alt = img.get("alt", "")
        if "logo" in src.lower() or "logo" in alt.lower():
            candidates.append((urljoin(base_url, src), "img"))

    # Add favicon fallback last
    candidates.append((f"https://{domain}/favicon.ico", "favicon"))
    return candidates

def main():
    input_path = config.LOGO_PARQUET_PATH

    print(f"Reading input from {input_path}")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Failed to read Parquet: {e}")
        return

    print(f"Columns found: {list(df.columns)}")

    if 'domain' not in df.columns:
        raise ValueError("Expected a column named 'domain' in the input file.")

    domains = (
        df['domain']
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(r"^https?://", "", regex=True)
        .str.replace(r"/.*$", "", regex=True)
        .unique()
        .tolist()
    )

    print(f"Found {len(domains)} unique domains to process.")
    extract_logos(domains)

if __name__ == "__main__":
    main()