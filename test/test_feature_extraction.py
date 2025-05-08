import sys
from pathlib import Path
import json
from scipy.spatial import distance
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))
from feature_extraction import extract_features_for_logo, extract_features_from_directory, save_features_to_json

# choose logo image
# test_logo = Path("output/logos/aamco-bellevue_com.png")
# test_logo = Path("output/logos/adra_ca.png")
# test_logo = Path("output\logos\zkteco_co_za.png") 
# test_logo = Path("test/images/test_star.png")
# test_logo = Path("test/images/test_circle.png")
# test_logo = Path("test/images/test_circle_3.png")
# test_logo = Path("test/images/test_square.png")
# test_logo = Path("test/images/test_red.png")
# test_logo = Path("output/logos/forpsi_pl.png")
# test_logo = Path("output/logos/flesan_com_pe.png")
# test_logo = Path("output/logos/amnesty_org_ng.png")
# test_logo = Path("output/logos/mazda-ma_com.ico")
# test_logo = Path("output/logos/mazda-lodz-matsuoka_pl.png")
# test_logo = Path("test/images/habitatns_ca.png")
# test_logo = Path("test/images/wwf_es.ico")


INPUT_DIR = "test/logos_w_null_features"
OUTPUT_JSON = "test/results/features.json"

with open("output/logos/ccusa_ca.png", "rb") as f:
    header = f.read(8)
    print(header)

#results = extract_features_from_directory(INPUT_DIR)
#save_features_to_json(results, OUTPUT_JSON)

