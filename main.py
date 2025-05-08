from logo_extraction import main as run_logo_extraction
from feature_extraction import main as run_extract_features
from clustering import main as run_clustering

def main():
    #Extract logos
    run_logo_extraction()

    #Extract features from logos
    run_extract_features()

    #Run clustering
    run_clustering()

if __name__ == "__main__":
    main()