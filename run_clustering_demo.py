import pandas as pd
from pipeline_clustering import clustering_pipeline

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("merged_raw_data.csv")
    result = clustering_pipeline(df, method='hdbscan')
    print(result[['Video Page URL', 'Cluster']].tail())