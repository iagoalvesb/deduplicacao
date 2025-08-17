import time
import os
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator import FuzzyDuplicatesConfig
from nemo_curator import FuzzyDuplicates
from dask.distributed import Client, LocalCluster
from nemo_curator.utils.config_utils import build_filter_pipeline


import warnings

warnings.filterwarnings("ignore")


def main():
    cluster = LocalCluster(n_workers=10, processes=True, memory_limit='16GB')
    client = Client(cluster)
    
    data_dir = "datasets_hf/"

    fuzzy_dedup_output_dir = os.path.join(data_dir, "dataset_without_fuzzy_duplicates")

    heuristics_input_data_dir = fuzzy_dedup_output_dir

    kept_document_dir =  os.path.join(data_dir, 'heuristic_filtering', 'data','hq.parquet')
    filter_config_file = '/workspace/heuristics.yaml'

    # Creating directories if they don't exist
    os.makedirs(kept_document_dir, exist_ok=True)

    
    #Load filters from config
    filter_pipeline = build_filter_pipeline(filter_config_file)

    print(f"\n***\nReading input dataset to remove samples based on heuristics.\n\n")

    dataset = DocumentDataset.read_parquet(heuristics_input_data_dir, backend='pandas')
    
    print(f"\n***\nFILTERING.\n\n")
    result_data = filter_pipeline(dataset)

    
    print(f"\n***\nFILTERING COMPLETED. SAVING DATASET.\n\n")
    print(f"\n***\nLength of original dataset: {len(dataset):_}")
    print(f"After heuristics: {len(result_data):_}.\n\n")
    # result_data.to_parquet(kept_document_dir)


if __name__ == "__main__":
    main()
