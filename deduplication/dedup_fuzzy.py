import time
import os
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator import FuzzyDuplicatesConfig
from nemo_curator import FuzzyDuplicates

import warnings

warnings.filterwarnings("ignore")


def main():
    print("\n========== Fuzzy Deduplication Started ==========")
    start = time.time()

    def pre_imports() -> None:
        import cudf  # noqa: F401


    client = get_client(cluster_type="gpu", set_torch_to_use_rmm=False)
    client.run(pre_imports)
    print(f"🚀 Dask Dashboard link: {client.dashboard_link}")

    data_dir = "datasets/"
    dataset_id_field = "id"
    dataset_text_field = "text"

    fuzzy_dedup_input_dataset_dir = os.path.join(data_dir, "dataset_without_exact_duplicates")
    fuzzy_dedup_input_dataset_dir = os.path.join(data_dir, "dataset_with_id")

    fuzzy_dedup_base_output_path = os.path.join(data_dir, "fuzzy_dedup")
    fuzzy_dedup_log_dir = os.path.join(fuzzy_dedup_base_output_path, "log")
    fuzzy_dedup_cache_dir = os.path.join(fuzzy_dedup_base_output_path, "data")
    fuzzy_dedup_output_dir = os.path.join(data_dir, "dataset_without_fuzzy_duplicates")

    # Creating directories if they don't exist
    os.makedirs(fuzzy_dedup_base_output_path, exist_ok=True)
    os.makedirs(fuzzy_dedup_log_dir, exist_ok=True)
    os.makedirs(fuzzy_dedup_cache_dir, exist_ok=True)
    os.makedirs(fuzzy_dedup_output_dir, exist_ok=True)

    dataset_id_field = "id"
    dataset_text_field = "text"

    fuzzy_dedup_config = FuzzyDuplicatesConfig(
        cache_dir=fuzzy_dedup_cache_dir, # must be cleared between runs
        id_field=dataset_id_field,
        text_field=dataset_text_field,
        perform_removal=False, # dictates if deduplicated dataset or IDs of duplicates are returned
        seed=42,
        char_ngrams=24,
        num_buckets=20,
        hashes_per_bucket=13,
        use_64_bit_hash=False,
        buckets_per_shuffle=2,
        # false_positive_check=True,
        false_positive_check=False,
        jaccard_threshold=0.8,
    )

    input_dataset = DocumentDataset.read_parquet(fuzzy_dedup_input_dataset_dir, backend="cudf")

    start = time.time()
    fuzzy_dup = FuzzyDuplicates(logger=fuzzy_dedup_log_dir, config=fuzzy_dedup_config)
    duplicates = fuzzy_dup(dataset=input_dataset)
    duplicates.to_parquet(fuzzy_dedup_cache_dir)
        
    fuzzy_docs_to_remove = duplicates.df.map_partitions(
        lambda x: x[x.group.duplicated(keep="first")]
    )
    num_unique_samples = len(duplicates) - len(fuzzy_docs_to_remove)
    end_fuzzy = time.time()

    result = input_dataset.df[
        ~input_dataset.df[dataset_id_field].isin(fuzzy_docs_to_remove[dataset_id_field].compute())
        ]

    result = DocumentDataset(result)

    # Summary
    end_calculation = time.time()
    time_to_fuzzy_dedup = end_calculation - start
    removed = len(input_dataset) - len(result)
    percent_removed = (removed / len(input_dataset)) * 100 if len(input_dataset) > 0 else 0

    print("\n========== Fuzzy Deduplication Completed ==========")
    print(f"Duration:   {time_to_fuzzy_dedup:,.2f} seconds ({time_to_fuzzy_dedup/60:.2f} minutes)\n")
    print(f"Input:      {len(input_dataset):,} samples")
    print(f"Removed:    {removed:,} samples ({percent_removed:.2f}%)\n")
    print(f"SAVING OUTPUT TO PATH: '{fuzzy_dedup_output_dir}'")
    

    result.to_parquet(fuzzy_dedup_output_dir)
    end_save = time.time()
    time_to_save = end_save - end_calculation
    time_run = end_save - start
    print(f"Time to save:   {time_to_save:,.2f} seconds ({time_to_save/60:.2f} minutes)\n")
    print(f"Total time:   {time_run:,.2f} seconds ({time_run/60:.2f} minutes)\n")
    print("===================================================\n")

if __name__ == "__main__":
    main()