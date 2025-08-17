from dask_cuda import LocalCUDACluster
from distributed import Client
import os
from nemo_curator import AddId
from nemo_curator.datasets import DocumentDataset
from nemo_curator.utils.distributed_utils import get_client
from nemo_curator.modules import ExactDuplicates

import warnings

warnings.filterwarnings("ignore")


def main():
    # ---------------------------------------------------------------------------
    # EXACT DEDUPLICATION
    # ---------------------------------------------------------------------------

    def pre_imports() -> None:
        import cudf  # noqa: F401


    client = get_client(cluster_type="gpu", set_torch_to_use_rmm=False)
    # client = Client(
    #     LocalCUDACluster(
    #     CUDA_VISIBLE_DEVICES="0",  # Use two workers (on devices 0 and 1)
    #     rmm_pool_size=0.9,  # Use 90% of GPU memory as a pool for faster allocations
    #     enable_cudf_spill=True,  # Improve device memory stability
    #     local_directory="dask/fast/scratch/",  # Use fast local storage for spilling
    #     )
    # )
    client.run(pre_imports)

    data_dir = "datasets_hf/"
    added_id_output_path = os.path.join(data_dir, "dataset_with_id")

    exact_dedup_input_dataset_dir = added_id_output_path

    exact_dedup_base_output_path = os.path.join(data_dir, "exact_dedup")
    exact_dedup_log_dir = os.path.join(exact_dedup_base_output_path, "log")
    exact_dedup_cache_dir = os.path.join(exact_dedup_base_output_path, "data")
    exact_dedup_output_dir = os.path.join(data_dir, "dataset_without_exact_duplicates")

    # Creating directories if they don't exist
    os.makedirs(exact_dedup_base_output_path, exist_ok=True)
    os.makedirs(exact_dedup_log_dir, exist_ok=True)
    os.makedirs(exact_dedup_cache_dir, exist_ok=True)
    os.makedirs(exact_dedup_output_dir, exist_ok=True)

    exact_dedup_dataset_id_field = "id"
    exact_dedup_dataset_text_field = "text"

    # Initialize and run exact deduplication
    exact_dup = ExactDuplicates(
        logger=exact_dedup_log_dir,
        id_field=exact_dedup_dataset_id_field,
        text_field=exact_dedup_dataset_text_field,
        hash_method="md5",
        cache_dir=exact_dedup_cache_dir,
        perform_removal=False,
    )

    input_dataset = DocumentDataset.read_parquet(exact_dedup_input_dataset_dir, backend="cudf")
    duplicates = exact_dup(dataset=input_dataset)

    print(f"\n***\nNumber of exact duplicate samples: {len(duplicates):_}\n***\n")
    
    # Extract list of duplicate document IDs
    exact_docs_to_remove = duplicates.df.map_partitions(
        lambda x: x[x._hashes.duplicated(keep="first")],  # noqa: SLF001
    )

    # Remove duplicated documents from the input dataset
    result = input_dataset.df[
        ~input_dataset.df[exact_dedup_dataset_id_field].isin(exact_docs_to_remove[exact_dedup_dataset_id_field].compute())
    ]
    result.to_parquet(exact_dedup_output_dir, write_to_filename=True)
    print(f"\n***\nNumber of samples after exact deduplication: {len(result):_}\n***\n")

if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.freeze_support()  # Optional, good for compatibility
    main()
