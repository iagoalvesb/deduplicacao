from dask.distributed import Client, LocalCluster
import time

import os
from nemo_curator import AddId
from datasets import load_dataset as load_hf_dataset
from nemo_curator.datasets import DocumentDataset
import warnings

warnings.filterwarnings("ignore")


def main():
    print("\n========== Corpus Download + AddId Pipeline Started ==========")
    start_all = time.time()


    data_dir = "datasets/"
    os.makedirs(data_dir, exist_ok=True)

    # Download wikipedia dataset
    ds_wiki = load_hf_dataset("wikimedia/wikipedia", "20231101.pt")
    filtered_wiki = ds_wiki['train'].filter(lambda x: x['text'] not in [None, '', 'null'], num_proc=8)
    filtered_wiki = filtered_wiki.remove_columns([col for col in filtered_wiki.column_names if col != 'text'])
    filtered_wiki.to_parquet(os.path.join(data_dir, "initial", "wiki_pt_231101.parquet"))
    print("\n*** DOWNLOAD OF WIKI DATASET FINISHED! ***")

    # Download brwac dataset
    ds_brwac = load_hf_dataset("nlpufg/brwac")
    filtered_brwac = ds_brwac['train'].filter(lambda x: x['text'] not in [None, '', 'null'], num_proc=8)
    filtered_brwac = filtered_brwac.remove_columns([col for col in filtered_brwac.column_names if col != 'text'])
    filtered_brwac.to_parquet(os.path.join(data_dir, "initial", "brwac.parquet"))
    print("\n*** DOWNLOAD OF BRWAC DATASET FINISHED! ***")

    # Download oscar dataset
    ds_oscar = load_hf_dataset("oscar-corpus/OSCAR-2301", language="pt", token=True, trust_remote_code=True)
    filtered_oscar = ds_oscar['train'].filter(lambda x: x['text'] not in [None, '', 'null'], num_proc=8)
    filtered_oscar = filtered_oscar.remove_columns([col for col in filtered_oscar.column_names if col != 'text'])
    filtered_oscar.to_parquet(os.path.join(data_dir, "initial", "oscar_pt.parquet"))
    print("\n*** DOWNLOAD OF OSCAR DATASET FINISHED! ***")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    cluster = LocalCluster(n_workers=12, processes=True, memory_limit="64GB")
    client = Client(cluster)
    dataset = DocumentDataset.read_parquet(os.path.join(data_dir, "initial"), columns=["text"], blocksize="256MB")
    added_id_output_path = os.path.join(data_dir, "dataset_with_id")
    add_id_prefix = "portuguese-corpus"
    add_id = AddId(id_field="id", id_prefix=add_id_prefix, start_index=0)

    # Apply the ID addition to the dataset
    id_dataset = add_id(dataset)
    id_dataset.to_parquet(added_id_output_path)

    # ---------------------- Summary ----------------------
    total_time = time.time() - start_all
    print("\n========== Preprocess Pipeline Completed ==========")
    print(f"Total duration: {total_time:.2f} sec ({total_time/60:.2f} min)\n")
    print(f"Final output:     {added_id_output_path}")
    print("===================================================\n")


if __name__ == "__main__":
    # import multiprocessing
    # multiprocessing.freeze_support()  # Optional, good for compatibility
    main()
