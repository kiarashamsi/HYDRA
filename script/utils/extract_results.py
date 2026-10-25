"""
Extract Results from the output files
"""

import pandas as pd
import numpy as np
import os
from script.configs.data_spec import DATA_PATH, dataset_pack_dict


def extract_results_TEST(res_filename, data_pack_train, data_pack_test, use_dos, dataset_names):
    """
    Extract results from the output file 
    Here, the file contains the TEST results
    """
    res_df = pd.read_csv(res_filename)
    # desired_columns = ['seed', 'dataset', 
    #                    'test_auc', 'test_ap', 'test_time', 'full_forward_pass_time']

    selected_results = res_df.loc[((res_df['data_pack_train'] == data_pack_train) &
                                   (res_df['data_pack_test'] == data_pack_test) &
                                   (res_df['DOS'] == use_dos) &
                                   (res_df['dataset'].isin(dataset_names))
                                   )]
    return selected_results



def main():
    """
    Extract results from the output file
    """
    res_filename = os.path.join(DATA_PATH, "output/results/tgf_results_TEST.csv")
    data_pack_train = "pack_8"
    data_pack_test = "pack_test"
    use_dos = True
    dataset_names_list = dataset_pack_dict["pack_test"][0]  # because pack_test has only one list of dataset names

    extracted_results_filename = f"../../data/extracted_results/extracted_results_TEST.csv"

    for dataset_names in dataset_names_list:
        print(f"Extracting results for {dataset_names}...")
        dataset_names = [dataset_names]
        selected_results = extract_results_TEST(res_filename, data_pack_train, data_pack_test, use_dos, dataset_names)

        if os.path.isfile(extracted_results_filename):
            selected_df_existing = pd.read_csv(extracted_results_filename)
            df_combined = pd.concat([selected_df_existing, selected_results], ignore_index=True)
            df_combined.to_csv(extracted_results_filename, index=False)
        else:
            selected_results.to_csv(extracted_results_filename, index=False)
        
        print("="*100)


if __name__ == "__main__":
    main()