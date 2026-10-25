import os
import datetime as dt
from pathlib import Path

import networkx as nx
import pandas as pd
import sys
import os

from script.utils.tgf_param import supported_properties
from script.utils.util import compute_algebraic_connectivity, compute_number_of_connected_components

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)
# from utils.config import supported_properties

from script.utils.util import compute_algebraic_connectivity
from script.utils.tgf_param import supported_properties


def binary_label_generator(current_snapshot_value, prev_snapshot_value):
    return 1 if current_snapshot_value > prev_snapshot_value else 0


def regression_label_generator(current_snapshot_value, prev_snapshot_value):
    return current_snapshot_value


class TGS_Handler:

    def __init__(self, TGS_storage_path, window_size=7, gap=3, label_window_size=7, min_valid_duration=3,
                 task=None, regression=False):
        self.token_root_path = TGS_storage_path
        self.task = task
        if regression:
            self.label_root_path = "../../data/input/raw/labels_regression/{}".format(task)
            print("INFO: Generating regression labels")
        else:
            self.label_root_path = "../../data/input/raw/labels/{}".format(task)
            print("INFO: Generating binary labels")

        self.dummy_label_root_path = "../../data/input/raw/labels/{}/dummy_labels/".format(task)
        self.edgelist_path = "../../data/input/raw/edgelists/"

        self.window_size = window_size  # Day
        self.gap = gap
        self.label_window_size = label_window_size  # Day
        self.min_valid_duration = min_valid_duration  # Day

        if not os.path.exists(self.label_root_path):
            os.makedirs(self.label_root_path)

        if not os.path.exists(self.edgelist_path):
            os.makedirs(self.edgelist_path)

        if not os.path.exists(self.dummy_label_root_path):
            os.makedirs(self.dummy_label_root_path)

        self.label_generator = regression_label_generator if regression else binary_label_generator

    def create_dummy_baseline_labels_weekly(self, dataset, normalization=False):
        print("INFO: Dummy label is not yet calculating. Generating dummy labels weekly...")

        save_file_name = dataset.split(".")[0].replace("_", "")
        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        selectedNetwork = pd.read_csv((self.token_root_path + dataset + ".csv"), sep=',')
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # drop unused data
        selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
        selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
        selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
        selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

        # rename columns
        selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            raise Exception("{} is not a valid network".format(dataset))

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        # value normalization
        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        base_progress = (data_last_date - window_start_date).days / (
                self.window_size + self.gap + self.label_window_size)

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date
            label_start_date = window_start_date - dt.timedelta(days=self.window_size)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            label = 0 if (len(selectedNetworkInLbelingWindow) - len(
                selectedNetworkInGraphDataWindow)) > 0 else 1

            # Storing the new snapshot data after processing

            # ------------------------------------------------
            # Storing each snapshot label data
            label_file_path = self.dummy_label_root_path + save_file_name + "_dummy_fd_ld_labels.csv"

            batch_lables.append(label)
            # Open a file in append mode and write a line to it
            if (indx % batch_size == 0):
                with open(label_file_path, 'a') as file_label:
                    for l in batch_lables:
                        file_label.write(str(l) + "\n")
                    file_label.close()

                batch_lables = []

            window_start_date = window_start_date + dt.timedelta(days=1)

    def creat_baseline_datasets_edge_GS(self, file, normalization=False, create_edge_lists=True, social=False):
        print(
            "INFO: Edge list and label doesn't exist yet. Generating from raw data from TGS...  TASK --> #Edges Growth or Shrink")
        save_file_name = file.split(".")[0].replace("_", "")
        label_file_path = "{}/{}_{}_labels.csv".format(self.label_root_path, save_file_name, self.task)
        edgelist_file_path = "{}/{}_edgelist.txt".format(self.edgelist_path, save_file_name)

        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        if os.path.exists(edgelist_file_path):
            os.remove(edgelist_file_path)

        appended_edgelist_df = pd.DataFrame()
        # print(os.path.exists("../../data/input/tokens/raw/ARC.csv"))
        # pd.read_csv("../../data/input/tokens/raw/ARC.csv")

        if "Reddit" in save_file_name:
            selectedNetwork = pd.read_csv((self.token_root_path + file + ".tsv"), sep='\t')
            selectedNetwork = selectedNetwork[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"]]
            column_mapping = {
                'SOURCE_SUBREDDIT': 'from',
                'TARGET_SUBREDDIT': 'to',
                'TIMESTAMP': 'date',
                'LINK_SENTIMENT': 'value'
            }
            selectedNetwork.rename(columns=column_mapping, inplace=True)
            selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date']).dt.date

        else:
            selectedNetwork = pd.read_csv((self.token_root_path + file + ".csv"), sep=',')
            selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date

        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # drop unused data
        if not social:
            selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
            selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
            selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
            selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

        # rename columns
        selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            print(file + "Is not a valid network")
            # shutil.move(root_path + file, root_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        # value normalization
        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            print("\nRemaining Process  {} ".format(
                (data_last_date - window_start_date).days / self.window_size + self.gap + self.label_window_size))

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(
                days=self.gap) + dt.timedelta(
                days=self.label_window_size)
            label_start_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(days=self.gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # generating the label for this window
            # 1 -> Increading Transactions 0 -> Decreasing Transactions
            # label = 1 if (len(selectedNetworkInLbelingWindow) - len(
            #     selectedNetworkInGraphDataWindow)) > 0 else 0
            label = self.label_generator(len(selectedNetworkInLbelingWindow), len(selectedNetworkInGraphDataWindow))
            print(label)

            if create_edge_lists:
                selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)
                appended_edgelist_df = appended_edgelist_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
                # ------------------------------------------------
                # Storing each snapshot label data
            batch_lables.append(label)
            # Open a file in append mode and write a line to it
            if (indx % batch_size == 0):
                if create_edge_lists:
                    appended_edgelist_df.to_csv(edgelist_file_path, index=False)

                with open(label_file_path, 'a') as file_label:
                    for l in batch_lables:
                        file_label.write(str(l) + "\n")
                batch_lables = []

            window_start_date = window_start_date + dt.timedelta(days=1)

    def creat_baseline_datasets_node_GS(self, file, use_unique_nodes=False, normalization=False,
                                        create_edge_lists=True):
        print(
            "INFO: Edge list and label doesn't exist yet. Generating from raw data from TGS...  TASK --> #Node Growth or Shrink")
        save_file_name = file.split(".")[0].replace("_", "")
        label_file_path = "{}/{}_{}_labels.csv".format(self.label_root_path, save_file_name, self.task)
        edgelist_file_path = "{}/{}_edgelist.txt".format(self.edgelist_path, save_file_name)

        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        if os.path.exists(edgelist_file_path):
            os.remove(edgelist_file_path)

        appended_edgelist_df = pd.DataFrame()
        # print(os.path.exists("../../data/input/tokens/raw/ARC.csv"))
        # pd.read_csv("../../data/input/tokens/raw/ARC.csv")
        selectedNetwork = pd.read_csv((self.token_root_path + file + ".csv"), sep=',')
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # drop unused data
        selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
        selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
        selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
        selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

        # rename columns
        selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            print(file + "Is not a valid network")
            # shutil.move(root_path + file, root_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        # value normalization
        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1
            print("\nRemaining Process  {} ".format(
                (data_last_date - window_start_date).days / (self.window_size + self.gap + self.label_window_size)))

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(
                days=self.gap) + dt.timedelta(
                days=self.label_window_size)
            label_start_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(days=self.gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            # Initialize list to store node counts for the two periods
            node_counts = []

            # Iterate over the two periods: selectedNetworkInGraphDataWindow and selectedNetworkInLabelingWindow
            for period in [selectedNetworkInGraphDataWindow, selectedNetworkInLbelingWindow]:
                if use_unique_nodes:
                    # Get unique nodes by combining 'from' and 'to' columns using a set
                    unique_nodes = set(period['source']).union(set(period['destination']))
                    node_counts.append(len(unique_nodes))
                else:
                    # Count total number of nodes including duplicates
                    total_node_count = period['source'].count() + period['destination'].count()
                    node_counts.append(total_node_count)

            # Set label based on the difference in node counts between periods
            # label = 1 if (node_counts[1] - node_counts[0]) > 0 else 0
            label = self.label_generator(node_counts[1], node_counts[0])
            print(label)

            # Storing the new snapshot data after processing
            if create_edge_lists:
                selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)
                appended_edgelist_df = appended_edgelist_df._append(selectedNetworkInGraphDataWindow, ignore_index=True)
            # ------------------------------------------------
            # Storing each snapshot label data
            batch_lables.append(label)
            # Open a file in append mode and write a line to it
            if (indx % batch_size == 0):
                if create_edge_lists:
                    appended_edgelist_df.to_csv(edgelist_file_path, index=False)

                with open(label_file_path, 'a') as file_label:
                    for l in batch_lables:
                        file_label.write(str(l) + "\n")
                batch_lables = []

            window_start_date = window_start_date + dt.timedelta(days=1)

    def creat_baseline_datasets_density_GS(self, file, normalization=False, create_edge_lists=True):
        print(
            "INFO: Edge list and label doesn't exist yet. Generating from raw data from TGS... TASK --> density Growth or Shrink")
        save_file_name = file.split(".")[0].replace("_", "")
        label_file_path = "{}/{}_{}_labels.csv".format(self.label_root_path, save_file_name, self.task)
        edgelist_file_path = "{}/{}_edgelist.txt".format(self.edgelist_path, save_file_name)

        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        if os.path.exists(edgelist_file_path):
            os.remove(edgelist_file_path)

        appended_edgelist_df = pd.DataFrame()
        # print(os.path.exists("../../data/input/tokens/raw/ARC.csv"))
        # pd.read_csv("../../data/input/tokens/raw/ARC.csv")
        selectedNetwork = pd.read_csv((self.token_root_path + file + ".csv"), sep=',')
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # drop unused data
        selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
        selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
        selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
        selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

        # rename columns
        selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            print(file + "Is not a valid network")
            # shutil.move(root_path + file, root_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        # value normalization
        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(
                days=self.gap) + dt.timedelta(
                days=self.label_window_size)
            label_start_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(days=self.gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            density_count = []
            for period in [selectedNetworkInGraphDataWindow, selectedNetworkInLbelingWindow]:
                transactionGraph = nx.Graph()
                for item in period.to_dict(orient="records"):
                    transactionGraph.add_node(item["source"])
                    transactionGraph.add_node(item["destination"])
                    transactionGraph.add_edge(item["source"], item["destination"], value=item["weight"])
                density_count.append(nx.density(transactionGraph))
            # label = 1 if (density_count[1] - density_count[0]) > 0 else 0
            label = self.label_generator(density_count[1], density_count[0])

            # Storing the new snapshot data after processing
            if create_edge_lists:
                selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)
                appended_edgelist_df = appended_edgelist_df._append(selectedNetworkInGraphDataWindow, ignore_index=True)
            # ------------------------------------------------
            # Storing each snapshot label data
            batch_lables.append(label)
            # Open a file in append mode and write a line to it
            if (indx % batch_size == 0):
                if create_edge_lists:
                    appended_edgelist_df.to_csv(edgelist_file_path, index=False)

                with open(label_file_path, 'a') as file_label:
                    for l in batch_lables:
                        file_label.write(str(l) + "\n")
                batch_lables = []

            window_start_date = window_start_date + dt.timedelta(days=1)

    def creat_baseline_datasets_alg_connectivity_GS(self, file, normalization=False, create_edge_lists=True,
                                                    social=False):
        print(
            "INFO: Edge list and label doesn't exist yet. Generating from raw data from TGS... TASK --> alg connectivity Growth or Shrink")
        save_file_name = file.split(".")[0].replace("_", "")
        label_file_path = "{}/{}_{}_labels.csv".format(self.label_root_path, save_file_name, self.task)
        edgelist_file_path = "{}/{}_edgelist.txt".format(self.edgelist_path, save_file_name)

        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        if os.path.exists(edgelist_file_path):
            os.remove(edgelist_file_path)

        appended_edgelist_df = pd.DataFrame()
        # print(os.path.exists("../../data/input/tokens/raw/ARC.csv"))
        # pd.read_csv("../../data/input/tokens/raw/ARC.csv")
        selectedNetwork = pd.read_csv((self.token_root_path + file + ".csv"), sep=',')
        selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date
        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # drop unused data
        if not social:
            selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
            selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
            selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
            selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

        # rename columns
        selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            print(file + "Is not a valid network")
            # shutil.move(root_path + file, root_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        # value normalization
        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            print("\nRemaining Process  {} ".format(
                (data_last_date - window_start_date).days / self.window_size + self.gap + self.label_window_size))

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(
                days=self.gap) + dt.timedelta(
                days=self.label_window_size)
            label_start_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(days=self.gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            algebraic_connectivities = []

            for period in [selectedNetworkInGraphDataWindow, selectedNetworkInLbelingWindow]:
                transactionGraph = nx.Graph()

                # Add nodes and edges to the graph based on the transaction data
                for item in period.to_dict(orient="records"):
                    transactionGraph.add_node(item["source"])
                    transactionGraph.add_node(item["destination"])
                    transactionGraph.add_edge(item["source"], item["destination"], value=item["weight"])

                # Compute algebraic connectivity and append it to the list
                algebraic_connectivity = compute_algebraic_connectivity(transactionGraph)
                algebraic_connectivities.append(algebraic_connectivity)

            # Determine growth (1) or shrink (0) based on algebraic connectivity
            # label = 1 if (algebraic_connectivities[1] - algebraic_connectivities[0]) > 0 else 0
            label = self.label_generator(algebraic_connectivities[1], algebraic_connectivities[0])
            print("ALGG CNN 1 {} - ALG CNN 2 {}".format(algebraic_connectivities[1], algebraic_connectivities[0]))
            print(label)

            # Storing the new snapshot data after processing
            if create_edge_lists:
                selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)
                appended_edgelist_df = appended_edgelist_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
            # ------------------------------------------------
            # Storing each snapshot label data
            batch_lables.append(label)
            # Open a file in append mode and write a line to it
            if (indx % batch_size == 0):
                if create_edge_lists:
                    appended_edgelist_df.to_csv(edgelist_file_path, index=False)

                with open(label_file_path, 'a') as file_label:
                    for l in batch_lables:
                        file_label.write(str(l) + "\n")
                batch_lables = []

            window_start_date = window_start_date + dt.timedelta(days=1)

    def creat_baseline_datasets_Largest_Connected_Components_GS(self, file, normalization=False, create_edge_lists=True,
                                                                social=False):
        print(
            "INFO: Edge list and label doesn't exist yet. Generating from raw data from TGS... TASK --> Largest Connecnted Components Growth or Shrink")
        save_file_name = file.split(".")[0].replace("_", "")
        label_file_path = "{}/{}_{}_labels.csv".format(self.label_root_path, save_file_name, self.task)
        edgelist_file_path = "{}/{}_edgelist.txt".format(self.edgelist_path, save_file_name)

        # windowSize = 7  # Day
        # gap = 3
        # lableWindowSize = 7  # Day
        # minValidDuration = 20  # Day
        indx = 0
        batch_size = 20
        batch_lables = []

        if os.path.exists(label_file_path):
            os.remove(label_file_path)

        if os.path.exists(edgelist_file_path):
            os.remove(edgelist_file_path)

        appended_edgelist_df = pd.DataFrame()
        # print(os.path.exists("../../data/input/tokens/raw/ARC.csv"))
        # pd.read_csv("../../data/input/tokens/raw/ARC.csv")
        if "Reddit" in save_file_name:
            selectedNetwork = pd.read_csv((self.token_root_path + file + ".tsv"), sep='\t')
            selectedNetwork = selectedNetwork[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"]]
            column_mapping = {
                'SOURCE_SUBREDDIT': 'from',
                'TARGET_SUBREDDIT': 'to',
                'TIMESTAMP': 'date',
                'LINK_SENTIMENT': 'value'
            }
            selectedNetwork.rename(columns=column_mapping, inplace=True)
            selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date']).dt.date

        else:
            selectedNetwork = pd.read_csv((self.token_root_path + file + ".csv"), sep=',')
            selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date

        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # drop unused data
        if not social:
            selectedNetwork = selectedNetwork.drop('tokenAddress', axis=1)
            selectedNetwork = selectedNetwork.drop('timestamp', axis=1)
            selectedNetwork = selectedNetwork.drop('blockNumber', axis=1)
            selectedNetwork = selectedNetwork.drop('fileBlock', axis=1)

        # rename columns
        selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

        # check if the network has more than 20 days of data
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            print(file + "Is not a valid network")
            # shutil.move(root_path + file, root_path + "Invalid/" + file)
            return

        # normalize the edge weights for the graph network {0-9}
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        # value normalization
        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer))))

        # Graph Generation Process and Labeling

        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            print("\nRemaining Process  {} ".format(
                (data_last_date - window_start_date).days / self.window_size + self.gap + self.label_window_size))

            # select window data
            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) & (
                        selectedNetwork['date'] < window_end_date)]

            # select labeling data
            label_end_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(
                days=self.gap) + dt.timedelta(
                days=self.label_window_size)
            label_start_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(days=self.gap)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) & (selectedNetwork['date'] < label_end_date)]

            connected_components = []

            for period in [selectedNetworkInGraphDataWindow, selectedNetworkInLbelingWindow]:
                transactionGraph = nx.Graph()

                # Add nodes and edges to the graph based on the transaction data
                for item in period.to_dict(orient="records"):
                    transactionGraph.add_node(item["source"])
                    transactionGraph.add_node(item["destination"])
                    transactionGraph.add_edge(item["source"], item["destination"], value=item["weight"])

                # Compute algebraic connectivity and append it to the list
                connected_component = compute_number_of_connected_components(transactionGraph)
                connected_components.append(connected_component)

            # Determine growth (1) or shrink (0) based on algebraic connectivity
            # label = 1 if (algebraic_connectivities[1] - algebraic_connectivities[0]) > 0 else 0
            label = self.label_generator(connected_components[1], connected_components[0])
            print(label)

            # Storing the new snapshot data after processing
            if create_edge_lists:
                selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)
                appended_edgelist_df = appended_edgelist_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)
            # ------------------------------------------------
            # Storing each snapshot label data
            batch_lables.append(label)
            # Open a file in append mode and write a line to it
            if (indx % batch_size == 0):
                if create_edge_lists:
                    appended_edgelist_df.to_csv(edgelist_file_path, index=False)

                with open(label_file_path, 'a') as file_label:
                    for l in batch_lables:
                        file_label.write(str(l) + "\n")
                batch_lables = []

            window_start_date = window_start_date + dt.timedelta(days=1)

    def creat_baseline_datasets_regression(self, file, normalization=False, create_edge_lists=True,
                                           social=False):
        """
        Generates edge lists (optional) and four separate regression label files using
        the exact same window, gap, labeling window, batching, and file write logic
        as creat_baseline_datasets_edge_GS.

        Labels per labeling snapshot:
          1) number of new nodes (present in labeling window but not in graph-data window)
          2) number of nodes with degree > 5 (degree = in + out within labeling window)
          3) number of edges
          4) average transaction value
        """
        import os
        import datetime as dt
        import pandas as pd

        print(
            "INFO: Edge list and regression labels do not exist yet. Generating from raw data from TGS...  TASK --> Regression labels")

        save_file_name = file.split(".")[0].replace("_", "")
        # Four separate label files (one value per line, no header), following the same pattern
        label_path = []
        tasks = ["New_Nodes_REG", "Node_Deg_5_REG", "Edge_Count_REG", "AVG_Value_REG"]
        for task in tasks:
            base_label_path = "{}/{}/{}_{}_labels.csv".format(self.label_root_path, task, save_file_name, task)
            label_path.append(base_label_path)

        label1_file_path = label_path[0]
        label2_file_path = label_path[1]
        label3_file_path = label_path[2]
        label4_file_path = label_path[3]

        # label1_file_path = base_label_path + "_new_nodes.csv"
        # label2_file_path = base_label_path + "_deg_gt5.csv"
        # label3_file_path = base_label_path + "_num_edges.csv"
        # label4_file_path = base_label_path + "_avg_value.csv"

        edgelist_file_path = "{}/{}_edgelist.txt".format(self.edgelist_path, save_file_name)

        indx = 0
        batch_size = 20
        batch_labels_1 = []
        batch_labels_2 = []
        batch_labels_3 = []
        batch_labels_4 = []

        # remove previous outputs (same as your reference)
        for p in [label1_file_path, label2_file_path, label3_file_path, label4_file_path, edgelist_file_path]:
            if os.path.exists(p):
                os.remove(p)

        appended_edgelist_df = pd.DataFrame()

        # Load data with the same branching you use
        if "Reddit" in save_file_name:
            selectedNetwork = pd.read_csv((self.token_root_path + file + ".tsv"), sep='\t')
            selectedNetwork = selectedNetwork[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "TIMESTAMP", "LINK_SENTIMENT"]]
            column_mapping = {
                'SOURCE_SUBREDDIT': 'from',
                'TARGET_SUBREDDIT': 'to',
                'TIMESTAMP': 'date',
                'LINK_SENTIMENT': 'value'
            }
            selectedNetwork.rename(columns=column_mapping, inplace=True)
            selectedNetwork['date'] = pd.to_datetime(selectedNetwork['date']).dt.date
        else:
            selectedNetwork = pd.read_csv((self.token_root_path + file + ".csv"), sep=',')
            selectedNetwork['date'] = pd.to_datetime(selectedNetwork['timestamp'], unit='s').dt.date

        selectedNetwork['value'] = selectedNetwork['value'].astype(float)
        selectedNetwork = selectedNetwork.sort_values(by='date')
        window_start_date = selectedNetwork['date'].min()
        data_last_date = selectedNetwork['date'].max()

        # drop unused columns (same as your reference)
        if not social:
            for col in ['tokenAddress', 'timestamp', 'blockNumber', 'fileBlock']:
                if col in selectedNetwork.columns:
                    selectedNetwork = selectedNetwork.drop(col, axis=1)

        # rename to match your pipeline
        selectedNetwork.rename(columns={'from': 'source', 'to': 'destination', 'value': 'weight'}, inplace=True)

        # check minimum valid duration (same condition)
        if ((data_last_date - window_start_date).days < self.min_valid_duration):
            print(file + "Is not a valid network")
            return

        # normalization (same logic)
        max_transfer = float(selectedNetwork['weight'].max())
        min_transfer = float(selectedNetwork['weight'].min())
        if max_transfer == min_transfer:
            max_transfer = min_transfer + 1

        if normalization:
            selectedNetwork['weight'] = selectedNetwork['weight'].apply(
                lambda x: 1 + (9 * ((float(x) - min_transfer) / (max_transfer - min_transfer)))
            )

        # Windowing loop (IDENTICAL boundary logic)
        while (data_last_date - window_start_date).days > (self.window_size + self.gap + self.label_window_size):
            indx += 1

            print("\nRemaining Process  {} ".format(
                (data_last_date - window_start_date).days / self.window_size + self.gap + self.label_window_size))

            # Graph data window
            window_end_date = window_start_date + dt.timedelta(days=self.window_size)
            selectedNetworkInGraphDataWindow = selectedNetwork[
                (selectedNetwork['date'] >= window_start_date) &
                (selectedNetwork['date'] < window_end_date)
                ]

            # Labeling window
            label_start_date = window_start_date + dt.timedelta(days=self.window_size) + dt.timedelta(days=self.gap)
            label_end_date = label_start_date + dt.timedelta(days=self.label_window_size)
            selectedNetworkInLbelingWindow = selectedNetwork[
                (selectedNetwork['date'] >= label_start_date) &
                (selectedNetwork['date'] < label_end_date)
                ]

            # ===== Regression Labels (computed strictly from the labeling window) =====

            # Sets of nodes for "new nodes" w.r.t. graph-data window
            nodes_graph = set(pd.concat([
                selectedNetworkInGraphDataWindow['source'],
                selectedNetworkInGraphDataWindow['destination']
            ], ignore_index=True).unique())

            nodes_label = set(pd.concat([
                selectedNetworkInLbelingWindow['source'],
                selectedNetworkInLbelingWindow['destination']
            ], ignore_index=True).unique())

            # 1) number of new nodes in the labeling snapshot
            label_1 = len(nodes_label - nodes_graph)

            # 2) number of nodes with degree > 5 in labeling snapshot
            # degree = in + out within the labeling window (treat as total degree in directed multigraph)
            if len(selectedNetworkInLbelingWindow) > 0:
                deg_counts = pd.concat([
                    selectedNetworkInLbelingWindow['source'],
                    selectedNetworkInLbelingWindow['destination']
                ], ignore_index=True).value_counts()
                label_2 = int((deg_counts > 5).sum())
            else:
                label_2 = 0

            # 3) number of edges in labeling snapshot
            label_3 = int(len(selectedNetworkInLbelingWindow))

            # 4) average transaction value for labeling snapshot
            if len(selectedNetworkInLbelingWindow) > 0:
                label_4 = float(selectedNetworkInLbelingWindow['weight'].mean())
            else:
                label_4 = 0.0

            # ===== Edge list saving (unchanged) =====
            if create_edge_lists:
                selectedNetworkInGraphDataWindow = selectedNetworkInGraphDataWindow.assign(snapshot=indx)
                appended_edgelist_df = appended_edgelist_df.append(selectedNetworkInGraphDataWindow, ignore_index=True)

            # ===== Batch accumulation (unchanged logic and cadence) =====
            batch_labels_1.append(label_1)
            batch_labels_2.append(label_2)
            batch_labels_3.append(label_3)
            batch_labels_4.append(label_4)

            if (indx % batch_size == 0):
                if create_edge_lists:
                    appended_edgelist_df.to_csv(edgelist_file_path, index=False)

                # Write each label list to its own file, one value per line, identical format
                ex1 = ensure_file(label1_file_path)
                ex2 = ensure_file(label2_file_path)
                ex3 = ensure_file(label3_file_path)
                ex4 = ensure_file(label4_file_path)

                with open(label1_file_path, 'a') as f1, \
                        open(label2_file_path, 'a') as f2, \
                        open(label3_file_path, 'a') as f3, \
                        open(label4_file_path, 'a') as f4:

                    for l in batch_labels_1:
                        f1.write(str(l) + "\n")
                    for l in batch_labels_2:
                        f2.write(str(l) + "\n")
                    for l in batch_labels_3:
                        f3.write(str(l) + "\n")
                    for l in batch_labels_4:
                        f4.write(str(l) + "\n")

                # reset batches (same behavior)
                batch_labels_1, batch_labels_2, batch_labels_3, batch_labels_4 = [], [], [], []

            # slide the window forward by 1 day (unchanged)
            window_start_date = window_start_date + dt.timedelta(days=1)


def ensure_file(path: str) -> bool:
    """
    Ensure parent dir exists. Return True if file already existed, False if newly created.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existed = p.exists()
    if not existed:
        p.touch()
    return existed

if __name__ == "__main__":
    Tgs = TGS_Handler("../../data/input/tokens/raw/", task="Regression", regression=False)
    datasets_list = pd.read_csv('../../data/TGS_available_datasets_complete.csv')['token_name'].to_list()
    for dataset in datasets_list:
        print(dataset)
        Tgs.creat_baseline_datasets_regression(dataset)
