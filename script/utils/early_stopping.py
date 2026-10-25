"""
Early stopping module
Adopted from DyGLib:
    utils/EarlyStopping.py

Date: 2024-11-16
"""

import os
import torch
import torch.nn as nn
import logging


class EarlyStopping(object):

    def __init__(self, patience: int, save_model_path: str,  logger: logging.Logger):
        """
        Early stop strategy.
        :param patience: int, max patience
        :param save_model_folder: str, save model folder
        :param logger: Logger
        """
        self.patience = patience
        self.counter = 0
        self.best_metrics = {}
        self.early_stop = False
        self.logger = logger
        self.save_model_path = save_model_path
        self.best_epoch = 0


    def step(self, metrics: list, current_epoch: int, model: dict):
        """
        execute the early stop strategy for each evaluation process
        :param metrics: list, list of metrics, each element is a tuple (str, float, boolean) -> (metric_name, metric_value, whether higher means better)
        :param model: dict, model checkpoint
        :param current_epoch: int, current epoch
        :return:
        """
        metrics_compare_results = []
        for metric_tuple in metrics:
            metric_name, metric_value, higher_better = metric_tuple[0], metric_tuple[1], metric_tuple[2]

            if higher_better:
                if self.best_metrics.get(metric_name) is None or metric_value > self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
            else:
                if self.best_metrics.get(metric_name) is None or metric_value < self.best_metrics.get(metric_name):
                    metrics_compare_results.append(True)
                else:
                    metrics_compare_results.append(False)
        # all the computed metrics are better than the best metrics
        if torch.all(torch.tensor(metrics_compare_results)):
            for metric_tuple in metrics:
                metric_name, metric_value = metric_tuple[0], metric_tuple[1]
                self.best_metrics[metric_name] = metric_value
            self.save_checkpoint(model)
            self.counter = 0
            self.best_epoch = current_epoch
            self.logger.info(f"Best epoch set to: {current_epoch}")
        # metrics are not better at the epoch
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True

        return self.early_stop, self.best_epoch

    def save_checkpoint(self, model_ckpt: dict):
        """
        saves model at self.save_model_path
        :param model: dict
        :return:
        """
        self.logger.info(f"save model at: {self.save_model_path}")

        torch.save(model_ckpt, self.save_model_path)

    def load_checkpoint(self, encoder: nn.Module, decoder: nn.Module, attention_dos: nn.Module):
        """
        load model at self.save_model_path
        :param model: nn.Module
        :return:
        """
        self.logger.info(f"load model from: {self.save_model_path}")

        pretrained_model_dict = torch.load(self.save_model_path)
        encoder = encoder.load_state_dict(pretrained_model_dict["encoder"])
        decoder = decoder.load_state_dict(pretrained_model_dict["decoder"])
        attention_dos = attention_dos.load_state_dict(pretrained_model_dict["attention_dos"])

        return encoder, decoder, attention_dos
