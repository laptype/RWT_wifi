import os
import logging
import torch

from torch.utils.data.dataloader import DataLoader

from pipeline import Trainer_DDP, pre_Trainer, Tester, pre_Trainer2

import init_util

def train(config: dict, Trainer=Trainer_DDP):
    train_dataset, eval_dataset = init_util.init_dataset(config, is_test=False)
    train_dataset.update_config(config)
    strategy = init_util.init_model(config)

    trainer = Trainer(
        strategy=strategy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        batch_size=config["learning"]["train_batch_size"],
        num_epoch=config["learning"]["num_epoch"],
        opt_method=config["learning"]["opt_method"],
        lr_rate=config["learning"]["lr_rate"],
        lr_rate_adjust_epoch=config["learning"]["lr_rate_adjust_epoch"],
        lr_rate_adjust_factor=config["learning"]["lr_rate_adjust_factor"],
        weight_decay=config["learning"]["weight_decay"],
        save_epoch=config["learning"]["save_epoch"],
        eval_epoch=config["learning"]["eval_epoch"],
        patience=config["learning"]["patience"],
        check_point_path=config["path"]["result_path"],
        use_gpu=config["training"]["use_gpu"],
        backbone_setting=config["model"]["backbone_setting"]["backbone_setting"],
        aug=config["dataset"]["dataset_setting"]["augment"]
    )
    trainer.training()

def pre_train(config: dict, Trainer=pre_Trainer):
    import importlib

    train_dataset, eval_dataset = init_util.init_dataset(config, is_test=False)
    train_dataset.update_config(config)
    strategy = init_util.init_model(config)

    # 得到 Head Config 和 head ##############################################################
    head_name = config["model"]["head_name"]
    Model = importlib.import_module("model")
    Head_config = getattr(Model, f"{head_name}_config")
    Head = getattr(Model, head_name)
    head_config = Head_config(config)
    hidden_dim = strategy.head.hidden_dim
    head = Head(hidden_dim, head_config)

    ########################################################################################
    trainer = Trainer(
        strategy=strategy,
        head=head,
        pre_train_dataset=train_dataset,
        pre_test_dataset=eval_dataset,
        pre_train_path=config["training"]["pretrain"]["path"],
        pre_train_ratio=config["training"]["pretrain"]["ratio"],
        batch_size=config["learning"]["train_batch_size"],
        num_epoch=config["learning"]["num_epoch"],
        opt_method=config["learning"]["opt_method"],
        lr_rate=config["learning"]["lr_rate"],
        lr_rate_adjust_epoch=config["learning"]["lr_rate_adjust_epoch"],
        lr_rate_adjust_factor=config["learning"]["lr_rate_adjust_factor"],
        weight_decay=config["learning"]["weight_decay"],
        save_epoch=config["learning"]["save_epoch"],
        eval_epoch=config["learning"]["eval_epoch"],
        patience=config["learning"]["patience"],
        check_point_path=config["path"]["result_path"],
        use_gpu=config["training"]["use_gpu"],
        backbone_setting=config["model"]["backbone_setting"]["backbone_setting"],
        aug=config["dataset"]["dataset_setting"]["augment"]
    )
    trainer.training()
