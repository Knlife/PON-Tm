import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
import os
import logging

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split


os.chdir("D:\WorkPath\PycharmProjects\MutTm-pred")
from DeepLearning.Util import (PonDataset4FineTuning, PonMetrics, EarlyStopping, logger_init,
                               embedding_dataset_creator_full_length,
                               embedding_model_getter)
from DeepLearning import Config


def training(dataset: PonDataset4FineTuning,
             model,
             train_batch: int,
             valid_batch: int,
             loss_fn,
             optimizer,
             device,
             logger,
             early_stop_patience: int,
             model_save_path: str,
             summary_write_dir: str,
             watch_metrics: list = None
             ):
    # region 加载并分割训练/验证集
    train_index, valid_index = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_fold, val_fold = dataset.subDataset(train_index), dataset.subDataset([valid_index])
    trainLoader = DataLoader(dataset=train_fold, batch_size=train_batch, shuffle=True, drop_last=True)
    validateLoader = DataLoader(dataset=val_fold, batch_size=valid_batch, shuffle=True, drop_last=True)
    # endregion

    # region 模型训练
    # 设定模型训练参数(回调点、早停点参数等)
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    metrics = PonMetrics(existingMetrics=watch_metrics, device=device).cuda()
    early_stopping = EarlyStopping(model_save_path=model_save_path, patience=early_stop_patience)
    writer = SummaryWriter(summary_write_dir)

    # writer.add_graph(model, input_to_model=[torch.randn(32, 52), torch.randn(32, 52), torch.randn(32, 52), torch.randn(32, 52), torch.randn(32, 52)])

    all_logger.debug("===损失函数如下===")
    all_logger.debug(loss_fn)
    all_logger.debug("===优化器如下===")
    all_logger.debug(optimizer)
    all_logger.debug("===训练模型如下===")
    all_logger.debug(model)

    all_logger.info("===开始训练===")
    # 训练/验证
    with tqdm(range(2000)) as progress_bar:
        for epoch in range(2000):
            metrics.clear()

            # region Training
            train_loss = 0.0
            train_steps = 0
            model.train()
            for wild_embedding, mutant_embedding, bio, label in trainLoader:
                # 载入GPU
                wild_embedding = wild_embedding.to(device)
                mutant_embedding = mutant_embedding.to(device)
                label = label.to(device)
                output = model(wild_embedding, mutant_embedding)  # 预测模型
                running_loss = loss_fn(output, label)  # 损失函数并运行计算梯度

                optimizer.zero_grad()  # 优化器梯度清零
                running_loss.requires_grad_(True)  # 允许梯度
                running_loss.backward()  # 反向传播
                optimizer.step()  # 重置参数
                train_loss += running_loss.item()
                train_steps += 1

            writer.add_scalar("train_loss", train_loss / train_steps, epoch)
            # endregion

            # region Validation
            valid_loss = 0.0
            valid_steps = 0
            model.eval()
            with torch.no_grad():
                for wild_embedding, mutant_embedding, bio, label in validateLoader:
                    # 载入GPU
                    wild_embedding = wild_embedding.to(device)
                    mutant_embedding = mutant_embedding.to(device)
                    label = label.to(device)
                    output = model(wild_embedding, mutant_embedding)  # 预测模型
                    running_loss = loss_fn(output, label)  # 损失函数并运行计算梯度

                    metrics.record(output, label)  # 计算每一轮的指标
                    valid_loss += running_loss.item()
                    valid_steps += 1

            logger.debug("\n" + metrics.display())
            writer.add_scalar("valid_loss", valid_loss / valid_steps, epoch)
            # endregion

            progress_bar.update(1)
            early_stopping(valid_loss, model)
            if early_stopping.stop_or_not():
                logger.info(f"在测试集上训练{epoch}轮后提前停止,最优指标如下:\n" + metrics.display())
                break

    writer.close()
    # endregion


def testing(dataset: PonDataset4FineTuning,
            test_batch: int,
            logger,
            model,
            model_params_path: str,
            device) -> None:
    logger.info("===开始测试...===")
    model.load_state_dict(torch.load(model_params_path))
    model = model.to(device)

    testLoader = DataLoader(dataset=dataset, batch_size=test_batch)
    metrics = PonMetrics(existingMetrics={"R^2", "MSE", "MAE", "PCC"}, device=device).cuda()

    # 开始测试
    model.eval()
    with torch.no_grad():
        with tqdm(range(len(dataset))) as progress_bar:
            for wild_embedding, mutant_embedding, bio, label in testLoader:
                # 载入GPU
                wild_embedding = wild_embedding.to(device)
                mutant_embedding = mutant_embedding.to(device)
                label = label.to(device)
                output = model(wild_embedding, mutant_embedding)  # 预测模型
                metrics.record(output, label)  # 计算每一轮的指标

                progress_bar.update(len(label))

        logger.info("所选模型在该测试集上的性能如下:\n" + metrics.display())


if __name__ == "__main__":
    # region Setting Vertification
    if not os.path.exists(Config.logger_dir):
        os.makedirs(Config.logger_dir, exist_ok=True)
    if not os.path.exists(Config.model_save_path):
        os.makedirs(Config.model_save_dir, exist_ok=True)
    if not os.path.exists(Config.summary_writer_dir):
        os.makedirs(Config.summary_writer_dir, exist_ok=True)
    all_logger = logger_init(logger=logging.getLogger(), output_dir=Config.logger_dir)
    # endregion

    # region Data processing
    # Load the pre-trained model
    embedding_tokenizer, embedding_model = embedding_model_getter(Config.embedding_model_path,
                                                                  Config.embedding_model_name,
                                                                  Config.device)

    # Build the PonDT testLoader
    train_dataset, test_dataset = embedding_dataset_creator_full_length(df=Config.dataset,
                                                                        embedding_model_name=Config.embedding_model_name,
                                                                        model=embedding_model,
                                                                        tokenizer=embedding_tokenizer,
                                                                        device=Config.device)
    # endregion

    # region 2.Train the model
    train_model = Config.train_model_class(length=Config.context_length,
                                           embedding_model=embedding_model,
                                           embedding_model_name=Config.embedding_model_name,
                                           fine_tuning=Config.fine_or_not)
    train_optimizer = Config.train_optimizer_class(train_model.parameters(), **Config.train_optimizer_params)
    all_loss_func = nn.L1Loss(reduction="mean") if Config.all_loss_func == "MAE" else nn.MSELoss()

    all_logger.debug("===" + Config.embedding_model_name + "===")

    training(dataset=train_dataset,
             model=train_model,
             train_batch=Config.training_batch,
             valid_batch=Config.valid_batch,
             loss_fn=all_loss_func,
             optimizer=train_optimizer,
             device=Config.device,
             logger=all_logger,
             early_stop_patience=Config.early_stop_patience,
             model_save_path=Config.model_save_path,
             summary_write_dir=Config.summary_writer_dir,
             watch_metrics=["R^2", "MSE", "MAE", "PCC"])
    # endregion

    # region test the model
    test_model = Config.test_model_class(length=Config.context_length,
                                         embedding_model=embedding_model,
                                         embedding_model_name=Config.embedding_model_name,
                                         fine_tuning=Config.fine_or_not)
    testing(dataset=test_dataset,
            test_batch=16,
            logger=all_logger,
            model=test_model,
            model_params_path=Config.model_save_path,
            device=Config.device)
    # endregion
