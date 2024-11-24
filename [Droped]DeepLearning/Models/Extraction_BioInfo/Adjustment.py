import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import optuna
from torch import optim
import sys
import os

if sys.platform == "linux":
    os.chdir("/public/home/yyang/kjh/MutTm-pred")
    sys.path.append("/public/home/yyang/kjh/MutTm-pred")
else:
    os.chdir("D:/WorkPath/PycharmProjects/MutTm-pred")

from Dataset.Process4Dataset.DatasetCeator4PonDT import Dataset4MutTm
from DeepLearning.Util import (embedding_model_getter, dataset_creator4feature_extraction,
                               training4feature_extraction, testing4feature_extraction)

model_save_dir = "DeepLearning/Models/Extraction_BioInfo/best_models/"
# storage = "sqlite:///db.sqlite3.Deeplearning"
storage = "sqlite:///LinuxDB/DeepLearning/Extraction_BioInfo.sqlite3"
float_scale = torch.float64
directions = ["minimize", "maximize", "maximize", "minimize"]


def objective(trial, optuna_train_dataset, optuna_test_dataset, optuna_model, model_params_path):
    # region Train
    model = optuna_model(trial)
    # region 损失函数
    loss_fn_name = trial.suggest_categorical("loss_fn", ["MAE", "MSE"])
    loss_fn = (nn.L1Loss(reduction="mean") if loss_fn_name == "MAE" else nn.MSELoss()).to(device)
    # endregion
    # region 优化器
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optim_params = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
    }
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               **optim_params,
                               betas=(0.9, 0.999),
                               eps=1e-04)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(),
                                  **optim_params,
                                  alpha=0.99)
    else:
        optimizer = optim.SGD(model.parameters(),
                              **optim_params,
                              momentum=0.9,
                              dampening=0)
    # endregion
    # region 加载并分割训练/验证集
    train_index, valid_index = train_test_split(range(len(optuna_train_dataset)), test_size=0.1, random_state=42)
    train_fold, val_fold = optuna_train_dataset.subDataset(train_index), optuna_train_dataset.subDataset([valid_index])
    trainLoader = DataLoader(dataset=train_fold, batch_size=64, shuffle=True, drop_last=True)
    validateLoader = DataLoader(dataset=val_fold, batch_size=64, shuffle=True, drop_last=True)
    # endregion
    training4feature_extraction(model,
                                trainLoader,
                                validateLoader,
                                loss_fn,
                                optimizer,
                                model_params_path,
                                device)
    # endregion

    # region Test
    return testing4feature_extraction(model,
                                      DataLoader(dataset=optuna_test_dataset, batch_size=32, drop_last=True),
                                      model_params_path,
                                      device)
    # endregion


def ESM3B():
    # region 创建嵌入数据集
    embedding_model_path = "DeepLearning/EmbeddingModels/ESM-2/esm2_t36_3B_UR50D"
    embedding_model_name = "ESM-2-3B"
    embedding_tokenizer, embedding_model = embedding_model_getter(embedding_model_path,
                                                                  embedding_model_name,
                                                                  device)
    train_dataset, test_dataset = dataset_creator4feature_extraction(df=dataset,
                                                                     embedding_model_name=embedding_model_name,
                                                                     model=embedding_model,
                                                                     tokenizer=embedding_tokenizer,
                                                                     device=device)
    del embedding_model, embedding_tokenizer
    # endregion

    from DeepLearning.Models.Extraction_BioInfo.Model import (ESM3B_AttentionNet_BioInfo_Sub,
                                                              ESM3B_AttentionNet_BioInfo_Comb,
                                                              ESM3B_ConvNet_BioInfo_Sub,
                                                              ESM3B_ConvNet_BioInfo_Comb)
    study01 = optuna.create_study(study_name="ESM3B-Attention-BioInf-Subtration",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study01.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ESM3B_AttentionNet_BioInfo_Sub,
                                             model_save_dir + "/ESM3B_AttentionNet_BioInfo_Sub.pth"), n_trials=300)

    study02 = optuna.create_study(study_name="ESM3B-Attention-BioInf-Combnation",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study02.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ESM3B_AttentionNet_BioInfo_Comb,
                                             model_save_dir + "ESM3B_AttentionNet_BioInfo_Comb.pth"), n_trials=300)

    study03 = optuna.create_study(study_name="ESM3B-Conv-BioInf-Subtration",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study03.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ESM3B_ConvNet_BioInfo_Sub,
                                             model_save_dir + "ESM3B_ConvNet_BioInfo_Sub.pth"), n_trials=300)

    study04 = optuna.create_study(study_name="ESM3B-Conv-BioInfo-Subtration",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study04.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ESM3B_ConvNet_BioInfo_Comb,
                                             model_save_dir + "ESM3B_ConvNet_BioInfo_Comb.pth"), n_trials=300)


def ESM650M():
    # region 创建嵌入数据集
    embedding_model_path = "DeepLearning/EmbeddingModels/ESM-2/esm2_t33_650M_UR50D"
    embedding_model_name = "ESM-2-650M"
    embedding_tokenizer, embedding_model = embedding_model_getter(embedding_model_path,
                                                                  embedding_model_name,
                                                                  device)
    train_dataset, test_dataset = dataset_creator4feature_extraction(df=dataset,
                                                                     embedding_model_name=embedding_model_name,
                                                                     model=embedding_model,
                                                                     tokenizer=embedding_tokenizer,
                                                                     device=device)
    del embedding_model, embedding_tokenizer
    # endregion

    from DeepLearning.Models.Extraction_BioInfo.Model import (ESM650M_AttentionNet_BioInfo_Sub,
                                                              ESM650M_AttentionNet_BioInfo_Comb,
                                                              ESM650M_ConvNet_BioInfo_Sub,
                                                              ESM650M_ConvNet_BioInfo_Comb)
    # study01 = optuna.create_study(study_name="ESM650M-Attention-BioInf-Subtration",
    #                               directions=directions,
    #                               storage=storage,
    #                               load_if_exists=True)
    # study01.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
    #                                          ESM650M_AttentionNet_BioInfo_Sub,
    #                                          model_save_dir + "ESM650M_AttentionNet_BioInfo_Sub.pth"),
    #                  n_trials=300)
    #
    # study02 = optuna.create_study(study_name="ESM650M-Attention-BioInf-Combnation",
    #                               directions=directions,
    #                               storage=storage,
    #                               load_if_exists=True)
    # study02.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
    #                                          ESM650M_AttentionNet_BioInfo_Comb,
    #                                          model_save_dir + "ESM650M_AttentionNet_BioInfo_Comb.pth"),
    #                  n_trials=300)
    #
    # study03 = optuna.create_study(study_name="ESM650M-Conv-BioInf-Subtration",
    #                               directions=directions,
    #                               storage=storage,
    #                               load_if_exists=True)
    # study03.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
    #                                          ESM650M_ConvNet_BioInfo_Sub,
    #                                          model_save_dir + "ESM650M_ConvNet_BioInfo_Sub.pth"),
    #                  n_trials=300)

    study04 = optuna.create_study(study_name="ESM650M-Conv-BioInfo-Subtration",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study04.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ESM650M_ConvNet_BioInfo_Comb,
                                             model_save_dir + "ESM650M_ConvNet_BioInfo_Comb.pth"),
                     n_trials=300)


def ProtBert():
    # region 创建嵌入数据集
    embedding_model_path = "DeepLearning/EmbeddingModels/ProtBert/ProtBert"
    embedding_model_name = "ProtBert"
    embedding_tokenizer, embedding_model = embedding_model_getter(embedding_model_path,
                                                                  embedding_model_name,
                                                                  device)
    train_dataset, test_dataset = dataset_creator4feature_extraction(df=dataset,
                                                                     embedding_model_name=embedding_model_name,
                                                                     model=embedding_model,
                                                                     tokenizer=embedding_tokenizer,
                                                                     device=device)
    del embedding_model, embedding_tokenizer
    # endregion

    from DeepLearning.Models.Extraction_BioInfo.Model import (ProtBert_AttentionNet_BioInfo_Sub,
                                                              ProtBert_AttentionNet_BioInfo_Comb,
                                                              ProtBert_ConvNet_BioInfo_Sub,
                                                              ProtBert_ConvNet_BioInfo_Comb)
    study01 = optuna.create_study(study_name="ProtBert-Attention-BioInf-Subtration",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study01.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ProtBert_AttentionNet_BioInfo_Sub,
                                             model_save_dir + "ProtBert_AttentionNet_BioInfo_Sub.pth"), n_trials=300)

    study02 = optuna.create_study(study_name="ProtBert-Attention-BioInf-Combnation",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study02.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ProtBert_AttentionNet_BioInfo_Comb,
                                             model_save_dir + "ProtBert_AttentionNet_BioInfo_Comb.pth"), n_trials=300)

    study03 = optuna.create_study(study_name="ProtBert-Conv-BioInf-Subtration",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study03.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ProtBert_ConvNet_BioInfo_Sub,
                                             model_save_dir + "ProtBert_ConvNet_BioInfo_Sub.pth"), n_trials=300)

    study04 = optuna.create_study(study_name="ProtBert-Conv-BioInfo-Subtration",
                                  directions=directions,
                                  storage=storage,
                                  load_if_exists=True)
    study04.optimize(lambda trial: objective(trial, train_dataset, test_dataset,
                                             ProtBert_ConvNet_BioInfo_Comb,
                                             model_save_dir + "ProtBert_ConvNet_BioInfo_Comb.pth"), n_trials=300)


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.get_device_name(0))
    dataset = Dataset4MutTm(package_path=r"Dataset/Process4Dataset",
                            train_dataset_path=r"Dataset/BasicData/ProThermDB/Common/excllent_ProThermDB_Training.csv",
                            test_dataset_path=r"Dataset/BasicData/ProThermDB/Common/excllent_ProThermDB_Testing.csv",
                            training_version="ProThermDB_Common",
                            testing_version="ProThermDBTest_Common",
                            selected_columns=["UniProt_ID", "Mutation", "ΔTm"],
                            mult_mode="Average",
                            features=["aaindex", "neighbor"],
                            context_length=0)  # 642
    print("Acting on ESM3B...")
    ESM3B()
    # print("Acting on ESM650M...")
    # ESM650M()
    # print("Acting on ProtBert...")
    # ProtBert()
