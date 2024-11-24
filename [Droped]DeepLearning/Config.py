# region Basic setting
# Based project path
import os

project_path = "D:/WorkPath/PycharmProjects/MutTm-pred"
os.chdir(project_path)

# Cuda
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Context length
context_length = 200

# Namedtuple for params
from collections import namedtuple

MainModel = namedtuple("MainModel", ["model", "name"])
EmbeddingModel = namedtuple("EmbeddingModel", ["name", "path"])
Optimizer = namedtuple("Optimizer", ["optimizer", "params"])
# endregion

# region Dataset init
from Dataset.Process4Dataset.DatasetCeator4PonDT import Dataset4MutTm

dataset = Dataset4MutTm(package_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset",
                        train_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\ProThermDB\Common\excllent_ProThermDB_Training.csv",
                        test_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\ProThermDB\Common\excllent_ProThermDB_Testing.csv",
                        training_version="ProThermDB_Common",
                        testing_version="ProThermDBTest_Common",
                        selected_columns=["UniProt_ID", "Mutation", "ΔTm"],
                        mult_mode="Average",
                        features=["neighbor"],
                        R_path=r"C:\Program Files\R\R-4.3.2",
                        context_length=context_length)
added_information = "pH-Tm"
# endregion

# region Optional main model
from DeepLearning.Model import PonDT, PontBert_FeedAttention, ESM3B_ConvNet_Sub, ESM3B_ConvNet_Comb

ponbert_feedattention: MainModel = MainModel(PontBert_FeedAttention, "PontBert_FeedAttention")
esm_convnet_sub: MainModel = MainModel(ESM3B_ConvNet_Sub, "ESM-ConvNet")
esm_convnet_comb: MainModel = MainModel(ESM3B_ConvNet_Comb, "ESM-ConvNet_Comb")
# endregion
selected_main_model = esm_convnet_comb
# region Optional embedding model
esm2_150M = EmbeddingModel("ESM-2-150M", "DeepLearning/EmbeddingModels/ESM-2/esm2_t30_150M_UR50D")
esm2_650M = EmbeddingModel("ESM-2-650M", "DeepLearning/EmbeddingModels/ESM-2/esm2_t33_650M_UR50D")
esm2_3B = EmbeddingModel("ESM-2-3B", "DeepLearning/EmbeddingModels/ESM-2/esm2_t36_3B_UR50D")
protbert = EmbeddingModel("ProtBert", "DeepLearning/EmbeddingModels/ProtBert/ProtBert")
prott5_half = EmbeddingModel("ProtT5-Half", "DeepLearning/EmbeddingModels/ProT5/prot_t5_xl_half_uniref50-enc")
# endregion
selected_embedding_model = esm2_3B
fine_or_not = False
# region Optional optimizer
adam = Optimizer(torch.optim.Adam, dict(lr=0.003124, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.000225))
sgd = Optimizer(torch.optim.SGD, dict(lr=9e-3, momentum=0.9, weight_decay=5e-4))
rms = Optimizer(torch.optim.RMSprop, dict(lr=1e-4, alpha=0.99, eps=1e-08, weight_decay=1e-4))
# endregion
selected_optimizer = adam
all_loss_func = "MAE"
early_stop_patience = 40
training_batch = 64
valid_batch = 32
# region quite work
train_model_class = test_model_class = selected_main_model.model
extraction_model_name = selected_main_model.name
embedding_model_path = selected_embedding_model.path
embedding_model_name = selected_embedding_model.name
train_optimizer_class = selected_optimizer.optimizer
train_optimizer_params = selected_optimizer.params
# endregion

# region Saving, Writing and Reloading path
logger_dir = f"{project_path}/DeepLearning/OutPut/{context_length}/"
summary_writer_dir = f"{project_path}/DeepLearning/WriterBoard/{context_length}/{added_information}_{embedding_model_name}_{extraction_model_name}/"
model_save_dir = f"{project_path}/DeepLearning/ModelSave/{context_length}"
model_save_path = f"{project_path}/DeepLearning/ModelSave/{context_length}/Result_{added_information}_{embedding_model_name}_{extraction_model_name}.pth"
# endregion
