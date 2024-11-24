import os
import numpy as np

os.chdir("D:/WorkPath/PycharmProjects/MutTm-pred")
from Dataset.Process4Dataset.DatasetCeator4PonDT import Dataset4MutTm

dataset = Dataset4MutTm(package_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset",
                      train_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\ProThermDB\pH-Tm\excllent_ProThermDB_Training.csv",
                      test_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\ProThermDB\pH-Tm\excllent_ProThermDB_Testing.csv",
                      training_version="ProThermDB_withpHTm",
                      testing_version="ProThermDBTest_withpHTm",
                      selected_columns=["UniProt_ID", "Mutation", "pH", "Tm", "ΔTm"],
                      mult_mode="Average",
                      features=["neighbor", "aaindex", "group", "param", "rpm", "hydrop", "GO"],
                      context_length=0)
global_train_x = dataset.train_feature_set
global_train_y = dataset.train_label_set
global_test_y = dataset.test_label_set
global_test_x = dataset.test_feature_set
x_train = np.array(global_train_x)
x_test = np.array(global_test_x)
y_train = np.array(global_train_y).reshape(len(global_train_y), 1).ravel()
y_test = np.array(global_test_y).reshape(len(global_test_y), 1).ravel()

import os
import numpy as np

os.chdir("D:/WorkPath/PycharmProjects/MutTm-pred")
from Dataset.Process4Dataset.DatasetCeator4PonDT import Dataset4MutTm

dataset = Dataset4MutTm(package_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\Process4Dataset",
                      train_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\PonDB\pH-Tm\PonDB.csv",
                      test_dataset_path=r"D:\WorkPath\PycharmProjects\MutTm-pred\Dataset\BasicData\ProThermDB\pH-Tm\excllent_ProThermDB_Testing.csv",
                      training_version="PonDB_withpHTm",
                      testing_version="ProThermDBTest_withpHTm",
                      selected_columns=["UniProt_ID", "Mutation", "pH", "Tm", "ΔTm"],
                      mult_mode="Average",
                      features=["neighbor", "aaindex", "group", "param", "rpm", "hydrop", "GO"],
                      context_length=0)
global_train_x = dataset.train_feature_set
global_train_y = dataset.train_label_set
global_test_y = dataset.test_label_set
global_test_x = dataset.test_feature_set
x_train = np.array(global_train_x)
x_test = np.array(global_test_x)
y_train = np.array(global_train_y).reshape(len(global_train_y), 1).ravel()
y_test = np.array(global_test_y).reshape(len(global_test_y), 1).ravel()