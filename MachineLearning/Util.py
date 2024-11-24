import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# region 评估指标计算函数
def rmse_scoring(true, pred):
    return np.sqrt(mean_squared_error(true, pred))


def pcc_scoring(true, pred):
    return stats.pearsonr(true, pred)[0]


def r2_scoring(true, pred):
    return r2_score(true, pred)


def mse_scoring(true, pred):
    return mean_squared_error(true, pred)


def mae_scoring(true, pred):
    return mean_absolute_error(true, pred)


# endregion


# region 交叉验证及测试函数
def clock(piece: int = 3):
    """
    A clock for time counting of multiple pieces.
    Example:
    \nclk = clock(3), next(clk) → initialization\n
    time.sleep(5)
    print(next(clk)) → 5s\n
    time.sleep(2)
    print(next(clk)) → 2s\n
    time.sleep(10)
    print(next(clk)) → 10s
    :param piece: How many pieces of time would you like to count?
    :yield: yield the last piece timer.
    """
    pre = time.time()

    # beginning
    yield pre

    while piece:
        part = time.time() - pre
        pre = time.time()
        piece -= 1
        yield part


def cross_validate_mine(train_x, train_y, model, kf) -> tuple:
    """
    用于观测(MAE, PCC, R2, RMSE)四项指标的交叉验证函数
    :return: tuple(MAE, PCC, R2, RMSE)
    """
    mae_scores = []
    pcc_scores = []
    r2_scores = []
    rmse_scores = []

    for train_index, validate_index in kf.split(train_x):
        train_fold_x, validate_x = train_x[train_index], train_x[validate_index]
        train_fold_y, validate_y = train_y[train_index], train_y[validate_index]

        # 创建临时模型
        cv_model = deepcopy(model)

        # 训练模型
        cv_model.fit(train_fold_x, train_fold_y)

        # 预测验证集
        pred_y = cv_model.predict(validate_x)

        # 计算评估指标
        mae_scores.append(mae_scoring(validate_y, pred_y))
        pcc_scores.append(pcc_scoring(validate_y, pred_y))
        r2_scores.append(r2_scoring(validate_y, pred_y))
        rmse_scores.append(np.sqrt(mse_scoring(validate_y, pred_y)))

    # 计算平均值
    mean_mae = np.mean(mae_scores)
    mean_pcc = np.mean(pcc_scores)
    mean_r2 = np.mean(r2_scores)
    mean_rmse = np.mean(rmse_scores)

    return mean_mae, mean_pcc, mean_r2, mean_rmse


def train_validation_test(model,
                          train_x,
                          train_y,
                          test_x,
                          test_y,
                          draw: bool = True):
    """
    针对制定模型数据的训练、交叉验证及测试结果
    :return: tuple(MAE, PCC, R2, RMSE)
    """
    clk = clock(3)
    # region Training
    next(clk)
    print("Starting Training...")
    model.fit(train_x, train_y)
    print("Time for Training:", next(clk))
    # endregion
    cv_res = cross_validate_mine(train_x, train_y, model, KFold(n_splits=5, shuffle=True, random_state=42))
    print(f"The Result for CV:\n"
          f"MAE:{cv_res[0]:.4f}\n"
          f"PCC:{cv_res[1]:.4f}\n"
          f"R2:{cv_res[2]:.4f}\n"
          f"RMSE:{cv_res[3]:.4f}\n"
          f"Time for CV:{next(clk)}")
    # endregion

    # region Testing
    print("Starting Testing...")
    pred_y = model.predict(test_x)
    print("The Result for Testing:"
          f"MAE:{mae_scoring(test_y, pred_y):.4f}\n"
          f"PCC:{pcc_scoring(test_y, pred_y):.4f}\n"
          f"R2:{r2_scoring(test_y, pred_y):.4f}\n"
          f"RMSE:{rmse_scoring(test_y, pred_y):.4f}\n"
          f"Time for Testing:{next(clk)}")
    # endregion

    # region Painting
    if draw:
        plt.figure(figsize=(5, 3))
        plt.scatter(test_y, pred_y, color='green', label='Predicted')
        plt.scatter(test_y, test_y, color='blue', label='Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')
        plt.legend()
        plt.show()
    # endregion


def train_test(model,
               train_x,
               train_y,
               test_x,
               test_y,
               draw: bool = True) -> None:
    clk = clock(2)
    # region Training
    next(clk)
    print("Starting Training...")
    model.fit(train_x, train_y)
    print("Time for Training:", next(clk))
    # endregion

    # region Testing
    print("Starting Testing...")
    pred_y = model.predict(test_x)
    print("The Result for Testing:"
          f"MAE:{mae_scoring(test_y, pred_y):.4f}\n"
          f"PCC:{pcc_scoring(test_y, pred_y):.4f}\n"
          f"R2:{r2_scoring(test_y, pred_y):.4f}\n"
          f"RMSE:{rmse_scoring(test_y, pred_y):.4f}\n"
          f"Time for Testing:{next(clk)}")
    # endregion

    # region Painting
    if draw:
        plt.figure(figsize=(5, 3))
        plt.scatter(test_y, pred_y, color='green', label='Predicted')
        plt.scatter(test_y, test_y, color='blue', label='Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')
        plt.legend()
        plt.show()
    # endregion


def test(model,
         test_x,
         test_y,
         draw: bool = False) -> None:
    """
    针对指定模型和数据集的测试函数
    :param model: 指定模型
    :param test_x: 测试数据
    :param test_y: 测试指标
    :param draw: 是否进行作图，默认为False，即不针对预测结果进行绘制
    :return:
    """
    clk = clock(1)
    next(clk)
    print("Starting Testing...")
    pred_y = model.predict(test_x)
    print("The Result for Testing:"
          f"MAE:{mae_scoring(test_y, pred_y):.4f}\n"
          f"PCC:{pcc_scoring(test_y, pred_y):.4f}\n"
          f"R2:{r2_scoring(test_y, pred_y):.4f}\n"
          f"RMSE:{rmse_scoring(test_y, pred_y):.4f}\n"
          f"Time for Testing:{next(clk)}")

    if draw:
        plt.figure(figsize=(5, 3))
        plt.scatter(test_y, pred_y, color='green', label='Predicted')
        plt.scatter(test_y, test_y, color='blue', label='Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')
        plt.legend()
        plt.show()


# endregion


# region optuna parameter adjustment objective
def objective_model(trial, model_x, model_y, estimator_name: str, n_jobs: int = 4) -> tuple:
    """
    :return: tuple(MAE, PCC, R2, RMSE)
    """
    # 定义超参数搜索空间
    if estimator_name == "lightGBM":
        lgb_param = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0, log=True),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 40.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "cat_smooth": trial.suggest_int("cat_smooth", 0, 100),
            "n_jobs": n_jobs,
            "verbose": -1,
            "force_col_wise": True
        }
        estimator = LGBMRegressor(**lgb_param)
    elif estimator_name == "RandomForest":
        rf_param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=10),
            'max_depth': trial.suggest_int('max_depth', 2, 30, step=1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_int('max_features', 1, 30),
            'bootstrap': trial.suggest_categorical("bootstrap", [True, False]),
            'criterion': "absolute_error",
            'n_jobs': n_jobs,
            "verbose": -1
        }
        estimator = RandomForestRegressor(**rf_param)
    elif estimator_name == "XGBoost":
        xg_param = {
            'booster': 'gbtree',
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 1),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 0, 1),
            "lambda": trial.suggest_float("lambda", 0, 1),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
            "n_jobs": n_jobs
        }
        estimator = XGBRegressor(**xg_param)
    elif estimator_name == "SVR":
        SVR_param = {
            "C": trial.suggest_float("C", 1e-5, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 1.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 1e-5, 1.0, log=True)
        }
        estimator = SVR(**SVR_param)
    elif estimator_name == "MLP":
        estimator = MLPRegressor()
    elif estimator_name == "DTR":
        estimator = DecisionTreeRegressor()
    else:
        raise ValueError("estimator should be in ['RandomForest', 'lightGBM', 'XGBoost', 'SVR', 'MLP', 'DTR']")

    # 交叉验证
    return cross_validate_mine(model_x, model_y, estimator, KFold(n_splits=5, shuffle=True, random_state=42))


def objective_rfecv(trial, rfecv_x, rfecv_y, estimator_name, n_jobs: int = 4) -> tuple:
    """
    :return: tuple(MAE, PCC, R2, RMSE)
    """
    params_rfecv = {
        "step": 50,
        "min_features_to_select": trial.suggest_categorical("min_features_to_select", [50, 100, 200, 300, 500, 1000]),
    }
    if estimator_name == "lightGBM":
        estimator = LGBMRegressor(force_col_wise=True, verbose=0, n_jobs=n_jobs)
        excutor = LGBMRegressor(force_col_wise=True, verbose=0, n_jobs=n_jobs)
    elif estimator_name == "RandomForest":
        estimator = RandomForestRegressor(verbose=0, n_jobs=n_jobs)
        excutor = RandomForestRegressor(verbose=0, n_jobs=n_jobs)
    elif estimator_name == "XGBoost":
        estimator = XGBRegressor(n_jobs=n_jobs)
        excutor = XGBRegressor(n_jobs=n_jobs)
    elif estimator_name == "SVR":
        estimator = SVR(kernel="linear", verbose=0)
        estimator.fit(rfecv_x, rfecv_y.ravel())
        excutor = SVR(kernel="linear", verbose=0)
    else:
        raise ValueError("estimator should be in ['RandomForest', 'lightGBM', 'XGBoost', 'SVR', 'MLP', 'DTR']")

    # feature selection
    selection_rfecv = RFECV(estimator=estimator,
                            n_jobs=n_jobs,
                            cv=KFold(n_splits=5, shuffle=True, random_state=42),
                            step=params_rfecv["step"],
                            min_features_to_select=params_rfecv["min_features_to_select"],
                            verbose=0)
    selection_rfecv.fit(rfecv_x,
                        rfecv_y.ravel())

    # train valid test
    return cross_validate_mine(selection_rfecv.transform(rfecv_x), rfecv_y, excutor,
                               KFold(n_splits=5, shuffle=True, random_state=42))


def objective_rfe(trial, rfe_x, rfe_y, estimator_name, n_jobs: int = 4) -> tuple:
    """
    :return: tuple(MAE, PCC, R2, RMSE)
    """
    params_rfe = {
        "step": 50,
        "n_features_to_select": trial.suggest_categorical("n_features_to_select", [50, 100, 200, 300, 500, 1000]),
    }
    if estimator_name == "lightGBM":
        estimator = LGBMRegressor(force_col_wise=True, verbose=0, n_jobs=4)
        excutor = LGBMRegressor(force_col_wise=True, verbose=0, n_jobs=4)
    elif estimator_name == "RandomForest":
        estimator = RandomForestRegressor(verbose=0, n_jobs=4)
        excutor = RandomForestRegressor(verbose=0, n_jobs=4)
    elif estimator_name == "XGBoost":
        estimator = XGBRegressor(n_jobs=n_jobs)
        excutor = XGBRegressor(n_jobs=n_jobs)
    elif estimator_name == "SVR":
        estimator = SVR(kernel="linear", verbose=0)
        excutor = SVR(kernel="linear", verbose=0)
    else:
        raise ValueError("estimator should be in ['RandomForest', 'lightGBM', 'XGBoost', 'SVR', 'MLP', 'DTR']")

    # feature selection
    selection_rfe = RFE(estimator=estimator,
                        step=params_rfe["step"],
                        n_features_to_select=params_rfe["n_features_to_select"],
                        verbose=0)
    selection_rfe.fit(rfe_x,
                      rfe_y.ravel())

    # train valid test
    return cross_validate_mine(selection_rfe.transform(rfe_x), rfe_y, excutor,
                               KFold(n_splits=5, shuffle=True, random_state=42))
# endregion
