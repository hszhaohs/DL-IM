import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import math


def record_output(gt, Test_Pred, cfg):
    if cfg.data.do_scale:
        Test_Pred_np = cfg.scaler_Y.inverse_transform(np.array(Test_Pred).squeeze().reshape(-1, 1))
        GT = cfg.scaler_Y.inverse_transform(gt)    # shape of gt is (N, 1)
    else:
        Test_Pred_np = 100 * np.array(Test_Pred).squeeze().reshape(-1, 1)
        GT = 100 * gt
    if len(Test_Pred_np.shape) == 1:
        Test_Pred_np = Test_Pred_np.reshape([len(gt), -1])
    N_sample, N_time = Test_Pred_np.shape
    R2_score = [r2_score(GT, Test_Pred_np[:, ii]) for ii in range(N_time)]
    R2_score_np = np.array(R2_score).reshape([1, -1])

    MAE_np = np.abs((GT - Test_Pred_np))
    RMSE = [math.sqrt(mean_squared_error(GT, Test_Pred_np[:, ii])) for ii in range(N_time)]
    RMSE_np = np.array(RMSE).reshape([1, -1])
    RMSE_all = np.vstack([MAE_np, RMSE_np, R2_score_np])
    RMSE_index = ['Sample_{}'.format(ii + 1) for ii in range(N_sample)] + ['RMSE', 'R2']
    RMSE_col = ['Result']
    RMSE_df = pd.DataFrame(RMSE_all, index=RMSE_index, columns=RMSE_col)
    RMSE_file = os.path.join(cfg.work_dir, 'Records_RMSE-R2_{}.csv'.format(cfg.record_type))
    RMSE_df.to_csv(RMSE_file)

    Test_Pred_index = ['Sample_{}'.format(ii + 1) for ii in range(N_sample)]
    Test_Pred_col = ['GT', 'Pred']
    GT_Test_Pred = np.hstack([GT, Test_Pred_np])
    Test_Pred_df = pd.DataFrame(GT_Test_Pred, index=Test_Pred_index, columns=Test_Pred_col)
    Test_Pred_file = os.path.join(cfg.work_dir, 'Records_GT-Test_Pred_{}.csv'.format(cfg.record_type))
    Test_Pred_df.to_csv(Test_Pred_file)

    return R2_score_np, RMSE_np


def record_pred(pred, cfg):
    if cfg.data.do_scale:
        pred_np = cfg.scaler_Y.inverse_transform(np.array(pred).squeeze().reshape(-1, 1))
    else:
        pred_np = 100 * np.array(pred).squeeze()

    N_sample = len(pred_np)

    pred_index = ['{}'.format(ii + 1) for ii in range(N_sample)]
    pred_col = ['pred']
    pred_df = pd.DataFrame(pred_np, index=pred_index, columns=pred_col)
    pred_file = os.path.join(cfg.work_dir, os.path.basename(cfg.filename).replace('spec', 'pred'))
    pred_df.to_csv(pred_file)
