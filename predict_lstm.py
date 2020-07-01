from rnns import seed_everything
from lstm import biLSTM
from gru import biGRU

import pickle
import os

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import seaborn as sns


def get_test_train(
    path, n_train_months, x_scaler=None, simple_scaler=None, device=torch.device("cpu")
):

    df = pd.read_csv(
        path, index_col=["id"], header=0
    )  # , usecols=["id"] + [f"d_{i}" for i in range(1000, 1914)],
    # )

    start_point = n_train_months

    X_test = df.iloc[
        :, -7 * 4 * start_point :,
    ]

    if x_scaler is not None:
        X_test = x_scaler.transform(X_test)
    elif simple_scaler is not None:
        X_test = X_test.values / simple_scaler.values[:, np.newaxis]

    # after dividing by zero several nans were returned
    X_test[X_test != X_test] = 0

    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze_(-1).to(device)

    print(f"Test datasets shapes: {X_test.shape}")

    return X_test


def eval(model, X_test, Y_test, criterion, scaler_xtrain, scaler_ytrain, save=False):

    preds = model(X_test)[:, -Y_test.shape[1] :, :].squeeze(-1).cpu().data.numpy()
    test_target = Y_test.squeeze(-1).cpu().data.numpy()
    preds = scaler_ytrain.inverse_transform(preds)
    test_target = scaler_ytrain.inverse_transform(test_target)

    return (
        criterion(model(X_test)[:, -Y_test.shape[1] :, :], Y_test).item(),
        criterion(torch.tensor(preds), torch.tensor(test_target)).item(),
    )


if __name__ == "__main__":

    data_path = r"data/raw/sales_train_evaluation.csv"

    model_path = os.path.join("models", "bilstm_2020-06-30_10:48:28")

    seed_everything(42)

    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    with open(os.path.join(model_path, "model_configs.pkl"), "rb") as f:
        model_configs = pickle.load(f)

    model = biGRU(**model_configs)
    # model = biLSTM(**model_configs)

    model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
    model.eval()

    # with open(os.path.join(model_path, "x_scaler.pkl"), "rb") as f:
    #     x_scaler = pickle.load(f)
    # with open(os.path.join(model_path, "y_scaler.pkl"), "rb") as f:
    #     y_scaler = pickle.load(f)

    with open(os.path.join(model_path, "simple_scaler.pkl"), "rb") as f:
        simple_scaler = pickle.load(f)

    X_test = get_test_train(
        data_path,
        n_train_months=3,
        x_scaler=None,
        simple_scaler=simple_scaler,
        device=DEVICE,
    )

    # X_test.to(DEVICE)

    preds = model(X_test).squeeze(-1).cpu().data.numpy()
    print(preds.shape)
    # preds = y_scaler.inverse_transform(preds)

    preds = preds * simple_scaler[:, np.newaxis]

    indices = pd.read_csv(data_path, usecols=["id"])["id"]

    submission = pd.DataFrame(
        preds, index=indices, columns=[f"F{i}" for i in range(1, 29)]
    )

    submission = submission.applymap(lambda x: 0 if x < 0 else x)

    print(submission)

    if False:  # errors if target exist
        evaluation = pd.read_csv(r"data/raw/evaluation.csv")

        errors = submission.values - evaluation.values[:, 1:]
        mse = (errors ** 2).mean()
        mae = np.abs(errors).mean()

        print("MSE: ", mse)
        print("MAE: ", mae)

        sns.distplot(errors.flatten())
        plt.savefig(os.path.join(model_path, "errors.png"))

        with open(os.path.join(model_path, "metrics.txt"), "w") as f:
            f.write(str(mse) + "\n")
            f.write(str(mae))

    submission.to_csv(
        os.path.join(model_path, "submission.csv"), sep=",", header=True, index=True
    )
