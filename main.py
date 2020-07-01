import gc
import os
import shutil
import logging
import pickle
import random
import datetime as dt
import time
import copy
import yaml
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.multiprocessing import set_start_method

import matplotlib.pyplot as plt

from ts_dataset import TimeSeriesDataSet
from gru import biGRU, init_bigru

try:
    set_start_method("spawn")
except RuntimeError:
    pass

#####
# Functions
#####


def get_dt_str():
    return str(dt.datetime.now()).split(".")[0].replace(" ", "_")


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_file):
    """Loading YAML config file and parsing to the dictionary"""
    with open(config_file, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def QLL(predicted, observed, p):
    # p = torch.tensor(p)
    QLL = torch.pow(predicted, (-p)) * (
        ((predicted * observed) / (1 - p)) - ((torch.pow(predicted, 2)) / (2 - p))
    )

    return QLL


def tweedieloss(predicted, observed, p):
    """
    Custom loss fuction designed to minimize the deviance using stochastic gradient descent
    tweedie deviance from McCullagh 1983

    """
    d = -2 * QLL(predicted, observed, p)
    #     loss = (weight*d)/1

    return torch.mean(d)


def clip_percentile(ts, perc=0.98):
    """Clip the timseries values with percentile threshold"""
    return np.clip(ts, 0, np.quantile(ts, q=perc))


def prepare_data(
    df: pd.DataFrame, n_train_months=12, n_test_months=1, device=torch.device("cpu"),
):
    """Prepare data for RNNs
    
    Month here equals 28 days or 4 weeks
    
    """
    start_point = n_train_months + n_test_months * 2

    X_train = df.iloc[:, -7 * 4 * start_point : -7 * 4 * (n_test_months * 2)]
    Y_train = df.iloc[:, -7 * 4 * (n_test_months * 2) : -7 * 4 * (n_test_months * 1)]

    X_test = df.iloc[
        :, -7 * 4 * (start_point - n_test_months * 1) : -7 * 4 * (n_test_months * 1),
    ]
    Y_test = df.iloc[:, -7 * 4 * (n_test_months * 1) :]

    # scaling training dataset considering activation funcrion to be used later (tanh)
    scaler_xtrain = MinMaxScaler(feature_range=(0, 1))
    scaler_ytrain = MinMaxScaler(feature_range=(0, 1))

    scaler_xtrain.fit(X_train)
    scaler_ytrain.fit(Y_train)

    X_train = scaler_xtrain.transform(X_train)
    Y_train = scaler_ytrain.transform(Y_train)
    X_test = scaler_xtrain.transform(X_test)
    Y_test = scaler_ytrain.transform(Y_test)

    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze_(-1).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze_(-1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze_(-1).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.float32).unsqueeze_(-1).to(device)

    logging.info(f"Train datasets shapes: {X_train.shape}, {Y_train.shape}")
    logging.info(f"Test datasets shapes: {X_test.shape}, {Y_test.shape}")

    return X_train, Y_train, X_test, Y_test, scaler_xtrain, scaler_ytrain


def sliding_window(ts, feature_length, target_length, gap=0):
    """Slides over timeseries to build training dataset

    Args:
        ts (1-D sequence): The timseries to iterate over.
        feature_length (int): The number of predictor points
        target_length (int): The number of predicting points
        gap (int, optional): The number of poiunts jumped between the last feature timestamp and the first target one. Defaults to 0.

    Returns:
        tuple: (features, targets) np.arrays
    """
    features = []
    targets = []

    last_start_point = len(ts) - feature_length - target_length - gap + 1

    for i in range(last_start_point):

        first_feature_point = i
        first_target_point = i + gap + feature_length
        last_target_point = first_target_point + target_length

        features.append(ts[first_feature_point : first_target_point - gap])
        targets.append(ts[first_target_point:last_target_point])

    return np.array(features), np.array(targets)


def sliding_window_bunch(tss, feature_length, target_length, gap=0):
    """Slides over the two dimnsional bunch of timeseries to build training dataset

    Args:
        ts (2-D sequence): The timseries to iterate over.
        feature_length (int): The number of predictor points
        target_length (int): The number of predicting points
        gap (int, optional): The number of poiunts jumped between the last feature timestamp and the first target one. Defaults to 0.

    Returns:
        tuple: (features, targets) np.arrays
    """

    features = []
    targets = []

    last_start_point = tss.shape[1] - feature_length - target_length - gap + 1

    for i in range(last_start_point):
        first_feature_point = i
        first_target_point = i + gap + feature_length
        last_target_point = first_target_point + target_length

        features.append(tss[:, first_feature_point : first_target_point - gap])
        targets.append(tss[:, first_target_point:last_target_point])

    return np.vstack(features), np.vstack(targets)


def train_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    dataloaders,
    dataset_sizes,
    metrics=None,
    device=torch.device("cpu"),
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf  # minimizing loss
    best_metrics = np.inf  # minimizing metrics

    for epoch in range(num_epochs):
        start_time = dt.datetime.now()

        # print("Epoch {}/{}".format(epoch, num_epochs - 1).center(50, "="))

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_metrics = 0.0

            # Iterate over data.
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # TODO: metrics
                    # loss = tweedieloss(
                    #     outputs, targets, 1.5
                    # )
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                if metrics:
                    running_metrics += metrics(outputs, targets).item() * inputs.size(0)

            if phase == "train":
                scheduler.step()
                train_loss = running_loss / dataset_sizes[phase]
            elif phase == "val":
                val_loss = running_loss / dataset_sizes[phase]

                if metrics:
                    val_metrics = running_metrics / dataset_sizes[phase]

            # epoch_loss = running_loss / dataset_sizes[phase]
            # print("{} loss: {:.4f}".format(phase, epoch_loss))

            # deep copy the model
            if phase == "val" and val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                if metrics:
                    best_metrics = val_metrics

        logging.info(
            f"Epoch: {epoch}, train_loss: {train_loss}, val_loss {val_loss}, metrics: {val_metrics}, lr: {scheduler.get_last_lr()[0]}, time_per_epoch: {(dt.datetime.now() - start_time).total_seconds()}"
        )

    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    logging.info("Best val loss: {:4f}".format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def eval_model(
    model,
    dataloader,
    criterion,
    simple_scaler,
    return_preds=False,
    device=torch.device("cpu"),
):
    running_loss = 0.0
    count_of_examples = 0

    preds = []

    # Iterate over data.
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        count_of_examples += inputs.shape[0]
        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # preds.append(outputs) # memory leakege

        # statistics
        running_loss += loss.item() * inputs.size(0)

    whole_loss = running_loss / count_of_examples

    if return_preds:
        # preds = np.vstack(preds)
        return whole_loss, preds
    else:
        return whole_loss


def save_model(model, name="bilstm.pt"):
    try:
        torch.save(model.state_dict(), name)
    except:
        logging.error("Saving error: {exc}")


if __name__ == "__main__":

    CONFIG_FILE = "config_bilstm.yml"
    CONFIG = load_config(CONFIG_FILE)

    path = "models"

    if not os.path.isdir(path):
        os.mkdir(path)

    folder_name = f"bilstm_{get_dt_str()}"
    path = os.path.join(path, folder_name)
    os.mkdir(path)

    shutil.copy2(CONFIG_FILE, path)

    #####
    # Logs
    #####
    LOG_FILE = os.path.join(path, "logs.txt")
    logging.basicConfig(
        filename=LOG_FILE, level="INFO", format="%(asctime)s: %(message)s",
    )

    DEVICE = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    #####
    # Configs
    #####
    TRAIN_PERIOD = CONFIG["TRAIN_PERIOD"]
    TEST_PERIOD = CONFIG["TEST_PERIOD"]
    NUM_LAYERS = CONFIG["NUM_LAYERS"]
    N_EPOCHS = CONFIG["N_EPOCHS"]
    seed_everything(CONFIG["RANDOM_SEED"])

    #####
    # Initialization
    #####

    usecols = [f"d_{i}" for i in range(1914 - 7 * 4 * 12 * 3, 1914)]  # two years

    df = pd.read_csv(
        CONFIG["DATA_PATH"], index_col=[0], header=0, usecols=["id"] + usecols,
    ).fillna(0)

    # X_train, Y_train, X_test, Y_test, x_scaler, y_scaler = prepare_data(
    #     df=df, n_train_months=TRAIN_PERIOD, n_test_months=TEST_PERIOD, device=DEVICE,
    # )

    #####
    # Scaling by the max
    #####
    simple_scaler = df.max(axis=1)
    df = df.div(simple_scaler, axis=0).fillna(0)

    datasets = {
        "train": TimeSeriesDataSet(
            df.iloc[:, : -2 * TEST_PERIOD].apply(clip_percentile, axis=0).values,
            df.iloc[:, -2 * TEST_PERIOD : -TEST_PERIOD].values,
            device="cpu",
        ),
        "val": TimeSeriesDataSet(
            df.iloc[:, TEST_PERIOD:-TEST_PERIOD].values,
            df.iloc[:, -TEST_PERIOD:].values,
            device="cpu",
        ),
    }
    #####
    # Cut the long series into smaller ones
    # Works worse
    #####

    # x, y = sliding_window_bunch(
    #     df.values, CONFIG["TRAIN_PERIOD"], CONFIG["TEST_PERIOD"], 0
    # )

    del df
    gc.collect()

    # remove zeros from train
    # y_train = y[: 30490 * 28][(x[: 30490 * 28] != 0).sum(axis=1) > 1]
    # x_train = x[: 30490 * 28][(x[: 30490 * 28] != 0).sum(axis=1) > 1]
    # del x, y

    # datasets = {
    #     "train": TimeSeriesDataSet(x, y, device="cpu"),
    #     "val": TimeSeriesDataSet(x[30490 * 28 :], y[30490 * 28 :], device="cpu"),
    # }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            datasets[x], num_workers=4, shuffle=True, batch_size=CONFIG["BATCH_SIZE"]
        )
        for x in datasets.keys()
    }

    dataset_sizes = {x: len(datasets[x]) for x in datasets.keys()}

    # model = init_bilstm(
    #     datasets["train"],
    #     num_layers=NUM_LAYERS,
    #     batch_first=True,
    #     dropout=CONFIG["DROPOUT"],
    #     bidirectional=True,
    #     device=DEVICE,
    # )

    model = init_bigru(
        datasets["train"],
        num_layers=NUM_LAYERS,
        batch_first=True,
        dropout=CONFIG["DROPOUT"],
        bidirectional=True,
        device=DEVICE,
    )
    #####
    # Hyperparams
    #####
    metrics_to_train = torch.nn.PoissonNLLLoss(
        log_input=False, full=False, reduction="mean"
    )
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["INITIAL_LEARNING_RATE"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS, eta_min=CONFIG["FINAL_LEARNING_RATE"], last_epoch=-1
    )

    model_configs = {
        x: model.__dict__[x]
        for x in [
            "input_size",
            "hidden_size",
            "num_layers",
            "bias",
            "batch_first",
            "dropout",
            "bidirectional",
            "sequence_length",
            "out_sequence_length",
            "device",
        ]
    }

    logging.info(f"Device: {DEVICE}")
    logging.info(f"RANDOM_SEED: {CONFIG['RANDOM_SEED']}")
    logging.info(f"Collected shapes: {dataset_sizes}")
    logging.info(f"Whole model params: {model.__dict__}")
    logging.info(f"Model configs: {model_configs}")
    logging.info(f"TRAIN_PERIOD: {TRAIN_PERIOD}")
    logging.info(f"TEST_PERIOD: {TEST_PERIOD}")
    logging.info(f"NUM_LAYERS: {NUM_LAYERS}")
    logging.info(f"N_EPOCHS: {N_EPOCHS}")
    logging.info(f"criterion: {criterion}")
    logging.info(f"optimizer: {optimizer}")
    logging.info(f"scheduler: {scheduler}")

    with open(os.path.join(path, "model_configs.pkl"), "wb") as f:
        pickle.dump(model_configs, f)
    with open(os.path.join(path, "simple_scaler.pkl"), "wb") as f:
        pickle.dump(simple_scaler, f)
    # with open(os.path.join(path, "x_scaler.pkl"), "wb") as f:
    #     pickle.dump(x_scaler, f)
    # with open(os.path.join(path, "y_scaler.pkl"), "wb") as f:
    #     pickle.dump(y_scaler, f)

    model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=CONFIG["N_EPOCHS"],
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        metrics=metrics_to_train,
        device=DEVICE,
    )

    if CONFIG["SAVE_VAL_ERRORS"]:
        metrics, errors = eval_model(
            model,
            dataloaders["val"],
            criterion,
            simple_scaler,
            return_preds=True,
            device=DEVICE,
        )
        with open(os.path.join(path, "errors.pkl"), "wb") as f:
            pickle.dump(errors, f)
    else:
        metrics = eval_model(
            model,
            dataloaders["val"],
            criterion,
            simple_scaler,
            return_preds=False,
            device=DEVICE,
        )
    save_model(model, name=os.path.join(path, "model.pt"))
