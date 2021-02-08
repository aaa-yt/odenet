import os
import time
import json
import csv
from datetime import datetime
from logging import getLogger
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle

from config import Config
from lib import optimizers
from lib import losses
from lib import regularizers

logger = getLogger(__name__)

def start(config: Config):
    return Trainer(config).start()


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.loss = None
        self.accuracy = None
        self.regularizer = None
    
    def start(self):
        self.model = self.load_model()
        self.training()
    
    def training(self):
        self.compile_model()
        self.dataset = self.load_dataset()
        self.fit(x=self.dataset[0][0], y=self.dataset[0][1], epochs=self.config.trainer.epoch, batch_size=self.config.trainer.batch_size, validation_data=self.dataset[1], is_accuracy=self.config.trainer.is_accuracy, save_step=self.config.trainer.save_step)
        self.evaluate(self.dataset[2][0], self.dataset[2][1])
        self.save_result()
    
    def compile_model(self):
        self.optimizer = optimizers.get(self.config, self.model.params)
        self.loss = losses.get(self.config.trainer.loss_type)
        self.accuracy = losses.Accuracy(self.config)
        self.regularizer = regularizers.get(self.config)
    
    def fit(self, x=None, y=None, epochs=1, batch_size=1, validation_data=None, is_shuffle=True, is_accuracy=False, save_step=1):
        if x is None or y is None:
            raise ValueError("There is no fitting data.")
        n_train = len(x)
        self.losses = []
        self.losses_reg = []
        if validation_data is not None: 
            self.losses_val = []
            self.losses_val_reg = []
        if is_accuracy: self.accuracies = []
        if validation_data is not None and is_accuracy: self.accuracies_val = []

        logger.info("training start")
        start_time = time.time()
        for epoch in range(1, epochs+1):
            if is_shuffle:
                x, y = shuffle(x, y)
            with tqdm(range(0, n_train, batch_size), desc="[Epoch: {}]".format(epoch)) as pbar:
                for i, ch in enumerate(pbar):
                    self.model.params = self.optimizer(self.model.params, self.model.gradient(self.loss.gradient(self.model(x[i:i+batch_size]), y[i:i+batch_size]), self.regularizer.gradient(self.model.params)))
            y_pred = self.model(x)
            error = self.loss(y_pred, y)
            error_reg = self.regularizer(self.model.params)
            self.losses.append(error)
            self.losses_reg.append(error+error_reg)
            if validation_data is None:
                if not is_accuracy:
                    message = "Epoch:{}  Training loss:{:.5f}".format(epoch, error)
                else:
                    accuracy = self.accuracy(y_pred, y)
                    self.accuracies.append(accuracy)
                    message = "Epoch:{}  Training loss:{:.5f}  Training accuracy:{:.5f}".format(epoch, error, accuracy)
            else:
                y_val_pred = self.model(validation_data[0])
                validation_data = (validation_data[0], validation_data[1], y_val_pred)
                error_val = self.loss(y_val_pred, validation_data[1])
                self.losses_val.append(error_val)
                self.losses_val_reg.append(error_val+error_reg)
                if not is_accuracy:
                    message = "Epoch:{}  Training loss:{:.5f}  Validation loss:{:.5f}".format(epoch, error, error_val)
                else:
                    accuracy = self.accuracy(y_pred, y)
                    self.accuracies.append(accuracy)
                    accuracy_val = self.accuracy(y_val_pred, validation_data[1])
                    self.accuracies_val.append(accuracy_val)
                    message = "Epoch:{}  Training loss:{:.5f}  Validation loss:{:.5f}  Training accuracy:{:.5f}  Validation accuracy:{:.5f}".format(epoch, error, error_val, accuracy, accuracy_val)
            logger.info(message)
            if epoch % save_step == 0:
                self.save_visualize_data(x=x, y=y, y_pred=y_pred, validation_data=validation_data)
        interval = time.time() - start_time
        logger.info("end of training")
        logger.info("time: {}".format(interval))
        logger.info(message)
    
    def evaluate(self, x, y):
        y_pred = self.model(x)
        error = self.loss(y_pred, y)
        if self.config.trainer.is_accuracy:
            accuracy = self.accuracy(y_pred, y)
            message = "Test loss:{}  Test accuracy:{}".format(error, accuracy)
        else:
            message = "Test loss:{}".format(error)
        logger.info(message)
    
    def save_visualize_data(self, x, y, y_pred, validation_data=None):
        self.model.save(self.config.resource.result_visualize_config_path, self.config.resource.result_visualize_model_path)
        self.save_learning_curve(self.config.resource.result_visualize_learning_curve_path)
        if validation_data is not None:
            dataset = {
                "Train": {
                    "Input": x.tolist(),
                    "Output": y.tolist(),
                },
                "Validation": {
                    "Input": validation_data[0].tolist(),
                    "Output": validation_data[1].tolist()
                }
            }
            logger.debug("save dataset to {}".format(self.config.resource.result_visualize_data_path))
            with open(self.config.resource.result_visualize_data_path, "wt") as f:
                json.dump(dataset, f, indent=4)
            dataset = {
                "Train": {
                    "Input": x.tolist(),
                    "Output": y_pred.tolist(),
                },
                "Validation": {
                    "Input": validation_data[0].tolist(),
                    "Output": validation_data[2].tolist()
                }
            }
            logger.debug("save dataset to {}".format(self.config.resource.result_visualize_data_predict_path))
            with open(self.config.resource.result_visualize_data_predict_path, "wt") as f:
                json.dump(dataset, f, indent=4)
        else:
            dataset = {
                "Train": {
                    "Input": x.tolist(),
                    "Output": y.tolist(),
                }
            }
            logger.debug("save dataset predict to {}".format(self.config.resource.result_visualize_data_path))
            with open(self.config.resource.result_visualize_data_path, "wt") as f:
                json.dump(dataset, f, indent=4)
            dataset = {
                "Train": {
                    "Input": x.tolist(),
                    "Output": y_pred.tolist(),
                }
            }
            logger.debug("save dataset to {}".format(self.config.resource.result_visualize_data_predict_path))
            with open(self.config.resource.result_visualize_data_predict_path, "wt") as f:
                json.dump(dataset, f, indent=4)

    
    def save_result(self):
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(self.config.resource.result_train_dir, result_id)
        os.makedirs(result_dir, exist_ok=True)
        self.model.save(os.path.join(result_dir, "parameter.conf"), os.path.join(result_dir, "model.json"))
        self.save_learning_curve(os.path.join(result_dir, "learning_curve.csv"))

    def load_model(self):
        from model import ODENetModel
        model = ODENetModel(self.config)
        model.load(self.config.resource.model_path)
        return model
    
    def load_dataset(self):
        data_path = self.config.resource.data_path
        if os.path.exists(data_path):
            logger.debug("loading data from {}".format(data_path))
            with open(data_path, "rt") as f:
                dataset = json.load(f)
            x_train = dataset.get("Train", {}).get("Input")
            y_train = dataset.get("Train", {}).get("Output")
            x_val = dataset.get("Validation", {}).get("Input")
            y_val = dataset.get("Validation", {}).get("Output")
            x_test = dataset.get("Test", {}).get("Input")
            y_test = dataset.get("Test", {}).get("Output")
            if x_train is None or y_train is None:
                raise TypeError("Dataset does not exists in {}.".format(data_path))
            if len(x_train[0]) != self.config.model.input_dimension:
                raise ValueError("Input dimensions in config and dataset are not equal: {} != {}.".format(self.config.model.input_dimension, len(x_train[0])))
            if len(y_train[0]) != self.config.model.output_dimension:
                raise ValueError("Output dimensions in config and dataset are not equal: {} != {}.".format(self.config.model.output_dimension, len(y_train[0])))
            train = (np.array(x_train, dtype=np.float32), np.array(y_train, dtype=np.float32))
            validation = (np.array(x_val, dtype=np.float32), np.array(y_val, dtype=np.float32))
            test = (np.array(x_test, dtype=np.float32), np.array(y_test, dtype=np.float32))
            return (train, validation, test)
        else:
            raise FileNotFoundError("Dataset file can not loaded!")

    def save_learning_curve(self, file_path):
        e = [i for i in range(len(self.losses))]
        try:
            if self.config.trainer.is_accuracy:
                result_csv = [e, self.losses, self.losses_reg, self.losses_val, self.losses_val_reg, self.accuracies, self.accuracies_val]
                columns = ["epoch", "loss_train", "loss_train_regularizer", "loss_validation", "loss_validation_regularizer", "accuracy_train", "accuracy_validation"]
            else:
                result_csv = [e, self.losses, self.losses_reg, self.losses_val, self.losses_val_reg]
                columns = ["epoch", "loss_train", "loss_train_regularizer", "loss_validation", "loss_validation_regularizer"]
        except AttributeError:
            if self.config.trainer.is_accuracy:
                result_csv = [e, self.losses, self.losses_reg, self.accuracies]
                columns = ["epoch", "loss_train", "loss_train_regularizer", "accuracy_train"]
            else:
                result_csv = [e, self.losses, self.losses_reg]
                columns = ["epoch", "loss_train", "loss_train_regularizer"]
        logger.debug("save learning curve to {}".format(file_path))
        with open(file_path, "wt") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(list(zip(*result_csv)))
