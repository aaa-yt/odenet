import os
import json
from datetime import datetime
from logging import getLogger
import numpy as np

from config import Config

logger = getLogger(__name__)

def start(config: Config):
    return ModelAPI(config).start()


class ModelAPI:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = None
    
    def start(self):
        self.model = self.load_model()
        self.dataset = self.load_dataset()
        dataset_pred = self.predict(self.dataset)
        self.save_dataset(dataset_pred)
    
    def predict(self, dataset):
        return tuple((data[0], self.model(data[0])) for data in dataset)
    
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
    
    def save_dataset(self, dataset_pred):
        
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(self.config.resource.result_predict_dir, result_id)
        os.makedirs(result_dir, exist_ok=True)
        data_predict_path = os.path.join(result_dir, "data_predict.json")
        logger.debug("save dataset predict to {}".format(data_predict_path))
        dataset = {
            "Train": {
                "Input": dataset_pred[0][0].tolist(),
                "Output": dataset_pred[0][1].tolist()
            },
            "Validation": {
                "Input": dataset_pred[1][0].tolist(),
                "Output": dataset_pred[1][1].tolist()
            },
            "Test": {
                "Input": dataset_pred[2][0].tolist(),
                "Output": dataset_pred[2][1].tolist()
            }
        }
        with open(data_predict_path, "wt") as f:
            json.dump(dataset, f, indent=4)
