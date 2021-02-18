import os
import configparser
from logging import getLogger

logger = getLogger(__name__)

def _project_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    def __init__(self):
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.trainer = TrainerConfig()
    
    def load_parameter(self, config_path):
        if os.path.exists(config_path):
            logger.debug("loading parameter from {}".format(config_path))
            config_parser = configparser.ConfigParser()
            config_parser.read(config_path, encoding="utf-8")
            config_model = config_parser["MODEL"]
            if config_model.get("Input_dimension") is not None: self.model.input_dimension = int(config_model.get("Input_dimension"))
            if config_model.get("Output_dimension") is not None: self.model.output_dimension = int(config_model.get("Output_dimension"))
            if config_model.get("Maximum_time") is not None: self.model.maximum_time = float(config_model.get("Maximum_time"))
            if config_model.get("Weights_division") is not None: self.model.weights_division = int(config_model.get("Weights_division"))
            if config_model.get("Function_type") is not None: self.model.function_type = config_model.get("Function_type")
            if config_model.get("Initializer_type") is not None: self.model.initializer_type = config_model.get("Initializer_type")
            if config_model.get("Initializer_parameter") is not None: self.model.initializer_parameter = float(config_model.get("Initializer_parameter"))
            if config_model.get("Solver_type") is not None: self.model.solver_type = config_model.get("Solver_type")
            config_trainer = config_parser["TRAINER"]
            if config_trainer.get("Loss_type") is not None: self.trainer.loss_type = config_trainer.get("Loss_type")
            if config_trainer.get("Optimizer_type") is not None: self.trainer.optimizer_type = config_trainer.get("Optimizer_type")
            if config_trainer.get("Learning_rate") is not None: self.trainer.learning_rate = float(config_trainer.get("Learning_rate"))
            if config_trainer.get("Momentum") is not None: self.trainer.momentum = float(config_trainer.get("Momentum"))
            if config_trainer.get("Decay") is not None: self.trainer.decay = float(config_trainer.get("Decay"))
            if config_trainer.get("Decay2") is not None: self.trainer.decay2 = float(config_trainer.get("Decay2"))
            if config_trainer.get("Regularizer_type") is not None: self.trainer.regularizer_type = config_trainer.get("Regularizer_type")
            if config_trainer.get("Regularizer_rate") is not None: self.trainer.regularizer_rate = float(config_trainer.get("Regularizer_rate"))
            if config_trainer.get("Epoch") is not None: self.trainer.epoch = int(config_trainer.get("Epoch"))
            if config_trainer.get("Batch_size") is not None: self.trainer.batch_size = int(config_trainer.get("Batch_size"))
            if config_trainer.get("Is_accuracy") is not None: self.trainer.is_accuracy = bool(int(config_trainer.get("Is_accuracy")))
            if config_trainer.get("Save_step") is not None: self.trainer.save_step = int(config_trainer.get("Save_step"))
            if config_trainer.get("Save_point") is not None: self.trainer.save_point = [int(s.strip("[] ")) for s in config_trainer.get("Save_point").split(",")]
    
    def save_parameter(self, config_path):
        logger.debug("save parameter to {}".format(config_path))
        config_parser = configparser.ConfigParser()
        config_parser["MODEL"] = {
            "Input_dimension": self.model.input_dimension,
            "Output_dimension": self.model.output_dimension,
            "Maximum_time": self.model.maximum_time,
            "Weights_division": self.model.weights_division,
            "Function_type": self.model.function_type,
            "Initializer_type": self.model.initializer_type,
            "Initializer_parameter": self.model.initializer_parameter,
            "Solver_type": self.model.solver_type
        }
        config_parser["TRAINER"] = {
            "Loss_type": self.trainer.loss_type,
            "Optimizer_type": self.trainer.optimizer_type,
            "Learning_rate": self.trainer.learning_rate,
            "Momentum": self.trainer.momentum,
            "Decay": self.trainer.decay,
            "Decay2": self.trainer.decay2,
            "Regularizer_type": self.trainer.regularizer_type,
            "Regularizer_rate": self.trainer.regularizer_rate,
            "Epoch": self.trainer.epoch,
            "Batch_size": self.trainer.batch_size,
            "Is_accuracy": self.trainer.is_accuracy,
            "Save_step": self.trainer.save_step,
            "Save_point": self.trainer.save_point
        }
        with open(config_path, "wt") as f:
            config_parser.write(f)



class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.config_dir = os.environ.get("CONFIG_DIR", os.path.join(self.project_dir, "config"))
        self.config_path = os.path.join(self.config_dir, "parameter.conf")
        self.data_dir = os.environ.get("DATA_DIR", os.path.join(self.project_dir, "data"))
        self.data_processed_dir = os.path.join(self.data_dir, "processed")
        self.data_path = os.path.join(self.data_processed_dir, "data.json")
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.project_dir, "model"))
        self.model_path = os.path.join(self.model_dir, "model.json")
        self.result_dir = os.path.join(self.project_dir, "result")
        self.result_train_dir = os.path.join(self.result_dir, "train")
        self.result_predict_dir = os.path.join(self.result_dir, "predict")
        self.result_visualize_dir = os.path.join(self.result_dir, "visualize")
        self.result_visualize_model_path = os.path.join(self.result_visualize_dir, "model.json")
        self.result_visualize_config_path = os.path.join(self.result_visualize_dir, "parameter.conf")
        self.result_visualize_learning_curve_path = os.path.join(self.result_visualize_dir, "learning_curve.csv")
        self.result_visualize_data_path = os.path.join(self.result_visualize_dir, "data.json")
        self.result_visualize_data_predict_path = os.path.join(self.result_visualize_dir, "data_predict.json")
    
    def create_directories(self):
        dirs = [self.project_dir, self.config_dir, self.data_dir, self.data_processed_dir, self.log_dir, self.model_dir, self.result_dir, self.result_train_dir, self.result_predict_dir, self.result_visualize_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)


class ModelConfig:
    def __init__(self):
        self.input_dimension = 1
        self.output_dimension = 1
        self.maximum_time = 1.
        self.weights_division = 100
        self.function_type = "sigmoid"
        self.initializer_type = "zero"
        self.initializer_parameter = 1.
        self.solver_type = "euler"

class TrainerConfig:
    def __init__(self):
        self.loss_type = "mse"
        self.optimizer_type = "sgd"
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.decay = 0.99
        self.decay2 = 0.999
        self.regularizer_type = "None"
        self.regularizer_rate = 0.
        self.epoch = 1
        self.batch_size = 1
        self.is_accuracy = 0
        self.save_step = 1
        self.save_point = [1]