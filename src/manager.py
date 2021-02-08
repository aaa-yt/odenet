import argparse
from logging import getLogger

from config import Config
from lib.logger import setup_logger

logger = getLogger(__name__)

CMD_LIST = ['train', 'predict', 'visualize']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", help="What to do 'train' or 'predict' or 'visualize'.", choices=CMD_LIST)
    return parser

def setup(config: Config):
    config.resource.create_directories()
    config.load_parameter(config.resource.config_path)
    setup_logger(config.resource.main_log_path)

def start():
    parser = create_parser()
    args = parser.parse_args()
    config = Config()
    setup(config)

    if args.cmd == 'train':
        import trainer
        return trainer.start(config)
    elif args.cmd == 'predict':
        import model_api
        return model_api.start(config)
    elif args.cmd == 'visualize':
        from visualize import app_manager
        return app_manager.start(config)

