import yaml
import os

class Config:
    def __init__(self):
        conf_path = "./ptsne/config.yaml"
        with open(conf_path) as f:
            conf_dict = yaml.safe_load(f)
        #print(os.getcwd(), conf_path, os.path.exists("config.yaml"))
        #print(conf_dict["device"])
        self.dev = conf_dict["device"]
        self.seed = conf_dict["seed"]
        self.save_dir_path = conf_dict["save_dir_path"]
        self.epochs_to_save_after = conf_dict["epochs_to_save_after"]

        self.optimization_conf = conf_dict["optimization"]
        self.training_params = conf_dict["training"]
