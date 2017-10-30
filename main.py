import torch

from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger, save_config

def main(config):
    prepare_dirs_and_logger(config)

    torch.manual_seed(config.random_seed)
    if config.num_gpu > 0:
        torch.cuda.manual_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    if config.src_names is not None:
        config.src_names = config.src_names.split(",")

    if config.load_attributes is not None:
        config.load_attributes = config.load_attributes.split(",")

    data_loader = get_loader(
        data_path, config.split, batch_size, config.input_scale_size, num_workers=config.num_worker, shuffle=do_shuffle, load_attributes=config.load_attributes, rotate_angle=config.rotate_angle, take_log=config.take_log, normalize=config.normalize)

    trainer = Trainer(config, data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
