import argparse
import os

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.utils import initialize_config


def main(config, resume):
    torch.manual_seed(config["seed"])  # for both CPU and GPU
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"]
    )

    valid_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        num_workers=1,
        batch_size=1, 
    )

    model = initialize_config(config["model"])

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"]),
    )

    loss_function = initialize_config(config["loss_function"])
    loss_function2 = initialize_config(config["loss_function2"]) 

    trainer_class = initialize_config(config["trainer"], pass_args=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        loss_function2=loss_function2,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()
    
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    parser = argparse.ArgumentParser(description="Wave-U-Net for Speech Enhancement")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    
    
    args = parser.parse_args()

   
    configuration = json.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)
