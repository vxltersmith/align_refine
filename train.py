from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate
import argparse


def main(cfg: OmegaConf):
    print(OmegaConf.to_yaml(cfg))
    # Instantiate datasets
    data_module = instantiate(cfg.dataset)
    # Instantiate model
    model = instantiate(cfg.model)
    model.set_optimizer_configs(cfg.optimizer, cfg.lr_scheduler if cfg.use_lr_scheduler else None)
    model.set_ctc_tokenizer(data_module.get_ctc_tokenizer())
    # Load checkpoint
    checkpoint_cfg = cfg.get('checkpointing', None)
    callbacks = [] # list of PL callbacks for trainer
    if checkpoint_cfg is not None:
        if checkpoint_cfg.get('continue_from', None) is not None:
            model.set_checkpoint(checkpoint_cfg.continue_from)
        # Create a ModelCheckpoint callback
        checkpoint_callback = instantiate(checkpoint_cfg.checkpoint_handler)
        callbacks.append(checkpoint_callback)
    # Instantiate trainer and add callbacks
    trainer = instantiate(cfg.trainer)
    trainer.callbacks += callbacks
    # Start training loop
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config_dir", type=str, default='configs', help="Path to the config directory")
    parser.add_argument("--config_name", type=str, default='config', help="Config name")
    args = parser.parse_args()
    
    config_path = args.config_dir
    config_name = args.config_name
    # Initialize Hydra and compose the configuration
    with initialize(config_path=config_path):
        cfg = compose(config_name=config_name)

    main(cfg)