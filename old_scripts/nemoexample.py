import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

# Define the model configuration
cfg = OmegaConf.create({
    "model": {
        "train_ds": {"manifest_filepath": "train_manifest.json", "batch_size": 32, "shuffle": True},
        "validation_ds": {"manifest_filepath": "val_manifest.json", "batch_size": 32, "shuffle": False},
        "test_ds": {"manifest_filepath": "test_manifest.json", "batch_size": 32, "shuffle": False},
        "tokenizer": {
            "dir": "tokenizer_dir",
            "type": "bpe",
        },
        "preprocessor": {
            "cls": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "params": {
                "sample_rate": 16000,
                "window_size": 0.02,
                "window_stride": 0.01,
                "window": "hann",
                "normalize": "per_feature",
            },
        },
        "encoder": {
            "cls": "nemo.collections.asr.modules.TransformerEncoder",
            "params": {
                "num_layers": 12,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "ffn_dim": 2048,
                "dropout": 0.1,
                "embedding_dropout": 0.1,
                "conv_subsampling_factor": 4,
                "conv_subsampling_layers": 2,
            },
        },
        "decoder": {
            "cls": "nemo.collections.asr.modules.CTCDecoder",
            "params": {
                "feat_in": 768,
                "num_classes": 1024,
            },
        },
        "optim": {
            "name": "adamw",
            "lr": 0.001,
            "weight_decay": 0.01,
            "sched": {
                "name": "WarmupAnnealing",
                "warmup_steps": 2000,
                "min_lr": 1e-5,
            },
        },
    },
})

# # Instantiate the model
# model = ASRModel(cfg=cfg.model)

# # Create a trainer
# trainer = Trainer(gpus=1, max_epochs=10, precision=16)

# # Train the model
# trainer.fit(model)
