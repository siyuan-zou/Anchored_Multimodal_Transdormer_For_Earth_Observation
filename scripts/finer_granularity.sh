HYDRA_FULL_ERROR=1 python train_TS.py classification_partition=1.0 logger.name=100E_patch_5_avgg_anchor-s2 load_checkpoint=False model.encoders.aerial.patch_size=5

HYDRA_FULL_ERROR=1 python train_TS.py classification_partition=1.0 logger.name=100E_patch_10_avgg_anchor-s2 load_checkpoint=False model.encoders.aerial.patch_size=10

HYDRA_FULL_ERROR=1 python train_TS.py classification_partition=1.0 logger.name=100E_patch_25_avgg_anchor-s2 load_checkpoint=False model.encoders.aerial.patch_size=25