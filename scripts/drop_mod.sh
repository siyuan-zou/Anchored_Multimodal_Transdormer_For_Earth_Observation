HYDRA_FULL_ERROR=1 python train_TS.py multimodal.modalities='["s2", "s1-asc"]' classification_partition=1.0 logger.name=100E_drop_aerial_pretrain_avg_anchor-s2 path_cpt=./misc/checkpoints/adapt_TS/contrastive/last-v9.ckpt

HYDRA_FULL_ERROR=1 python train_TS.py multimodal.modalities='["aerial", "s1-asc"]' classification_partition=1.0 logger.name=100E_drop_s2_pretrain_avg_anchor-s2 path_cpt=./misc/checkpoints/adapt_TS/contrastive/last-v9.ckpt

HYDRA_FULL_ERROR=1 python train_TS.py multimodal.modalities='["s2", "aerial"]' classification_partition=1.0 logger.name=100E_drop_s1_pretrain_avg_anchor-s2 path_cpt=./misc/checkpoints/adapt_TS/contrastive/last-v9.ckpt