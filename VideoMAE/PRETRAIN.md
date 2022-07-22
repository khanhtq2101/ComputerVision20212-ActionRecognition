# Pre-training VideoMAE 

-  For example, to pre-train VideoMAE ViT-Base on **HMDB10**, you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/hmdb10_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e800'
  DATA_PATH='YOUR_PATH/hmdb10/train.csv'
  
  python run_mae_pretraining.py \
      --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 16 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --save_ckpt_freq 2 \
        --epochs 1501 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
  ```
  To train from a checkpoint, add argument --resume "PATH_TO_THE_CHECKPOINT".  
  To autoload your saved checkpoint for training, add argument --auto_resume.
- To pre-train VideoMAE ViT-Base on **UCF10** you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/ucf10_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e800'
  DATA_PATH='YOUR_PATH/ucf10/train.csv'
  
  python run_mae_pretraining.py \
      --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 16 \
        --num_frames 16 \
        --sampling_rate 2 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --save_ckpt_freq 2 \
        --epochs 1501 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR}
  ```
  To train from a checkpoint, add argument --resume "PATH_TO_THE_CHECKPOINT".  
  To autoload your saved checkpoint for training, add argument --auto_resume.
