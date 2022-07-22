# Fine-tuning VideoMAE

-  To fine-tune VideoMAE ViT-Base on **HMDB10**, you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/hmdb10_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e1500/eval_lr_5e-4_epoch_100'
  DATA_PATH='YOUR_PATH/hmdb10'
  MODEL_PATH='YOUR_PATH/hmdb10_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e1500/checkpoint-runtime.pth'
  
  python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set HMDB10 \
    --nb_classes 10 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 2 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100  \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3  
  ```

- To fine-tune VideoMAE ViT-Base on **UCF10**, you can run

  ```bash
  OUTPUT_DIR='YOUR_PATH/ucf10_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1500/eval_lr_1e-3_epoch_100'
  DATA_PATH='YOUR_PATH/ucf10'
  MODEL_PATH='YOUR_PATH/ucf10_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1500/checkpoint-runtime.pth'
  
  python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set HMDB10 \
    --nb_classes 10 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 2 \
    --num_frames 16 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 100  \
    --dist_eval \
    --test_num_segment 2 \
    --test_num_crop 3  
  ```


