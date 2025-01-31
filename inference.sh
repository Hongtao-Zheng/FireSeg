CUDA_VISIBLE_DEVICES=0 python inference.py \
    --grounding_ckpt 'checkpoint_best/Train_10_images_t1_attention_transformer_VOC_10layers_ablation_200images/best_checkpoint.pth' \
    --n_each_class 1000 \
    --outdir 'results_mask/' \
    --thread_num 1 \
    --H 1024 \
    --W 1024 \
    --config ./config/VOC/VOC_Sematic_Seg_5Images.yaml
