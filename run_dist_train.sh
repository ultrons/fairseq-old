python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=4 --node_rank=0 --master_addr="10.138.0.6" \
    --master_port=8085 \
    $(which fairseq-train) /home/jupyter/data/wmt18_en_de_bpej32k \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0005 --min-lr 1e-09 \
    --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 3584 \
    --fp16 \
    --distributed-no-spawn