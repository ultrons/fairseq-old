#HOST_PORT= # one of the hosts used by the job
RANK=$1
LOCAL_RANK=0 # the local rank of this process, from 0 to 7 in case of 8 GPUs per mac
WORLD_SIZE=32 #number of GPUs to use
python train.py /home/sivaibhav/data/wmt18_en_de_bpej32k_btdata \
--clip-norm 0.0 -a transformer_vaswani_wmt_en_de_big \
--lr 0.0005 --source-lang en --target-lang de \
--label-smoothing 0.1 --upsample-primary 16 \
--attention-dropout 0.1 --dropout 0.3 --max-tokens 3584 \
--no-progress-bar --log-interval 100 --weight-decay 0.0 \
--criterion label_smoothed_cross_entropy --fp16 \
--max-update 100000 --seed 3 --save-interval-updates 16000 \
--share-all-embeddings --optimizer adam --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 \
--warmup-updates 4000 --no-save --min-lr 1e-09 \
--distributed-port 8085 --distributed-world-size ${WORLD_SIZE} \
--distributed-init-method 'tcp://10.138.0.17:8085' --distributed-rank $RANK \
--device-id $LOCAL_RANK
