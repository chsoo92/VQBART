EMBDIM=32
MLPS=2
BSZ=1
FL=0
EPOCH=10
CODEBOOK=$((512*4))
TEMBD=$((256*3))
MODEL="$EMBDIM"_"$MLPS"_"$CODEBOOK"
NAME="$MODEL"_uncond
LR=4.5e-06
GPUS=4

CUDA_VISIBLE_DEVICES="0,1,2,3" python -O train.py ./data-bin/wikitext-103 \
--tokens-per-sample 200 \
--total-num-update 500000 \
--max-update 500000 \
--warmup-updates 10000 \
--task denoising \
--save-interval 1 \
--optimizer adam \
--lr-scheduler polynomial_decay \
--lr $LR \
--dropout 0.1 \
--max-tokens 200 \
--weight-decay 0.01 \
--attention-dropout 0.1 \
--share-all-embeddings \
--clip-norm 0.1 \
--skip-invalid-size-inputs-valid-test \
--log-format json \
--log-interval 50 \
--save-interval-updates 500 \
--keep-interval-updates 1 \
--update-freq 4 \
--seed 4 \
--distributed-world-size $GPUS \
--distributed-port 54187 \
--mask-length span-poisson \
--replace-length 1 \
--encoder-learned-pos \
--decoder-learned-pos \
--rotate 0.0 \
--insert 0.0 \
--dataset-impl mmap \
--bpe gpt2 \
--num-workers 4 \
--distributed-init-method tcp://localhost:54187 \
--log-file logs_$NAME.txt \
--arch uncond_transformer \
--criterion uncond_cross_entropy \
--codebook $CODEBOOK \
--max-epoch $EPOCH \
--emb-dim $EMBDIM \
--MLPLayers $MLPS \
--batch-size $BSZ \
--tensorboard-logdir logs_$NAME \
--save-dir checkpoints/$NAME \
--feature-loss $FL \
--vocab-size $CODEBOOK \
--n-embd $TEMBD \
--restore-file checkpoints/$NAME/checkpoint_1_16500.pt \
--disable-validation

#--fp16 \
#--restore-file checkpoints/8_2_e2e_re/checkpoint_6_16500.pt \
#--no-epoch-checkpoints \
#--e2e \
#--max_valid_steps 100\
#--pretrained-file checkpoints/$MODEL/checkpoint10.pt \
