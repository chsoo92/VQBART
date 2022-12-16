# Fine-tuning BART with vector quantization inspired from [VQVAE](https://github.com/MishaLaskin/vqvae))


### 1) Preprocess the data using BPE by following instructions [here](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md)

### 2) To fine tune VQ layer with pretrained BART, run train.sh

```bash
EMBDIM=32             #Codebook vector dimension
MLPS=2                #number of MLP layers before and after VQ layer
BSZ=32                #batch size
EPOCH=30              #numbrt of epochs to train
CODEBOOK=$((512*4))   #number of codebook vectors
GPUS=4                #number of GPUS to use
ARCHTYPE=vqbart       #vqbart for default vq model. gaubart for sqvae variant
CRIT=vq_cross_entropy 
NAME="$EMBDIM"_"$MLPS"_"$CODEBOOK"_"$ARCHTYPE"
ARCH="$ARCHTYPE"_"large"

CUDA_VISIBLE_DEVICES="0,1,2,3" python -O train.py ./data-bin/wikitext-103 \
--mask 0.3 \
--tokens-per-sample 512 \
--total-num-update 500000 \
--max-update 500000 \
--warmup-updates 10000 \
--task denoising \
--save-interval 1 \
--optimizer adam \
--lr-scheduler polynomial_decay \
--lr 0.0004 \
--dropout 0.1 \
--max-tokens 3200 \
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
--mask-random 0.1 \
--permute-sentences 1 \
--insert 0.0 \
--poisson-lambda 3.5 \
--dataset-impl mmap \
--bpe gpt2 \
--num-workers 4 \
--distributed-init-method tcp://localhost:54187 \
--log-file logs_$NAME.txt \
--arch $ARCH \
--criterion $CRIT \
--codebook $CODEBOOK \
--max-epoch $EPOCH \
--emb-dim $EMBDIM \
--MLPLayers $MLPS \
--batch-size $BSZ \
--tensorboard-logdir logs_$NAME \
--save-dir checkpoints/$NAME \
--disable-validation \
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
```

### 3) To run GLUE task with trained model

Follow the instructions [here](https://github.com/facebookresearch/fairseq/edit/main/examples/bart/README.glue.md
) to preprocess for GLUE tasks.

Then, run GLUE.sh with hyperparameters in the table [here](https://github.com/facebookresearch/fairseq/edit/main/examples/bart/README.glue.md
).

### 1) Download the data from GLUE website (https://gluebenchmark.com/tasks) using following commands:
```bash
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
```

### 2) Preprocess GLUE task data (same as RoBERTa):
```bash
./examples/roberta/preprocess_GLUE_tasks.sh glue_data <glue_task_name>
```
`glue_task_name` is one of the following:
`{ALL, QQP, MNLI, QNLI, MRPC, RTE, STS-B, SST-2, CoLA}`
Use `ALL` for preprocessing all the glue tasks.

### 3) Fine-tuning on GLUE task:
Example fine-tuning cmd for `RTE` task
```bash
TOTAL_NUM_UPDATES=2036  # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=61      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=16        # Batch size.
BART_PATH=/path/to/bart/model.pt

CUDA_VISIBLE_DEVICES=0,1 fairseq-train RTE-bin/ \
    --restore-file $BART_PATH \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --arch bart_large \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
```

For each of the GLUE task, you will need to use following cmd-line arguments:

Model | MNLI | QNLI | QQP | RTE | SST-2 | MRPC | CoLA | STS-B
---|---|---|---|---|---|---|---|---
`--num-classes` | 3 | 2 | 2 | 2 | 2 | 2 | 2 | 1
`--lr` | 5e-6 | 1e-5 | 1e-5 | 1e-5 | 5e-6 | 2e-5 | 2e-5 | 2e-5
`bsz` | 128 | 32 | 32 | 32 | 128 | 64 | 64 | 32
`--total-num-update` | 30968 | 33112 | 113272 | 1018 | 5233 | 1148 | 1334 | 1799
`--warmup-updates` | 1858 | 1986 | 6796 | 61 | 314 | 68 | 80 | 107

For `STS-B` additionally add `--regression-target --best-checkpoint-metric loss` and remove `--maximize-best-checkpoint-metric`.

**Note:**

a) `--total-num-updates` is used by `--polynomial_decay` scheduler and is calculated for `--max-epoch=10` and `--batch-size=32/64/128` depending on the task.

b) Above cmd-args and hyperparams are tested on Nvidia `V100` GPU with `32gb` of memory for each task. Depending on the GPU memory resources available to you, you can use increase `--update-freq` and reduce `--batch-size`.

### Inference on GLUE task
After training the model as mentioned in previous step, you can perform inference with checkpoints in `checkpoints/` directory using following python code snippet:

```python
from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    'checkpoints/',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='RTE-bin'
)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)   
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('glue_data/RTE/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2, target = tokens[1], tokens[2], tokens[3]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', float(ncorrect)/float(nsamples))
```
