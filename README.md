# Fine-tuning BART with vector quantization inspired from [VQVAE](https://github.com/MishaLaskin/vqvae))

This repository is built on top of [Fairseq](https://github.com/facebookresearch/fairseq)

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
```

### 3) To run GLUE task with trained model:

Follow the instructions [here](https://github.com/facebookresearch/fairseq/edit/main/examples/bart/README.glue.md
) to preprocess for GLUE tasks.

Then, run GLUE.sh with hyperparameters in the table [here](https://github.com/facebookresearch/fairseq/edit/main/examples/bart/README.glue.md
).

### 4) After finetuning a model, to train a codebook sampling transformer run train_transformer.sh:

```bash
EMBDIM=32             #Codebook vector dimension
MLPS=2                #number of MLP layers before and after VQ layer
BSZ=1                 #batch size
EPOCH=10              #numbrt of epochs to train
CODEBOOK=$((512*4))   #number of codebook vectors
TEMBD=$((256*3))      #transformer embedding vector dimension
MODEL="$EMBDIM"_"$MLPS"_"$CODEBOOK"
NAME="$MODEL"_uncond
LR=4.5e-06           #learning rate
GPUS=4               #number of GPUS to use

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
--vocab-size $CODEBOOK \
--n-embd $TEMBD \
--disable-validation
```

### 5) Inference on GLUE task
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
