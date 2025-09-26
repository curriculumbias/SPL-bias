### Base model
#!/bin/bash

# Base model and result file paths
SAVE_BASE="./checkpoints/base_model_"
RES_BASE="./results/base_model_"
EXT=".bin"
TRE_EXT=".trec"

# Loop to run the command 5 times
for i in {1..5}
do
  CUDA_VISIBLE_DEVICES=0 \
  python train.py \
    -task ranking \
    -model <model> \
    -pretrain <model> \
    -train triples.train.small_reproduce.tsv \
    -dev dev.100.jsonl \
    -max_input 3000000 \
    -save ${SAVE_BASE}${i}${EXT} \
    -qrels qrels.dev.tsv \
    -vocab <model> \
    -res ${RES_BASE}${i}${TRE_EXT} \
    -metric mrr_cut_10 \
    -n_kernels 21 \
    -max_query_len 32 \
    -max_doc_len 221 \
    -epoch 1 \
    -batch_size 16 \
    -lr 3e-6 \
    -eval_every 10000 \
    -n_warmup_steps 160000
done


### Bias
#!/bin/bash

# Base model and result file paths
SAVE_BASE="./checkpoints/base_model_cl_"
RES_BASE="./results/base_model_cl_"
EXT=".bin"
TRE_EXT=".trec"

# Loop to run the command 5 times
for i in {6..10}
do
  CUDA_VISIBLE_DEVICES=0 \
  python train_bias.py \
    -task ranking \
    -model <model> \
    -pretrain <model> \
    -train triples.train.small_reproduce.tsv \
    -dev dev.100.jsonl \
    -max_input 3000000 \
    -save ${SAVE_BASE}${i}${EXT} \
    -qrels qrels.dev.tsv \
    -vocab <model> \
    -res ${RES_BASE}${i}${TRE_EXT} \
    -metric mrr_cut_10 \
    -n_kernels 21 \
    -max_query_len 32 \
    -max_doc_len 221 \
    -epoch 1 \
    -batch_size 16 \
    -lr 3e-6 \
    -eval_every 10000 \
    -n_warmup_steps 160000 \
    -num_buckets <bucket_size> \
    -sigma <sigma>
done
