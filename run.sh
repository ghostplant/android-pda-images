export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=${NGPU:-1}
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EXP_NAME="llama_1B3_debug_${TIMESTAMP}"
CHECKPOINT_PATH="$PWD/checkpoints/$EXP_NAME"
rm -rf "$PWD/checkpoints"
mkdir -p "$CHECKPOINT_PATH"


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 32768
    --num-layers 1 # 8
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    # --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts ${MOE:-2}
    --expert-model-parallel-size ${EP:-1}
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk ${TOP:-1}
    --moe-router-pre-softmax
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type allgather
    # --moe-grouped-gemm
)

if [[ "${MOE:-0}" == "0" ]]; then
    MOE_ARGS=()
fi

DATA_ARGS=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size 50257
    --split 949,50,1gti
)

STEP_BATCH=1

TRAINING_ARGS=(
    --micro-batch-size ${STEP_BATCH}
    --global-batch-size $((WORLD_SIZE * STEP_BATCH)) # 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --optimizer sgd
    # --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP:-1}
    --pipeline-model-parallel-size ${PP:-1}
    --sequence-parallel
    --context-parallel-size ${CP:-1}
    # --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-throughput
    --log-timers-to-tensorboard
    --log-progress
    --log-interval 10
    --save-interval 10000
    --eval-interval 1000
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"}
    )
fi

if false && [[ "${WORLD_SIZE}" == "1" ]]; then
  MASTER_ADDR=localhost MASTER_PORT=27272 WORLD_SIZE=1 RANK=0 python3 pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
else
  torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
fi

