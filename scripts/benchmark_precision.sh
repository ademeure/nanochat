#!/bin/bash
# Benchmark BF16 vs FP8 vs FP4 across model sizes and GPU counts
# Results: val loss, throughput (tok/sec), memory usage

set -e

PYTHON=".venv/bin/python"
STEPS=200
EVAL_EVERY=200  # eval only at end
COMMON_ARGS="--core-metric-every=-1 --sample-every=-1 --eval-tokens=2097152 --eval-every=$EVAL_EVERY --num-iterations=$STEPS"

echo "=========================================="
echo "  Precision Benchmark: BF16 vs FP8 vs FP4"
echo "=========================================="
echo ""

run_benchmark() {
    local name="$1"
    local depth="$2"
    local ngpu="$3"
    local precision_flag="$4"
    local batch_size="$5"
    local total_batch="$6"

    echo "--- $name: depth=$depth, ${ngpu}GPU, batch=$batch_size, total_batch=$total_batch $precision_flag ---"

    if [ "$ngpu" -eq 1 ]; then
        CMD="CUDA_VISIBLE_DEVICES=0 $PYTHON -m scripts.base_train"
    else
        CMD="OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=$ngpu -m scripts.base_train"
    fi

    eval $CMD \
        --depth=$depth \
        --device-batch-size=$batch_size \
        --total-batch-size=$total_batch \
        $precision_flag \
        $COMMON_ARGS \
        2>&1 | grep -E "(step 00199|Validation bpb|Peak memory|Total training time|FP[48] training enabled|converted)" | head -5

    echo ""
}

echo "============ d12 (768-dim), 1 GPU ============"
run_benchmark "BF16-d12-1gpu" 12 1 "" 16 65536
run_benchmark "FP8-d12-1gpu"  12 1 "--fp8" 16 65536
run_benchmark "FP4-d12-1gpu"  12 1 "--fp4" 16 65536

echo "============ d12 (768-dim), 8 GPU ============"
run_benchmark "BF16-d12-8gpu" 12 8 "" 16 524288
run_benchmark "FP8-d12-8gpu"  12 8 "--fp8" 16 524288
run_benchmark "FP4-d12-8gpu"  12 8 "--fp4" 16 524288

echo "============ d32 (2048-dim), 1 GPU ============"
run_benchmark "BF16-d32-1gpu" 32 1 "" 8 65536
run_benchmark "FP8-d32-1gpu"  32 1 "--fp8" 8 65536
run_benchmark "FP4-d32-1gpu"  32 1 "--fp4" 8 65536

echo "============ d32 (2048-dim), 8 GPU ============"
run_benchmark "BF16-d32-8gpu" 32 8 "" 8 524288
run_benchmark "FP8-d32-8gpu"  32 8 "--fp8" 8 524288
run_benchmark "FP4-d32-8gpu"  32 8 "--fp4" 8 524288

echo "=========================================="
echo "  Benchmark complete!"
echo "=========================================="
