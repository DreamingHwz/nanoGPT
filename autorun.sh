#!/bin/bash

# Array of dataset sizes to test
# Smaller steps near 0.4M, larger steps beyond
sizes=(
    10000
    20000
    30000
    40000
    50000
    60000
    70000
    80000
    90000
    100000
    110000
    120000
    130000
    140000
    150000
)

echo "Starting dataset size sweep..."
echo "Testing ${#sizes[@]} sizes"

for size in "${sizes[@]}"; do
    size_mb=$((size / 1000000))
    size_k=$((size / 1000))
    
    echo ""
    echo "======================================"
    echo "Testing size: ${size_k}K tokens (${size_mb}.${size:$((${#size}-6)):1}M)"
    echo "======================================"
    
    # Prepare data
    echo "[1/3] Preparing data with max_tokens=$size"
    python data/poems/prepare.py --max_tokens $size --train_split 0.9
    if [ $? -ne 0 ]; then
        echo "❌ Data preparation failed for size $size"
        continue
    fi
    
    # Copy checkpoint
    echo "[2/3] Copying Shakespeare checkpoint"
    rm -f ./out-poems/ckpt.pt
    cp ./out-shakespeare/ckpt.pt ./out-poems/
    if [ $? -ne 0 ]; then
        echo "❌ Checkpoint copy failed"
        continue
    fi
    
    # Train
    echo "[3/3] Fine-tuning on poems"
    WANDB_DISABLED=true python train.py config/finetune_poem.py > ./out-poems/train.log 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Training failed for size $size"
        continue
    fi
    
    # Save results
    echo "✓ Completed size $size"
    mv ./out-poems ./out-poems-${size_k}k
    mkdir ./out-poems
    
done

echo ""
echo "======================================"
echo "Sweep complete!"
echo "Results saved in out-poems-*k directories"
echo "======================================"