DATASET_DIR=gpt2_godlang_dataset

mkdir $DATASET_DIR

# for num in {1..100}
for num in {1..10}
do
    echo "Running loop: $num"
    env CUDA_LAUNCH_BLOCKING=1 python3.9 test_gpt2.py --command_batch_size 5 --total_batches 100 | tee $DATASET_DIR/$(date -u "+%Y_%m_%dT%H_%M_%S").txt
done

# python3.9 create_godlang_dataset.py --command_batch_size 5 --total_batches 100 | tee output.txt

# fine tuning on small machines: train and use gpt2 or smaller models to generate commands, then merge its weight into larger models

# only real interactions with the real environment will create new dataset.