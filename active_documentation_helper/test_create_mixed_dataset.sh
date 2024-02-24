mkdir mixed_dataset

# for num in {1..100}
for num in {1..10}
do
    echo "Running loop: $num"
    python3.9 download_and_mix_godlang_dataset.py --command_batch_size 5 --total_batches 100 | tee mixed_dataset/$(date -u "+%Y_%m_%dT%H_%M_%S").txt
done