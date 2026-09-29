

for layer in 16
do
    for denom in 5 10 15 20 25 30 35 40 45 50
    do
        python3 generate_data.py --keep 100 --reps 10 --target_layer $layer --generation sample_seed --denom $denom --temp 0.2 --dataset AG
        python3 generate_data.py --keep 100 --reps 10 --target_layer $layer --generation sample_seed --denom $denom --temp 0.4 --dataset AG
    done
done



