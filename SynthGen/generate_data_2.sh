
for temp in 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0
do
    python3 generate_data.py --keep 100 --reps 10 --generation sample --topK 50 --topP 0.99 --temp $temp --dataset BoolQ
done
