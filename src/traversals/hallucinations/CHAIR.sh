



for layer in 16 17 18 19 20 21 22 23 24
do
    for strength in 7 8 9 10 11 12
    do
        python3 CHAIR_llava.py --keep 300 --max_tokens 512 --strength $strength --target_layer $layer
    done
done



for layer in 15 16 17 18 19 20 21 22 23 24
do
    for strength in 3 4 5 6
    do
        python3 CHAIR_instructblip.py --keep 300 --max_tokens 512 --strength $strength --target_layer $layer
    done
done