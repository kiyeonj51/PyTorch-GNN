#!/bin/bash

# Generate symbolic data for all tasks

# download bAbI task data generation code
git clone https://github.com/facebook/bAbI-tasks.git ../data/babi/bAbI-tasks

cd ../data/babi/bAbI-tasks
. ~/torch/install/bin/torch-activate
luarocks make babitasks-scm-1.rockspec
mv lua/babi .
cd ..

for fold in {1..2}; do
    echo ================ Generating fold $fold =====================
    echo

    cd bAbI-tasks

    mkdir symbolic_$fold
    mkdir symbolic_$fold/train
    mkdir symbolic_$fold/test
#
    echo
    echo Generating 1000 training and 1000 test examples for each bAbI task.
    echo This will take a while...
    echo
#
    # for i in `seq 1 20`; do
    for i in {4,15,16,18,19}; do
        ./babi-tasks $i 1000 --symbolic true > symbolic_$fold/train/${i}.txt
        ./babi-tasks $i 1000 --symbolic true > symbolic_$fold/test/${i}.txt
    done

    # fix q18 data
    python ../../../ggnn/fix_q18.py symbolic_$fold/train/18.txt
    python ../../../ggnn/fix_q18.py symbolic_$fold/test/18.txt

    # back down
    cd ..

    for i in {4,15,16,18,19}; do
        python ../../ggnn/sym_prep.py $i bAbI-tasks/symbolic_$fold processed_$fold
    done

    # RNN data

    mkdir processed_$fold/rnn
    mkdir processed_$fold/rnn/train
    mkdir processed_$fold/rnn/test

    for i in {4,15,16,18,19}; do
        python ../../ggnn/rnn_prep.py processed_$fold/train/${i}_graphs.txt processed_$fold/rnn/train/${i}_tokens.txt --mode graph --nval 50
        python ../../ggnn/rnn_prep.py processed_$fold/rnn/train/${i}_tokens.txt processed_$fold/rnn/train/${i}_rnn.txt --mode rnn
        python ../../ggnn/rnn_prep.py processed_$fold/rnn/train/${i}_tokens.txt.val processed_$fold/rnn/train/${i}_rnn.txt.val --mode rnn --dict processed_$fold/rnn/train/${i}_rnn.txt.dict

        python ../../ggnn/rnn_prep.py processed_$fold/test/${i}_graphs.txt processed_$fold/rnn/test/${i}_tokens.txt --mode graph
        python ../../ggnn/rnn_prep.py processed_$fold/rnn/test/${i}_tokens.txt processed_$fold/rnn/test/${i}_rnn.txt --dict processed_$fold/rnn/train/${i}_rnn.txt.dict --mode rnn
    done
done

rm -rf bAbI-tasks