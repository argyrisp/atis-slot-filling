# ATIS slot filling

Slot filling, NLP, based on ATIS dataset using LSTM and RNN.

This directory contains:
1. ATIS dataset as "atis.fold{x}.pkl" x = {0, 1, 2, 3, 4},
2. Source code for the models used for training/evaluating {SimpleRNN, LSTM_model, Improved_model}
3. Code for evaluation on metrics {accuracy.py}
4. Presentation as "ATIS_slot_filling-RNN_presentation.pdf"
5. Conference paper as "ATIS_slot_filling_paper.pdf"


To run the model(s) use terminal, cd to project directory (Semantic) and type:

//python SimpleRNN.py// for Simple RNN model (yields ~92% score)
//python LTSM_model.py// for LSTM model (yields ~93% score)
//python Improved_model.py// for Improved Conv1D model (yields ~95% score)

All models are ran for 30 epochs and by default they train-evaluate on "atis.fold0" data.
To train-evaluate on different data you need to set s['fold'] as 1, 2, ... manually from dictionary initialization of s, which is located at line 20 for each model on their respective .py source code.

PS1: models LSTM, Improved require more than 10 minutes to run, whilst SimpleRNN takes around 5 minutes to complete, we recommend checking and running SimpleRNN.py.
PS2: when ran, there will be plenty of warnings, just wait and it will run fine.
