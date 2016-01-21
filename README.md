Learning to Execute
===================

This software allows to train a Recurrent Neural Network (RNN) with Long-Short 
Term Memory (LSTM) units on short snippets of python code. 
The Network is trained to predict the output of the generated programs.



Execution
=========

Please install Torch 7 http://torch.ch/ with the cunn package. Moreover, our 
software requires an NVIDIA GPU.

To execute the program, call:

    torch main.lua

This program starts training the LSTM and displays intermediate results. main.lua can be
executed with the following options:

    torch main.lua -gpuidx 1 -target_length 6 -target_nesting 3

- gpuidx: chooses a GPU for the program
- target_length: is a maximum number of digits in every number generated in test 
    programs in a test dataset.
- target_nesting: is the depth of the nesting in the generated programs in the test dataset.

Moreover, the command

    torch data.lua

verifies that training data is correct by evaluating 1000 samples with a python 
interpreter (python2.7 is required).


More information about the scientific work is provided at 
http://arxiv.org/abs/1410.4615

This software is located at
https://github.com/wojciechz/learning_to_execute 
