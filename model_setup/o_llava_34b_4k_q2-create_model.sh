#!/usr/bin/bash

# variables
model_name="llava:34b-v1.6-q2_K"

custom_model_name="mf-o-lva-34b-4k-q2"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./o_llava_34b_4k_q2-ModelFile

# List the models to confirm install
ollama list

# How to create the local model: 
################################################

## -- This shell script must be run in Git Bash or equivalent Bash terminal -- !! PowerShell doesn't work !!

# 1. Open a new Git Bash or Command Prompt Terminal
# 2. Spawn poetry shell within the terminal with the command: poetry shell
# 3. cd (change directory) to the folder the shell script is located: cd model_setup
# 4. Make the script executable by running:
#    chmod +x o_llava_34b_4k_q2-create_model.sh
# 5. Run the executable .sh file with the command: ./o_llava_34b_4k_q2-create_model.sh