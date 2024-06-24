#!/usr/bin/bash

# variables
model_name="wizardlm2:7b-q6_K"

custom_model_name="mf-o-wizlm2-7b-32k-q6"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./o_wizardlm2_7b_32k_q6-ModelFile

# List the models to confirm install
ollama list

# How to create the local model: 
################################################

## -- This shell script must be run in Git Bash or equivalent Bash terminal -- !! PowerShell doesn't work !!

# 1. Open a new Git Bash or Command Prompt Terminal
# 2. Spawn poetry shell within the terminal with the command: poetry shell
# 3. cd (change directory) to the folder the shell script is located: cd model_setup
# 4. Make the script executable by running:
#    chmod +x o_wizardlm2_7b_32k_q6-create_model.sh
# 5. Run the executable .sh file with the command: ./o_wizardlm2_7b_32k_q6-create_model.sh
