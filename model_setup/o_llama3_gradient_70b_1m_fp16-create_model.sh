#!/usr/bin/bash

# variables
model_name="llama3-gradient:70b-instruct-1048k-fp16"

custom_model_name="mf-o-l3g-70b-1m-fp16"

#get the base model
ollama pull $model_name

#create the model file
ollama create $custom_model_name -f ./o_llama3_gradient_70b_1m_fp16-ModelFile

# List the models to confirm install
ollama list

# How to create the local model: 
################################################

## -- This shell script must be run in Git Bash or equivalent Bash terminal -- !! PowerShell doesn't work !!

# 1. Open a new Git Bash or Command Prompt Terminal
# 2. Spawn poetry shell within the terminal with the command: poetry shell
# 3. cd (change directory) to the folder the shell script is located: cd model_setup
# 4. Make the script executable by running:
#    chmod +x o_llama3_gradient_70b_1m_fp16-create_model.sh
# 5. Run the executable .sh file with the command: ./o_llama3_gradient_70b_1m_fp16-create_model.sh
