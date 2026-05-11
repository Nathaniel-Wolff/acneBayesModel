#!/bin/bash


#hardcoding environment source, script sources for now

venv_dir="/Users/nathaniel/acneBayesModel/venv"
script_location="/Users/nathaniel/acneBayesModel/src/acne_model/model_script.py"

if [ -d "$venv_dir" ]; then 
	echo "Environment exists. Activation about to start..."
	source "$venv_dir/bin/activate"
else
	echo "Environment "$venv_dir" not found. Create one?"

fi


#consolidating needed config
#piping data into python script
#fix the full path issue

read -p "Type the name of the data, the other, and the other, separated by commas" user_input; echo "$user_input" | python3 "$script_location"

echo "Done"


