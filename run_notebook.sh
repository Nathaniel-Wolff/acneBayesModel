#!/bin/bash

chmod u+r+x run_notebook.sh

ls -la run_notebook.sh

echo "Ensure that the notebook is in the notebooks directory."

dir_base=/Users/nathaniel/acneBayesModel

read -p "Enter notebook title to open:" notebook_title

target_path="$dir_base/notebooks/$notebook_title"

jupyter notebook $target_path&

echo "Complete."
