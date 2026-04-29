#!/bin/bash

echo "Ensure that the notebook is in the notebooks directory."

dir_base=/Users/nathaniel/acneBayesModel

read -p "Enter notebook title to open:" notebook_title

target_path="$dir_base/notebooks/$notebook_title"


open -a "Jupyter Notebook" $target_path

echo "Complete."
