#!/bin/bash

# This script associates each molecule name stored in ~/scratch/ML_OSC
# to the name it will have ~/scratch/ML_DFT. It writes a file with these
# name associations to each directory in ~/scratch/ML_DFT.

# Parse input
if [[ $# == 1 ]]; then
	rundir="C${1}"
elif [[ $# == 2 ]]; then
	rundir="C${1}/C${1}_C${2}"
else
    echo "Please provide directory index (or indices) as an argument."
	echo 'If a single argument X is given; this script will consider directory C$X/'
	echo 'If two arguments X and Y are given; this script will consider directory C$X/C$X_C$Y/'
    exit 1
fi


XYZ_DIR="/scratch/ngaste/ML_OSC/C/${rundir}"  
DFT_DIR="/scratch/ngaste/ML_DFT/C/${rundir}"  

if [ ! -d "$DFT_DIR" ]; then
	mkdir -p "$DFT_DIR"
fi

namefile="${DFT_DIR}/conf_names.txt"

if [ -f "$namefile" ]; then
	rm "$namefile"
fi

molecule_counts=()

for xyz_file in "$XYZ_DIR"/*.molecule; do
    # Extract molecule name from file name
    base_name=$(basename "$xyz_file" .molecule)
    molecule_name=$(echo "$base_name" | awk -F '.' '{print $3}')

    # Count molecule occurrences
    if [[ -z "${molecule_counts[$molecule_name]}" ]]; then
        molecule_counts[$molecule_name]=0
    fi
    ((molecule_counts[$molecule_name]++))

    # Add _confX suffix if molecule name is repeated
    if (( molecule_counts[$molecule_name] > 1 )); then
        molecule_name="${molecule_name}_conf${molecule_counts[$molecule_name]}"
    fi
	echo "$base_name ---> $molecule_name" >> "$namefile"
done
