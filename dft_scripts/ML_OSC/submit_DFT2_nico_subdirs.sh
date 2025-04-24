#!/bin/bash

ndir=$1
ndir2=$2

echo "Running on directory C/C$ndir/C${ndir}_C${ndir2} (<-- this script expects 2 command line args; should of the form CX/CX_Y)"

XYZ_DIR="/scratch/ngaste/ML_OSC/C/C$ndir/C${ndir}_C${ndir2}"  
OUTPUT_DIR="/scratch/ngaste/ML_DFT/C/C$ndir/C${ndir}_C${ndir2}"  
g16_script='run_gaussian_nico.sh'

# Create the output directory 
mkdir -p "$OUTPUT_DIR"
cp "$g16_script" "$OUTPUT_DIR"

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

    # Read geometry from xyz file, skipping the first two lines, removing ^M and $end
    geometry=$(tail -n +3 "$xyz_file" | tr -d '\r' | sed '/^\$end$/d')

    # Create input file for singlet (0 1)
    cat << EOF > "$OUTPUT_DIR/${molecule_name}_singlet.com"
chk=${molecule_name}_singlet.chk 
%CPU=0-23
%GPUCPU=0-3=0-3
%mem=50GB
#p b3lyp/6-31g(d) pop=full density=current gfprint test

Get molecular properties

0 1
$geometry

EOF

    # Create input file for triplet (0 3)
    cat << EOF > "$OUTPUT_DIR/${molecule_name}_triplet.com"
chk=${molecule_name}_triplet.chk
%CPU=0-23
%GPUCPU=0-3=0-3
%mem=50GB
#p b3lyp/6-31g(d) pop=full density=current gfprint test

Get molecular properties

0 3
$geometry

EOF

    cd $OUTPUT_DIR

    # Submit the job for singlet calculation
    sbatch --job-name="${molecule_name}_singlet" --export=ALL,JOB_ID="${molecule_name}_singlet" "$g16_script" "${molecule_name}_singlet"

    # Submit the job for triplet calculation
    sbatch --job-name="${molecule_name}_triplet" --export=ALL,JOB_ID="${molecule_name}_triplet" "$g16_script" "${molecule_name}_triplet"
    
    cd -
done

