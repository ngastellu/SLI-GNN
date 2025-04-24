#!/bin/bash

if [[ $# != 3 ]]; then
	echo 'Missing command line args! Need 3. Exiting.'
	exit 1
fi

orig_dir=C$1/C$1_C$2
dest_dir=C$1/C$1_C$3
nfiles=$(ls $orig_dir | wc -l)
n_mv=$(( nfiles / 2 ))
files=$(ls $orig_dir)

echo "Copying half of the files ($n_mv files) from $orig_dir to $dest_dir"

k=0

if [ ! -d $dest_dir ]; then
	mkdir $dest_dir
fi

for ff in ${orig_dir}/*; do
	echo "$k : $ff"
	mv $ff $dest_dir
	k=$(( k + 1 ))
	if [ $k -ge $n_mv ]; then
		break
	fi
done

