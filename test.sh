#!/bin/bash


source ~/.bashrc
conda init
conda activate boltz-1x
which python
#--output-dir ./test_no_rmsd \

python run_intermediate_coords_extraction.py examples/a-secretase_tmpl.yaml \
       --save-intermediate-coords \
       --detailed-analysis --validate-integrity \
       --sampling-steps 100 --save-every 1 \
       --verbose --create-animation

exit
python -m boltz.main predict test_no_rmsd/boltz_results_ligand/processed/structures/ligand.npz \
       --out_dir test_no_rmsd --save_intermediate_coords --intermediate_output_format pdb \
       --sampling_steps 100 --save-every 1 
