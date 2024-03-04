#!/bin/bash
#SBATCH -J boundary-100k-2.75                 # DO NOT use '/' in job-name!
#SBATCH -p gpu                      # gpu partition
#SBATCH -c 1                        # no. of CPU cores
#SBATCH -N 1                        # 1 node
#SBATCH -D /scratch                 # local working directory on compute node. DO NOT touch this line!
#SBATCH -o %J-%N.log                # jobid-node.log
#SBATCH -e %J-%N.err                # jobid-node.err
#SBATCH --mail-type=ALL             # email notification
#SBATCH --mail-user=user@mail.com

## list of files needed for the run & run cmd ##

PY_FILE=CalcBoundary.py

## log & err files ##
LOG_FILE=/scratch/${SLURM_JOB_ID}-${SLURM_JOB_NODELIST}.log
ERR_FILE=/scratch/${SLURM_JOB_ID}-${SLURM_JOB_NODELIST}.err
echo "Job submitted from ${SLURM_SUBMIT_DIR} starts at ${SLURM_JOB_NODELIST} since $(date)" >>$LOG_FILE

## unique local working dir under /scratch of the compute node ##
SLURM_COMPUTE_DIR=/scratch/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
mkdir -p $SLURM_COMPUTE_DIR

## sync data from login node to compute node ##
rsync -alu ${SLURM_SUBMIT_HOST}:${SLURM_SUBMIT_DIR}/* $SLURM_COMPUTE_DIR

## run the simulation under compute dir ##
cd $SLURM_COMPUTE_DIR || exit
python3 ${PY_FILE} >run.log
for f in [sr]*/; do
  tail -n 1000 "$f"/perimeter.dat >>perimeter.dat
done

## save complete time ##
echo "Job ends at $(date)" >>$LOG_FILE

## sync data back ##
rsync -alu $LOG_FILE $ERR_FILE $SLURM_COMPUTE_DIR/perimeter.dat ${SLURM_SUBMIT_HOST}:${SLURM_SUBMIT_DIR}
rsync -alu $LOG_FILE $ERR_FILE $SLURM_COMPUTE_DIR/traj_cluster.gif ${SLURM_SUBMIT_HOST}:${SLURM_SUBMIT_DIR}

exit
