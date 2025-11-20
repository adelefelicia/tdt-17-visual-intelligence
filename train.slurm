#!/bin/bash
#SBATCH --account=ie-idi
#SBATCH --job-name=odelia_densenet
#SBATCH --time=0-08:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --output=output.out
#SBATCH --error=error.err

#SBATCH --mail-user=adelefe@stud.ntnu.no
#SBATCH --mail-type=ALL

# Go to the directory where you ran sbatch
cd "$SLURM_SUBMIT_DIR"

echo "Running from: $SLURM_SUBMIT_DIR"
echo "Job name:     $SLURM_JOB_NAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"

# 0) Ensure clean environment
module purge

# 1) Load Python module
module load Python/3.10.4-GCCcore-11.3.0

# 2) Activate existing venv
source /cluster/home/adelefe/task-4-env/bin/activate

# 3) Run training script using that env
echo "Path to code: $CODE_PATH"
python $CODE_PATH/train.py

echo " --- Finished running train.py ---"