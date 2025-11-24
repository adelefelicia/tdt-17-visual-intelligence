#!/bin/bash
#SBATCH --account=your-account
#SBATCH --job-name=odelia_densenet
#SBATCH --time=0-03:00:00

#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --output=output.out
#SBATCH --error=error.err

#SBATCH --mail-user=your.email@address.com
#SBATCH --mail-type=ALL

# Go to the directory where you ran sbatch
cd "/path/to/repo/location/tdt-17-visual-intelligence"

echo "Running from: $SLURM_SUBMIT_DIR"
echo "Job name:     $SLURM_JOB_NAME"
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NODELIST"

# 0) Ensure clean environment
module purge

# 1) Load Python module
module load Python/3.10.4-GCCcore-11.3.0

# 2) Activate existing venv
source /path/to/venv/location/<venv-name>/bin/activate

# # 3) Run training script using that env with a specified dropout rate
python train.py --dropout 0.0

echo " --- Finished running train.py ---"