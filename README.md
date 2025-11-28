# TDT17 Visual Intelligence - Breast MRI Lesion Classification

## Introduction

This project implements a deep learning solution for automated breast lesion classification from multi-parametric MRI scans. The goal is to classify breast lesions into three categories: normal (no lesion), benign, and malignant. This work is part of the [TDT17 Visual Intelligence](https://i.ntnu.no/wiki/-/wiki/Norsk/TDT17+-+Visual+Intelligence) course at the Norwegian University of Science and Technology (NTNU).

The project uses a 3D DenseNet-121 architecture to process multi-sequence MRI volumes and predict lesion malignancy.

## Running the Project

### Environment setup

Both running on a SLURM-based cluster and locally requires a virtual environment with some libraries installed:

1. **Create a vitual environment**:
   ```bash
   python -m venv /path/to/venv/location/<venv-name>
   ```

2. **Install requirements**:
If you are running on a CUDA enabled GPU, ensure a compatible PyTorch version by running these commands before installing the requirements. For IDUN the following is compatible:

   ``` bash
   module load Python/3.10.4-GCCcore-11.3.0
   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

   Install the rest of the requirements (includes PyTorch if you did not install it in the previous step):

   ``` bash
   pip install -r requirements.txt
   ```

### Training with a SLURM 

This project is designed to run on NTNU's IDUN cluster using SLURM for job scheduling.

1. **Update the data root**: Modify DATA_ROOT in `config.py` to point to the filepath of the ODELIA2025 folder

2. **Configure the SLURM script**: Edit `train.slurm` to update these lines with your information:

   ``` bash
   #SBATCH --account=your-account

   #SBATCH --mail-user=your.email@address.com

   cd "/path/to/repo/location/tdt-17-visual-intelligence"

   source /path/to/venv/location/<venv-name>/bin/activate

   python train.py --dropout <your_dropout_probability>
   ```

   You can also optionally modify the GPU specifics in the same file.

3. **Submit the training job**:
   ```bash
   sbatch train.slurm
   ```

4. **Monitor the job**:
   ```bash
   squeue -u your-username
   ```

5. **View output logs**:
   - Standard output: `idun_logs/output_<job-id>.out`
   - Error logs: `idun_logs/error_<job-id>.err`

### Running Training Locally

Requires Python version 3.10.4 or higher.

```bash
source /path/to/venv/location/<venv-name>/bin/activate
python train.py
```

### Training outputs

Training outputs are saved to `logs/train/odelia_YYYY-MM-DD-HH-MM/` including:
- Model checkpoints (`best_model.pth`)
- Training configuration (`config.json`)
- TensorBoard logs (`tensorboard/`)

Monitor training with TensorBoard by running the following command:

```bash
tensorboard --logdir logs/train/odelia_YYYY-MM-DD-HH-MM/tensorboard
```

Then open your browser to `http://localhost:6006`.

### Predicting with a SLURM

Generate predictions on the RSH test dataset without ground truth using a trained model.

1. **Update paths for data and model**: Modify the following three paths in predict_rsh.py to point to the correct folders and files

   ``` python
   data_root = "/path/to/dataset/ODELIA2025/data/RSH/data_unilateral"
   split_csv_path = "/path/to/dataset/ODELIA2025/data/RSH/metadata_unilateral/split.csv"
   checkpoint_path = "/path/to/your/best_model.pth"
   ```

2. **Configure the SLURM script OR run directly**: Modify predict.slurm in the same way you did train.slurm previously.

3. **Submit the training job**:
   ```bash
   sbatch predict.slurm
   ```

5. **View output logs**:
   - Standard output: `idun_logs/output_<job-id>.out`
   - Error logs: `idun_logs/error_<job-id>.err`

### Running prediction Locally

Requires Python version 3.10.4 or higher.

```bash
source /path/to/venv/location/<venv-name>/bin/activate
python predict_rsh.py
```

## Dataset

This project uses the **ODELIA 2025** dataset, which contains breast MRI scans from multiple medical institutions. Read more about the dataset [here](https://arxiv.org/abs/2506.00474).

### Data Splits

Two splitting strategies are supported (configured in `config.py`):
- **CSV mode**: Uses the `Split` column from CSV files for train/val/test division (60/40 split)
- **Institution mode (recommended)**: Train on CAM+RUMC, validate on MHA+UKA (cross-institution generalization) (80/20 split)

## Strategy and Methodology

### Model Architecture

The project uses a **3D DenseNet-121** backbone (from MONAI) as the feature extractor. It uses early stopping and weight decay to reduce overfitting.

### Data Processing

**Pre-processing**:
- Intensity normalization (z-score)
- Center cropping/padding to fixed shape (256×256×32)

**Data Augmentation** (training only):
- Random rotations
- Random intensity shifts
- Cropping to fixed shape (256×256×32) with random center

### Evaluation Metrics

The model is evaluated using challenge-specific metrics:
- **AUC** (Area Under ROC Curve)
- **Sensitivity at 90% Specificity**
- **Specificity at 90% Sensitivity**
- **Composite Score**: Weighted combination of the above metrics

The composite score is used for model selection and early stopping.
