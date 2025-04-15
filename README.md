# Emoji Diffusion Model

This project implements and trains a Denoising Diffusion Probabilistic Model (DDPM) conditioned on text prompts (using CLIP) to generate small emoji-like images (32x32). The implementation is based on common Stable Diffusion components like a UNet, VAE, and CLIP text encoder.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd emoji-diffusion
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Prepare Data:**
    - Place your image files in the `data/images/` directory.
    - Ensure your `data/metadata.csv` file has columns like `file_name` (relative path within `images/`) and `prompt`.

## Configuration

Training hyperparameters, model IDs, and paths are primarily defined in `config.yaml` located in the project root.

## Training

Navigate to the `src` directory and run the training script:

```bash
cd src
```

**Using default configuration:**

```bash
python train.py
```

**Specifying a different configuration file:**

```bash
python train.py --config ../configs/experiment_2.yaml
```

**Overriding specific parameters from the command line:**

You can override parameters defined in the YAML file directly via command-line arguments. The script uses the YAML file specified by `--config` (or the default `../config.yaml`) as the base, and any command-line argument provided will take precedence.

Example: Use the default config but override epochs, learning rate, and batch size.

```bash
python train.py --num-epochs 50 --learning-rate 5e-5 --batch-size 16
```

Example: Use a specific config file and override the output directory.

```bash
python train.py --config ../configs/low_lr.yaml --output-dir ../results/low_lr_run
```

## Inference / Image Generation

After training, you can generate images using the `src/generate.py` script.

**Prerequisites:**

- A trained model checkpoint file (e.g., `outputs/models/diffusion_model_emoji_epoch18.pth`).
- The `config.yaml` file that was used during training (or one with compatible model parameters).

**Usage:**

Navigate to the `src` directory:

```bash
cd src
```

Run the script with the required arguments:

```bash
python generate.py \
    --config ../config.yaml \
    --checkpoint ../outputs/models/diffusion_model_emoji_epoch18.pth \
    --prompt "a happy blob emoji with sunglasses, green skin" \
    --output ../generated_images/happy_blob.png \
    --num-inference-steps 150 \
    --seed 1234
```

**Arguments:**

- `--config`: (Required) Path to the YAML configuration file (used to get model parameters like dimensions, VAE ID, CLIP ID).
- `--checkpoint`: (Required) Path to the specific `.pth` checkpoint file of the trained diffusion model.
- `--prompt`: (Required) The text description for the image you want to generate.
- `--output`: (Optional) Path where the generated PNG image will be saved (default: `generated_emoji.png` in the current directory). Make sure the directory exists or the script will create it.
- `--num-inference-steps`: (Optional) Number of denoising steps (default: 100). More steps might improve quality but take longer.
- `--seed`: (Optional) A random seed for reproducible results.
- `--device`: (Optional) The device to run inference on (`cuda` or `cpu`, default: `cuda`).

```

```
