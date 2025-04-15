import os
import torch
import yaml
import argparse
from PIL import Image
from tqdm import tqdm
import numpy as np

# Import project components
from diffusers import AutoencoderKL
from models.clip import CLIPTextEncoder
from models.diffusion import Diffusion
from schedulers.ddpm import DDPMScheduler
from utils.helpers import set_seed, rescale
from utils.time_embedding import embed_a_timestep # Use embed_a_timestep for single steps

def load_config(config_path):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        exit(1)

def generate(args, config):
    """Generates an image based on a prompt using a trained model."""
    # --- 1. Setup ---
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Models ---
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(config['model']['vae_id']).to(device)
    vae.requires_grad_(False)
    vae.eval()

    print("Loading CLIP Text Encoder...")
    text_encoder = CLIPTextEncoder().to(device) # Uses clip_id from config
    # text_encoder.eval() # Done internally

    print(f"Loading Diffusion Model checkpoint from: {args.checkpoint}")
    diffusion_model = Diffusion(
        h_dim=config['model']['diffusion_hidden_dim'],
        n_head=config['model']['diffusion_num_heads']
    ).to(device)
    try:
        # Map location ensures model loads correctly even if trained on different device
        diffusion_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
    diffusion_model.eval()

    print("Models loaded successfully.")

    # --- 3. Prepare for Generation ---
    noise_scheduler = DDPMScheduler(
        random_generator=None, # Generator handled per-call or within scheduler
        train_timesteps=config['scheduler']['train_timesteps'],
        diffusion_beta_start=config['scheduler']['beta_start'],
        diffusion_beta_end=config['scheduler']['beta_end']
    )

    # Latent dimensions
    latents_height = config['data']['image_height'] // 8 # VAE downsampling factor
    latents_width = config['data']['image_width'] // 8

    # Prepare prompt embedding
    full_prompt = f"A photo of {args.prompt}" # Add prefix like in training
    print(f"Using prompt: '{full_prompt}'")
    with torch.no_grad():
        text_embeddings = text_encoder([full_prompt]).to(device)

    # Set inference steps
    noise_scheduler.set_steps(args.num_inference_steps)
    print(f"Using {args.num_inference_steps} inference steps.")

    # Prepare initial noise
    latent_shape = (1, config['model']['latent_channels'], latents_height, latents_width)
    rng_generator = torch.Generator(device=device)
    if args.seed is not None:
        rng_generator.manual_seed(args.seed)
    else:
        rng_generator.seed() # Use random seed if none provided

    noisy_latents = torch.randn(latent_shape, generator=rng_generator, device=device)

    # --- 4. Denoising Loop ---
    timesteps = noise_scheduler.schedule_timesteps # Get the timesteps AFTER set_steps

    print("Starting denoising loop...")
    for t in tqdm(timesteps):
        latent_model_input = noisy_latents # In Stable Diffusion, CFG would combine conditional/unconditional here

        # Get time embedding for the current step t
        time_embedding = embed_a_timestep(t, config['model']['diffusion_time_emb_dim']).to(device)
        # embed_a_timestep expects a scalar, ensure t is scalar if needed (it should be from scheduler)
        # Need to unsqueeze for batch dimension if embed_a_timestep doesn't handle it
        if time_embedding.ndim == 1:
            time_embedding = time_embedding.unsqueeze(0)

        # Predict noise
        with torch.no_grad():
            noise_pred = diffusion_model(
                latent_model_input,
                text_embeddings,
                time_embedding
            )

        # Compute previous noisy sample x_t -> x_{t-1}
        noisy_latents = noise_scheduler.step(
            t,              # Current timestep t
            noisy_latents,  # Latents at t (x_t)
            noise_pred      # Predicted noise (epsilon_theta)
        )

    print("Denoising complete.")

    # --- 5. Decode Latents ---
    # Scale latents before decoding
    final_latents = noisy_latents / config['model']['vae_scale_factor']

    print("Decoding latents...")
    with torch.no_grad():
        # VAE decode expects batch dim: [1, C, H, W]
        decoded_image_tensor = vae.decode(final_latents).sample

    # --- 6. Post-process and Save ---
    # Rescale from [-1, 1] to [0, 255] and convert to image format
    image_output = rescale(decoded_image_tensor, (-1, 1), (0, 255), clamp=True)
    image_output = image_output.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy() # B C H W -> B H W C
    generated_image = Image.fromarray(image_output[0]) # Get the first image in the batch

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generated_image.save(args.output)
    print(f"Image saved to: {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Emoji Image from Text Prompt")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file used during training.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained diffusion model checkpoint (.pth file).")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt describing the desired emoji.")
    parser.add_argument("--output", type=str, default="generated_emoji.png",
                        help="Path to save the generated image.")
    parser.add_argument("--num-inference-steps", type=int, default=100,
                        help="Number of denoising steps.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use ('cuda' or 'cpu').")

    args = parser.parse_args()
    config = load_config(args.config) # Load config using the provided path

    # Pass both args and config to the generate function
    generate(args, config)