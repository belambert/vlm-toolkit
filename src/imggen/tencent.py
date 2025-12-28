import torch
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "mps"
print(f"Using device: {device}")

# Load the model
print("Loading HunyuanVideo model...")
pipe = HunyuanVideoPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    torch_dtype=torch.bfloat16
)
pipe.to(device)

# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

# Your prompt
prompt = "A cat wearing sunglasses walking on a beach at sunset, cinematic, high quality"

print(f"Generating video for: {prompt}")

# Generate video
output = pipe(
    prompt=prompt,
    num_frames=49,      # Number of frames to generate
    height=544,         # Video height
    width=960,          # Video width  
    num_inference_steps=30,  # More steps = better quality but slower
    guidance_scale=7.5  # How closely to follow the prompt
)

# Get the frames
frames = output.frames[0]

# Save as MP4
output_path = "output_video.mp4"
export_to_video(frames, output_path, fps=24)

print(f"Video saved to {output_path}")
print(f"Generated {len(frames)} frames")