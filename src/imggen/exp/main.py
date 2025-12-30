import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev", dtype=torch.bfloat16, device_map="mps"
)

prompt = "Turn this cat into a dog"
input_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)
print(input_image)

image = pipe(image=input_image, prompt=prompt).images[0]
print(image)
