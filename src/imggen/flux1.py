import torch
from diffusers import FluxPipeline


# https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux

# maybe this code?
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux.py

# also see:
# https://huggingface.co/blog/sd3#memory-optimizations-for-sd3

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()  # save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt,
    height=512,
    width=512,
    guidance_scale=3.5,
    num_inference_steps=25,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
image.save("photo5.png")
