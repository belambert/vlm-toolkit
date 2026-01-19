import torch
from diffusers import DiffusionPipeline


def main():
    # switch to "mps" for apple devices
    pipe = DiffusionPipeline.from_pretrained(
        "zai-org/GLM-Image", dtype=torch.bfloat16, device_map="mps"
    )

    # prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    # image = pipe(prompt).images[0]


if __name__ == "__main__":
    main()
