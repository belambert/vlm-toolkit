import sys
from pathlib import Path

from clip_interrogator import Config, Interrogator
from PIL import Image

# clip interrogator:
# https://github.com/pharmapsychotic/clip-interrogator/tree/main

# For the best prompts for Stable Diffusion 1.X use ViT-L-14/openai for clip_model_name
# For Stable Diffusion 2.0 use ViT-H-14/laion2b_s32b_b79k

MODEL = "ViT-L-14/openai"
# MODEL = "ViT-H-14/laion2b_s32b_b79k"


# low memory options
# self.caption_model_name = 'blip-base'
# self.caption_offload = True
# self.clip_offload = True
# self.chunk_size = 1024
# self.flavor_intermediate_count = 1024


def main():
    image_paths = sys.argv[1:]
    config = Config(
        # caption_model_name='blip-base',
        chunk_size=1024,
        flavor_intermediate_count=1024,
        caption_offload=True,
        clip_offload=True,
    )
    ci = Interrogator(config)
    # print the plain caption

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        filename = Path(image_path).name
        caption = ci.generate_caption(image)
        print(f"{caption} ({filename})")
        # then try to get stylistic stuff
        # print(ci.interrogate(image))


if __name__ == "__main__":
    main()
