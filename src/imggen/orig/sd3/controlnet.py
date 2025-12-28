import torch
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image


def main():
    print("loading model...")
    controlnet = SD3ControlNetModel.from_pretrained(
        "InstantX/SD3-Controlnet-Canny", torch_dtype=torch.float16
    )

    print("loading pipe...")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    control_image = load_image(
        "https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg"
    )
    prompt = "A girl holding a sign that says InstantX"
    image = pipe(
        prompt, control_image=control_image, controlnet_conditioning_scale=0.7
    ).images[0]
    image.save("sd3.png")


if __name__ == "__main__":
    main()
