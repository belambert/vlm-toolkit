from diffusers import StableDiffusionXLPipeline


def main():
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0"
    )
    img = pipeline(prompt="bluejay", num_inference_steps=1).images[0]
    img.save("img.jpg")


if __name__ == "__main__":
    main()
