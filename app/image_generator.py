import torch
from diffusers import StableDiffusionPipeline


MODEL_ID = "runwayml/stable-diffusion-v1-5"
IMAGE_SIZE = 512
INFERENCE_STEPS = 20


def load_image_model():
    """Load Stable Diffusion and place it on the available device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )
    pipeline = pipeline.to(device)

    if device == "cuda":
        pipeline.enable_attention_slicing()

    return pipeline


def generate_image(prompt, pipeline, output_path="generated_image.png"):
    """Generate one 512x512 image, save it as PNG, and return the PIL image."""
    if not prompt or not prompt.strip():
        raise ValueError("The image prompt cannot be empty.")

    with torch.inference_mode():
        result = pipeline(
            prompt=prompt,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=INFERENCE_STEPS,
            num_images_per_prompt=1,
        )

    image = result.images[0]
    image.save(output_path, format="PNG")
    return image
