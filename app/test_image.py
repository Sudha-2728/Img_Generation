from app.image_generator import generate_image, load_image_model


TEST_PROMPT = (
    "A dreamy oil painting of a serene beach at dusk, "
    "with a lone figure walking toward the shore, "
    "a full moon rising behind them, "
    "soft waves, warm golden sand, "
    "pastel colors and ethereal lighting."
)


def main():
    print("Loading the image-generation model...")
    pipeline = load_image_model()

    print("Generating the test image...")
    generate_image(
        prompt=TEST_PROMPT,
        pipeline=pipeline,
        output_path="test_image.png",
    )

    print("Image generated successfully!")


if __name__ == "__main__":
    main()
