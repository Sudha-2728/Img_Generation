import logging
from io import BytesIO

import streamlit as st
import torch

from image_generator import generate_image, load_image_model
from utils.prompt_generator import generate_image_prompt


# Page settings
st.set_page_config(
    page_title="AI Creative Image Studio",
    page_icon="🎨",
)


@st.cache_resource
def get_image_pipeline():
    """Load the diffusion model once and reuse it across Streamlit reruns."""
    return load_image_model()


# Title
st.title("🎨 AI Creative Image Studio")
st.write("Turn your ideas into beautiful AI-generated images.")


# User idea
st.subheader("📝 Describe Your Idea")
user_input = st.text_area(
    "What image do you want to create?",
    placeholder="Example: A young girl walking through a magical forest.",
    height=150,
)


# Image style
st.subheader("🎨 Choose Image Style")
style = st.selectbox(
    "Select a style",
    [
        "Realistic",
        "Anime",
        "Oil Painting",
        "Watercolor",
        "Digital Art",
        "3D Art",
        "Cartoon",
        "Fantasy Art",
        "Cyberpunk",
        "Pencil Sketch",
        "Comic Art",
        "Pixel Art",
        "Cinematic",
        "Minimalist",
    ],
)


# Image mood
st.subheader("😊 Choose Image Mood")
mood = st.selectbox(
    "Select a mood",
    [
        "Happy",
        "Peaceful",
        "Dreamy",
        "Mysterious",
        "Dramatic",
        "Dark",
        "Epic",
        "Romantic",
        "Futuristic",
        "Energetic",
    ],
)


# Image lighting
st.subheader("💡 Choose Lighting")
lighting = st.selectbox(
    "Select lighting",
    [
        "Natural Light",
        "Sunlight",
        "Sunset",
        "Sunrise",
        "Golden Hour",
        "Moonlight",
        "Soft Light",
        "Dramatic Lighting",
        "Cinematic Lighting",
        "Neon Lighting",
    ],
)


# Color style
st.subheader("🌈 Choose Color Style")
color_style = st.selectbox(
    "Select color style",
    [
        "Natural Colors",
        "Warm Colors",
        "Cool Colors",
        "Pastel Colors",
        "Vibrant Colors",
        "Neon Colors",
        "Dark Colors",
        "Earthy Colors",
        "Black and White",
    ],
)


# Additional user requirements
st.subheader("✨ Optional Additional Requirements")
additional_requirements = st.text_area(
    "Add any extra details you want in the image",
    placeholder="Example: Add mountains in the background and make the character wear a red dress.",
    height=120,
)


# Generate image prompt
if st.button("✨ Generate Image Prompt"):
    if user_input.strip():
        try:
            st.session_state["final_prompt"] = generate_image_prompt(
                user_idea=user_input,
                style=style,
                mood=mood,
                lighting=lighting,
                color_style=color_style,
                additional_requirements=additional_requirements,
            )
            st.session_state.pop("generated_image", None)
        except Exception as exc:
            st.session_state.pop("final_prompt", None)
            logging.exception("Failed to generate the image prompt.")
            st.error(f"Prompt generation failed: {exc}")
    else:
        st.warning("Please describe the image you want.")


# Show the Llama prompt and generate an image from that exact prompt.
final_prompt = st.session_state.get("final_prompt", "")
if final_prompt:
    st.subheader("🧠 AI Generated Prompt")
    st.write(final_prompt)

    st.subheader("🖼️ Generate Image")
    if st.button("Generate Image"):
        if not final_prompt.strip():
            st.warning("Generate an image prompt first.")
        else:
            try:
                with st.spinner("Loading the model and generating your image..."):
                    pipeline = get_image_pipeline()
                    image = generate_image(final_prompt, pipeline)
                st.session_state["generated_image"] = image
            except (torch.cuda.OutOfMemoryError, MemoryError):
                st.error("⚠️ Not enough memory to generate the image. Try closing other applications.")
            except Exception:
                logging.exception("Failed to load the model or generate the image.")
                st.error("⚠️ Image generation failed. Check the model download and try again.")

    generated_image = st.session_state.get("generated_image")
    if generated_image is not None:
        st.subheader("🖼️ Generated Image")
        st.image(generated_image, use_container_width=True)

        image_file = BytesIO()
        generated_image.save(image_file, format="PNG")
        st.download_button(
            label="Download Image",
            data=image_file.getvalue(),
            file_name="generated_image.png",
            mime="image/png",
        )