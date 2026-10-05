# Img_Generation
# 🎨 AI Creative Image Studio

AI Creative Image Studio is a Generative AI application that creates images based on user requirements.

Instead of directly sending a simple text prompt to an image-generation model, the application first uses a locally running Llama model through Ollama to understand and enhance the user's requirements.

The enhanced prompt is then passed to a Diffusers-based image generation model to create the final image.

---

## 🚀 Project Workflow

```text
User Input
    ↓
Image Idea
    ↓
Style + Mood + Lighting + Color
    ↓
Llama via Ollama
    ↓
AI Enhanced Image Prompt
    ↓
Diffusers
    ↓
Stable Diffusion
    ↓
Generated Image
