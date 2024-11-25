import base64
import gradio as gr # type: ignore
from main import ocr
import asyncio
import os
from PIL import Image

async def process_image(
    image, 
    api_key, 
    model_type="Llama-3.2-90B-Vision",
    temperature=0,
    seed=42
):
    try:
        if image is None:
            return "Please upload an image"
            
        # Save the temporary image
        temp_path = "temp_image.jpg"
        if isinstance(image, str):  # If it's an example image path
            Image.open(image).save(temp_path)
        else:
            image.save(temp_path)
        
        # Process with OCR
        result = await ocr(
            file_path=temp_path,
            api_key=api_key,
            model=model_type,
            temperature=temperature,
            seed=seed
        )
        
        # Clean up
        os.remove(temp_path)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

css = """
<style>
    .markdown-output {
        height: 650px;
        overflow-y: auto;
        padding: 1rem;
        background: var(--background-fill-primary);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color-primary);
    }
</style>
"""

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(css)
    
    gr.Markdown(
        """
        # 📷 Llama OCR
        Convert images to structured text using Llama Vision models
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Image"
            )
            api_key = gr.Textbox(
                type="password",
                label="Together AI API Key",
                placeholder="Enter your API key here..."
            )
            
            model_type = gr.Radio(
                choices=["Llama-3.2-90B-Vision", "Llama-3.2-11B-Vision", "free"],
                label="Model",
                value="Llama-3.2-90B-Vision"
            )
            
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0,
                    step=0.1,
                    label="Temperature (0 for consistent output, higher for more variety)"
                )
                seed = gr.Number(
                    value=42,
                    label="Seed (same seed = same output)",
                    precision=0
                )
            
            explore_btn = gr.Button("Explore Output (Try different seed)", variant="secondary")
            lock_btn = gr.Button("Lock This Output", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Markdown(
                label="OCR Result",
                elem_classes="markdown-output"
            )
            current_seed = gr.Number(value=42, visible=False)  # Hidden field to store locked seed
    
    def generate_random_seed():
        import random
        return random.randint(1, 10000)
    
    # Handle exploration (random seed)
    explore_btn.click(
        fn=lambda img, key, model, temp: asyncio.run(process_image(
            img, key, model, temp, generate_random_seed()
        )),
        inputs=[input_image, api_key, model_type, temperature],
        outputs=output_text
    )
    
    # Handle locking current output
    lock_btn.click(
        fn=lambda img, key, model, temp, seed: asyncio.run(process_image(
            img, key, model, temp, seed
        )),
        inputs=[input_image, api_key, model_type, temperature, seed],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(share=True) 