import base64
import os
import asyncio
import requests
from PIL import Image
from typing import Optional

async def ocr(
    file_path: str, 
    api_key: Optional[str] = None, 
    model: str = "Llama-3.2-90B-Vision",
    temperature: float = 0,
    seed: int = 42
) -> str:
    if api_key is None:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("No API key provided. Set TOGETHER_API_KEY environment variable or pass api_key parameter.")

    vision_llm = (
        "meta-llama/Llama-Vision-Free" if model == "free" 
        else f"meta-llama/{model}-Instruct-Turbo"
    )
    
    result = await asyncio.to_thread(
        get_markdown, 
        vision_llm, 
        file_path, 
        api_key,
        temperature,
        seed
    )
    
    return result

def get_markdown(
    vision_llm: str,
    file_path: str,
    api_key: str,
    temperature: float,
    seed: int
) -> str:
    system_prompt = (
        "Extract the movie title from the provided image. "
        "Return only the title without any additional information."
    )
    
    final_image_url = file_path if is_remote_file(file_path) else f"data:image/jpeg;base64,{encode_image(file_path)}"
    
    response = requests.post(
        "https://api.together.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": vision_llm,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": final_image_url}}
                    ]
                }
            ],
            "temperature": temperature,
            "seed": seed
        }
    )
    
    if response.status_code != 200:
        print(f"Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        raise Exception(f"API call failed with status code {response.status_code}")
    
    output = response.json()
    if 'choices' not in output:
        print(f"Unexpected API response: {output}")
        raise Exception("API response does not contain 'choices' key")
        
    return output['choices'][0]['message']['content']

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_remote_file(file_path):
    return file_path.startswith("http://") or file_path.startswith("https://")