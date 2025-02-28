from cog import BasePredictor, Input, Path
import os
import io
import time
import torch
import requests
import subprocess
import soundfile as sf
from typing import List, Optional, Tuple
from urllib.request import urlopen
from PIL import Image
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    GenerationConfig
)

# Define model path and special tokens
MODEL_PATH = "checkpoints"
USER_PROMPT = '<|user|>'
ASSISTANT_PROMPT = '<|assistant|>'
SYSTEM_PROMPT = '<|system|>'
PROMPT_SUFFIX = '<|end|>'

MODEL_CACHE = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/microsoft/Phi-4-multimodal-instruct/model.tar"

def download_weights(url, dest, isFile=False):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running predictions efficient."""
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Download weights if they don't exist
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        print("Loading Phi-4 multimodal model...")
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
        )
        
        self.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
        print("Model loaded successfully!")

    def download_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load an audio file and return the audio data and sample rate."""
        try:
            audio, samplerate = sf.read(str(audio_path))
            return audio, samplerate
        except Exception as e:
            raise RuntimeError(f"Error processing audio file: {e}")

    def construct_prompt(
        self, 
        text: str, 
        system_prompt: str,
        image_count: int = 0,
        has_audio: bool = False,
        task: Optional[str] = None,
    ) -> str:
        """Construct the prompt with proper formatting based on input types."""
        # Start with system prompt if provided
        formatted_prompt = f"{SYSTEM_PROMPT}{system_prompt}{PROMPT_SUFFIX}" if system_prompt else ""
        
        # Start user section
        formatted_prompt += USER_PROMPT
        
        # Add image placeholders if images are provided
        for i in range(1, image_count + 1):
            formatted_prompt += f"<|image_{i}|>"
        
        # Add audio placeholder if audio is provided
        if has_audio:
            formatted_prompt += "<|audio_1|>"
        
        # Add the task-specific instruction
        if text:
            formatted_prompt += text
        elif task and isinstance(task, list) and len(task) > 0:
            task_value = task[0]  # Take the first task if it's a list
            if task_value == "transcribe":
                formatted_prompt += "Transcribe the audio clip into text."
            elif task_value == "summarize":
                formatted_prompt += "Summarize the content of the audio."
            elif task_value == "describe" and image_count > 0:
                formatted_prompt += "Describe the image in detail."
            else:
                # Default instruction if task is not specified
                if image_count > 0 and has_audio:
                    formatted_prompt += "Process this image and audio."
                elif image_count > 0:
                    formatted_prompt += "What can you tell me about this image?"
                elif has_audio:
                    formatted_prompt += "What can you tell me about this audio?"
        
        # Close user section and start assistant section
        formatted_prompt += f"{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"
        
        return formatted_prompt

    def predict(
        self,
        text: str = Input(description="Input text", default=""),
        system_prompt: str = Input(description="System prompt", default="You are a helpful assistant."),
        images: List[Path] = Input(description="Image URLs", default=None),
        audio: List[Path] = Input(description="Audio files", default=None),
        task: List[str] = Input(description="Available tasks: transcribe, summarize, describe", default=None),
        max_tokens: int = Input(description="Max tokens", default=1000),
        temperature: float = Input(description="Temperature", default=0.7),
    ) -> str:
        """Run a single prediction on the model."""
        # Validate inputs
        if not text and not images and not audio:
            return "Error: You must provide at least one of: text, image URL, or audio URL"
        
        # Download and process media if provided
        image_list = []
        audio_data = None
        sample_rate = None
        
        if images:
            try:
                # Load images using PIL
                for img_path in images:
                    # Open and convert image to RGB mode
                    image = Image.open(str(img_path)).convert('RGB')
                    image_list.append(image)
            except Exception as e:
                return f"Error processing images: {str(e)}"
        
        if audio and len(audio) > 0:
            try:
                # Take the first audio file from the list
                audio_data, sample_rate = self.download_audio(audio[0])
            except Exception as e:
                return f"Error processing audio: {str(e)}"
        
        # Construct the prompt
        prompt = self.construct_prompt(
            text=text,
            system_prompt=system_prompt,
            image_count=len(image_list) if image_list else 0,
            has_audio=audio_data is not None and audio_data.size > 0,
            task=task,
        )
        
        # Prepare the model inputs
        inputs = {
            "text": prompt,
            "return_tensors": "pt"
        }
        
        if image_list:
            inputs["images"] = image_list
        
        if audio_data is not None:
            inputs["audios"] = [(audio_data, sample_rate)]
        
        # Move inputs to GPU
        model_inputs = self.processor(**inputs).to("cuda")
        
        # Update generation config with user parameters
        config_dict = self.generation_config.to_dict()
        config_dict.update({
            'max_new_tokens': max_tokens,
            'temperature': temperature
        })
        gen_config = GenerationConfig(**config_dict)
        
        # Generate response
        with torch.no_grad():
            generate_ids = self.model.generate(
                **model_inputs,
                generation_config=gen_config,
            )
        
        # Extract only the new tokens (the model's response)
        generate_ids = generate_ids[:, model_inputs["input_ids"].shape[1]:]
        
        # Decode the response
        response = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return response 