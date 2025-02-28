# Phi-4 Multimodal Model for Replicate

This is a [Replicate](https://replicate.com) implementation of Microsoft's [Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) model, a powerful multimodal model that can process text, images, and audio.

## Model Description

Phi-4-multimodal-instruct is a lightweight open multimodal foundation model that leverages language, vision, and speech research and datasets used for Phi-3.5 and 4.0 models. The model:

- Processes text, image, and audio inputs, generating text outputs
- Comes with 128K token context length
- Supports multiple languages:
  - **Text**: Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian
  - **Vision**: English
  - **Audio**: English, Chinese, German, French, Italian, Japanese, Spanish, Portuguese

## Usage

You can use this model with the Replicate API or through the web interface.

### Basic text chat

```python
import replicate

output = replicate.run(
    "username/phi-4-multimodal:latest",
    input={
        "text": "How does photosynthesis work?",
        "system_prompt": "You are a helpful science assistant."
    }
)
print(output)
```

### Image understanding

```python
import replicate

output = replicate.run(
    "username/phi-4-multimodal:latest",
    input={
        "text": "What's in this image and what can you tell me about it?",
        "images": ["https://example.com/path/to/image.jpg"]
    }
)
print(output)
```

### Audio processing

```python
import replicate

# Transcription
output = replicate.run(
    "username/phi-4-multimodal:latest",
    input={
        "audio": "https://example.com/path/to/audio.mp3",
        "task": "transcribe"
    }
)
print(output)

# Translation
output = replicate.run(
    "username/phi-4-multimodal:latest",
    input={
        "audio": "https://example.com/path/to/audio.mp3",
        "task": "translate",
        "target_language": "French"
    }
)
print(output)
```

### Multimodal interactions

You can also combine modalities:

```python
import replicate

output = replicate.run(
    "username/phi-4-multimodal:latest",
    input={
        "images": ["https://example.com/path/to/chart.jpg"],
        "audio": "https://example.com/path/to/question.mp3"
    }
)
print(output)
```

## Parameters

- `text` (string): Text input for the model
- `system_prompt` (string): Optional system prompt to guide the model's behavior (default: "You are a helpful assistant.")
- `images` (list of strings): List of image URLs to process (optional)
- `audio` (string): URL to an audio file for speech processing (optional)
- `task` (string): For audio/image tasks, specify task type (e.g., 'transcribe', 'translate', 'summarize', 'describe')
- `target_language` (string): For translation tasks, specify target language
- `max_tokens` (integer): Maximum number of tokens to generate in the response (default: 1000)
- `temperature` (float): Controls randomness in generation (default: 0.7)

## Limitations

As noted in the model card, be aware of these limitations:

- Quality of service varies by language and modality
- Performance is best for English content, with varying levels of degradation for non-English languages
- Speech recognition works best with standard American English accents
- The model may produce inaccurate information (hallucinations)
- Usage should adhere to responsible AI practices

## License

The Phi-4 model is licensed under the MIT license.

## Citations

If you use this model, please cite:

```
@misc{phi4multimodal2025,
  title={Phi-4: A lightweight open multimodal foundation model},
  author={Microsoft},
  year={2025},
  howpublished={\url{https://huggingface.co/microsoft/Phi-4-multimodal-instruct}}
}
``` 