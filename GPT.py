import torch  # Import PyTorch for deep learning
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # Import GPT-2 model and tokenizer

# Load pre-trained GPT-2 tokenizer
print("Loading GPT-2 Tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # Tokenizer converts text to model-readable format

# Load pre-trained GPT-2 model
print("Loading GPT-2 Model...")
model = GPT2LMHeadModel.from_pretrained("gpt2")  # Load the GPT-2 model


def generate_text(prompt, max_length=100):
    """
    Generates text using the GPT-2 model.
    
    Parameters:
        prompt (str): The input text prompt.
        max_length (int): The maximum number of words in the generated text.

    Returns:
        str: Generated text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # Convert input text to token IDs
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,  # Generate only one sequence
        temperature=0.7,  # Controls randomness (lower = more predictable, higher = more creative)
        top_k=50,  # Consider only top 50 words at each step
        top_p=0.95,  # Nucleus sampling - picks from top words until probability sum reaches 95%
        repetition_penalty=1.2,  # Reduces repetition in generated text
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)  # Convert generated tokens back to text


# Example Usage
if __name__ == "__main__":
    print("\n=== GPT-2 Text Generator ===")
    user_prompt = input("Enter a text prompt: ")  # Take user input
    print("\nGenerating text...\n")
    generated_text = generate_text(user_prompt, max_length=150)  # Generate text with max length of 150 tokens
    print("Generated Text:\n", generated_text)  # Display output
