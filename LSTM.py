
# Import required libraries
import tensorflow as tf  # TensorFlow for deep learning
from tensorflow.keras.preprocessing.text import Tokenizer  # Tokenizer to process text
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Pad sequences for equal input size
import numpy as np  # NumPy for numerical operations

# Sample training text (Replace with a larger dataset for better results)
training_text = """Artificial intelligence is shaping the future of technology. 
Machine learning and deep learning are subsets of AI. 
Neural networks play an important role in modern AI development."""

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([training_text])  # Fit tokenizer on given text
total_words = len(tokenizer.word_index) + 1  # Get vocabulary size

# Generate input sequences
input_sequences = []
words = training_text.split()

for i in range(1, len(words)):  
    sequence = words[:i+1]  # Take words progressively to create sequences
    input_sequences.append(tokenizer.texts_to_sequences([' '.join(sequence)])[0])

# Pad sequences to the same length
max_seq_length = max(len(seq) for seq in input_sequences)  # Find longest sequence
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='pre')

# Split input and output labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)  # Convert labels to one-hot encoding

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_seq_length-1),  # Embedding layer
    tf.keras.layers.LSTM(100, return_sequences=True),  # LSTM layer
    tf.keras.layers.LSTM(100),  # Another LSTM layer
    tf.keras.layers.Dense(100, activation='relu'),  # Fully connected layer
    tf.keras.layers.Dense(total_words, activation='softmax')  # Output layer
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (Use a larger dataset for better results)
model.fit(X, y, epochs=100, verbose=1)

def generate_text_lstm(seed_text, next_words=10):
    """
    Generates text using the trained LSTM model.
    
    Parameters:
        seed_text (str): The starting text prompt.
        next_words (int): The number of words to generate.

    Returns:
        str: Generated text.
    """
    for _ in range(next_words):
        sequence = tokenizer.texts_to_sequences([seed_text])[0]  # Convert text to tokens
        sequence = pad_sequences([sequence], maxlen=max_seq_length-1, padding='pre')  # Pad sequence
        predicted = np.argmax(model.predict(sequence), axis=-1)  # Predict next word index
        output_word = ""  

        for word, index in tokenizer.word_index.items():  # Convert index back to word
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word  # Append predicted word to the text
    return seed_text

# Example Usage
if __name__ == "__main__":
    user_prompt = input("Enter a seed text: ")  # Take user input
    generated_text = generate_text_lstm(user_prompt, next_words=20)  # Generate text
    print("\nGenerated Text:\n", generated_text)  # Display output
