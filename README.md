# Text Generation Using GPT-2 and LSTM

## 📌 Project Overview
This project implements **Text Generation** using:
1. **GPT-2 (Pretrained Transformer model)** → Generates text based on a given prompt.
2. **LSTM (Recurrent Neural Network)** → Trained on sample text to predict the next words in a sequence.

## 🚀 Features
- Uses **GPT-2 (Hugging Face)** for high-quality text generation.
- Implements **LSTM-based text generation** for custom datasets.
- Supports **user input prompts** to generate dynamic text.
- Includes **two separate scripts:**
  - `GPT.py` → Uses GPT-2 for text generation.
  - `LSTM.py` → Uses LSTM to generate text from a trained dataset.

## ⚙️ Installation
### **1️⃣ Install Dependencies**
Ensure you have Python installed, then run:
```bash
pip install torch transformers tensorflow keras numpy matplotlib
```

## 📜 Usage
### **Run GPT-2 Model:**
```bash
python GPT.py
```
### **Run LSTM Model:**
```bash
python LSTM.py
```

### **Example Inputs & Outputs**
#### **📝 GPT-2 Input:**
```
Enter a text prompt: The future of artificial intelligence is
```
#### **🎯 GPT-2 Output:**
```
The future of artificial intelligence is evolving rapidly with advancements in deep learning and automation.
```

#### **📝 LSTM Input:**
```
Enter a seed text: Artificial intelligence is
```
#### **🎯 LSTM Output:**
```
Artificial intelligence is transforming industries by enabling automation, decision-making, and problem-solving.
```

## 📂 File Structure
```
📁 text-generation/
├── GPT.py  # GPT-2 text generation script
├── LSTM.py  # LSTM text generation script
├── README.md  # Project documentation
```
👨‍💻 Developed by **Naveen K**