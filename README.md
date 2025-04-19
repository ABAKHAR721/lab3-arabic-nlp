
# 📝 What I Learned (Synthesis)
This lab helped me deepen my understanding of NLP and deep learning.
I learned how to collect and prepare real-world Arabic datasets, build sequence models using PyTorch, and fine-tune large-scale language models like GPT-2.
I also practiced evaluation with metrics like BLEU and MSE, and successfully pushed my fine-tuned model and dataset to the Hugging Face Hub.


# 🧠 Lab 3 – Arabic Text Classification & Generation with Deep Learning
This project explores both sequence modeling and text generation using deep learning applied to Arabic language data. It is part of a university lab assignment on NLP with PyTorch and Transformers.

---

## 📚 Project Structure

The lab is divided into two major parts:

---

## 🧩 Part 1: Sequence Modeling with RNNs

### ✅ Objectives:
- Scrape Arabic text articles using BeautifulSoup
- Preprocess the data (cleaning, tokenization, stopword removal)
- Score texts for relevance using SBERT semantic similarity
- Train four types of recurrent models:
  - RNN
  - GRU
  - LSTM
  - Bidirectional LSTM
- Evaluate using:
  - **MSE** (Mean Squared Error)
  - **R² Score**
  - **BLEU Score** (for text prediction similarity)

### 🧪 Results:

| Model      | MSE ↓   | R² ↑   | BLEU ↑  |
|------------|--------|--------|--------|
| RNN        | 0.6279   | -1.6907  | 0.0    |
| GRU        | 1.5393   | -5.5962  | 0.0    |
| LSTM       | 5.7165   | -23.4969  | 0.0    |
| BiLSTM     | 2.0246 | -7.6761 | 0.0 |

---

## 🤖 Part 2: Fine-Tuned GPT-2 for Arabic Generation

We fine-tuned the [Arabic GPT-2 model](https://huggingface.co/aubmindlab/aragpt2-base) on a custom dataset of scraped Arabic news content.

The fine-tuned model is publicly available on Hugging Face:

👉 **[🟢 GPT2 Arabic Fine-Tuned – Hugging Face](https://huggingface.co/ABAKHAR721/gpt2-arabic-finetuned-lab3)**  
👉 The dataset used for training is included in the model repository under the `/dataset` folder.

### 🧪 Inference Example

```python
from transformers import pipeline

generator = pipeline("text-generation", model="ABAKHAR721/gpt2-arabic-finetuned-lab3")
output = generator("أهمية التعليم في المستقبل", max_length=100, do_sample=True)
print(output[0]['generated_text'])
```



## 🧑‍💻 How to Run the Project

### ⚙️ Environment:
- Python 3.11+
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- Google Colab GPU (recommended)

### 📦 Installation:
```bash
pip install transformers sentence-transformers datasets nltk beautifulsoup4
```
Then open the notebook (Lab3_Arabic_NLP.ipynb) and run all cells step-by-step.

## 👨‍🎓 Author
Name: ABAKHAR Abdessamad
Lab: Deep Learning Lab 3 – Text Classification & Generation
