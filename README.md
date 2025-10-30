# 🧠 Multi-Class Text Classification: Comparing Word Representations with ML and Neural Models

This repository presents a **comparative study** of multiple word representation techniques and machine learning (ML) / neural network (NN) architectures for **multi-class text classification**.  
We explore how classical models and deep learning methods perform under different textual representations such as **Bag of Words (BoW)**, **TF–IDF**, **GloVe**, and **Skip-gram** embeddings.

---

## 📚 Overview

Text classification is a fundamental task in **Natural Language Processing (NLP)** with applications in sentiment analysis, spam detection, document categorization, and question answering.  
This project evaluates and compares traditional ML and neural approaches across multiple embedding schemes using a **balanced dataset of 340,000 question–answer pairs** across **10 categories**.

---

## 🧩 Dataset

- **Size:** ~340,000 question–answer pairs  
- **Categories:** 10 (e.g., Science, Education, Business, Sports, Politics, etc.)  
- **Split:** 80% training / 20% testing  
- **Preprocessing:**
  - Removed noise (URLs, boilerplate tokens, numbers)
  - Lowercased, tokenized, and lemmatized
  - Filtered stopwords (preserving negations)
  - Added placeholders for dates, percentages, etc.
  - Padded and truncated sequences for NN models

---

## 🔤 Word Representation Methods

| Representation | Description |
|----------------|--------------|
| **Bag of Words (BoW)** | Sparse count vectors representing word frequency |
| **TF–IDF** | Weighted term frequencies capturing informativeness |
| **GloVe** | Pre-trained dense embeddings capturing global co-occurrence |
| **Skip-gram** | Contextual embeddings predicting neighboring words |

---

## ⚙️ Models Implemented

### 🧮 Machine Learning Models
- Naive Bayes  
- Logistic Regression  
- Random Forest  

### 🤖 Neural Network Models
- Deep Neural Network (DNN)  
- Simple RNN  
- GRU / Bidirectional GRU  
- LSTM / Bidirectional LSTM  

Each model was fine-tuned with optimized hyperparameters for fair comparison.  
Bidirectional variants were implemented to capture context from both past and future tokens.

---

## 🔍 Experimental Setup

- **Vectorization:**
  - TF–IDF with `max_features=50000`
  - `ngram_range=(1, 2)`
  - `sublinear_tf=True`
- **Hyperparameter tuning:**
  - Grid search over α (Naive Bayes), C (Logistic Regression), hidden sizes (NNs)
- **Training:**
  - Optimizer: Adam
  - Dropout: 0.3–0.5
  - Early stopping based on validation loss
- **Metrics:**
  - Accuracy  
  - Macro & Weighted F1-score  
  - Confusion Matrix visualization

---

## 📈 Results Summary

### 🧮 Classical ML Models (TF–IDF)
| Model | Accuracy | Macro F1 |
|--------|-----------|-----------|
| **Logistic Regression** | **0.703** | **0.704** |
| Naive Bayes | 0.691 | 0.688 |
| Random Forest | 0.593 | 0.589 |

> ✅ **Logistic Regression + TF–IDF** was the best classical approach.

---

### 🤖 Neural Network Models (Skip-gram)
| Model | Accuracy | Macro F1 |
|--------|-----------|-----------|
| **Bidirectional LSTM** | **0.725** | **0.73** |
| Bidirectional GRU | 0.724 | 0.72 |
| GRU | 0.723 | 0.72 |
| LSTM | 0.723 | 0.72 |

> 🚀 **Bidirectional LSTM with Skip-gram embeddings** achieved the highest overall performance.

---

## 🧠 Key Insights

- **TF–IDF + Logistic Regression** is a strong, interpretable, and efficient baseline.  
- **Bidirectional LSTM + Skip-gram** delivers the best accuracy and F1 through richer contextual modeling.  
- **Random Forest** performs poorly with high-dimensional sparse text vectors.  
- **Bidirectional and gated architectures** significantly improve classification for long sequences.

---

## ⚡ Limitations

- Limited compute and memory restricted batch sizes and tuning scope.  
- Only static embeddings (GloVe, Skip-gram) were used — transformers not explored.  
- Evaluation limited to a single balanced dataset split.

---

## 🚀 Future Work

- Integrate **transformer-based models** (BERT, RoBERTa)  
- Apply **Bayesian optimization** for hyperparameter tuning  
- Use **data augmentation** and **domain adaptation**  
- Explore **ensembles** and **knowledge distillation**  
- Test robustness under **class imbalance** and **domain shift**

---

## 🛠️ Tech Stack

- Python 3.x  
- TensorFlow / Keras  
- Scikit-learn  
- NumPy, Pandas  
- Matplotlib  

---

## 👨‍💻 Authors

- **Nahid Hassan** — *BRAC University, Dhaka, Bangladesh*  
- **Alvee Ishraque** — *BRAC University, Dhaka, Bangladesh*  
- **Tarek Alam Bhuiyan** — *BRAC University, Dhaka, Bangladesh*

---

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{hassan2025multiclass,
  title={Multi-Class Text Classification: A Comparison of Word Representations with ML/NN Models},
  author={Hassan, Nahid and Ishraque, Alvee and Bhuiyan, Tarek Alam},
  year={2025},
  institution={BRAC University},
  note={Department of Computer Science and Engineering}
}
