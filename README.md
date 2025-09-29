# Interaction-Aware Video Narrative Generation (IaVNG)

**Paper:** *Interaction-Aware Video Narrative Generation for Short-Form Gaming Content*  
Ari Yu\*, Sung-Yun Park\*, Sang-Kwang Lee†  
Electronics and Telecommunications Research Institute (ETRI) & University of Science and Technology (UST)  
(*Equal contribution, †Corresponding author*)  

Presented at the **NeurIPS 2025 Workshop on Next-Generation Video Understanding (NeurIPS 2025)**  

---

## 📌 Overview
The rapid growth of short-form video consumption underscores the need for a next-generation paradigm in video generation.  
This repository provides the implementation of **IaVNG**, a two-stage model designed for **League of Legends (LoL)** gameplay videos:

1. **Key Interaction Segment (KIS) Module**  
   - Extracts dense and meaningful interaction segments using Kernel Density Estimation (KDE).  
   - Interaction logs (e.g., champion kills, dragon/baron objectives) are detected via YOLOv5-based models.  

2. **Narrative Generation (NG) Module**  
   - Generates coherent short-form narratives grounded in the selected KIS.  
   - Combines visual features and commentary text via cross-attention and LSTM decoding.  

---

## 🗂 Dataset
- Based on **LoL Champions Korea (LCK)** 2023–2024 Spring and Summer splits.  
- 952 match videos (~570 hours, 1080p 59.94fps).  
- 25,120 timestamped commentary sentences (transcribed & anonymized).  
- Key interaction logs include **CHAMPION_KILL** and **OBJECT_KILL** events.  

---

## 🧪 Experimental Results
- **Interaction Log Detection**: 96.71% Precision, 97.08% Recall, 96.90% F1-score.  
- **Narrative Generation**: Outperforms baselines on METEOR (+1.33) and ROUGE-L (+2.06).  
- Qualitative results show IaVNG produces **context-aware and coherent commentary** for short-form content.  

---

## 🚀 Getting Started
### Installation
```bash
git clone https://github.com/SungyunPark/League-of-legends-event-log-recognition.git
cd League-of-legends-event-log-recognition
pip install -r requirements.txt
