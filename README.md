# 🤖 BAIMAX – Body Analysis and Intelligent Monitor AI Expert

An AI-based decision support system **concept** for child growth monitoring and healthy-lifestyle assistance, designed to help detect **stunting**, **obesity**, and **posture problems** early.

> 🎓 **Final project** (PKM-AI proposal) at **[BINUS University]**, Cyber Security program, 2026.
> ⚠️ This is a **research proposal and system design**. The repository contains the project write-up and a small demo of one algorithm, not a deployed application.

## 🎯 Problem

- Stunting prevalence in Indonesia is about **21.6%**, and overweight/obesity among school-age children is about **21.5%** (Ministry of Health surveys, 2023).
- Manual growth measurement is prone to human error and detects problems late.
- Many parents have limited knowledge of how to recognize growth risks.

## 💡 Proposed Solution

BAIMAX combines three modules:

| Module | Technique | Purpose |
|---|---|---|
| **Smart Growth & Posture Analysis** | Computer Vision, pose estimation (MediaPipe) | Estimate height from a camera image and check spine angle / posture |
| **Predictive Analytics & Recommendation** | Linear Regression | Project growth trends 6 months ahead and trigger an Early Warning System |
| **Parenting Assistant** | NLP (information retrieval, lexicon-based sentiment analysis) | 24/7 Q&A on child nutrition and a daily journal that detects parents' emotional state |

**Privacy-first design:** photos and videos are processed on the phone and never uploaded. Only anonymized statistics (height, weight, age) are sent to the server.

## 🏗️ System Flow

```
Camera image ─► Pose estimation (33 keypoints) ─► Height + posture analysis ─┐
                                                                             ├─► Decision Support ─► Early warning
Growth history ─► Linear Regression ─► 6-month projection ───────────────────┘        & recommendations
Parent question / journal ─► NLP assistant ─► Answer from nutrition knowledge base
```

## 🛠️ Planned Tech Stack

Python · MediaPipe · TensorFlow / Scikit-Learn · Hugging Face Transformers · Kemenkes anthropometric standards (Permenkes No. 2/2020)

## 📐 Evaluation Plan

- Height accuracy compared with conventional measurement using **MAPE**
- Nutritional-status classification at a 95% confidence level
- Sentiment analysis validated with a **confusion matrix**

## ▶️ Demo: Growth Prediction

A small, dependency-free script that illustrates the Linear Regression early-warning idea with dummy data:

```bash
python demo/growth_prediction_demo.py
```

```
Model      : y = 16.67 + 1.25x
Prediction : month 4 -> 21.67 kg
Status     : EARLY WARNING - projected weight is above the threshold
```

> The data and threshold in the demo are dummy values for illustration only.

## 🧠 Skills Demonstrated

AI system design · Computer vision & NLP concepts · Predictive analytics · Privacy-by-design · Technical research writing · Teamwork (team of 5)
