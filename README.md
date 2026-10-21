# BAIMAX: Body Analysis & Intelligent Monitor AI Expert

A Python proof-of-concept for an AI-assisted child growth monitoring decision-support system, built as part of a **PKM-AI 2026** research proposal at BINUS University (team of five).

BAIMAX targets two problems in Indonesian child health: late detection of **stunting** and **obesity**, and low parental health literacy. This repository contains a command-line prototype that simulates the core logic of the system described in the proposal.

> **Status:** research prototype. The logic is rule-based and statistical (no trained ML models). It has been exercised on simulated data only and is **not a medical device**.

## Modules

| Module | What it does |
|---|---|
| **1. Smart Growth & Posture** | Converts pixel measurements to centimetres using a reference object, evaluates accuracy with MAPE, and computes spine tilt angle from shoulder/hip keypoints to flag possible posture abnormalities. |
| **2. Predictive Analytics** | Fits a linear regression (`y = a + bx`) on historical weight data, projects 6 months ahead, classifies each projection against SD thresholds, and raises an early warning with a recommendation for both under- and overweight trends. |
| **3. Parenting Assistant (NLP)** | Whole-word keyword intent classification, a small QnA knowledge base, lexicon-based sentiment analysis with negation handling, empathetic responses, and a confusion-matrix evaluator (accuracy, precision, recall, F1). |
| **4. Privacy-First Anonymization** | Strips name and photo before data leaves the device; only height, weight, and age are sent. |

An optional function, `analyze_image_mediapipe`, shows how MediaPipe Pose keypoints could feed Module 1 (**experimental, untested**).

## Run it

Requires Python 3.9+ and no external dependencies for the core prototype.

```bash
python baimax_revisi.py
```

Pick option `5` to run the default scenario from the proposal.

### Example (default scenario)

```
y = 16.67 + 1.25x | Month 4 = 21.67 kg -> Risiko berat badan lebih
Month 6: 24.17 kg -> Di atas +2 SD
Intent: ['Masalah Gizi'] (keywords: susah makan, sayur)
Sentiment: Butuh Dukungan (negatives: lelah, bingung)
```

## Design decisions

- **Explainable by design:** regression, thresholds, and lexicons are transparent, which suits a screening tool where parents and health workers need to understand why a warning fired.
- **Privacy by architecture:** raw images and identifiers stay local; only anonymized statistics are transmitted.
- **Input validation:** every module validates input and fails with clear messages instead of crashing.
- **Bug fixes found during review:** switched from substring to whole-word matching (avoids "keberatan" triggering "berat"), added negation handling ("tidak lelah"), and made early warning bidirectional.

## Limitations

- SD thresholds in `SD_TABLE_DEMO` are **placeholders** and must be replaced with official values from Permenkes No. 2/2020.
- Weight-for-age is used for the demo; obesity screening for ages 5-18 should use BMI-for-age.
- Linear regression over 3 data points is illustrative, not clinically meaningful.
- The 10° spine-angle threshold is an assumption pending clinical validation.
- The knowledge base is sample text, not a vetted medical source.
- Sentiment analysis is lexicon-based and limited to a small Indonesian word list.

## Roadmap

- Integrate real MediaPipe pose estimation with camera input and skeleton overlay
- Load official Kemenkes Z-score tables and compute proper Z-scores
- Replace the sample knowledge base with a retrieval system over vetted sources
- Validate height accuracy against manual measurement (target MAPE < 5%)
- Evaluate sentiment on a labelled dataset using the built-in confusion matrix

## Tech

Python, MediaPipe (optional), linear regression, NLP (tokenization, intent classification, lexicon sentiment), MAPE evaluation, data anonymization.

## Team

Developed by a five-member Cyber Security team at BINUS University.
