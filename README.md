
# Content Safety Classifier

## Overview
This project implements an automated content moderation system for social media platforms. The system analyzes post metadata and content to classify each post as Safe, Unsafe, or Neutral, helping maintain community standards and ensure a positive user experience.

## Features
- Automated Content Analysis: Processes text and hashtags to detect potentially harmful content

- Multi-faceted Scoring: Combines toxicity detection, sentiment analysis, and keyword matching

- Structured Classification: Categorizes posts as Safe, Neutral, or Unsafe with detailed justifications

- Comprehensive Reporting: Generates detailed moderation reports with statistics and examples

- Customizable Thresholds: Easily adjust sensitivity levels for different moderation needs

## Requirements
- Python 3.7+

- pandas

- numpy

- NLTK

- vaderSentiment

- Detoxify

- TextBlob

- transformers (for BERT-based models)

## Installation

```
# Clone the repository
git clone https://github.com/namish18/ContentSafetyClassifier.git
cd ContentSafetyClassifier

# Install dependencies
pip install pandas numpy nltk vaderSentiment detoxify textblob transformers

# Run the file
python classify_feed.py
```
