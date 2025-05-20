import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from detoxify import Detoxify
from textblob import TextBlob
import json

# Load dataset
df = pd.read_csv('social_feed_metadata.csv')

# Basic preprocessing
df['post_text'] = df['post_text'].fillna('')  # Handle missing text
df['hashtags'] = df['hashtags'].fillna('')  # Handle missing hashtags

def analyze_content(row):
    text = row['post_text'] + ' ' + row['hashtags']
    
    # VADER sentiment analysis
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    # Detoxify toxicity analysis
    toxicity_scores = Detoxify('original').predict(text)
    
    # TextBlob for additional sentiment metrics
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Define unsafe keywords
    unsafe_keywords = ['kill', 'hate', 'nsfw', 'violence', 'attack', 'threat']
    contains_unsafe = any(keyword in text.lower() for keyword in unsafe_keywords)
    
    # Calculate final toxicity score (weighted combination)
    toxicity_score = (
        0.5 * toxicity_scores['toxicity'] + 
        0.3 * (1 - (sentiment_scores['compound'] + 1) / 2) +  # Convert to 0-1 scale
        0.2 * (1 if contains_unsafe else 0)
    )
    
    return {
        'vader_compound': sentiment_scores['compound'],
        'toxicity': toxicity_scores['toxicity'],
        'severe_toxicity': toxicity_scores['severe_toxicity'],
        'insult': toxicity_scores['insult'],
        'threat': toxicity_scores['threat'],
        'identity_attack': toxicity_scores['identity_attack'],
        'polarity': polarity,
        'subjectivity': subjectivity,
        'contains_unsafe_keywords': contains_unsafe,
        'toxicity_score': toxicity_score
    }

def classify_content(analysis):
    # Define thresholds
    if analysis['toxicity_score'] > 0.7:
        label = 'Unsafe'
    elif analysis['toxicity_score'] > 0.3:
        label = 'Neutral'
    else:
        label = 'Safe'
    
    # Generate reason for classification
    reasons = []
    if analysis['toxicity'] > 0.7:
        reasons.append("High toxicity detected")
    if analysis['severe_toxicity'] > 0.5:
        reasons.append("Severe toxicity detected")
    if analysis['threat'] > 0.5:
        reasons.append("Threatening content detected")
    if analysis['insult'] > 0.5:
        reasons.append("Insulting content detected")
    if analysis['identity_attack'] > 0.5:
        reasons.append("Identity-based attack detected")
    if analysis['contains_unsafe_keywords']:
        reasons.append("Unsafe keywords detected")
    if analysis['vader_compound'] < -0.5:
        reasons.append("Highly negative sentiment")
        
    if not reasons:
        if label == 'Neutral':
            reasons.append("Borderline content")
        else:
            reasons.append("No issues detected")
    
    return {
        'final_label': label,
        'reason': "; ".join(reasons)
    }

# Apply analysis to each post
results = []
for _, row in df.iterrows():
    analysis = analyze_content(row)
    classification = classify_content(analysis)
    
    results.append({
        'post_id': row['post_id'],
        'post_text': row['post_text'],
        'platform': row['platform'],
        'hashtags': row['hashtags'],
        'timestamp': row['timestamp'],
        'likes': row['likes'],
        'comments': row['comments'],
        'toxicity_score': analysis['toxicity_score'],
        'final_label': classification['final_label'],
        'reason': classification['reason']
    })

# Create output dataframe
output_df = pd.DataFrame(results)
output_df.to_csv('moderated_feed.csv', index=False)

def generate_report():
    # Count statistics
    total_posts = len(output_df)
    unsafe_posts = output_df[output_df['final_label'] == 'Unsafe']
    neutral_posts = output_df[output_df['final_label'] == 'Neutral']
    safe_posts = output_df[output_df['final_label'] == 'Safe']
    
    # Calculate percentages
    unsafe_percent = (len(unsafe_posts) / total_posts) * 100
    neutral_percent = (len(neutral_posts) / total_posts) * 100
    safe_percent = (len(safe_posts) / total_posts) * 100
    
    # Identify common reasons
    all_reasons = []
    for reason in output_df[output_df['final_label'] != 'Safe']['reason']:
        all_reasons.extend([r.strip() for r in reason.split(";")])
    
    reason_counts = {}
    for reason in all_reasons:
        if reason not in reason_counts:
            reason_counts[reason] = 0
        reason_counts[reason] += 1
    
    common_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Sample flagged posts
    flagged_examples = unsafe_posts.head(5)[['post_id', 'post_text', 'toxicity_score', 'reason']].to_dict('records')
    
    # Create report
    report = {
        "summary_statistics": {
            "total_posts": total_posts,
            "unsafe_posts": len(unsafe_posts),
            "neutral_posts": len(neutral_posts),
            "safe_posts": len(safe_posts),
            "unsafe_percentage": round(unsafe_percent, 2),
            "neutral_percentage": round(neutral_percent, 2),
            "safe_percentage": round(safe_percent, 2)
        },
        "moderation_reasons": {
            "common_causes": [{"reason": reason, "count": count} for reason, count in common_reasons]
        },
        "flagged_examples": flagged_examples,
        "overall_assessment": f"This feed contains {round(unsafe_percent, 2)}% unsafe content that requires moderation. The most common issue is {common_reasons[0][0]}."
    }
    
    # Save report
    with open('report_summary.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    # Also create markdown version
    md_report = f"""# Content Moderation Report

## Summary Statistics
- Total Posts Analyzed: {total_posts}
- Unsafe Posts: {len(unsafe_posts)} ({round(unsafe_percent, 2)}%)
- Neutral Posts: {len(neutral_posts)} ({round(neutral_percent, 2)}%)
- Safe Posts: {len(safe_posts)} ({round(safe_percent, 2)}%)

## Common Causes for Moderation
"""
    
    for reason, count in common_reasons:
        md_report += f"- {reason}: {count} occurrences\n"
    
    md_report += "\n## Sample Flagged Posts\n"
    
    for i, example in enumerate(flagged_examples):
        md_report += f"""
### Example {i+1}
- Post ID: {example['post_id']}
- Content: "{example['post_text']}"
- Toxicity Score: {example['toxicity_score']}
- Reason: {example['reason']}
"""
    
    md_report += f"\n## Overall Assessment\nThis feed contains {round(unsafe_percent, 2)}% unsafe content that requires moderation. The most common issue is {common_reasons[0][0]}."
    
    with open('report_summary.md', 'w') as f:
        f.write(md_report)

# Generate the report
generate_report()
