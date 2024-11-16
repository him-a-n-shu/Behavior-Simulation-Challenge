# Adobe Behavior Simulation Challenge
This repository contains our team's solution to the Adobe Behavior Simulation Challenge, developed as part of an internal hackathon for the selection process in Inter IIT 2024.

## Problem Statement
The challenge includes two primary tasks:
1. **Behavior Simulation**: Predicts user engagement (number of likes) on tweets based on tweet content, media type, company, username, and timestamp. 
2. **Content Simulation**: Generates potential tweet content based solely on metadata (company, username, media type, timestamp).

[Training Dataset](https://docs.google.com/spreadsheets/d/1JcESl7qCCBvS6xpWMZplhCXunvmkcNU_/edit?usp=drive_link&ouid=101476968084918341858&rtpof=true&sd=true)

## Approach
### Task 1: Behavior Simulation
- **Objective**: Predict engagement levels for social media posts.
- **Model Pipeline**:
  - **Feature Extraction**: Processed metadata, media captions, and timestamps to create a feature-rich dataset.
  - **Model**: Used XGBClassifier for initial classification into engagement categories, followed by a regression layer to refine engagement predictions.
- **Evaluation**: RMSE

### Task 2: Content Simulation
- **Objective**: Generate tweet content from metadata alone.
- **Model Pipeline**:
  - **Metadata Extraction**: Collected structured tweet metadata to use as input features.
  - **Text Generation**: Used models like BLIP-2 to generate media captions and fine-tuned BART model for tweet reconstruction.
- **Evaluation**: BLEU, ROGUE, CIDEr

## Repository Structure
- `task 1/` - Folder containing files for Behavior Simulation.
- `task 2/` - Folder containing files for Content Simulation.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/him-a-n-shu/Behavior-Simulation-Challenge.git
   cd Behavior-Simulation-Challenge

2. For Task 1, run `app_task_1.py` and open `index.html` in browser.

3. For Task 2, run `app.py` and open `index.html` in browser.

## Team
- [HIMANSHU](https://github.com/him-a-n-shu)
- [D Barghav](https://github.com/Barghav777)
- [Purushartha Gupta](https://github.com/Puru348)
- Aman Nagar
- Palak
- Misti D Shah
