✨ Project Overview

This project implements a complete pipeline to fine-tune OpenAI's GPT-2 language model on a dataset of folk tales. The goal is to enable the model to generate coherent and creative stories from custom prompts. 

The process includes:

  Data ingestion and preprocessing
  
  Visual and statistical data exploration
  
  Tokenization and dataset preparation
  
  GPT-2 fine-tuning with HuggingFace Transformers
  
  Evaluation and story generation from prompts

📁 Project Structure

  
  ├── data/                     # Input dataset (CSV/TXT)
  
  ├── visualizations/          # Plots and WordClouds
  
  ├── fine_tuned_folktales_gpt2/ # Saved model and tokenizer
  
  ├── logs/                    # Training logs
  
  ├── cleaned_stories.csv      # Preprocessed dataset
  
  └── main.py                  # Main training and generation script


🧠 Key Features

  🧹 Data Cleaning & Normalization: Handles CSV/TXT files, removes noise, processes paragraphs.
  
  📊 Visualization: Word count, sentence length, sentiment analysis, word clouds, and top words.
  
  🤖 Model: GPT-2 fine-tuning using HuggingFace's Trainer API.
  
  🔤 Tokenization: Adds special tokens and handles sequence length efficiently.
  
  ✍️ Story Generation: Generates multiple folk stories from diverse prompts.
  
  📁 Customizable: Easily adapt to any story-based dataset.

🧪 Requirements
  Python 3.7+

  Libraries:
  
    nginx
    transformers
    datasets
    torch
    nltk
    pandas
    seaborn
    matplotlib
    plotly
    wordcloud
    scikit-learn
    textblob
    tqdm


Install with:

  pip install -r requirements.txt


🚀 How to Run

  Place Dataset
  
  Put your dataset file (.csv or .txt) in the root directory.
  
  Run the Script
  
  python main.py


Generated Outputs

  Cleaned dataset: cleaned_stories.csv
  
  Visualizations: visualizations/
  
  Fine-tuned model: fine_tuned_folktales_gpt2/
  
  Generated stories: generated_stories_from_prompts.txt

🔍 Example Prompts

  Once upon a time in a small village,
  
  In the ancient kingdom of dragons,
  
  Deep in the enchanted forest,

Each prompt generates a unique story from the fine-tuned GPT-2 model.

📈 Visualizations

  Word count distribution
  
  Sentence count distribution
  
  Character length histograms
  
  Word cloud of most used terms
  
  Sentiment analysis with TextBlob

📚 Acknowledgements

  OpenAI GPT-2
  
  HuggingFace Transformers & Datasets
  
  NLTK & TextBlob
  
  Inspired by traditional Indian folk tales and children's literature

💡 Future Work

  Integration with a web interface (e.g., Streamlit or Flask)
  
  Addition of attention heatmaps
  
  Deployment as a story generation API
