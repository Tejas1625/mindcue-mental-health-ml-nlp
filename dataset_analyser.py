# --- Comprehensive Dataset Analyzer for MindCue ---

import os
import re
import string

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# --- Configuration ---
DATA_PATH = "Combined Data.csv"
OUTPUT_DIR = "analysis_plots"
TOP_N_NGRAMS = 20 # Number of top n-grams to display in plots

# --- Setup: Ensure output directory and NLTK stopwords exist ---
def setup():
    """Create output directory and download NLTK stopwords if needed."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    try:
        stopwords.words("english")
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        nltk.download("stopwords")
        print("Download complete.")

# --- 1. Data Loading and Basic Info ---
def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Loads and cleans the dataset, standardizing column names.
    Returns a cleaned pandas DataFrame.
    """
    print(f"--- 1. Loading and Cleaning Data from '{path}' ---")
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found. Please make sure it's in the same directory.")
        return None

    # Standardize column names
    col_map = {c.lower(): c for c in df.columns}
    text_col = col_map.get("statement", col_map.get("text", None))
    label_col = col_map.get("status", col_map.get("label", None))

    if text_col is None or label_col is None:
        print(f"Error: Expected text ('statement'/'text') and label ('status'/'label') columns. Found: {list(df.columns)}")
        return None

    df = df.rename(columns={text_col: "text", label_col: "label"})
    df = df[["text", "label"]] # Keep only relevant columns

    # Clean the data
    initial_rows = len(df)
    df = df.dropna().drop_duplicates()
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(str)
    print(f"Cleaned data: {len(df)} rows remaining (removed {initial_rows - len(df)} empty or duplicate rows).")

    print("\n[Dataset Overview]")
    print(f"Shape: {df.shape}")
    print("\n[Data Types]")
    df.info(memory_usage="deep")
    print("\n[Sample Data]")
    print(df.head())
    print("-" * 50 + "\n")
    return df

# --- 2. Label Distribution Analysis ---
def analyze_label_distribution(df: pd.DataFrame):
    """Analyzes and visualizes the distribution of labels."""
    print("--- 2. Label Distribution Analysis ---")

    counts = df["label"].value_counts()
    percentages = df["label"].value_counts(normalize=True) * 100

    label_dist = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

    print(label_dist)

    # Visualization
    plt.figure(figsize=(12, 7))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title("Distribution of Classes in the Dataset", fontsize=16)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Number of Entries", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "label_distribution.png")
    plt.savefig(save_path)
    print(f"\nSaved label distribution plot to: {save_path}")
    plt.close()
    print("-" * 50 + "\n")

# --- 3. Text Statistics Analysis ---
def analyze_text_stats(df: pd.DataFrame):
    """Calculates and visualizes statistics about the text data."""
    print("--- 3. Text Statistics Analysis ---")

    # Character length
    df["char_count"] = df["text"].apply(len)

    # Word count
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))

    # Average word length
    df["avg_word_length"] = df["text"].apply(lambda x: sum(len(word) for word in x.split()) / (len(x.split()) + 1e-6))

    print("[Descriptive Statistics for Text Features]")
    print(df[["char_count", "word_count", "avg_word_length"]].describe())

    # Visualization: Histograms
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df["char_count"], bins=50, ax=axes[0], color="skyblue", kde=True)
    axes[0].set_title("Distribution of Text Length (Characters)")
    axes[0].set_xlabel("Character Count")

    sns.histplot(df["word_count"], bins=50, ax=axes[1], color="salmon", kde=True)
    axes[1].set_title("Distribution of Text Length (Words)")
    axes[1].set_xlabel("Word Count")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "text_length_distributions.png")
    plt.savefig(save_path)
    print(f"\nSaved text length distribution plots to: {save_path}")
    plt.close()
    print("-" * 50 + "\n")


# --- 4. N-gram Analysis ---
def get_top_ngrams(corpus: pd.Series, ngram_range=(1, 1), top_n=20):
    """
    Extracts and returns the top n-grams from a text corpus.
    """
    # Preprocessing: remove punctuation, convert to lowercase
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        return " ".join([word for word in text.split() if word not in stop_words])

    processed_corpus = corpus.apply(preprocess_text)

    vec = CountVectorizer(ngram_range=ngram_range).fit(processed_corpus)
    bag_of_words = vec.transform(processed_corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_n]


def analyze_ngrams(df: pd.DataFrame):
    """Performs and visualizes Unigram, Bigram, and Trigram analysis."""
    print("--- 4. N-gram Analysis (Most Common Phrases) ---")

    # Unigrams (single words)
    top_unigrams = get_top_ngrams(df['text'], ngram_range=(1, 1), top_n=TOP_N_NGRAMS)

    # Bigrams (two-word phrases)
    top_bigrams = get_top_ngrams(df['text'], ngram_range=(2, 2), top_n=TOP_N_NGRAMS)

    # Trigrams (three-word phrases)
    top_trigrams = get_top_ngrams(df['text'], ngram_range=(3, 3), top_n=TOP_N_NGRAMS)

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 24))
    ngram_data = [
        (top_unigrams, "Top 20 Unigrams (Single Words)"),
        (top_bigrams, "Top 20 Bigrams (Two-Word Phrases)"),
        (top_trigrams, "Top 20 Trigrams (Three-Word Phrases)"),
    ]

    for i, (data, title) in enumerate(ngram_data):
        df_ngram = pd.DataFrame(data, columns=["N-gram", "Count"])
        sns.barplot(x="Count", y="N-gram", data=df_ngram, ax=axes[i], palette="mako")
        axes[i].set_title(title, fontsize=16)
        axes[i].set_xlabel("Count")
        axes[i].set_ylabel("Phrase")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "top_ngrams_analysis.png")
    plt.savefig(save_path)
    print(f"Saved n-gram analysis plots to: {save_path}")
    plt.close()
    print("-" * 50 + "\n")


# --- Main execution ---
if __name__ == "__main__":
    setup()
    df = load_and_clean_data(DATA_PATH)

    if df is not None:
        analyze_label_distribution(df)
        analyze_text_stats(df)
        analyze_ngrams(df)
        print("✅ Analysis complete. Check the console output and the 'analysis_plots' directory.")
