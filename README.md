# Document Clustering and Topic Modeling on 20 Newsgroups Dataset

## Project Overview
This project implements document clustering and topic modeling on the **20 Newsgroups dataset** using two machine learning approaches: **K-means clustering** and **Latent Dirichlet Allocation (LDA)**. The goal is to group similar documents and extract meaningful topics from a large text corpus, demonstrating unsupervised learning techniques for text analysis.

- **K-means Clustering**: Groups documents into 15 clusters based on TF-IDF features, reduced to 100 dimensions using TruncatedSVD for improved performance.
- **LDA Topic Modeling**: Identifies 15 probabilistic topics, capturing latent themes in the dataset.
- **Visualizations**:
  - `kmeans_clusters_annotated.png`: A 2D scatter plot of K-means clusters with category annotations.
  - `lda_topic_X.png`: Word clouds for each LDA topic (where X ranges from 0 to 14).

This project was developed as part of an internship, showcasing text preprocessing, clustering, topic modeling, and visualization techniques.

## Dataset
The **20 Newsgroups dataset** consists of approximately **18,846 documents** across 20 newsgroup categories (e.g., `rec.sport.hockey`, `talk.politics.guns`, `soc.religion.christian`). The dataset is preprocessed by:

- Removing headers, footers, and quotes.
- Converting text to lowercase, removing punctuation, and lemmatizing words.
- Filtering stopwords and rare words (appearing in fewer than 5 documents).

## Requirements
To run the project, install the following Python libraries:

```bash
pip install numpy pandas scikit-learn gensim nltk matplotlib wordcloud
```

Additionally, download the required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

## Usage

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Run the Script**:
   ```bash
   python document_clustering.py
   ```

### Output
- **Console Output**:
  - K-means silhouette score (e.g., 0.0529).
  - LDA coherence score (e.g., 0.5972).
  - K-means cluster sizes and sample documents with categories.
  - LDA topics with top 10 words per topic.

- **Visualizations**:
  - `kmeans_clusters_annotated.png`: A 2D scatter plot of K-means clusters with category annotations.
  - `lda_topic_0.png` to `lda_topic_14.png`: Word clouds for each LDA topic, with word size proportional to importance.

## Results

### K-means Clustering
- **Silhouette Score**: ~0.0529, indicating moderate cluster separation due to overlapping clusters (visible in the scatter plot).
- Clusters align with some categories (e.g., `comp.sys.mac.hardware`, `rec.sport.hockey`), but overlap exists.

### LDA Topic Modeling
- **Coherence Score**: ~0.5972, suggesting reasonable topic interpretability.
- Each topic is visualized as a word cloud, with top words indicating key themes (e.g., sports, politics, religion).

