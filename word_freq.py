from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import silhouette_score
from nltk.stem import SnowballStemmer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re

# Load the full 20 Newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all')
data = newsgroups.data

print("Success1!")

# Create a DataFrame to analyze the data
df = pd.DataFrame(data, columns=["text"])

# Display basic statistics about the dataset
print(df.describe())

# Vectorize the text data to count word occurrences
vectorizer = CountVectorizer(stop_words='english')
X_counts = vectorizer.fit_transform(data)

# Sum up the counts of each word in the vocabulary
word_counts = X_counts.toarray().sum(axis=0)
word_freq = [(word, word_counts[idx]) for word, idx in vectorizer.vocabulary_.items()]
word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

# Plot the top 30 most frequent words
words = [wf[0] for wf in word_freq[:30]]
counts = [wf[1] for wf in word_freq[:30]]
plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.xticks(rotation=90)
plt.title("Top 30 Words Frequency")
plt.show()

# run with and without 
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to the text data
df['clean_text'] = df['text'].apply(preprocess_text)

# Transform the cleaned text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['clean_text'])

# Apply K-Means with a predetermined number of clusters
num_clusters = 20
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_tfidf)

# Get the cluster assignments
labels_kmeans = kmeans.labels_
df['cluster_kmeans'] = labels_kmeans

# Apply Agglomerative Hierarchical Clustering
agglo = AgglomerativeClustering(n_clusters=num_clusters, metric='cosine', linkage='average')
agglo.fit(X_tfidf.toarray())

# Get the cluster assignments
labels_agglo = agglo.labels_
df['cluster_agglo'] = labels_agglo

# Calculate silhouette scores for each clustering algorithm
sil_score_kmeans = silhouette_score(X_tfidf, labels_kmeans)
sil_score_agglo = silhouette_score(X_tfidf, labels_agglo)

print(f'Silhouette Score for K-Means: {sil_score_kmeans}')
print(f'Silhouette Score for Agglomerative Clustering: {sil_score_agglo}')

# Function to plot clusters
def plot_clusters(X_pca, labels, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=2)
    plt.title(title)
    plt.show()

# Reduce dimensions to 2 for visualization
X_pca = PCA(n_components=2).fit_transform(X_tfidf.toarray())

# Visualize clusters
plot_clusters(X_pca, labels_kmeans, 'K-Means Clusters')
plot_clusters(X_pca, labels_agglo, 'Agglomerative Clustering Clusters')