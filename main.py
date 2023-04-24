import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# preprocessing text
text = "This is a sample sentence for unsupervised learning."
tokens = word_tokenize(text)

# Vectorizing text
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(tokens)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

# Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(X_pca)
