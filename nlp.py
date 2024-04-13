import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# from gensim.models import Word2Vec

# Download NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
# nltk.download('vader_lexicon')
# nltk.download('movie_reviews')

# Sample text for demonstration
text = "NLTK is a powerful Python library for natural language processing tasks."

# Remove punctuation
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# POS Tagging
pos_tags = pos_tag(filtered_tokens)

# Named Entity Recognition (NER)
ner_tree = ne_chunk(pos_tags)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Frequency Distribution
fdist = FreqDist(lemmatized_tokens)

# TF-IDF Vectorization
corpus = ["This is a sample document.", "Another document for demonstration."]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Word Embedding with Word2Vec
sentences = [word_tokenize(sentence.lower()) for sentence in sent_tokenize(text)]
# word2vec_model = Word2Vec(sentences, min_count=1)

# Topic Modeling with Latent Dirichlet Allocation (LDA)
lda_model = LatentDirichletAllocation(n_components=2)
doc_topic_matrix = lda_model.fit_transform(tfidf_matrix)

# Sentiment Analysis
sentiment_analyzer = SentimentIntensityAnalyzer()
sentiment_score = sentiment_analyzer.polarity_scores(text)

# Text Classification with Naive Bayes
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
word_features = [word for (word, freq) in fdist.most_common(2000)]

def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d, word_features), category) for (d, category) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = NaiveBayesClassifier.train(train_set)
accuracy_score = accuracy(classifier, test_set)


# Print results
print("Filtered Tokens:", filtered_tokens)
print("POS Tags:", pos_tags)
print("NER Tree:", ner_tree)
print("Lemmatized Tokens:", lemmatized_tokens)
print("Word Frequencies:", fdist.most_common())
print("TF-IDF Matrix:", tfidf_matrix.toarray())
# print("Word Embedding Model:", word2vec_model.wv.most_similar('python'))
print("Document-Topic Matrix (LDA):", doc_topic_matrix)
print("Sentiment Score:", sentiment_score)
print("Accuracy of Naive Bayes Classifier:", accuracy_score)
