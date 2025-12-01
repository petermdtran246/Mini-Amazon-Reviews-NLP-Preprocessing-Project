# 🛒 Mini Amazon Reviews - NLP Text Preprocessing Pipeline  
A complete end-to-end **NLP preprocessing project** built using **Python, Pandas, and NLTK**, based on a custom dataset of Amazon-style product reviews containing emojis, star ratings, punctuation noise, slang, and irregular formatting.

This project showcases a full text-cleaning workflow similar to real-world NLP tasks such as sentiment analysis, keyword extraction, and model preparation.

---

## ✨ Project Goals  
- Clean raw text containing emojis, “5*”, “<3”, “40%”, slang, punctuation, and noise  
- Perform stopword removal (while keeping sentiment-carrying words like *“not”*)  
- Apply regex-based special replacements  
- Remove punctuation and normalize text  
- Tokenize text into word-level units  
- Apply Stemming & Lemmatization  
- Flatten all tokens into a single corpus  
- Generate Unigrams and Bigrams from cleaned text  
- Visualize top frequent terms  

This project simulates an **industry-standard NLP preprocessing pipeline**.

---

## 🧠 Dataset
**mini_amazon_reviews.csv** (10 rows)

Contains product reviews such as:

"Battery life is amazing!!! Lasted 3 days on a single charge 👍"
"Not worth the money. Cheap build, feels like plastic."
"5* product, but shipping was super slow :("
"Love it!! Best purchase of 2024 <3"
"Item arrived 40% damaged... not happy at all."



Dataset includes:
- Emojis  
- Star expressions (`5*`)  
- Percentages  
- Ellipsis (`...`)  
- Slang (`tbh`, `meh`)  
- Sad faces (`:(`)  

A perfect practice dataset for messy real-world text.

---

## 🔧 Libraries Used

```python
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download(['stopwords','punkt','wordnet'])


🧹 Full NLP Preprocessing Pipeline
1️⃣ Load & Inspect Data

data = pd.read_csv('mini_amazon_reviews.csv')
data.info()
data.head()

2️⃣ Lowercase Text

Uses vectorized string operations with .str.lower().

data['review_lowercase'] = data['Review'].str.lower()

Emojis remain unchanged ✔
Makes text uniform for later steps.

3️⃣ Stopword Removal (Keep “not”)

We keep "not" because it changes sentiment:

not good

not worth

not compatible

en_stopwords = stopwords.words('english')
en_stopwords.remove('not')

data['reviews_no_stopwords'] = data['review_lowercase'].apply(
    lambda x: ' '.join([w for w in x.split() if w not in en_stopwords])
)


4️⃣ Special Pattern Replacement

Custom rules inspired by real production NLP systems.

data['review_no_stopwords_no_punct'] = (
    data['review_no_stopwords']
        .str.replace(r'5\*', ' 5 star', regex=True)
        .str.replace(r'<3', ' love', regex=True)
        .str.replace(r'40%', ' 40 percent', regex=True)
        .str.replace(r'\.+', ' ', regex=True)
)


Handles:

5* → “5 star”

<3 → “love”

40% → “40 percent”

... → space


5️⃣ Remove Punctuation

data['review_clean'] = data['review_no_stopwords_no_punct'] \
    .str.replace(r"[^\w\s]", " ", regex=True)


Removes:

?!

:|

emojis (optional)

special characters

Only letters, digits, underscore, and whitespace remain.


6️⃣ Tokenization

data['tokenized'] = data['review_clean'].apply(word_tokenize)

Turns text into lists of tokens:

['battery','life','amazing','lasted','3','days']

7️⃣ Stemming (Porter Stemmer)

ps = PorterStemmer()
data['stemmed'] = data['tokenized'].apply(
    lambda tokens: [ps.stem(t) for t in tokens]
)


Example:

“amazing” → “amaz”

“batteries” → “batteri”

8️⃣ Lemmatization (WordNet Lemmatizer)

lemmatizer = WordNetLemmatizer()
data['lemmatized'] = data['tokenized'].apply(
    lambda tokens: [lemmatizer.lemmatize(t) for t in tokens]
)

Example:

“batteries” → “battery”

“feet” → “foot”

“better” → “better” (noun default)


9️⃣ Flatten All Tokens (Corpus)

tokens_clean = sum(data['lemmatized'], [])

Converts list-of-lists into one global token list.

🔟 Unigrams & Bigrams
Unigrams

unigrams = pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()
print(unigrams.head(20))


Bigrams

bigrams = pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()
print(bigrams.head(20))


📊 Visualization (Top 10)

unigrams[:10].sort_values().plot.barh(color="lightsalmon", figsize=(12,8))
plt.title("Top 10 Unigrams")


bigrams[:10].sort_values().plot.barh(color="skyblue", figsize=(12,8))
plt.title("Top 10 Bigrams")


🔥 Final Results (Examples)

Top Unigrams:
product
love
battery
life
good
worth


Top Bigrams:
battery life
not worth
sound quality
