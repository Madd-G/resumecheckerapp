try:
    from PIL import Image
except ImportError:
    import Image

import pytesseract
import pickle
import re
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

from nltk.tokenize import word_tokenize
from pdf2image import convert_from_path

import spacy

en_core = "venv/Lib/site-packages/en_core/en_core_web_sm/en_core_web_sm-3.2.0/"
nlp = spacy.load(en_core)

from nltk.stem import WordNetLemmatizer

from string import punctuation

punctuation = punctuation + '\n'


def convert_pdf_to_text(path):
    pages = convert_from_path(pdf_path=path)
    num_pages = 0
    extractedInformation = ''
    for page in pages:
        page.save('static/files/resume_' + str(num_pages) + '.jpg', 'JPEG')
        image_path = ('static/files/resume_' + str(num_pages) + '.jpg')
        text = pytesseract.image_to_string(Image.open(image_path))
        extractedInformation += text
        num_pages += 1
    return extractedInformation


def cv_classification(path, inp):
    filename = 'resume_classification_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    text = convert_pdf_to_text(path).lower()
    vec_path = 'vectorizer.pickle'
    tfidf_file = open(vec_path, 'rb')
    tfidfconverter = pickle.load(tfidf_file)
    tfidf_file.close()

    text_vector = tfidfconverter.transform([text]).toarray()
    pred_text = loaded_model.predict(text_vector)

    pkl_file = open('encoder.pkl', 'rb')
    le = pickle.load(pkl_file)
    pkl_file.close()

    pred_text = le.inverse_transform(pred_text)
    score = round(max(loaded_model.predict_proba(text_vector)[0]) * 100, 2)
    score_str = str(score)
    result = str(pred_text[0].capitalize())

    doc = nlp(text)
    # tokens = [token.text for token in doc]
    word_frequencies = {}

    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    lemmatizer = WordNetLemmatizer()

    doc1 = list(word_frequencies.keys())
    doc1 = [lemmatizer.lemmatize(word, pos='a') for word in doc1]
    doc1 = [lemmatizer.lemmatize(word, pos='v') for word in doc1]
    doc1 = [lemmatizer.lemmatize(word, pos='n') for word in doc1]

    def listToString(s):
        str1 = ""
        for ele in s:
            str1 += " " + ele
        return str1

    doc1 = listToString(doc1)

    processed_tweet = re.sub(r'\W', ' ', doc1)
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    processed_tweet = re.sub(r'\d', '', processed_tweet)
    processed_tweet1 = processed_tweet.lower()

    doc2 = nlp(inp)
    word_frequencies2 = {}

    for word in doc2:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies2.keys():
                    word_frequencies2[word.text] = 1
                else:
                    word_frequencies2[word.text] += 1

    doc2 = list(word_frequencies2.keys())

    doc2 = [lemmatizer.lemmatize(word, pos='a') for word in doc2]
    doc2 = [lemmatizer.lemmatize(word, pos='v') for word in doc2]
    doc2 = [lemmatizer.lemmatize(word, pos='n') for word in doc2]

    def listToString(s):
        str1 = ""
        for ele in s:
            str1 += " " + ele
        return str1

    doc2 = listToString(doc2)

    processed_tweet = re.sub(r'\W', ' ', doc2)
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)
    processed_tweet = re.sub(r'\d', '', processed_tweet)
    processed_tweet2 = processed_tweet.lower()

    X_list = word_tokenize(processed_tweet1)
    Y_list = word_tokenize(processed_tweet2)

    l1 = []
    l2 = []

    X_set = {w for w in X_list if not w in stopwords}
    Y_set = {w for w in Y_list if not w in stopwords}

    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = round(c / float((sum(l1) * sum(l2)) ** 0.5) * 100, 2)

    return result, score_str, cosine
