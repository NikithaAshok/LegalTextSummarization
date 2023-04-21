import PyPDF2
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import re
import string
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser



#creating a pdf file object
pdf = open(".\\abstract.pdf","rb")

#creating a pdf reader object
pdf_reader = PyPDF2.PdfReader(pdf)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#function for tokenization and removing stopwords
def remove_stop_words(text):
    punct_removed_text = text.translate(str.maketrans('','',string.punctuation))
    words = nltk.word_tokenize(punct_removed_text)
    
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)

#function for stemming
def stemming(text):
    tokens = text.split(' ')

    #defining a Stemmer
    stemmer = PorterStemmer()

    #stem the tokens 
    stemmed_tokens = []
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)

    #join the stemmed tokens back into a string
    stemmed_text = ' '.join(stemmed_tokens)

    return stemmed_text



extracted_text=' ' #initialise for multiple pages
for i in range(0,len(pdf_reader.pages)):
    page = pdf_reader.pages[i]
    extracted_text += page.extract_text() # extraction happens here --- String


stopWords = set(stopwords.words("english"))
words = word_tokenize(extracted_text)

#creating a frequency table to keep score of each word

freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

sentences = sent_tokenize(extracted_text)
sentenceValue = dict()

for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence.lower():
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq
            else:
                sentenceValue[sentence] = freq

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence]

#average value of a sentence from the original text

average = int(sumValues/len(sentenceValue))

#storing the sentences into our summary
reference_summary = ''
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
        reference_summary += " " + sentence

print(reference_summary)