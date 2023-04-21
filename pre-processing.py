#1) we extracted pdf using PdfReader
#2) performed pre-processing of data with these steps:
#    a)lowercasing
#    b)removing punctuation
#    c)removing stop words
#    d)performed word tokenization
#    e)performed stemming

import PyPDF2
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import string


#creating a pdf file object
pdf = open(".\\abstract.pdf","rb")

#creating a pdf reader object
pdf_reader = PyPDF2.PdfReader(pdf)

#checking number of pages in a pdf file
#print("Number of pages in the pdf ",len(pdf_reader.pages))

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


#stemming tokens 

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

for i in range(0,len(pdf_reader.pages)):
    page = pdf_reader.pages[i]
    extracted_text = page.extract_text() # extraction happens here
    text_without_stopwords = remove_stop_words(extracted_text) #removing stopwords (function called)
    #stemmed_tokens = [stemmer.stem(word) for word in text_without_stopwords]
    text_after_stem = stemming(text_without_stopwords)
    #print("Text in page :",i+1," is ",text_without_stopwords)
    print("After Stemming is performed : ",text_after_stem)



