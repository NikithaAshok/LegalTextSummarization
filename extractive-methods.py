

import PyPDF2
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
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
    extracted_text += page.extract_text() # extraction happens here
    text_without_stopwords = remove_stop_words(extracted_text) #removing stopwords (function called)
    #stemmed_tokens = [stemmer.stem(word) for word in text_without_stopwords]
    sentences = sent_tokenize(extracted_text)
    #sentences = [sentence for sentence in sentences if not any(word.lower() in stop_words for word in sentence.split())]
    
    # text rank --- TextRank looks at the semantic relationships between sentences.
    txt_parser = PlaintextParser.from_string('\n'.join(sentences),Tokenizer('english'))
    txt_rank_summarizer = TextRankSummarizer()
    txt_summary = txt_rank_summarizer(txt_parser.document,4) #summarizer has arguments (document,sentences_count)
    print("Text Rank Summary :",txt_summary)

    # lex rank --- LexRank looks at the lexical overlap between sentences.
    lex_parser = PlaintextParser.from_string(extracted_text,Tokenizer('english'))
    lex_summarizer = LexRankSummarizer()
    lex_summary = lex_summarizer(lex_parser.document, sentences_count=3)
    print("Lex Rank Summary: ",lex_summary)     

    # LSA( latent semantic analysis ) --- LSA extracts semantically significant sentences by applying SVD to the term-document frequency
    lsa_parser = PlaintextParser.from_string('\n'.join(sentences),Tokenizer('english'))
    lsa_summarizer = LsaSummarizer()
    lsa_summary = lsa_summarizer(lsa_parser.document,3)
    print("LSA summary : ",lsa_summary)

    #Luhn Summarization algorithm’s approach is based on TF-IDF
    luhn_parser = PlaintextParser.from_string('\n'.join(sentences),Tokenizer('english'))
    luhn_summarizer = LuhnSummarizer()
    luhn_summary = luhn_summarizer(luhn_parser.document,3)
    print("Luhn Summary : ",luhn_summary)

    #KL-Sum Summarization selects sentences based on similarity of word distribution as the original text
    kl_parser = PlaintextParser.from_string('\n'.join(sentences),Tokenizer('english'))
    kl_summarizer = KLSummarizer()
    kl_summary = kl_summarizer(kl_parser.document,3)
    print("KL-Sum Summary : ",kl_summary)

