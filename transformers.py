#T5 is an encoder-decoder model. Converts all language problems into text-to-text format.

from transformers import GPT2Tokenizer,GPT2LMHeadModel
from transformers import XLMWithLMHeadModel, XLMTokenizer
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
import PyPDF2
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

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

#T5 model tokenizer and model
# t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

#BART model tokenizer and model
# bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
# bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#GPT model
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')
gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#XLM MODEL 
# xlm_tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
# xlm_model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')



for i in range(0,len(pdf_reader.pages)):
    page = pdf_reader.pages[i]
    extracted_text = page.extract_text() # extraction happens here
    text_without_stopwords = remove_stop_words(extracted_text) #removing stopwords (function called)
    sentences = sent_tokenize(extracted_text)
    text = "summarize:" + sentences

    #T5 MODEL -----------
    #converting input sequence to input-ids through process of encoding - encode()
    # t5_input_ids = t5_tokenizer.encode(text,return_tensors='pt',max_length=512)
    # #generate() method returns a sequence of ids corresponding to the summary
    # t5_summary_ids = t5_model.generate(t5_input_ids)
    # #using decode() function to generate summary text from the above ids
    # t5_summary = t5_tokenizer.decode(t5_summary_ids[0])
    # print(t5_summary)

    # #BART MODEL ------------
    # bart_input_ids = bart_tokenizer.encode(extracted_text)
    # bart_summary_ids = bart_model.generate(bart_input_ids)
    # bart_summary = bart_model.decode(bart_summary_ids[0])
    # print(bart_summary)

    #GPT MODEL ------------------
    gpt_input_ids = gpt_tokenizer.encode(extracted_text,return_tensors='pt',max_length=512)
    gpt_summary_ids = gpt_model.generate(gpt_input_ids['input_ids'],early_stopping=True)

    gpt_summary = gpt_tokenizer.decode(gpt_summary_ids[0],skip_special_tokens=True)
    print(gpt_summary)





