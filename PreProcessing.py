import re
import nltk
from spacy.lang.en.stop_words import STOP_WORDS

auxiliary_verbs=['is', 'am', 'are', 'was', 'were', 'been', 'being','have', 'has', 'had', 'having','do', 'does', 'did','can', 'could',
                 'shall', 'should', 'will', 'would', 'may', 'might', 'must', 'dare', 'need', 'used to', 'ought to']

stopwords = set(STOP_WORDS)
def PreProcessing(text):
    output=" "
    word_list2=tokenization(text)
    for words in word_list2:
        if not words in stopwords and not words in auxiliary_verbs:
            output=output + " " + words

    return output

def tokenization(text):
    text = text.lower()
    word_list2 = re.findall(r'\w+', text)

    return word_list2