#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only one person_name'
base_blob = TextBlob(text)
print(base_blob)
#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Tokenizing Text into Sentences and Words
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name'
base_blob = TextBlob(text)
print(base_blob) # convert into string
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

print(base_blob.sentences) # convert into sentences
#output: [Sentence("Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are
#          only for one person_name")]

print(base_blob.words) # convert into words
#output:    ['Muhammad', 'Allah', 'Rakha', 'Naich.Hassan', 'Raza', 'Nacih.Both','name', 'are', 'only', 'for', 'one', 'person_name']

#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Parts of Speech Tagging
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name'
base_blob = TextBlob(text)
print(base_blob) # convert into string
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

print(base_blob.tags) # convert into tags
#output:    [('Muhammad', 'NNP'), ('Allah', 'NNP'), ('Rakha', 'NNP'), ('Naich.Hassan', 'NNP'), ('Raza', 'NNP'),
#           ('Nacih.Both', 'NNP'), ('name', 'NN'), ('are', 'VBP'), ('only', 'RB'),
#            ('for', 'IN'), ('one', 'CD'), ('person_name', 'NN')]
#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Extracting Noun Parses
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name'
base_blob = TextBlob(text)
print(base_blob) # convert into string
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

print(base_blob.noun_phrases) # convert into Noun Phrases Words
#output: ['muhammad allah rakha naich.hassan raza nacih.both']

#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Sentiment Analysis with TextBlob's Default sentiment Analyzer
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name'
base_blob = TextBlob(text)
print(base_blob) # convert into string
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

print(base_blob.sentiment) # convert into Sentiment. Where check it positive or negative and whether's it's objective or subjective.
#output: Sentiment(polarity=0.0, subjectivity=1.0)

#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Sentiment Analysis with TextBlob's Default sentiment Analyzer
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name'
base_blob = TextBlob(text)
print(base_blob) # convert into string
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

print(base_blob.sentiment) # convert into Sentiment. Where check it positive or negative and whether's it's objective or subjective.
#output: Sentiment(polarity=0.0, subjectivity=1.0)

print(base_blob.sentiment.polarity) # check the objective form
#output: 0.0

print(base_blob.sentiment.subjectivity) # check the subjective form
#output: 1.0

#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Sentiment Analysis with the NaiveBayesAnalyzer
#       Naive Bayes is a commonly use machine learning text classification algorithm.
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name'
base_blob = TextBlob(text)
print(base_blob) # convert into string
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

from textblob.sentiments import NaiveBayesAnalyzer
base_blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
print(base_blob)
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

print(base_blob.sentiment) # convert into sentiment values
#output: Sentiment(classification='pos', p_pos=0.9191678020362545, p_neg=0.08083219796374506)

for sentence in base_blob.sentences:
    print(sentence.sentiment)
#output: Sentiment(classification='pos', p_pos=0.9191678020362545, p_neg=0.08083219796374506)

#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Language Detection and Translation
# Inter-language translation is a challenging problem in natural language processing and artificial intelligence.
# Service like Google Translate can translate between language instantly.
from textblob import TextBlob
text = 'Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name'
base_blob = TextBlob(text)
print(base_blob) # convert into string
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

detection = base_blob.detect_language()
print(detection) # check the language name to relative any country.
#output: something print any here if internet is connect to your PC or Laptop.

spanish = base_blob.translate(to='es')
print(spanish) # something print spanish language here if internet is connect to your PC or Laptop.
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih Ambos nombres son solo para una persona_name


chinese = base_blob.translate(to='zh')
print(chinese) # something print chinese language here if internet is connect to your PC or Laptop.
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih。两个名字仅代表一个人

from textblob.sentiments import NaiveBayesAnalyzer
base_blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
print(base_blob)
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih.Both name are only for one person_name

detection = base_blob.detect_language()
print(detection) # check the language name to relative any country.
#output: something print any here if internet is connect to your PC or Laptop.

spanish = base_blob.translate(to='es')
print(spanish) # something print spanish language here if internet is connect to your PC or Laptop.
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih Ambos nombres son solo para una persona_name

chinese = base_blob.translate(to='zh')
print(chinese) # something print chinese language here if internet is connect to your PC or Laptop.
#output: Muhammad Allah Rakha Naich.Hassan Raza Nacih。两个名字仅代表一个人

#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Inflection: Pluralization and Singularization
from  textblob import Word

name = Word('index')
print(name.pluralize()) # convert into Plural form.
#output: indices
print(name.singularize()) # convert into singular form.
#output: index

name = Word('person')
print(name.pluralize()) # convert into Plural form.
#output: people
print(name.singularize()) # convert into singular form.
#output: person

name = Word('people')
print(name.pluralize()) # convert into Plural form.
#output: peoples
print(name.singularize()) # convert into singular form.
#output: person

#-------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Inflection: Pluralization and Singularization
from  textblob import Word

name = Word('person')
print(name.pluralize()) # convert into Plural form.
#output: people
print(name.singularize()) # convert into singular form.
#output: person

from textblob import TextBlob
animals_name = TextBlob('dog cat fish monkey donkey bird').words

name = animals_name.pluralize() # convert into Plural form.
print(name)
#output: ['dogs', 'cats', 'fish', 'monkeys', 'donkeys', 'birds']

name = animals_name.singularize() # convert into singular form.
print(name)
#output: ['dog', 'cat', 'fish', 'monkey', 'donkey', 'bird']

#------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Spell Checking and Correction

from textblob import Word
from textblob import TextBlob

name = Word('theyr')
print(name.spellcheck()) # check relative common words
#output: [('they', 0.5713042216741622), ('their', 0.42869577832583783)]

print(name.correct()) # check relative original word
#output: they

sentence = TextBlob('This is my name. AAAA STARK')
print(sentence.correct()) # correct the sentence paragraph words.
#output: His is my name. AAAA STARK

#------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Normalization: Stemming and Lemmatization
# Stemming removes a prefix or suffix from a wod leaving only a stem, which may or may not be a real word.
# Lemmatization is similar, but factors in the word's part of speech and meaning and results is a real word.

from textblob import Word
from textblob import TextBlob

name = Word('varieties')
print(name.stem()) # check relative common stemming (stem) words
#output: varieti

print(name.lemmatize()) # check relative original lemmatization word
#output: variety

name = Word('people')
print(name.stem()) # check relative common stemming (stem) words
#output: peopl

print(name.lemmatize()) # check relative original lemmatization word
#output: people

sentence = TextBlob('This is my name. AAAA STARK')
print(sentence.serialized) # serial the sentence paragraph words.
#output: [{'raw': 'This is my name.', 'start_index': 0, 'end_index':16, 'stripped':
# 'this is my name', 'noun_phrases': WordList([]), 'polarity': 0.0, 'subjectivity': 0.0},
# {'raw': 'AAAASTARK', 'start_index': 17, 'end_index': 27, 'stripped':
# 'aaaa stark', 'noun_phrases': WordList(['aaaa stark']), 'polarity': -0.2, 'subjectivity': 0.6}]

#------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Getting Definitions, Synonyms and Antonyms from WordNet
# WordNet is a word database created by Princeton University. The TextBlob library uses the NLTK library's WordNet interface.

from textblob import Word
from textblob import TextBlob

name = Word('happy')
print(name.definitions)  # analysis the definition of given word
#output: ['enjoying or showing or marked by joy or pleasure', 'marked by good fortune',
#         'eagerly disposed to act or to be of service', 'well expressed and to the point']

print(name.synsets) # check the synonyms (synsets) mean of "happy"
#output: [Synset('happy.a.01'), Synset('felicitous.s.02'), Synset('glad.s.02'), Synset('happy.s.04')]

print(name.synsets[0].lemmas())  # check the antonyms (synsets[0].lemmas()) mean of "happy"
#output: [Lemma('happy.a.01.happy')]


name = Word('sad')
print(name.definitions) # analysis the definition of given word
#output: ['experiencing or showing sorrow or unhappiness; ; - Christina Rossetti',
#         'of things that make you feel sad; ; ; ; - Christina Rossetti', 'bad; unfortunate']

print(name.synsets) # check the synonyms (synsets) mean of "sad"
#output: [Synset('sad.a.01'), Synset('sad.s.02'), Synset('deplorable.s.01')]

print(name.synsets[0].lemmas())  # check the antonyms (synsets[0].lemmas()) mean of "sad"
#output: [Lemma('sad.a.01.sad')]

#------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               Deleting Stop Words (Stop words are common words in text)
# WordNet is a word database created by Princeton University. The TextBlob library uses the NLTK library's WordNet interface.
from  nltk.corpus import stopwords
stops = stopwords.words('english')
from textblob import TextBlob

blob_name = TextBlob('today is a beautiful day')
name = [word for word in blob_name.words if word not in stops]
# check the word that place into (stopwords.words('english')) library
print(name)
#output: ['today', 'beautiful', 'day']

for word in blob_name.words:
    if word not in stops:
        print(word)
# check the word that place into (stopwords.words('english')) library
#output:    today
#           beautiful
#           day

blob_name = TextBlob('i love you')
name = [word for word in blob_name.words if word not in stops]
# check the word that place into (stopwords.words('english')) library
print(name)
#output: ['love']

for word in blob_name.words:
    if word not in stops:
        print(word)
# check the word that place into (stopwords.words('english')) library
#output: love

#------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------#
#               BaseBlob (convert character into string format)
#               N-grams (It used to identify letters or words that frequently appear adjacent to one another.
# WordNet is a word database created by Princeton University. The TextBlob library uses the NLTK library's WordNet interface.
from textblob import TextBlob
text = 'Today is a beautiful day. Tomorrow looks like bad weather.'
blob_text = TextBlob(text)
print(blob_text.ngrams()) # creating the Trigram connection to another words. chose the three terms
#output: [WordList(['Today', 'is', 'a']), WordList(['is', 'a', 'beautiful']),
#         WordList(['a', 'beautiful', 'day']), WordList(['beautiful', 'day', 'Tomorrow']),
#         WordList(['day', 'Tomorrow', 'looks']), WordList(['Tomorrow', 'looks', 'like']),
#         WordList(['looks', 'like', 'bad']), WordList(['like', 'bad', 'weather'])]

print(blob_text.ngrams(n= 5)) # creating the Trigram connection to another words. chose the five terms
#output: [WordList(['Today', 'is', 'a', 'beautiful', 'day']),
#         WordList(['is', 'a', 'beautiful', 'day', 'Tomorrow']),
#         WordList(['a', 'beautiful', 'day', 'Tomorrow', 'looks']),
#         WordList(['beautiful', 'day', 'Tomorrow', 'looks', 'like']),
#         WordList(['day', 'Tomorrow', 'looks', 'like', 'bad']),
#         WordList(['Tomorrow', 'looks', 'like', 'bad', 'weather'])]

#------------------------------------------------------------------------------------------#
