# Twitter-SNLP
1 Introduction

Twitter is a social micro-blogging service that allows subscribers to write short text messages

called “tweets”. Due to its popularity and easy-to-use features, a lot of people nowadays use Twitter as an open medium to express their concerns and opinions at liberty. This massive volume

of opinion-rich data stream presents lots of opportunities, but also a lot of challenges. On the

one hand, Twitter enables everyone to publicly express themselves, to give a voice to those who

may otherwise not be heard, and to some extent, to have a virtual space that is not governed by

the authority nor shaped by the cultural biases of the society. On the other hand, a lot of the

content on social media might be offensive to minorities or inappropriate for many online users.

Therefore, computational techniques that identify, categorize, and filter abusive language on social media have become a topic of a significant interest within the natural language processing

(NLP) community.

User-generated text on Twitter is characterized by the creative use of language. Due to the

absence of any form of online “grammar police”, tweets are usually written in an informal conversational style with an extensive use of non-standard lexical items such as emojis (e.g, ),

irregular words (e.g., gr8), hashtag phrases (e.g., #TwitterLeftBias), and slang abbreviations

(e.g., smh). Because most of existing NLP systems (e.g., tokenizers, POS taggers, etc.) are

trained on a well-edited text (e.g., newswire corpora), these systems do not perform well on

Twitter text. Thus, developing techniques that are specifically designed for microblogs is an interesting challenge. The goal of this project is to explore NLP techniques and engineer a system

that can cope with the creative use of language in Twitter, or the so-called Twitterese.

2 Task Description

The task to be completed within this project is to develop and evaluate a system that processes

a single tweet and decides whether or not the tweet contains offensive language. This task

could be tackled as a binary text classification problem. To this end, we provide a dataset of

tweets annotated for offensiveness. The files “train.tsv” and “dev.tsv” all have the same format.

The second column (text) contains the text of a tweet, the first column (label) contains an

offensiveness label:

• (NOT) Not Offensive - This post does not contain offense or profanity.

• (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct)

offense

The file “test.tsv” contains only a set of tweets without the labels. Please be alarmed that the

dataset might contain very offensive or racist content, thus reading the tweets in this dataset

may not be the most pleasing experience. The table below shows a few examples of tweets that

have been annotated as offensive.

1/4

1 @USER Liberals are all Kookoo !!!

2 @USER @USER Most stupid tweet yet...and yes...its a libtard

3 @USER I bet the first ones she calls when she is being robbed is those

very same police . People are freaking stupid ...

4 @USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 URL

Moreover, even though the tweets were annotated by people who were trained to perform the

annotations with clear guidelines, you may not agree with some of the annotations in the dataset

due to the subjective nature of the task.

3 Baseline Classifier

To assess the difficulty of the task, we developed a simple baseline. First, the tweets were tokenized using the Twitter-specified tokenizer in CMU TweetNLP tool [Owoputi et al., 2013]. We

held-out two splits of 1000 random tweets each as development and test sets, and the remaining

tweets were used to train a Naive Bayer (NB) classifier. The NB likelihoods were estimated using

the training set with add-one smoothing, while the model priors were assumed to be uniform.

Thus, our NB classifier can be formalized as

C^ = argmax

C

P(Cjhw1; : : : ; wMi) ≈ argmax

C

MXi

=1

logP(wijC)

where hw1; : : : ; wMi is the tweet represented as a sequence of word tokens. The performance of

our NB classifier as measured by the accuracy metric was slightly less than 70% for the both

development and test set. This is promising given the simplicity of our approach and the gain

over the random assignment baseline (∼50% accuracy). We also observed that the performance

of the classifier has degraded when the tokens were lower-cased, which suggests that word shapes

are informative for this task.
