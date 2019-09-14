# Twitter-SNLP
##School Assignment


### Task Description
The task to be completed within this project is to develop and evaluate a system that processes

a single tweet and decides whether or not the tweet contains offensive language. This task

could be tackled as a binary text classification problem. To this end,  a dataset of

tweets annotated for offensiveness is provided in the repo. The files “train.tsv” and “dev.tsv” all have the same format.

The second column (text) contains the text of a tweet, the first column (label) contains an

offensiveness label:

• (NOT) Not Offensive - This post does not contain offense or profanity.

• (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct)

offense

The table below shows a few examples of tweets that

have been annotated as offensive.

1/4

1 @USER Liberals are all Kookoo !!!

2 @USER @USER Most stupid tweet yet...and yes...its a libtard

3 @USER I bet the first ones she calls when she is being robbed is those

very same police . People are freaking stupid ...

4 @USER @USER Go home you’re drunk!!! @USER #MAGA #Trump2020 URL




###Baseline Classifier

Naive bayes and SVM classifiers were used as models.

to run the model please install using pip
>pip install -r requirements.txt
