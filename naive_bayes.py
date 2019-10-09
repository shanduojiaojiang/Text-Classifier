# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
from nltk import bigrams

LOWER_CASE = True
STEMMING = True

def generate_unigram_BOW(train_set, train_labels):
    pos_bow = {}
    neg_bow = {}
    for i in range(len(train_set)):
        mesg = train_set[i]
        type = train_labels[i]
        for word in mesg:
            if type == 1:
                if word in pos_bow:
                    pos_bow[word] += 1
                else:
                    pos_bow[word] = 1
            else:
                if word in neg_bow:
                    neg_bow[word] += 1
                else:
                    neg_bow[word] = 1

    return pos_bow, neg_bow


def generate_bigram_BOW(train_set, train_labels):
    pos_bow = {}
    neg_bow = {}
    for i in range(len(train_set)):
        currLabel = train_labels[i]
        for j in range(len(train_set[i])-1):
            currTuple = (train_set[i][j], train_set[i][j+1])
            if currLabel == 1:
                if currTuple in pos_bow:
                    pos_bow[currTuple] += 1
                else:
                    pos_bow[currTuple] = 1
            else:
                if currTuple in neg_bow:
                    neg_bow[currTuple] += 1
                else:
                    neg_bow[currTuple] = 1
    return pos_bow, neg_bow



def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    # TODO: Write your code here
    # return predicted labels of development set
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - The prior of the review being positive. P(Type=Positive)
    """
    retval = []
    smoothing_parameter = 0.0055
    # Generate a unigram BOW for both positive and negative reviews, choose the top 2500 words
    pos_bow, neg_bow = generate_unigram_BOW(train_set, train_labels)
    sorted_pos = sorted(pos_bow.items(), key=lambda x: x[1], reverse = True)
    sorted_neg = sorted(neg_bow.items(), key=lambda x: x[1], reverse = True)
    pos_words = sorted_pos[:].copy()
    neg_words = sorted_neg[:].copy()

    pos_bi_bow, neg_bi_bow = generate_bigram_BOW(train_set, train_labels)
    sorted_bi_pos = sorted(pos_bi_bow.items(), key=lambda x: x[1], reverse = True)
    sorted_bi_neg = sorted(neg_bi_bow.items(), key=lambda x: x[1], reverse = True)
    bi_pos_words = sorted_bi_pos[:].copy()
    bi_neg_words = sorted_bi_neg[:].copy()

    # Calculate the log probabilities each word given type
    pos_count = sum(pair[1] for pair in pos_words)
    neg_count = sum(pair[1] for pair in neg_words)
    bi_pos_count = sum(pair[1] for pair in bi_pos_words)
    bi_neg_count = sum(pair[1] for pair in bi_neg_words)

    log_probability_pos = {} #(word)->P(word|positive)
    log_probability_neg = {} #(word)->P(word|negative)
    log_prob_bi_pos = {}
    log_prob_bi_neg = {}

    for pair in pos_words:
        pos_prob = np.log((pair[1]+smoothing_parameter)/(pos_count+smoothing_parameter*(len(pos_words) + 1)))
        log_probability_pos[pair[0]] = pos_prob

    for pair in neg_words:
        neg_prob = np.log((pair[1]+smoothing_parameter)/(neg_count+smoothing_parameter*(len(neg_words) + 1)))
        log_probability_neg[pair[0]] = neg_prob

    for pair in bi_pos_words:
        bi_pos_prob = np.log((pair[1]+smoothing_parameter)/(bi_pos_count+smoothing_parameter*(len(bi_pos_words) + 1)))
        log_prob_bi_pos[pair[0]] = bi_pos_prob

    for pair in bi_neg_words:
        bi_neg_prob = np.log((pair[1]+smoothing_parameter)/(bi_neg_count+smoothing_parameter*(len(bi_neg_words) + 1)))
        log_prob_bi_neg[pair[0]] = bi_neg_prob
    # Finished training

    # For each of the new reviews from development data
    for review in dev_set:
        uni_pos = np.log(pos_prior)
        uni_neg = np.log(1 - pos_prior)
        for word in review:
            if word in log_probability_pos:
                uni_pos += log_probability_pos[word]
            elif word not in log_probability_pos:
                uni_pos += np.log(smoothing_parameter/(pos_count+smoothing_parameter*(len(pos_words) + 1)))

            if word in log_probability_neg:
                uni_neg += log_probability_neg[word]
            elif word not in log_probability_neg:
                uni_neg += np.log(smoothing_parameter/(neg_count+smoothing_parameter*(len(neg_words) + 1)))

        bi_pos = np.log(pos_prior)
        bi_neg = np.log(1 - pos_prior)
        for i in range(len(review)-1):
            currTuple = (review[i], review[i+1])
            if currTuple in log_prob_bi_pos:
                bi_pos += log_prob_bi_pos[currTuple]
            elif currTuple not in log_prob_bi_pos:
                bi_pos += np.log(smoothing_parameter/(bi_pos_count+smoothing_parameter*(len(bi_pos_words) + 1)))

            if currTuple in log_prob_bi_neg:
                bi_neg += log_prob_bi_neg[currTuple]
            elif currTuple not in log_prob_bi_neg:
                bi_neg += np.log(smoothing_parameter/(bi_neg_count+smoothing_parameter*(len(bi_neg_words) + 1)))

        MAP_pos = (1-0.4)*uni_pos + 0.4*bi_pos
        MAP_neg = (1-0.4)*uni_neg + 0.4*bi_neg

        if MAP_pos >= MAP_neg:
            retval.append(1)
        else:
            retval.append(0)

    return retval
