#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: 
"""

import random
import ps2
import ps3a


word_list = ps2.load_words()
print('done loading words')

def test_mc_player(hand, seed, N=100):
    """
    play a hand 3 times, make sure produced same scores.
    
    """
    #create empty list to store scores
    scores =[]
    #play 3 total hands
    for i in range(3):
        #set the random seed for each loop
        random.seed(seed)
        #call play_mc_hand
        wordls, handscore = ps3a.play_mc_hand(hand, N)
        #append the score list
        scores.append(handscore)
        print(wordls)
    
    print(scores)
    
    #check if all scores are the same
    if scores[0] == scores[1] == scores[2]:
        return True
    else:
        return False

    
    
if __name__ == "__main__":
    
    # Set the MC seed
    seed = 100
    
    test_hands = ['helloworld', 'UMasswins', 'statisticscomputing']
    for handword in test_hands:
        hand = ps2.get_frequency_dict(handword)
        if not test_mc_player(hand, seed=seed):
            print('Reproducibility problem for %s' % handword)
        else:
            print('Passed Test for %s' % handword)
