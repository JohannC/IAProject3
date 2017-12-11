# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:46:59 2017

@author: Lorenzo
"""

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import os
import io
import unicodedata
import re




def initiatingArffFile(fileName):
    #Initiating the arff file
    fh = open(fileName,"w",encoding = "utf-8")
    fh.write("%% Dataset about positive and negative movie reviews\n")
    fh.write("@relation review_type\n")
    fh.write("@attribute number_of_words numeric\n")
    fh.write("@attribute number_of_positive_words numeric\n")
    fh.write("@attribute number_of_negative_words numeric\n")
    fh.write("@attribute percent_of_positive_words numeric\n")
    fh.write("@attribute percent_of_negative_words numeric\n")
    fh.write("@attribute positive_minus_negative numeric\n")
    fh.write("@attribute review {positive, negative}\n")
    fh.write("@data\n")
    return fh


def getNegativeDictionary():
    negativeDictionary = list()
    with io.open("data\\dictionnaries\\negative-words.txt", encoding="utf-8") as f:
        res = f.read()
        res_decoded = unicodedata.normalize('NFC', res)
        negativeDictionary = res_decoded.split("\n")
    return negativeDictionary

#Creating the list of positive words
def getPositiveDictionary(): 
    positiveDictionary = list()
    with io.open("data\\dictionnaries\\positive-words.txt", encoding="utf-8") as f:
        res = f.read()
        res_decoded = unicodedata.normalize('NFC', res)
        positiveDictionary = res_decoded.split("\n")
    return positiveDictionary

def processFolder(dir, type, positiveDictionary, negativeDictionary):
    print("Processing following folder : "+dir)
    i = 1
    files = os.listdir(dir)
    for file in files:
        if file.endswith(".txt"):
            with io.open(dir+"\\"+file, encoding="utf-8") as f:
                print("Files processed "+str(i)+" / "+str(len(files)))
                i = i+1
                res = f.read()
                res_decoded = unicodedata.normalize('NFC', res)
                (nbWords, nbPositive, nbNegative) = processAttributesForEachReview(res_decoded, positiveDictionary, negativeDictionary)
                percentPos = nbPositive/nbWords
                percentNeg = nbNegative/nbWords
                posMinusNeg = nbPositive-nbNegative
            fh.write(str(nbWords)+","+str(nbPositive)+","+str(nbNegative)+","+str(percentPos)+","+str(percentNeg)+","+str(posMinusNeg)+","+type+"\n")

def processReviews(fh, dir1, dir2):
    positiveDictionary = getPositiveDictionary()
    negativeDictionary = getNegativeDictionary()
    processFolder(dir1, "negative", positiveDictionary, negativeDictionary)
    processFolder(dir2, "positive", positiveDictionary, negativeDictionary)
	
def processAttributesForEachReview(review, positiveDictionary, negativeDictionary):
    words = review.split()
    nbWords = len(words)
    nbPositive = calculateNumberOfOccurancePresentInDictionary(words, positiveDictionary)
    nbNegative = calculateNumberOfOccurancePresentInDictionary(words, negativeDictionary)
    return (nbWords, nbPositive, nbNegative)

def calculateNumberOfOccurancePresentInDictionary(words, dictionnary):
    count = 0
    for word in  words:
        cleanWord = re.sub('[.,;!?\-]', '', word)
        if cleanWord != "" and cleanWord.lower() in dictionnary:
            count = count + 1            
    return count     

dir1 = os.getcwd()+"\\data\\part1\\neg";
dir2 = os.getcwd()+"\\data\\part1\\pos";
fh = initiatingArffFile("dataset1.arff")
processReviews(fh, dir1, dir2)
fh.close()

dir1 = os.getcwd()+"\\data\\part2\\neg";
dir2 = os.getcwd()+"\\data\\part2\\pos";
fh = initiatingArffFile("dataset2.arff")
processReviews(fh, dir1, dir2)
fh.close()
