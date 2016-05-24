# -*- coding: utf-8 -*- 
# Name: Luolei Zhao (lzg431), Yue Hu (yhn490)
# Date: May 23, 2016
# Description: bayes.py runs the basic naive bayes algorithm to classify sentiment of any movie review.
#
import math, os, pickle, re
from random import shuffle


class Bayes_Classifier:
   
   def __init__(self, evaluation=False):
      if evaluation: #if the evaluation parameter is set to True, then the script will do the 10-fold cross validation.
         print "For 10-fold evaluation of basic algorithm, run prepareData() to train data and test"
         #clear the pickled content for new training set
         with open("pickle.txt", "w"):
            pass

         if os.stat("pickle.txt").st_size != 0:
            self.posiFreq, self.negFreq = self.load("pickle.txt")
            print "pickle.txt is not empty. Unpickle dictionaries."
         else: 
            self.posiFreq={}
            self.negFreq={}
            #self.train()
      else:
         """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
         cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
         the system will proceed through training.  After running this method, the classifier 
         is ready to classify input text."""
         
         print "Run classify(\"Your text\") to classify your sentiment."
         if os.stat("pickle.txt").st_size != 0:
            self.posiFreq, self.negFreq, self.pseudoPosiPossibility, self.pseudoNegPossibility= self.load("pickle.txt")
         else: 
            self.posiFreq={}
            self.negFreq={}
            self.train()
#prepareData does the 10-fold cross-validation for the algorithm
   def prepareData(self):
      allFiles = []
      for f in os.walk("movies_reviews/"):
         allFiles = f[2]
         break
      numFiles = len(allFiles)
      self.trainingSet = []
      self.testingSet = []

      #shuffle files and slice them into 10 groups
      shuffle(allFiles)
      cut = numFiles/10
#microTable counts the number of reviews. Row0 counts star-5 reviews. Row1 counts star-1 reviews. 
#Col0 counts reviews classified as positive and Col1 counts reviews classified as negative.
      self.microTable = [[0,0,0],[0,0,0]]
      groups = []
      for i in range(10):
         groups.append(allFiles[(cut*i):(cut*(i+1))])
      for i in range(10):
         self.posiFreq={}
         self.negFreq={}
         print "running Test " + str(i)
         self.testingSet = groups[i]
         self.trainingSet = [x for x in allFiles if x not in groups[i]]
         #print "Preaparing training set"
         #print self.trainingSet
         self.train_for_evaluation()
         self.classifyTest()
         print "classify ",
         print i,
         print " ends."
         #print "Clearing pickle"
         with open("pickle.txt", "w"):
            pass

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      directory = "movies_reviews/"
      allFiles = []
      for f in os.walk(directory):
         allFiles = f[2]
         break

      #select a randomed 10% for testing
      # shuffle(allFiles)
      # allFiles = allFiles[:len(allFiles)/10]
      numFiles = len(allFiles)
      # print "testing numFiles is: ",
      # print numFiles

      numPosFiles = 0
      numNegFiles = 0
      numVocabulary = 0
      posiFreq = {}
      negFreq = {}

      for f in allFiles:
         fTokens = self.fileNameTokenize(f)
         if "1" in fTokens: #the review is supposed to be negative
            numNegFiles +=1
            content = self.loadFile(directory+f)
            words = self.tokenize(content)
            
            for w in words:
               if w in negFreq.keys():
                  negFreq[w] += 1
               else:
                  negFreq[w] = 1

            
         elif "5" in fTokens: #the review is supposed to be positive
            numPosFiles += 1
            content = self.loadFile(directory+f)
            words = self.tokenize(content)

            for w in words:
               if w in posiFreq.keys():
                  posiFreq[w] += 1
               else:
                  posiFreq[w] = 1

      #calculate the size of vocabulary for add-one smoothing
      visited = []
      for x in posiFreq.keys():
         if x not in visited:
            visited.append(x)
      for x in negFreq.keys():
         if x not in visited:
            visited.append(x)
      vocSize = len(visited)
      #convert dictionaries of counts into dictionaries of possibilities 
      print "numPosFiles: ",
      print numPosFiles
      print "numNegFiles",
      print numNegFiles
      #change the values of the dictionaries from counts to possibility of appearance
      posiFreqSum = 0
      for b in posiFreq.values():
         posiFreqSum += b
      for a in posiFreq.keys():
         c = posiFreq[a]
         posiFreq[a] = math.log((c+1)/float(posiFreqSum+vocSize), 10)
      #pseudoPossibility is for words that haven't showed up in the dictionary
      self.pseudoPosiPossibility = math.log(1/float(posiFreqSum+vocSize), 10)

      negFreqSum = 0
      for b in negFreq.values():
         negFreqSum += b
      for a in negFreq.keys():
         c = negFreq[a]
         negFreq[a] = math.log((c+1)/float(negFreqSum+vocSize),10)
      self.pseudoNegPossibility = math.log(1/float(negFreqSum+vocSize), 10)

      self.posiFreq = posiFreq
      self.negFreq = negFreq
      #save the dictionaries and two pseudo possibilities for furture use
      result = [self.posiFreq, self.negFreq, self.pseudoPosiPossibility, self.pseudoNegPossibility]
      self.save(result, "pickle.txt")
             
   def train_for_evaluation(self):   
      """Similar to train(), but contails trainingSet and testingSet as the attributes of the object."""
      numPosFiles = 0
      numNegFiles = 0
      visited = []

      posiFreq = {}
      negFreq = {}
      for f in self.trainingSet:
         #print f
         visited = []
         fTokens = self.fileNameTokenize(f)
         if "1" in fTokens:
            numNegFiles +=1
            #print "it is a negative comment\n"
            content = self.loadFile("movies_reviews/"+f)
            words = self.tokenize(content)
            len(words)
            for w in words:
               if w in negFreq.keys():
                  negFreq[w] += 1
               else:
                  negFreq[w] = 1
               if w not in visited:
                 visited.append(w)
              
         elif "5" in fTokens:
            numPosFiles += 1
            #print "positive comment\n"
            content = self.loadFile("movies_reviews/"+f)
            words = self.tokenize(content)
            len(words)
            for w in words:
               if w in posiFreq.keys():
                  posiFreq[w] += 1
               else:
                  posiFreq[w] = 1
               if w not in visited:
                  visited.append(w)

      #convert dictionaries of counts into dictionaries of possibilities 

      posiFreqSum = 0
      for b in posiFreq.values():
         posiFreqSum += b
      for a in posiFreq.keys():
         c = posiFreq[a]
         posiFreq[a] = math.log((c+1)/float(posiFreqSum+len(visited)), 10)
      self.pseudoPosiPossibility = math.log(1/float(posiFreqSum+len(visited)), 10)

      negFreqSum = 0
      for b in negFreq.values():
         negFreqSum += b
      for a in negFreq.keys():
         c = negFreq[a]
         negFreq[a] = math.log((c+1)/float(negFreqSum+len(visited)),10)
      self.pseudoNegPossibility = math.log(1/float(negFreqSum+len(visited)), 10)
      
      self.posiFreq = posiFreq
      self.negFreq = negFreq
      result = [self.posiFreq, self.negFreq]
      self.save(result, "pickle.txt")
 
    
   def classify(self, sText, considerNeutral=True):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      considerNeutral is set to be True as default, but should be set to False for 
      cross-validation"""
      
      words = self.tokenize(sText)
      isPosi = 0
      isNeg = 0
      for w in words:
         if w in self.posiFreq.keys():
            isPosi += self.posiFreq[w]
         else:
            isPosi += self.pseudoPosiPossibility
         if w in self.negFreq.keys():
            isNeg += self.negFreq[w]
         else:
            isNeg += self.pseudoNegPossibility

      ratio = isPosi/isNeg
      if considerNeutral and (ratio < 1.005 and ratio > 0.995): return "Neutral"
      #bc.classify("how can I be neutral") returns "Neutral"
      elif isPosi > isNeg: return "Positive"
      return "Negative"
   
   def classifyTest(self):
      for i in self.testingSet:
         result = self.classify(self.loadFile("movies_reviews/" + i), False)
         fTokens = self.fileNameTokenize(i)
         if ("5" in fTokens):
            if result == "Positive":
               self.microTable[0][0] = self.microTable[0][0] + 1
            elif result == "Negative":
               self.microTable[0][1] = self.microTable[0][1] + 1
            else:
               self.microTable[0][2] = self.microTable[0][2] + 1
         elif ("1" in fTokens):
            if result == "Negative":
               self.microTable[1][1] = self.microTable[1][1] + 1
            elif result == "Positive":
               self.microTable[1][0] = self.microTable[1][0] + 1
            else:
               self.microTable[1][2] = self.microTable[1][2] + 1
      print self.microTable

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, 'r')
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""
      lowerText = sText.lower()
      lTokens = []
      sToken = ""
      for c in lowerText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)

               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)
      
      return lTokens

   def fileNameTokenize(self, sText): 
      """Tokenize filename so we can find out whether the review is 1-star or 5-star"""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens