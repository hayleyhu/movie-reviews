# -*- coding: utf-8 -*- 
# Name: 
# Date:
# Description: bayes_0.py is designed for training all files and classify user entered text


import math, os, pickle, re
from random import shuffle


class Bayes_Classifier:
   
   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      print "bayes_0.py is designed for training files and classify user entered text"
      if os.stat("pickle_0.txt").st_size != 0:
         self.posiFreq, self.negFreq, self.pseudoPosiPossibility, self.pseudoNegPossibility= self.load("pickle_0.txt")
      else: 
         self.posiFreq={}
         self.negFreq={}
         self.train()
      # self.directory = "test_reviews/"
   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      directory = "movie_reviews/"
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
         # print f
         fTokens = self.fileNameTokenize(f)
         if "1" in fTokens:
            numNegFiles +=1
            # print "it is a negative comment\n"
            content = self.loadFile(directory+f)
            words = self.tokenize(content)
            
            for w in words:
               if w in negFreq.keys():
                  negFreq[w] += 1
               else:
                  negFreq[w] = 1

            
         elif "5" in fTokens:
            numPosFiles += 1
            # print "positive comment\n"
            content = self.loadFile(directory+f)
            words = self.tokenize(content)

            for w in words:
               if w in posiFreq.keys():
                  posiFreq[w] += 1
               else:
                  posiFreq[w] = 1

      #calculate the size of vocabulary
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
      posiFreqSum = 0
      for b in posiFreq.values():
         posiFreqSum += b
      for a in posiFreq.keys():
         c = posiFreq[a]
         posiFreq[a] = math.log((c+1)/float(posiFreqSum+vocSize), 10)
      self.pseudoPosiPossibility = math.log(1/float(posiFreqSum+vocSize), 10)

      negFreqSum = 0
      for b in negFreq.values():
         negFreqSum += b
      for a in negFreq.keys():
         c = negFreq[a]
         negFreq[a] = math.log((c+1)/float(negFreqSum+vocSize),10)
      self.pseudoNegPossibility = math.log(1/float(negFreqSum+vocSize), 10)

      # print "negFreqSum",
      # print negFreqSum
      # print "posiFreqSum",
      # print posiFreqSum
      self.posiFreq = posiFreq
      self.negFreq = negFreq
      result = [self.posiFreq, self.negFreq, self.pseudoPosiPossibility, self.pseudoNegPossibility]
      self.save(result, "pickle_0.txt")
    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      
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
      # isPosi = math.pow(10, isPosi)
      # isNeg = math.pow(10, isNeg)


      print "isPosi: ",
      print isPosi
      print "isNeg: ",
      print isNeg
      print "isPosi/isNeg", isPosi/isNeg
      ratio = isPosi/isNeg
      if ratio < 1.005 and ratio > 0.995: return "Neutral"
      elif isPosi > isNeg: return "Positive"
      return "Negative"
      

   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""

      f = open(sFilename, "r")
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
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

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

