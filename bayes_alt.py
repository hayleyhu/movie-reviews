# -*- coding: utf-8 -*- 
# Name: 
# Date:
# Description:
#


import math, os, pickle, re
from random import shuffle
from nltk.stem.porter import *
import sys  

reload(sys)  
sys.setdefaultencoding('cp1252')

class Bayes_Classifier:
   directory = "movie_reviews/"
   
   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""

      #clear the pickled content
      with open("pickle.txt", "w"):
         pass

      if os.stat("pickle.txt").st_size != 0:
         self.posiFreq, self.negFreq = self.load("pickle.txt")
         print "pickle.txt is not empty. Unpickled dictionaries."
      else: 
         print "begin trainning"
         self.posiFreq={}
         self.negFreq={}
         #self.train()

   def prepareData(self):
      allFiles = []
      for f in os.walk(self.directory):
         allFiles = f[2]
         break
      numFiles = len(allFiles)
      self.trainingSet = []
      self.testingSet = []

      #shuffle files and slice them into 10 groups
      shuffle(allFiles)
      cut = numFiles/10
      #print self.directory
      self.microTable = [[0,0,0],[0,0,0]]
      groups = []
      for i in range(10):
         groups.append(allFiles[(cut*i):(cut*(i+1))])
      for i in range(10):
         self.posiFreq={}
         self.negFreq={}
         print "running Test with removing noise" + str(i)
         self.testingSet = groups[i]
         self.trainingSet = [x for x in allFiles if x not in groups[i]]
         #print "Preaparing training set"
         #print self.trainingSet
         self.train()
         self.classifyTest()
         print "classify ",
         print i,
         print " ends."
         #print "Clearing pickle"
         with open("pickle.txt", "w"):
            pass

      
   
   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      numPosFiles = 0
      numNegFiles = 0
      visited = []

      # negFreq = {}
      # negPres = {}
      for f in self.trainingSet:
         #print f
         visited = []
         fTokens = self.fileNameTokenize(f)
         if "1" in fTokens:
            numNegFiles +=1
            #print "it is a negative comment\n"
            content = self.loadFile(self.directory+f)
            words = self.tokenize(content)
            len(words)
            for w in words:
               if w in self.negFreq.keys():
                  self.negFreq[w] += 1
               else:
                  self.negFreq[w] = 1
               if w not in visited:
                 visited.append(w)
                  # if w in negPres.keys():
                  #    negPres[w] += 1
                  # else:
                  #    negPres[w] = 1               
         elif "5" in fTokens:
            numPosFiles += 1
            #print "positive comment\n"
            content = self.loadFile(self.directory+f)
            words = self.tokenize(content)
            len(words)
            for w in words:
               if w in self.posiFreq.keys():
                  self.posiFreq[w] += 1
               else:
                  self.posiFreq[w] = 1
               if w not in visited:
                  visited.append(w)
               #    if w in negPres.keys():
               #       negPres[w] += 1
               #    else:
               #       negPres[w] = 1
      #convert dictionaries of counts into dictionaries of possibilities 
      # print "numPosFiles: ",
      # print numPosFiles
      # print "numNegFiles",
      # print numNegFiles
      posiFreqSum = 0
      for b in self.posiFreq.values():
         posiFreqSum += b
      for a in self.posiFreq.keys():
         c = self.posiFreq[a]
         self.posiFreq[a] = math.log((c+1)/float(posiFreqSum+len(visited)), 10)
      self.pseudoPosiPossibility = math.log(1/float(posiFreqSum+len(visited)), 10)

      negFreqSum = 0
      for b in self.negFreq.values():
         negFreqSum += b
      for a in self.negFreq.keys():
         c = self.negFreq[a]
         self.negFreq[a] = math.log((c+1)/float(negFreqSum+len(visited)),10)
      self.pseudoNegPossibility = math.log(1/float(negFreqSum+len(visited)), 10)

      result = [self.posiFreq, self.negFreq]
      self.save(result, "pickle.txt")

      # test the chosen 1/10 of files 
    
   def classify(self, sText):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      stemmer = PorterStemmer()
      words = self.tokenize(sText)
      isPosi = 0
      isNeg = 0
      for w in words:
         w = stemmer.stem(w)
         if w in self.posiFreq.keys() and w in self.negFreq.keys():
            if abs(self.posiFreq[w] - self.negFreq[w]) < abs(max(self.posiFreq[w],self.negFreq[w]) * 0.05): 
               continue

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
      #print "isPosi: ",
      #print isPosi
      #print "isNeg: ",
      #print isNeg
      # print "isPosi/isNeg", isPosi/isNeg

      # ratio = isPosi/isNeg
      # if ratio < upperBound and ratio > lowerBound: return "Neutral"
      # el
      if isPosi > isNeg: return "Positive"
      return "Negative"
   
   def classifyTest(self):
      #self.macroTable = [[0,0,0],[0,0,0]]
      for i in self.testingSet:
         result = self.classify(self.loadFile(self.directory + i))
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
      stemmer = PorterStemmer()
      lowerText = sText.lower()
      lTokens = []
      sToken = ""
      for c in lowerText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               try:
                  stemmedToken = stemmer.stem(sToken)
                  lTokens.append(stemmedToken)
               except UnicodeDecodeError:
                  # lTokens.append(sToken)
                  print sToken
               sToken = ""
            if c.strip() != "":
               try:
                  stemmedToken = stemmer.stem(str(c.strip()))
                  lTokens.append(stemmedToken)
               except UnicodeDecodeError:
                  # lTokens.append(str(c.strip()))
                  print str(c.strip())
               
      if sToken != "":
         try:
            stemmedToken = stemmer.stem(sToken)
            lTokens.append(stemmedToken)
         except UnicodeDecodeError:
            # lTokens.append(sToken)
            print sToken

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