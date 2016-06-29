# Author: Carlos Huertas <ing.carloshuertas@gmail.com>
# License: BSD 3 clause

"""Heat Map Based Feature Selection"""

import numpy as np
from sklearn import preprocessing
from sklearn import cross_validation

class HmbFS():

    def __init__(self, estimator, validator, MinTh=1.0, MaxTh=4.0, verbose=False):
        self.estimator = estimator
        self.validator = validator
        self.MinTh = MinTh
        self.MaxTh = MaxTh
        self.verbose = verbose
        
        self.coloresWeb = [
	[0,0,0],
	[0,0,128],
	[0,128,0],
	[0,128,128],
	[128,0,0],
	[128,0,128],
	[128,128,0],
	[192,192,192],
	[128,128,128],
	[0,0,255],
	[0,255,0],
	[0,255,255],
	[255,0,0],
	[255,0,255],
	[255,255,0],
	[255,255,255]
	]


    def getBaseColor(self,rValue=128, gValue=128, bValue=128):
      allDistances=[450]*16
      for x in range(0,16):
	valoresColor = self.coloresWeb[x]
	#allDistances[x]= ((valoresColor[0]-rValue)**2 + (valoresColor[1]-gValue)**2 + (valoresColor[2]-bValue)**2)**0.5
	allDistances[x]= (abs(valoresColor[0]-rValue) + abs(valoresColor[1]-gValue) + abs(valoresColor[2]-bValue))
      return allDistances.index(min(allDistances))
    
    
    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        
        # Initialization
        #print "Awesome"
        
        numClases = len(set(y))
	numFeats = len(X[0])
	numInstances = len(y)
	
	X_original = np.copy(X)
	X_temp = np.copy(X) #Used to avoid modification to original X
	
        if self.verbose:
	  print "Debug HmbFS using SKF"
        
        #Perfor Scaling
        scores = cross_validation.cross_val_score(self.estimator, X_original, y, scoring='accuracy', cv=self.validator, n_jobs=7)
	scoreSinFS=scores.mean()*100.0
	if self.verbose:
	  print "Before Feature Selection" +" = " + str(scoreSinFS) + " with " + str(numFeats) + " feats"
        
        
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 255), copy=False)
	X_temp = min_max_scaler.fit_transform(X_temp,y)
	
	#Create compression dataset
	numGrupos = numFeats/3
	if numFeats%3 > 0:
	  numGrupos+=1
	  
	coloredDataset = np.zeros((numInstances, numGrupos), dtype=np.int8) #(instances,features)

	#print "Empty Colored Dataset"
	#print coloredDataset
	
	#Build colored dataset
	for ins in range(0,numInstances):
	  for gro in range(0,numGrupos):
	    if (gro*3)+2 < numFeats:
	      coloredDataset[ins][gro]= self.getBaseColor(X_temp[ins][gro*3],X_temp[ins][(gro*3)+1],X_temp[ins][(gro*3)+2])
	    else:
	      coloredDataset[ins][gro]= self.getBaseColor(X_temp[ins][-3],X_temp[ins][-2],X_temp[ins][-1])
        
        #print "Filled Colored Dataset"
        #print coloredDataset
        
        #This is the support
        usefulFeats = np.zeros(numFeats,dtype=bool)
        
        #print "Initial Support"
        #print usefulFeats

	previousScore = scoreSinFS
	numFails = 0
	#gotImprovement = False
	bestScoreFromCV=scoreSinFS
	bestFeatsFromCV = np.ones(numFeats,dtype=bool)
	previousNumFeat = 0

	#for CVth in range(10,41):
	#for CVth in range(10,(numClases*20)+1):
	for CVth in range(int(self.MinTh*10),int(self.MaxTh*10)+1):
	  th=CVth/10.0
	  #th=1.5
	  
	  usefulFeats = np.zeros(numFeats,dtype=bool)
	  
	  for cF in range(0,numGrupos):
	    colorDistrib = np.zeros((numClases, 16), dtype=np.float16) #Because 16 base colors
	    for cI in range(0,numInstances):
	      colorDistrib[y[cI],coloredDataset[cI][cF]]+=1.0
	    
	    #print colorDistrib[0]
	    #print colorDistrib[1]
	    #Ok, now I know the distribution
	    for cA in range(0,numClases):
	      for cB in range(0,numClases):
		if cA != cB:
		  if ( (max(colorDistrib[cA])/sum(colorDistrib[cA])) >  (th * (colorDistrib[cB][colorDistrib[cA].argmax()]/sum(colorDistrib[cB]))) ):
		    if ((cF*3)+2) < numFeats:
		      usefulFeats[(cF*3)]=True
		      usefulFeats[(cF*3)+1]=True
		      usefulFeats[(cF*3)+2]=True
		    else:
		      usefulFeats[-1]=True
		      usefulFeats[-2]=True
		      usefulFeats[-3]=True
		      
	  #print "Final Support"
	  #print usefulFeats
		      
		      
	  scores = cross_validation.cross_val_score(self.estimator, X_original[:, usefulFeats], y, scoring='accuracy', cv=self.validator, n_jobs=7)
	  scoreWithFS = scores.mean()*100.0
	  if self.verbose:
	    print "Using Th:" + str(th) +" = " + str(scoreWithFS) +" with " + str(sum(usefulFeats)) + " feats"
	
	  if previousScore > scoreWithFS:
	    numFails+=1
	    #print "Fail 1"
	  else:
	    if scoreWithFS>=bestScoreFromCV:
	      bestScoreFromCV=scoreWithFS
	      bestFeatsFromCV = np.copy(usefulFeats)
	      #gotImprovement=True
	    if previousNumFeat== sum(usefulFeats):
	      numFails+=1
	    else:
	      numFails=0

	  
	  if numFails>1:
	    break
	  
	  previousScore = scoreWithFS
	  previousNumFeat = sum(usefulFeats)
		    
	
	#if gotImprovement:
	#  self.support_ = bestFeatsFromCV
	#else:
	#  self.support_ = np.ones(numFeats,dtype=bool) #No improvement, return all features
	self.support_ = bestFeatsFromCV
	
	return self

    def fit_transform(self, X, y=None):
      return self.fit(X,y).transform(X)
    
    def get_support(self, indices=False):
      return self.support_
    
    def transform(self, X):
      support = self.get_support()
      X_reduced = X[:, support]
      return X_reduced
