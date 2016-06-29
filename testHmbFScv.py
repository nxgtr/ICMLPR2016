import sys
import numpy as np
#from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_svmlight_file,load_svmlight_files,dump_svmlight_file
import Image
from sklearn.linear_model import ElasticNet,LogisticRegression
from sklearn.cross_validation import LeaveOneOut,StratifiedKFold
from sklearn import cross_validation
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import SGDClassifier, Perceptron,PassiveAggressiveClassifier
from HmbFS import *
from sklearn.feature_selection import SelectPercentile, chi2, SelectFpr, f_classif, RFECV, SelectFdr, SelectFwe, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
import warnings
warnings.filterwarnings("ignore")

useMetric="auc"

class RandomForestClassifierWithCoef(RandomForestClassifier):
    def fit(self, *args, **kwargs):
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_
        
def performFS(texto,clfFS):
  skf = StratifiedKFold(y, random_state=0, n_folds=3)
  #print "FS tuned by " + texto
  
  fs = HmbFS(clfFS, skf, 1.5,1.5)
  X_withHmbFS = fs.fit_transform(X,y)
  #print "Done0"

  fs = HmbFS(clfFS, skf, 1.0,4.0)
  X_withHmbFSCV = fs.fit_transform(X,y)
  #print "Done1"

  #competencia = RFECV(clfFS, step=0.1, cv=skf, scoring='accuracy')
  stepSize=int(np.ceil(len(X[0])/10.0))
  competencia = RFECV(clfFS, step=stepSize, cv=skf, scoring='accuracy', verbose=0)
  X_withRFECV = competencia.fit_transform(X,y)
  #print "Done2"

  return X_withHmbFS,X_withHmbFSCV,X_withRFECV
  
# Generate the dataset
print "Loading dataset..."
X_sparse, y = load_svmlight_file(sys.argv[1])
X = X_sparse.toarray()


clfs = [PassiveAggressiveClassifier(random_state=0),RandomForestClassifier(random_state=0),LinearSVC(random_state=0),LogisticRegression(random_state=0)]
namesClfs = ["Passive Aggresive","Random Forest","Linear SVM","Logistic Regression"]

#megaLOO = LeaveOneOut(len(y))
megaLOO = StratifiedKFold(y, random_state=0, n_folds=10)

fsMethods=["NoFS","HmbFS","HmbFSCV","RFECV"]

me=0.0
their=0.0
#for numClassify in range(0,4):
for numClassify in range(3,4):
  #Feature Selection
  X_withHmbFS,X_withHmbFSCV,X_withRFECV = performFS(namesClfs[numClassify],clfs[numClassify])
  print " Features with NoFS: " + str(len(X[0]))
  print " Features with HmbFS: " + str(len(X_withHmbFS[0]))
  print " Features with HmbFSCV: " + str(len(X_withHmbFSCV[0]))
  print " Features with RFECV: " + str(len(X_withRFECV[0]))

# Performance Evaluation
  for numEvalua in range(0,4):
    if numEvalua!=numClassify:
      print "Testing performance for: " + namesClfs[numEvalua]

      scores = cross_validation.cross_val_score(clfs[numEvalua], X, y, scoring='accuracy', cv=megaLOO, n_jobs=7)
      scoreFin = scores.mean()*100.0
      print " With " + fsMethods[0] +": " + str(scoreFin)
      
      scores = cross_validation.cross_val_score(clfs[numEvalua], X_withHmbFS, y, scoring='accuracy', cv=megaLOO, n_jobs=7)
      scoreFin = scores.mean()*100.0
      print " With " + fsMethods[1] +": " + str(scoreFin)
      
      scores = cross_validation.cross_val_score(clfs[numEvalua], X_withHmbFSCV, y, scoring='accuracy', cv=megaLOO, n_jobs=7)
      scoreFin = scores.mean()*100.0
      me+=scoreFin
      print " With " + fsMethods[2] +": " + str(scoreFin)
      
      scores = cross_validation.cross_val_score(clfs[numEvalua], X_withRFECV, y, scoring='accuracy', cv=megaLOO, n_jobs=7)
      scoreFin = scores.mean()*100.0
      their+=scoreFin
      print " With " + fsMethods[3] +": " + str(scoreFin)

"""
if me>=their:
  print "WIN"
else:
  print "FAIL"
"""
