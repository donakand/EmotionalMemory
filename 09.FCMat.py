########
# This code is from Dubois et al. (2018)
########

# Force matplotlib to not use any Xwindows backend.
import matplotlib
# core dump with matplotlib 2.0.0; use earlier version, e.g. 1.5.3
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os.path as op
from os import mkdir, makedirs, getcwd, remove, listdir, environ
import sys
import numpy as np
from numpy.polynomial.legendre import Legendre
from scipy import stats, linalg,signal
import scipy.io as sio
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.morphology import binary_closing, binary_dilation, binary_erosion, binary_opening, generate_binary_structure
import nipype.interfaces.fsl as fsl
from subprocess import call, check_output, CalledProcessError, Popen
import nibabel as nib
import sklearn.model_selection as cross_validation
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model,feature_selection,preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import MinCovDet,GraphLassoCV
from nilearn.signal import clean
from nilearn import connectome
import operator
import gzip
import string
import random
import xml.etree.cElementTree as ET
from time import localtime, strftime, sleep, time
import fnmatch
import re
import os
import glob
from statsmodels.nonparametric.smoothers_lowess import lowess
import seaborn as sns
import gzip
import shutil
from sklearn.covariance import MinCovDet,GraphLassoCV,LedoitWolf


#----------------------------------
# initialize global variable config
#----------------------------------
class config(object):
	overwrite          = False
	scriptlist         = list()
	joblist            = list()
	queue              = False
	tStamp             = ''
	useFIX             = False
	useMemMap          = False
	steps              = {}
	Flavors            = {}
	sortedOperations   = list()
	maskParcelswithGM  = False
	preWhitening       = False
	maskParcelswithAll = True
	save_voxelwise     = False
	useNative          = False
	parcellationName   = ''
	parcellationFile   = '/Downloads/shen_2mm_268_parcellation.nii.gz'
	isCifti= False
	nParcels=268
	# these variables are initialized here and used later in the pipeline, do not change
	filtering   = []
	doScrubbing = False
topfolder="/MemModel"
studyfolder=topfolder+"/Preprocessing"
subjectlist=[ name for name in os.listdir(studyfolder) if os.path.isdir(os.path.join(studyfolder, name)) ]
inputfolder=studyfolder+"/"
outputfolder=inputfolder

def getAllFC(subjectList,runs,sessions=None,parcellation=None,operations=None,outputDir=None,isCifti=False,fcMatFile='/storage/shared/research/cinn/2015/MemModel/Dona_Analysis/FCMat/fcMatsFinal.mat',
             kind='correlation',overwrite=True,FCDir=None,mergeSessions=True,mergeRuns=False,cov_estimator=None):
    iii=""
    jj=0
    yy=0
    inputfolder=outputDir
    if (not op.isfile(fcMatFile)) or overwrite:
        if cov_estimator is None:
            cov_estimator=LedoitWolf(assume_centered=False, block_size=1000, store_precision=False)
        measure = connectome.ConnectivityMeasure(
        cov_estimator=cov_estimator,
        kind = kind,
        vectorize=True, 
        discard_diagonal=True)
        if isCifti:
            ext = '.dtseries.nii'
        else:
            ext = '.nii.gz'

        FC_sub = list()
        ts_all = list()
        for subject in subjectList:
            chk=inputfolder+subject+"/"+subject+"_NoMCNoSmoothingNonLinearRegDOF12.feat/output/signalReg.nii.gz"
            if (op.isfile(chk)):
		        print("#################SUBJECT ",subject)
		        config.subject = str(subject)
		        try:
				    ts_sub = list()
				    if sessions:
				      #  print("session section!!")
				        ts_ses = list()
				        for config.session in sessions:
				            ts_run = list()
				            for config.fmriRun in runs:

				                outputPath = outputDir
				                tsFile = inputfolder+subject+"/"+subject+"_NoMCNoSmoothingNonLinearRegDOF12.feat/output_Dubois/NewAllllParcels_Pearson.txt"
				                if tsFile:
				                    # retrieve time courses of parcels
				                    ts        = np.genfromtxt(tsFile,delimiter="\t")
				                else:
				                    continue
				                # standardize
				                ts -= ts.mean(axis=0)
				                ts /= ts.std(axis=0)
				                ts_sub.append(ts) 
				                ts_run.append(ts)
				            if len(ts_run)>0:
				                ts_ses.append(np.concatenate(ts_run,axis=0))  
				        if not mergeSessions and mergeRuns:
				            FC_sub.append(measure.fit_transform(ts_ses)) 
				    else:
				        mergeSessions = False
				        for config.fmriRun in runs:
				            outputPath = outputDir
				            tsFile    = inputfolder+subject+"/"+subject+"_NoMCNoSmoothingNonLinearRegDOF12.feat/output/NewAllParcels_Pearson.txt"
				            ts        = np.genfromtxt(tsFile,delimiter=",")
				            # standardize
				            ts -= ts.mean(axis=0)
				            ts /= ts.std(axis=0)
				            ts_sub.append(ts)
				    if len(ts_sub)>0:
				        ts_all.append(np.concatenate(ts_sub, axis=0))
				        print(measure.fit_transform(ts_sub))
				        print("len tsall",len(ts_all))
				    if not mergeSessions and not mergeRuns:
				       FC_sub.append(measure.fit_transform(ts_sub))
				       print("len fcsub",len(FC_sub))
				    jj=jj+1
				    
		        except:
				    pass
				    print("****************************************************")
				    print("____________________________________________________")
				    print(subject,"FAILED!")
				    print("****************************************************")
				    iii=iii+" "+subject
				    yy=yy+1
                        
        # compute connectivity matrix
        if mergeSessions or (sessions is None and mergeRuns): 
            print(measure.fit_transform(ts_sub))
            fcMats = measure.fit_transform(ts_all)
        else: 
            fcMats = np.vstack([np.mean(el,axis=0) for el in FC_sub])
        # SAVE fcMats
        results      = {}
        results['fcMats'] = fcMats
        results['subjects'] = subjectList
        results['runs'] = np.array(runs)
        if sessions: results['sessions'] = np.array(sessions)
        results['kind'] = kind
        print("/////////will save mat now")
        sio.savemat(fcMatFile, results)
        return iii,jj,yy
    else:
        results = sio.loadmat(fcMatFile)
        return iii,jj,yy
runs="1"

[i,j,yy]=getAllFC(subjectlist,runs,sessions=None,parcellation=None,operations=None,outputDir=outputfolder,isCifti=False,fcMatFile='/MemModel/FCMat/fcMatsFinal.mat', kind='correlation',overwrite=True,FCDir=None,mergeSessions=False,mergeRuns=False,cov_estimator=None)
print("##########Failed FCMAT")
print(i)
print("##############Number of completed FCMATs")
print(j)
print("##############Number of failed FCMATs")
print(yy)



