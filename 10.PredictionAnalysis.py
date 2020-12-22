#!/home/opt-user/Enthought/Canopy_64bit/User/bin/python

# from Dubois et al. (2018)

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

#import nistats
#from nistats import design_matrix
i=""
topfolder="/storage/shared/research/cinn/2015/MemModel"
inputfolder=topfolder+"/Dona_Analysis/regularisedreg"



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
	DATADIR='/storage/shared/research/cinn/2015/MemModel/Dona_Analysis/regularisedreg'
	pipelineName='one'
	release='a'
	behavFile='a'

def defConVec(df,confound):
    if confound == 'gender':
        conVec = df['gender_code']
    elif confound == 'age':
        conVec = df['age']
    elif confound == 'intelligenceFactor':
        conVec = df['intelligenceFactor']
    elif confound == 'handedness':
        conVec = np.square(df['Hand'])
    return conVec

#  @brief Compute prediction of subject measure (output saved to file)
#  
#  @param     [str] fcMatFile filename of functional connectivity matrices file
#  @param     [str] dataFile filename of subject measures data frame
#  @param     [int] test_index index of the subject whose score will be predicted
#  @param     [float] filterThr threshold for p-value to select edges correlated with subject measure
#  @param     [str] keepEdgeFile name of file containing a mask to select a subset of edge (optional)
#  @iPerm     [array_like] vector of permutation indices, if [0] no permutation test is run
#  @SM        [str] subject measure name
#  @session   [str] session identifier (one of 'REST1', 'REST2, 'REST12')
#  @model     [str] regression model type (either 'Finn' or 'elnet')
#  @outDir    [str] output directory
#  @confounds [list] confound vector
#  
#  @details The edges of FC matrix are used to build a linear regression model to predict the subject measure. Finn model uses a first degree 
#  polynomial to fit the subject measure as a function of the sum of edge weights. Elastic net builds a multivariate model using the edges of 
#  the FC matrix as features. In both models, only edges correlated with the subject measure on training data are selected, and counfounds are
#  regressed out from the subject measure. If requested, a permutation test is also run.
#  

fcMatFile=topfolder+"/Dona_Analysis/FCMat/FCMatsFinal.mat"
dataFile=inputfolder+"/CAMCanData.csv"
colfile=topfolder+"/Dona_Analysis/Scripts/edgesNames.mat"
df1=pd.read_csv(dataFile)
outDir=inputfolder


def runPredictionJD(fcMatFile, dataFile, test_index, filterThr=0.01, keepEdgeFile='', iPerm=[0], SM='EmEffect', session='REST12', decon='decon', fctype='Pearson', model='Finn', outDir='', confounds=['age','intelligenceFactor']):
    data         = sio.loadmat(fcMatFile)
    edges        = data['fcMats']

    if len(keepEdgeFile)>0:
        keepEdges = np.loadtxt(keepEdgeFile).astype(bool)
        edges     = edges[:,keepEdges]

    n_subs       = edges.shape[0]
    print("nsubs is",n_subs)
    n_edges      = edges.shape[1]
    print("nedges is",n_edges)
    df           = pd.read_csv(dataFile)
    score        = np.array(np.ravel(df[SM]))
    train_index = np.setdiff1d(np.arange(n_subs),test_index)
    print("test index",test_index)
    print("train_index",train_index)

    # REMOVE CONFOUNDS
    conMat = None
    if len(confounds)>0:
        for confound in confounds:
                conVec = defConVec(df,confound)
            # add to conMat
                if conMat is None:
                        conMat = np.array(np.ravel(conVec))
                else:
                        print(confound,conMat.shape,conVec.shape)
                        conMat = np.vstack((conMat,conVec))
        # if only one confound, transform to matrix
        if len(confounds)==1:
            conMat = conMat[:,np.newaxis]
            print("only one confound")
        else:
            conMat = conMat.T

        corrBef = []
        for i in range(len(confounds)):
            corrBef.append(stats.pearsonr(conMat[:,i].T,score)[0])
        print('maximum corr before decon: ',max(corrBef))

        regr        = linear_model.LinearRegression()
        regr.fit(conMat[train_index,:], score[train_index])
        fittedvalues = regr.predict(conMat)
        score        = score - np.ravel(fittedvalues)

        corrAft = []
        for i in range(len(confounds)):
            corrAft.append(stats.pearsonr(conMat[:,i].T,score)[0])
        print('maximum corr after decon: ',max(corrAft))

    # keep a copy of score
    score_ = np.copy(score)
    for thisPerm in iPerm: 

        score = np.copy(score_)
        if thisPerm > 0:
            # read permutation indices
            permInds = np.loadtxt(op.join(outDir,'permInds.txt'),dtype=np.int16)
            score    = score[permInds[thisPerm-1,:]]

        if not op.isdir(op.join(outDir,'{:04d}'.format(thisPerm))):
            mkdir(op.join(outDir,'{:04d}'.format(thisPerm)))
        outFile = op.join(outDir,'{:04d}'.format(thisPerm),'{}.mat'.format(
            '_'.join(['%s' % test_sub for test_sub in df['Subject'][test_index]])))

        if op.isfile(outFile) and not config.overwrite:
            continue
     
        # compute univariate correlation between each edge and the Subject Measure
        pears  = [stats.pearsonr(np.squeeze(edges[train_index,j]),score[train_index]) for j in range(0,n_edges)]

        pearsR = [pears[j][0] for j in range(0,n_edges)]
        print("pears",pears)
        idx_filtered     = np.array([idx for idx in range(0,n_edges) if pears[idx][1]<filterThr])
        print("idxfiltered",idx_filtered)
        idx_filtered_pos = np.array([idx for idx in range(0,n_edges) if pears[idx][1]<filterThr and pears[idx][0]>0])
        print("idx filtered pos",idx_filtered_pos)
        idx_filtered_neg = np.array([idx for idx in range(0,n_edges) if pears[idx][1]<filterThr and pears[idx][0]<0])
        print("idx filtered neg",idx_filtered_neg)
            
        if model=='Finn':
            print(model)
            lr  = linear_model.LinearRegression()
            # select edges (positively and negatively) correlated with score with threshold filterThr
            filtered_pos = edges[np.ix_(train_index,idx_filtered_pos)]
            filtered_neg = edges[np.ix_(train_index,idx_filtered_neg)]
            # compute network statistic for each subject in training
            strength_pos = filtered_pos.sum(axis=1)
            strength_neg = filtered_neg.sum(axis=1)
            strength_posneg = strength_pos - strength_neg
            # compute network statistic for test subjects
            str_pos_test = edges[np.ix_(test_index,idx_filtered_pos)].sum(axis=1)
            str_neg_test = edges[np.ix_(test_index,idx_filtered_neg)].sum(axis=1)
            str_posneg_test = str_pos_test - str_neg_test
            # regression
            lr_posneg           = lr.fit(np.stack((strength_pos,strength_neg),axis=1),score[train_index])
            predictions_posneg  = lr_posneg.predict(np.stack((str_pos_test,str_neg_test),axis=1))
            lr_pos_neg          = lr.fit(strength_posneg.reshape(-1,1),score[train_index])
            predictions_pos_neg = lr_posneg.predict(str_posneg_test.reshape(-1,1))
            lr_pos              = lr.fit(strength_pos.reshape(-1,1),score[train_index])
            predictions_pos     = lr_pos.predict(str_pos_test.reshape(-1,1))
            lr_neg              = lr.fit(strength_neg.reshape(-1,1),score[train_index])
            predictions_neg     = lr_neg.predict(str_neg_test.reshape(-1,1))
            results = {'score':score[test_index],'pred_posneg':predictions_posneg,'pred_pos_neg':predictions_pos_neg, 'pred_pos':predictions_pos, 'pred_neg':predictions_neg,'idx_filtered_pos':idx_filtered_pos, 'idx_filtered_neg':idx_filtered_neg}
            print('saving results')
            sio.savemat(outFile,results)
        
        elif model=='elnet':
            X_train, X_test, y_train, y_test = edges[np.ix_(train_index,idx_filtered)], edges[np.ix_(test_index,idx_filtered)], score[train_index], score[test_index]
            rbX            = RobustScaler()
            X_train        = rbX.fit_transform(X_train)
            # equalize distribution of score for cv folds
            n_bins_cv      = 4
            hist_cv, bin_limits_cv = np.histogram(y_train, n_bins_cv)
            bins_cv        = np.digitize(y_train, bin_limits_cv[:-1])
            # set up nested cross validation 
            nCV_gridsearch = 3
            cv             = cross_validation.StratifiedKFold(n_splits=nCV_gridsearch)       
            elnet          = ElasticNetCV(l1_ratio=[.01],n_alphas=50,cv=cv.split(X_train, bins_cv),max_iter=1500,tol=0.001)
            # TRAIN
            start_time     = time()
            elnet.fit(X_train,y_train)
            elapsed_time   = time() - start_time
            print("Trained ELNET in {0:02d}h:{1:02d}min:{2:02d}s".format(int(elapsed_time//3600),int((elapsed_time%3600)//60),int(elapsed_time%60)))
            # PREDICT
            X_test         = rbX.transform(X_test)
            if len(X_test.shape) == 1:
                X_test     = X_test.reshape(1, -1)
            prediction     = elnet.predict(X_test)
            results        = {'score':y_test,'pred':prediction, 'coef':elnet.coef_, 'alpha':elnet.alpha_, 'l1_ratio':elnet.l1_ratio_, 'idx_filtered':idx_filtered}
            print('saving results')
            sio.savemat(outFile,results)  

        plt.close('all')
        feature_importance = pd.Series(index = idx_filtered, data = np.abs(elnet.coef_))
        n_selected_features = (feature_importance>0).sum()
        print('{0:d} features, reduction of {1:2.2f}%'.format(n_selected_features,(1-n_selected_features/len(feature_importance))*100))
        feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (18,6))
        testindex=str(test_index).strip('[]')
        plt.savefig(inputfolder+'/ImportantFeatures_'+testindex+'.jpg')
       
        

def runPredictionParJD(fcMatFile, dataFile, SM='EmEffect', iPerm=[0], confounds=['gender','age','age^2','gender*age','gender*age^2','brainsize','motion','recon'], launchSubproc=False, session='REST12',decon='decon',fctype='Pearson',model='Finn', outDir = '', filterThr=0.01, keepEdgeFile=''):
    data = sio.loadmat(fcMatFile)
    df   = pd.read_csv(dataFile)
    # leave one family out
    iCV = 0
    config.scriptlist = []
    for el in np.unique(df['Family_ID']):
        test_index    = list(df.ix[df['Family_ID']==el].index)
        test_subjects = list(df.ix[df['Family_ID']==el]['Subject'])
        jPerm = list()
        for thisPerm in iPerm:
            outFile = op.join(outDir,'{:04d}'.format(thisPerm),'{}.mat'.format(
                '_'.join(['%s' % test_sub for test_sub in test_subjects])))

            if not op.isfile(outFile) or config.overwrite:
                jPerm.append(thisPerm)
        if len(jPerm)==0:
            iCV = iCV + 1 
            continue
        jobDir = op.join(outDir, 'jobs')
        if not op.isdir(jobDir): 
            mkdir(jobDir)
        jobName = 'f{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(el,config.pipelineName,config.parcellationName,SM, model,config.release,session,decon,fctype)
        # make a script
        thispythonfn  = '<< END\nimport sys\nsys.path.insert(0,"{}")\n'.format(getcwd())
        thispythonfn += 'from HCP_helpers import *\n'
        thispythonfn += 'logFid                  = open("{}","a+")\n'.format(op.join(jobDir,jobName+'.log'))
        thispythonfn += 'sys.stdout              = logFid\n'
        thispythonfn += 'sys.stderr              = logFid\n'
        # print date and time stamp
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'print(strftime("%Y-%m-%d %H:%M:%S", localtime()))\n'
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'config.DATADIR          = "{}"\n'.format(config.DATADIR)
        thispythonfn += 'config.pipelineName     = "{}"\n'.format(config.pipelineName)
        thispythonfn += 'config.parcellationName = "{}"\n'.format(config.parcellationName)
        #        thispythonfn += 'config.outScore         = "{}"\n'.format(config.outScore)
        thispythonfn += 'config.release          = "{}"\n'.format(config.release)
        thispythonfn += 'config.behavFile        = "{}"\n'.format(config.behavFile)
        thispythonfn += 'config.overwrite        = {}\n'.format(config.overwrite)
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'print("runPredictionJD(\'{}\',\'{}\')")\n'.format(fcMatFile, dataFile)
        thispythonfn += 'print("=========================")\n'
        thispythonfn += 'print("=========================")\n'
        str1 =  '['+','.join(['%s' % test_ind for test_ind in test_index])+']'
        str2 =  '['+','.join(['"%s"' % el for el in confounds])+']'
        str3 =  '['+','.join(['%s' % el for el in jPerm])+']'
        thispythonfn += 'runPredictionJD("{}","{}",'.format(fcMatFile, dataFile)
        thispythonfn += ' {}, filterThr={}, keepEdgeFile="{}", SM="{}"'.format(str1,filterThr, keepEdgeFile, SM)
        thispythonfn += ', session="{}", decon="{}", fctype="{}", model="{}", outDir="{}", confounds={},iPerm={})\n'.format(session, decon, fctype, model, outDir, str2,str3)
        thispythonfn += 'logFid.close()\n'
        thispythonfn += 'END'
        thisScript=op.join(jobDir,jobName+'.sh')
        while True:
            if op.isfile(thisScript) and (not config.overwrite):
                thisScript=thisScript.replace('.sh','+.sh') # use fsl feat convention
            else:
                break
        with open(thisScript,'w') as fidw:
            fidw.write('#!/bin/bash\n')
            fidw.write('python {}\n'.format(thispythonfn))
        cmd='chmod 774 '+thisScript
        call(cmd,shell=True)
        #this is a "hack" to make sure the .sh script exists before it is called... 
        while not op.isfile(thisScript):
            sleep(.05)
        if config.queue:
            # call to fnSubmitToCluster
            config.scriptlist.append(thisScript)
            sys.stdout.flush()
        elif launchSubproc:
            sys.stdout.flush()
            process = Popen(thisScript,shell=True)
            config.joblist.append(process)
        else:
            runPredictionJD(fcMatFile,dataFile,test_index,filterThr=filterThr,keepEdgeFile=keepEdgeFile,SM=SM, session=session, decon=decon, fctype=fctype, model=model, outDir=outDir, confounds=confounds,iPerm=jPerm)
        iCV = iCV +1
    
    if len(config.scriptlist)>0:
        ## launch array job
        JobID = fnSubmitJobArrayFromJobList()
        config.joblist.append(JobID.split(b'.')[0])

#############################################################
############### Running the functions #######################
#############################################################

nPerm=1000
outFileStats = op.join(inputfolder,'fitStats.mat')
thisOutDir=inputfolder
model='elnet'
df=df1
jPerm=list()
for thisPerm in range(nPerm+1): 
                        jPerm.append(thisPerm)
permInds=np.vstack([np.random.permutation(range(df1.shape[0])) for i in range(nPerm)])
np.savetxt(op.join(inputfolder,'permInds.txt'),permInds)
#SM is memItemEmEnh for the emotional enhancement model. memItemPosEff for the positivity effect model. 
runPredictionParJD(fcMatFile, dataFile, SM='memItemEmEnh', iPerm=jPerm, confounds=[], launchSubproc=False, session='REST12',decon='decon',fctype='Pearson',model='elnet', outDir = inputfolder, filterThr=0.01, keepEdgeFile='')
if not op.isfile(outFileStats):
    fitMeas      = ['Pearson r','R-squared','nRMSD']
    fitPerm      = np.empty([nPerm+1,3])  
    fitPerm[:]   = np.nan
    for thisPerm in range(nPerm+1): 
        resDir    = op.join(thisOutDir,'{0:04d}'.format(thisPerm))
        ## MAKE SURE RESULTS.MAT EXISTS, AND HAS OBSERVED DATA SAVED
        resFile   = op.join(resDir,'result.mat')
        rePackage = False
        if (not op.isfile(resFile)) or config.overwrite:
            rePackage = True
        else:
            res       = sio.loadmat(resFile)
            if 'obs' not in res.keys():
                rePackage = True
        if rePackage: 
            obs              = np.zeros([df.shape[0],1])  ###### make obs variable with the no. of participants in the study
            pred             = np.zeros([df.shape[0],1]) ##### makes a pred variable with the no. of participants in the study
            for el in np.unique(df['Family_ID']): ##########  goes on to each subject. 
                test_index    = list(df.ix[df['Family_ID']==el].index)
                test_subjects = list(df.ix[df['Family_ID']==el]['Subject'])
                cvResFile = op.join(resDir,'{}.mat'.format(
                    '_'.join(['%s' % test_sub for test_sub in test_subjects])))
                try:
                    results = sio.loadmat(cvResFile)  #######  opens participant's mat file. 
                except:
                    print 'Missing prediction!'+cvResFile

                if model=='Finn':
                    pred_pos[test_index]    = results['pred_pos'].T
                    pred_neg[test_index]    = results['pred_neg'].T
                    pred_posneg[test_index] = results['pred_posneg'].T
                else:
                    pred[test_index] = results['pred'].T   
                obs[test_index]      = results['score'].T   

            if model=='Finn':
                rho_posneg,p_posneg   = stats.pearsonr(np.ravel(pred_posneg),np.ravel(obs))
                rho_pos,p_pos         = stats.pearsonr(np.ravel(pred_pos),np.ravel(obs))
                rho_neg,p_neg         = stats.pearsonr(np.ravel(pred_neg),np.ravel(obs))
                res = {'obs':obs,
                           'pred_posneg':pred_posneg, 'rho_posneg':rho_posneg, 'p_posneg': p_posneg,
                           'pred_pos':pred_pos, 'rho_pos':rho_pos, 'p_pos': p_pos,
                           'pred_neg':pred_neg, 'rho_neg':rho_neg, 'p_neg': p_neg}
            else:
                rho,p   = stats.pearsonr(np.ravel(pred),np.ravel(obs)) ###### calculate the r and p for this permutation
                res = {'obs':obs,
                           'pred':pred, 'rho':rho, 'p': p}
            # save result
            sio.savemat(resFile,res)  ####### save this permutation's: obs, preds, r, p-value
        else:
            res       = sio.loadmat(resFile)

        if model == 'Finn': 
            pred = res['pred_pos'].reshape(-1,)
            obs  = res['obs'].reshape(-1,)
        else:
            pred = res['pred'].reshape(-1,)    
            obs  = res['obs'].reshape(-1,)     

        fitPerm[thisPerm,0] = stats.pearsonr(obs,pred)[0] 
        # compute R^2 for this permutation
        fitPerm[thisPerm,1] = 1 - np.sum(np.square(obs-pred)) / np.sum(np.square(obs-np.mean(obs)))
        # compute nRMSD = sqrt(1-R^2) for this permutation
        fitPerm[thisPerm,2] = np.sqrt(1 - fitPerm[thisPerm,1])          
    res = {'fitPerm':fitPerm,'fitMeas':fitMeas}
    # save result
    sio.savemat(outFileStats,res)
else:
    res     = sio.loadmat(outFileStats)
    fitPerm = res['fitPerm'] 
    fitMeas = res['fitMeas'] 

## Gets the p-value of R-squared of our model.
p1 = ((nPerm-np.where(np.argsort(np.ravel(fitPerm[:,1]))==0)[0][0])+1)/np.float(nPerm)


print("****************** p-value of the model is",p1)
