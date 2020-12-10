#!/home/opt-user/Enthought/Canopy_64bit/User/bin/python

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
    parcellationFile   = ''
    # these variables are initialized here and used later in the pipeline, do not change
    filtering   = []
    doScrubbing = False
#------------------------------------
################################## FIRST LOAD #####################################
#  @brief Load Nifti data 
#  
#  @param  [str] volFile filename of volumetric file to be loaded
#  @param  [numpy.array] maskAll whole brain mask
#  @param  [bool] unzip True if memmap should be used to load data
#  @return [tuple] image data, no. of rows in image, no. if columns in image, no. of slices in image, no. of time points, affine matrix and repetition time
#  
def load_img(volFile,maskAll,unzip):
    if unzip:
        volFileUnzip = volFile.replace('.gz','') 
        if not op.isfile(volFileUnzip):
            with open(volFile, 'rb') as fFile:
                decompressedFile = gzip.GzipFile(fileobj=fFile)
                with open(volFileUnzip, 'wb') as outfile:
                    outfile.write(decompressedFile.read())
        img = nib.load(volFileUnzip)
    else:
        img = nib.load(volFile)

    try:
        nRows, nCols, nSlices, nTRs = img.header.get_data_shape()
    except:
        nRows, nCols, nSlices = img.header.get_data_shape()
        nTRs = 1
    TR = img.header.structarr['pixdim'][4]
    data = np.asarray(img.dataobj).reshape((nRows*nCols*nSlices,nTRs), order='F')[maskAll,:]

    return data, nRows, nCols, nSlices, nTRs, img.affine, TR, img.header
	
#########################################################################################
topfolder="/storage/shared/research/cinn/2015/MemModel"
studyfolder=topfolder+"/Dona_Analysis/Preprocessing_FSLthenDubois"
allsubs=[ name for name in os.listdir(studyfolder) if os.path.isdir(os.path.join(studyfolder, name)) ]
for subject in allsubs:

	print(subject)
	inputfolder=studyfolder+"/"+subject+"/"+subject+"_NoMCNoSmoothingNonLinearRegDOF12.feat"
	niiImg=inputfolder+"/filtered_func_mc_standard.nii.gz"
	motionFile = studyfolder+"/"+subject+"/mc/Movement_Regressors2.txt"
	subno=subject[6:12]
	print(subno)
	structfolder=inputfolder+"/reg/"

	if op.isfile(niiImg):
		if not op.isfile(inputfolder+"/output_Dubois/signalReg.nii.gz"):
			print("EPI exsits")
			outDir=inputfolder+"/output_Dubois/"
			if not op.isdir(outDir):
				os.mkdir(outDir)
			#no overwrite if "orIm.nii.gz", other values would mean: overwrite
			chk=outDir+"orIm.nii.gz"
			if not op.isfile(chk):
				print("SUBJECT IS ",subject)
				[loaded_image, nRows, nCols, nSlices, nTRs, affine, TR,header]=load_img(niiImg,None,False)
				##fix the TR number. 
				TR=1.97
				imgInfo=[nRows,nCols,nSlices,nTRs,affine,TR,header]
				################# Reading Mask Files #################
				maskAll=inputfolder+"/filtered_func_mc_standard_mask.nii.gz"
				maskWM=structfolder+"pve_2_standard_bin.nii.gz"
				print(maskWM)
				maskCSF=structfolder+"pve_0_standard_bin.nii.gz"
				maskGM=structfolder+"pve_1_standard_bin.nii.gz"


				#############LOAD WM MASK##############
				ref = nib.load(maskWM)
				maskWM = np.asarray(nib.load(maskWM).dataobj)>0
				#############LOAD GM MASK#############
				ref = nib.load(maskGM)
				maskGM = np.asarray(nib.load(maskGM).dataobj)>0
				###########LOAD CSF MASK##############
				ref = nib.load(maskCSF)
				maskCSF = np.asarray(nib.load(maskCSF).dataobj)>0
				#############Creating maskALL###############
				maskAll = np.asarray(nib.load(maskAll).dataobj)>0
				maskAll_=maskAll
				#######################################
				############LOAD REST EPI#############
				#######################################
				#loading EPI image
				restImOr = np.asarray(nib.load(niiImg).dataobj)
				print("restimOr shape",restImOr.shape)
				################################################
				##########RESHAPE TO (R*C*S, nTR) ############
				################################################
			
				maskAll_=maskAll.reshape(nRows*nCols*nSlices, order='F')
				maskWM_=maskWM.reshape(nRows*nCols*nSlices, order='F')
				maskCSF_=maskCSF.reshape(nRows*nCols*nSlices, order='F')
				maskGM_=maskGM.reshape(nRows*nCols*nSlices, order='F')
				maskGM_ = maskGM_[maskAll_]
				maskCSF_ = maskCSF_[maskAll_]
				maskWM_  = maskWM_[maskAll_]

				masks=[maskAll_,maskWM_,maskCSF_,maskGM_]
				restim=restImOr.reshape(nRows*nCols*nSlices,nTRs, order='F')
				###### Create a Brain Extracted Image (betIm)
				betim=restim[maskAll_,:]
				print("betIm shape",betim.shape)

				#########################################################
				#######################STEP1#############################
				#########################################################

				outFile='orImg'
				nib.save(nib.Nifti1Image(restim.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))

				def VoxelNormalization(niiImg, flavor):
					if flavor == 'zscore':
						niiImg = stats.zscore(niiImg, axis=1, ddof=1)      
					return niiImg
				print("##############step1#############")
				betim=VoxelNormalization(betim,'zscore')
				outFile='voxelNormalized'
				####saving the ouput. First creating an empty image, then adding the betted image to it
				niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
				niiImg[maskAll_,:] = betim
				nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))
				########################################################
				######################STEP2#############################
				########################################################
				def regress(data, nTRs, TR, regressors, preWhitening=False):
					print('Starting regression with {} regressors...'.format(regressors.shape[1]))
					X  = np.concatenate((np.ones([nTRs,1]), regressors), axis=1)
					N = data.shape[0]
					start_time = time()
					fit = np.linalg.lstsq(X, data.T)[0]
					fittedvalues = np.dot(X, fit)
					print("fval")
					resid = data - fittedvalues.T
					data = resid
					elapsed_time = time() - start_time
					print('Regression completed in {:02d}h{:02d}min{:02d}s'.format(int(np.floor(elapsed_time/3600)),int(np.floor((elapsed_time%3600)/60)),int(np.floor(elapsed_time%60)))) 
					return data
				def legendre_poly(order, nTRs):
					# ** a) create polynomial regressor **
					x = np.arange(nTRs)
					x = x - x.max()/2
					num_pol = range(order+1)
					y = np.ones((len(num_pol),len(x)))   
					coeff = np.eye(order+1)
	
					for i in num_pol:
						myleg = Legendre(coeff[i])
						y[i,:] = myleg(x) 
						if i>0:
							y[i,:] = y[i,:] - np.mean(y[i,:])
							y[i,:] = y[i,:]/np.max(y[i,:])
					return y
				def Detrending(niiImg, flavor, masks, imgInfo):
					maskAll, maskWM_, maskCSF_, maskGM_ = masks
					nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
					nPoly = flavor[1] + 1   
					volData = niiImg
					if flavor[2] == 'WMCSF':
						niiImgWMCSF = volData[np.logical_or(maskWM_,maskCSF_),:]
						if flavor[0] == 'legendre':
							y = legendre_poly(flavor[1],nTRs)                
						niiImgWMCSF = regress(niiImgWMCSF, nTRs, TR, y[1:nPoly,:].T, config.preWhitening)
						volData[np.logical_or(maskWM_,maskCSF_),:] = niiImgWMCSF
					elif flavor[2] == 'GM':
						niiImgGM = volData[maskGM_,:]
						if flavor[0] == 'legendre':
							y = legendre_poly(flavor[1], nTRs)
						niiImgGM = regress(niiImgGM, nTRs, TR, y[1:nPoly,:].T, config.preWhitening)
						volData[maskGM_,:] = niiImgGM
					niiImg = volData            
					return niiImg   
				flavor=['legendre',3,'WMCSF']
				print("#########step 2#########")
				betim=Detrending(betim,flavor,masks,imgInfo)
				outFile='detrended'
				#saving the detrended image
				niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
				niiImg[maskAll_,:] = betim
				nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))
				#################################################
				#######################STEP3####################
				################################################
				flavor=['WMCSF', 'GM']
				def TissueRegression(niiImg, flavor, masks, imgInfo):
					maskAll, maskWM_, maskCSF_, maskGM_ = masks
					nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo
	
					volData = niiImg
					if flavor[0] == 'WMCSF':
						meanWM = np.mean(np.float32(volData[maskWM_,:]),axis=0)
						meanWM = meanWM - np.mean(meanWM)
						meanWM = meanWM/max(meanWM)
						meanCSF = np.mean(np.float32(volData[maskCSF_,:]),axis=0)
						meanCSF = meanCSF - np.mean(meanCSF)
						meanCSF = meanCSF/max(meanCSF)
						X  = np.concatenate((meanWM[:,np.newaxis], meanCSF[:,np.newaxis]), axis=1)   
					else:
						print('Warning! Wrong tissue regression flavor. Nothing was done')  
					if flavor[-1] == 'GM':       
						niiImgGM = volData[maskGM_,:]
						niiImgGM = regress(niiImgGM, nTRs, TR, X, config.preWhitening)
						volData[maskGM_,:] = niiImgGM
						niiImg = volData    
						return niiImg
					elif flavor[-1] == 'wholebrain':
						return X
					else:
						print('Warning! Wrong tissue regression flavor. Nothing was done')
				print("#######step3########")
				betim=TissueRegression(betim, flavor, masks, imgInfo)
				#saving the output
				outFile='tissueReg'
				niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
				niiImg[maskAll_,:] = betim
				nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))
				######################################################
				######################################################
				#						STEP 4						 #
				######################################################
				######################################################
				flavor=['R dR']


				def MotionRegression(niiImg, motionFile, flavor, masks, imgInfo,filtering, configdoScrubbing):
					# assumes that data is organized as in the HCP   
					data = np.genfromtxt(motionFile)
					if flavor[0] == 'R':
						X = data[:,:6]
					elif flavor[0] == 'R dR':
						X = data
					else:
						print('Wrong flavor, using default regressors: R dR')
						X = data         
					return X
				print("#########step4#######")
				r0=MotionRegression(betim, motionFile, flavor, masks, imgInfo,[], False)
				print("NTRS",nTRs)
				print("TR",TR)
				betim = regress(betim, nTRs, TR, r0, False)
				#saving the output
				outFile='motionReg'
				niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
				niiImg[maskAll_,:] = betim
				nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))
				##############################################################
				#####################STEP 5###################################
				##############################################################
				flavor=['Gaussian', 1]
				doScrubbing=False
				def TemporalFiltering(niiImg, flavor, masks, imgInfo):
					maskAll, maskWM_, maskCSF_, maskGM_ = masks
					nRows, nCols, nSlices, nTRs, affine, TR, header = imgInfo   
					data = niiImg
					if flavor[0] == 'Gaussian':
						w = signal.gaussian(7,std=flavor[1])
						niiImg = signal.lfilter(w,1,data)
					else:
						print('Warning! Wrong temporal filtering flavor. Nothing was done')      
				#    config.filtering = flavor
					return niiImg
				print("#######Step5########")
				betim = TemporalFiltering(betim,flavor,masks,imgInfo)
				outFile='tempfil'
				#saving the output
				niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
				niiImg[maskAll_,:] = betim
				nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))
				#######################################################
				###################STEP 6##############################
				######################################################
				flavor=['legendre', 3 ,'GM']
				print("######step6######")
				betim=Detrending(betim, flavor, masks, imgInfo)
				#saving the output
				outFile='det2_rio'
				niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
				niiImg[maskAll_,:] = betim
				nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))

				####################################################
				####################STEP 7##########################
				#####################################################

				flavor=['GS']
				def GlobalSignalRegression(niiImg, flavor, masks, imgInfo):
					meanAll = np.mean(niiImg,axis=0)
					meanAll = meanAll - np.mean(meanAll)
					meanAll = meanAll/np.max(meanAll)
					if flavor[0] == 'GS':
						return meanAll[:,np.newaxis]
					else:
						print('Warning! Wrong normalization flavor. Using defalut regressor: GS')
						return meanAll[:,np.newaxis]
				print("####step7#####")
				r0=GlobalSignalRegression(betim, flavor, masks, imgInfo)
				betim = regress(betim, nTRs, TR, r0, config.preWhitening)
				#saving the output
				outFile='signalReg'
				niiImg = np.zeros((nRows*nCols*nSlices, nTRs),dtype=np.float32)
				niiImg[maskAll_,:] = betim
				nib.save(nib.Nifti1Image(niiImg.reshape((nRows, nCols, nSlices, nTRs), order='F').astype('<f4'), affine, header=header),op.join(outDir,outFile+'.nii.gz'))
			print("finished")
