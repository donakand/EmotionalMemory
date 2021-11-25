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

#import nistats
#from nistats import design_matrix
i=""

topfolder="/MemModel"
studyfolder=topfolder+"/Preprocessing"
allsubs=[ name for name in os.listdir(studyfolder) if os.path.isdir(os.path.join(studyfolder, name)) ]
for subject in allsubs:
	print(subject)
	inputfolder=studyfolder+"/"+subject+"/"+subject+"_NoMCNoSmoothingNonLinearRegDOF12.feat"
	niiImg=inputfolder+"/output/signalReg.nii.gz"
	motionFile = studyfolder+"/"+subject+"/mc/Movement_Regressors2.txt"
	subno=subject[6:12]
	structfolder=topfolder+"/Struct/"+subno+"/"
	maskAll=inputfolder+"/filtered_func_mc_standard_mask.nii.gz"
	overwrite=True
	if op.isfile(niiImg):
		print("### EPI found")
		outDir=inputfolder+"/output/"
		if not op.isfile(inputfolder+"/output/allParcels.txt"):
			print("Applying parcellation")
		#	try:
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
				fmriFile=niiImg
				fmriFile_dn=niiImg
				nParcels=268
				# these variables are initialized here and used later in the pipeline, do not change
				filtering   = []
				doScrubbing = False
				isCifti = False
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


#################### Starting Parcellation
		if not op.isfile(inputfolder+"/output/allParcels.txt"):
			# After preprocessing, functional connectivity is computed
			[loaded_image, nRows, nCols, nSlices, nTRs, affine, TR,header]=load_img(niiImg,None,False)
			TR=1.97
			imgInfo=[nRows,nCols,nSlices,nTRs,affine,TR,header]
			maskAll = np.asarray(nib.load(maskAll).dataobj)>0
			maskAll_=maskAll.reshape(nRows*nCols*nSlices, order='F')
			tsDir = outDir

			# read parcels
			if not config.isCifti:
				if not config.maskParcelswithAll:     
					maskAll  = np.ones(np.shape(maskAll), dtype=bool)
				allparcels, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(config.parcellationFile, maskAll_,False)		
			alltsFile = op.join(tsDir,'allParcels.txt')
			if not op.isfile(alltsFile) or overwrite:
				# read original volume
				if not config.isCifti:
					data, nRows, nCols, nSlices, nTRs, affine, TR, header = load_img(config.fmriFile, maskAll_,False)
			
				for iParcel in np.arange(config.nParcels):
					tsFile = op.join(tsDir,'parcel{:03d}.txt'.format(iParcel+1))
					if not op.isfile(tsFile) or overwrite:
					    np.savetxt(tsFile,np.nanmean(data[np.where(allparcels==iParcel+1)[0],:],axis=0),fmt='%.16f',delimiter='\n')

				# concatenate all ts
				print('Concatenating data')
				cmd = 'paste '+op.join(tsDir,'parcel???.txt')+' > '+alltsFile
				call(cmd, shell=True)
	######################## Parcellation done

	####################### Creating FCmats (computeFC function)
				FCDir = outDir
				alltsFile = op.join(tsDir,'allParcels.txt')
				fcFile     = alltsFile.replace('.txt','_Pearson.txt')
				if not op.isfile(fcFile) or overwrite:
					ts = np.loadtxt(alltsFile)
						# correlation
					corrMat = np.corrcoef(ts,rowvar=0)
				# np.fill_diagonal(corrMat,1)
				# save as .txt
					np.savetxt(fcFile,corrMat,fmt='%.6f',delimiter=',')
