#!/bin/bash

# Change the directories as needed
topDIR=/MemModel

outDIR=${topDIR}/Struct
structDir=${outDIR}
cd $structDir
echo ${structDir}
standard=/MNI152_T1_2mm_brain.nii.gz
for subDIR in sub-CC*;do
echo "Working on ${subDIR}"
sub=${subDIR}
echo "sub is " $sub
outDIR2=${outDIR}/${subDIR}
epidir=${topDIR}/Preprocessing/${sub}/${sub}_NoMCNoSmoothingNonLinearRegDOF12.feat
reg=${epidir}/reg
t1mni=`ls -d ${reg}/highres2standard.nii.gz`
echo $t1mni
if [ -e $t1mni ];then
#checking if FEAT registration was completed
if [ -e $epidir/reg/example_func2highres_fast_wmedge.nii.gz ];then

echo "doing FAST"
fast ${t1mni}
echo "thresholding the masks"
for mask in pve_0 pve_1 pve_2; do
maskIm=${reg}/highres2standard_${mask}.nii.gz
fslmaths ${maskIm} -thr 0.5 -bin ${reg}/${mask}_standard_bin.nii.gz 
done
echo "doing flirt for mc corrected EPI"
flirt -in $epidir/filtered_func_data.nii.gz  -ref ${reg}/example_func2standard.nii.gz  -init ${reg}/example_func2standard.mat -applyxfm -out $epidir/filtered_func_mc_standard
echo "creating epi brain mask"
bet $epidir/filtered_func_mc_standard $epidir/filtered_func_mc_standard_bet
fslmaths ${epidir}/filtered_func_mc_standard_bet.nii.gz -bin ${epidir}/filtered_func_mc_standard_mask
echo "done"
 
fi
fi
done

