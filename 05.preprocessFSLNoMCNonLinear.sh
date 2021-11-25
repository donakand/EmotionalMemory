#!/bin/bash
topDIR=/MemModel
funcDIR=${topDIR}/CamCAN_Data/func
outputDIR=${topDIR}/Preprocessing
fsfDIR=${topDIR}/Scripts/FSL/fsf
structDIR=${topDIR}/Struct
fmapdir=${topDIR}/FieldMaps
memDIR=/EmotionalMemory
cd ${funcDIR}
betepidir=${topDIR}/BET_EPI
dof=12
if [ ! -e $betepidir ];then
mkdir $betepidir
fi
for subDIR in sub-CC*;do


fsforig=${fsfDIR}/preprocessingNoMCNoSmoothingNonLinearDOF${dof}.fsf
if [ ! -e ${betepidir}/${subDIR} ];then
mkdir ${betepidir}/${subDIR}
fi
# check if the analysis was already done
if [ ! -e ${outputDIR}/${subDIR}/${subDIR}_NoSmoothingNonLinearRegDOF${dof}.feat/filtered_func_data.ica ]; then
subNo=`echo $subDIR | awk -F "-" '{ print $2 }'`
sub=`echo $subDIR | awk -F "-" '{ print $2 }' | sed s@CC@@g`
# check if T1s exist
if [ -e ${structDIR}/${subDIR} ]; then
#check if memory scores exist
if [ -e ${memDIR}/EmotionalMemory_${subNo}_scored.txt ];then

echo "%%%%%%%%%%%%%%%%%%%%%:"
echo "subdir"
echo $subDIR


input=`ls -d $outputDIR/$subDIR/mc/filtered_func_data_mc.nii.gz`
chmod 770 $input

fsf=${fsfDIR}/temp/temp2_${sub}_${dof}.fsf

cp $fsforig $fsf


# Output directory
echo 'set fmri(outputdir) "'${outputDIR}'/'${subDIR}'/'${subDIR}'_NoMCNoSmoothingNonLinearRegDOF'${dof}'.feat"' >> $fsf
# change input
echo 'set feat_files(1) "'${input}'"' >> $fsf

# change structural image
brain=`ls -d ${structDIR}/${subDIR}/T1_brain.nii.gz`
echo 'set highres_files(1) "'${brain}'"' >> $fsf

# set TR(s)
echo 'set fmri(tr) "'1.97'"' >>$fsf
# change field map
# B0 unwarp input image for analysis 1
unwarp=`ls ${fmapdir}/${subDIR}/fmap.nii.gz`
echo 'set unwarp_files(1) "'${unwarp}'"' >> $fsf

# B0 unwarp mag input image for analysis 1
bet_fm=`ls ${fmapdir}/${subDIR}/magBET.nii.gz`
echo 'set unwarp_files_mag(1) "'${bet_fm}'"' >> $fsf

feat $fsf
fi
fi
fi

done
