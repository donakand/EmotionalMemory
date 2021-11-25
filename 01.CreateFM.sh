#!/bin/bash
topDIR=/MemModel
funcDIR=${topDIR}/CamCAN_Data/func
outputDIR=${topDIR}/Preprocessing
fsfDIR=${topDIR}/Scripts/fsfs
origfsf=${fsfDIR}/preprocess.fsf
fmapdir=${topDIR}/CamCAN_Data/fmap
opdir=${topDIR}/FieldMaps
cd /MemModel/Struct3
for subject in sub-CC*;do
echo "#########################"
echo "Applying BET to "${subject}
subdir=${fmapdir}/${subject}/fmap
#1- use bet for mag image. Any of the two mag images can be used. BET should be tight
magIm=${subdir}/${subject}_task-Rest_magnitude1.nii.gz
if [ ! -e ${opdir}/${subject} ]; then
mkdir ${opdir}/${subject}
fi
magBet=${opdir}/${subject}/magBET.nii.gz
bet ${magIm} ${magBet} -f 0.8
#2- create the fieldmap
phaseIm=${subdir}/${subject}_task-Rest_phasediff
outIm=${opdir}/${subject}/fmap.nii.gz
deltaTE=2.46
echo "Creating fieldmap"
fsl_prepare_fieldmap SIEMENS ${phaseIm} ${magBet} ${outIm} ${deltaTE}
done

