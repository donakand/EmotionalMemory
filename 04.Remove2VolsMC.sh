#!/bin/bash
# Change the directories as needed
topDIR=/MemModel
structDir=${topDIR}/CamCAN_Data/anat
inputDIR=${topDIR}/Preprocessing
cd $inputDIR


for subDIR in sub-CC*;do 
echo $subDIR
subDIR=${inputDIR}/${subDIR}/mc
if [ -e $subDIR ];then
movReg=`ls -d ${subDIR}/Movement_Regressors.txt`
movRegDt=`ls -d ${subDIR}/Movement_Regressors_dt.txt`
movReg2=$subDIR/Movement_Regressors2.txt
movRegDt2=${subDIR}/Movement_Regressors_dt2.txt
echo "$(tail -n +3 $movReg)" > $movReg2
echo "$(tail -n +3 $movRegDt)" > $movRegDt2
fi


done
