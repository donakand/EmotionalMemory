#!/bin/bash 

#######
# This code is from Glasser et al. (2013)
#######


# --------------------------------------------------------------------------------
#  Load Function Libraries
# --------------------------------------------------------------------------------

#source $HCPPIPEDIR_Global/log.shlib # Logging related functions

# --------------------------------------------------------------------------------
#  Establish tool name for logging
# --------------------------------------------------------------------------------

echo "MotionCorrection.sh"
echo "START"
topdir=/MemModel
cd /MemModel/Struct
for subject in sub-CC*;do
funcDIR=${topdir}/CamCAN_Data/func
fMRIinFolder=${funcDIR}/${subject}/func
fMRIopFolder=${topdir}/Preprocessing/${subject}
if [ ! -e $fMRIopFolder ];then
mkdir $fMRIopFolder
fi
WorkingDirectory=${fMRIopFolder}/mc
if [ ! -e $WorkingDirectory ];then
mkdir ${WorkingDirectory}
fi
if [ ! -e $WorkingDirectory/filtered_func_data_mc.nii.gz ];then
if [ -e /storage/shared/research/cinn/2015/MemModel/CamCAN_Data/func/$subject/func/${subject}_task-Rest_bold.nii.gz ];then
InputfMRI=${fMRIinFolder}/${subject}_task-Rest_bold.nii.gz
echo $InputfMRI

#fslroi "$InputfMRI" $WorkingDirectory"/scout" 0 1
HCPPIPEDIR_Global=/storage/shared/research/cinn/2015/MemModel/Dona_Analysis/Scripts/Preprocessing_HCP/copy3HCPpipelines-4.0.0/global/scripts
Scout=""
OutputfMRI=$WorkingDirectory/filtered_func_data_mc
OutputMotionRegressors=${WorkingDirectory}/Movement_Regressors
OutputMotionMatrixFolder=${WorkingDirectory}/MotionMatrices
OutputMotionMatrixNamePrefix=MAT_
MotionCorrectionType="MCFLIRT"

OutputfMRIBasename=${subject}_task-Rest_bold



# Do motion correction
echo "Do motion correction"
case "$MotionCorrectionType" in
    MCFLIRT)
        ${HCPPIPEDIR_Global}/mcflirt.sh ${InputfMRI} ${WorkingDirectory}/${OutputfMRIBasename} ${Scout}
    ;;
    
    FLIRT)
        ${HCPPIPEDIR_Global}/mcflirt_acc.sh ${InputfMRI} ${WorkingDirectory}/${OutputfMRIBasename} ${Scout}
    ;;
    
    *)
    echo "ERROR: MotionCorrectionType must be 'MCFLIRT' or 'FLIRT'"
        exit 1
    ;;
esac
# Move output files about
mv -f ${WorkingDirectory}/${OutputfMRIBasename}/mc.par ${WorkingDirectory}/${OutputfMRIBasename}.par
if [ -e $OutputMotionMatrixFolder ] ; then
  rm -r $OutputMotionMatrixFolder
fi
mkdir $OutputMotionMatrixFolder

mv -f ${WorkingDirectory}/${OutputfMRIBasename}/* ${OutputMotionMatrixFolder}
mv -f ${WorkingDirectory}/${OutputfMRIBasename}.nii.gz ${OutputfMRI}.nii.gz

# Change names of all matrices in OutputMotionMatrixFolder
echo "Change names of all matrices in OutputMotionMatrixFolder"
DIR=`pwd`
if [ -e $OutputMotionMatrixFolder ] ; then
  cd $OutputMotionMatrixFolder
  Matrices=`ls`
  for Matrix in $Matrices ; do
    MatrixNumber=`basename ${Matrix} | cut -d "_" -f 2`
    mv $Matrix `echo ${OutputMotionMatrixNamePrefix}${MatrixNumber} | cut -d "." -f 1`
  done
  cd $DIR
fi

# Make 4dfp style motion parameter and derivative regressors for timeseries
# Take the backwards temporal derivative in column $1 of input $2 and output it as $3
# Vectorized Matlab: d=[zeros(1,size(a,2));(a(2:end,:)-a(1:end-1,:))];
# Bash version of above algorithm
function DeriveBackwards {
  i="$1"
  in="$2"
  out="$3"
  # Var becomes a string of values from column $i in $in. Single space separated
  Var=`cat "$in" | sed s/"  "/" "/g | cut -d " " -f $i`
  Length=`echo $Var | wc -w`
  # TCS becomes an array of the values from column $i in $in (derived from Var)
  TCS=($Var)
  # random is a random file name for temporary output
  random=$RANDOM

  # Cycle through our array of values from column $i
  j=0
  while [ $j -lt $Length ] ; do
    if [ $j -eq 0 ] ; then
      # Backward derivative of first volume is set to 0
      Answer=`echo "0"`
    else
      # Compute the backward derivative of non-first volumes

      # Format numeric value (convert scientific notation to decimal) jth row of ith column
      # in $in (mcpar)
      Forward=`echo ${TCS[$j]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
    
      # Similarly format numeric value for previous row (j-1)
      Back=`echo ${TCS[$(($j-1))]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`

      # Compute backward derivative as current minus previous
      Answer=`echo "scale=10; $Forward - $Back" | bc -l`
    fi
    # 0 prefix the resulting number
    Answer=`echo $Answer | sed s/"^\."/"0."/g | sed s/"^-\."/"-0."/g`
    echo `printf "%10.6f" $Answer` >> $random
    j=$(($j + 1))
  done
  paste -d " " $out $random > ${out}_
  mv ${out}_ ${out}
  rm $random
}

# Run the Derive function to generate appropriate regressors from the par file
echo "Run the Derive function to generate appropriate regressors from the par file"
in=${WorkingDirectory}/${OutputfMRIBasename}.par
out=${OutputMotionRegressors}.txt
cat $in | sed s/"  "/" "/g > $out
i=1
while [ $i -le 6 ] ; do
  DeriveBackwards $i $in $out
  i=`echo "$i + 1" | bc`
done

cat ${out} | awk '{for(i=1;i<=NF;i++)printf("%10.6f ",$i);printf("\n")}' > ${out}_
mv ${out}_ $out

awk -f ${HCPPIPEDIR_Global}/mtrendout.awk $out > ${OutputMotionRegressors}_dt.txt

echo "END"

# Make 4dfp style motion parameter and derivative regressors for timeseries
# Take the unbiased temporal derivative in column $1 of input $2 and output it as $3
# Vectorized Matlab: d=[a(2,:)-a(1,:);(a(3:end,:)-a(1:end-2,:))/2;a(end,:)-a(end-1,:)];
# Bash version of above algorithm
# This algorithm was used in Q1 Version 1 of the data, future versions will use DeriveBackwards
function DeriveUnBiased {
  i="$1"
  in="$2"
  out="$3"
  Var=`cat "$in" | sed s/"  "/" "/g | cut -d " " -f $i`
  Length=`echo $Var | wc -w`
  length1=$(($Length - 1))
  TCS=($Var)
  random=$RANDOM
  j=0
  while [ $j -le $length1 ] ; do
    if [ $j -eq 0 ] ; then # This is the forward derivative for the first row
      Forward=`echo ${TCS[$(($j+1))]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
      Back=`echo ${TCS[$j]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
      Answer=`echo "$Forward - $Back" | bc -l`
    elif [ $j -eq $length1 ] ; then # This is the backward derivative for the last row
      Forward=`echo ${TCS[$j]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
      Back=`echo ${TCS[$(($j-1))]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
      Answer=`echo "$Forward - $Back" | bc -l`
    else # This is the center derivative for all other rows.
      Forward=`echo ${TCS[$(($j+1))]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
      Back=`echo ${TCS[$(($j-1))]} | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
      Answer=`echo "scale=10; ( $Forward - $Back ) / 2" | bc -l`
    fi
    Answer=`echo $Answer | sed s/"^\."/"0."/g | sed s/"^-\."/"-0."/g`
    echo `printf "%10.6f" $Answer` >> $random
    j=$(($j + 1))
  done
  paste -d " " $out $random > ${out}_
  mv ${out}_ ${out}
  rm $random
}
fi
fi
done

