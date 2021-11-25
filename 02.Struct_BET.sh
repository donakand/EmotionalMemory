#!/bin/bash

# Change the directories as needed
topDIR=/MemModel
structDir=${topDIR}/CamCAN_Data/anat
outDIR=${topDIR}
cd $structDir
echo ${structDir}
memDIR=/EmotionalMemory

# Cycle through specified subjects
cd ${memDIR}
i=0
for subNo in EmotionalMemory_CC*;do
subDIR=`echo $subNo | awk -F "_" '{ print $2 }'`
sub="sub-${subDIR}"
echo "Working on ${sub}"
outDIR2=${outDIR}/Struct/${sub}
mkdir  ${outDIR2}
cd ${structDir}/${sub}/anat
struct=`ls -d $structDir/$sub/anat/${sub}_T1w.nii.gz`
echo "Struct is ${struct}"
fslreorient2std ${struct} ${outDIR2}/T1.nii.gz
robustfov -i ${outDIR2}/T1.nii.gz -r ${outDIR2}/T1.nii.gz
bet ${outDIR2}/T1 ${outDIR2}/T1_brain -f .3
done


##bet##
# Usage:    bet <input> <output> [options]
# 
# Main bet2 options:
#   -o          generate brain surface outline overlaid onto original image
#   -m          generate binary brain mask
#   -s          generate approximate skull image
#   -n          don't generate segmented brain image output
#   -f <f>      fractional intensity threshold (0->1); default=0.5; smaller values give larger brain outline estimates
#   -g <g>      vertical gradient in fractional intensity threshold (-1->1); default=0; positive values give larger brain outline at bottom, smaller at top
#   -r <r>      head radius (mm not voxels); initial surface sphere is set to half of this
#   -c <x y z>  centre-of-gravity (voxels not mm) of initial mesh surface.
#   -t          apply thresholding to segmented brain image and mask
#   -e          generates brain surface as mesh in .vtk format
# 
# Variations on default bet2 functionality (mutually exclusive options):
#   (default)   just run bet2
#   -R          robust brain centre estimation (iterates BET several times)
#   -S          eye & optic nerve cleanup (can be useful in SIENA)
#   -B          bias field & neck cleanup (can be useful in SIENA)
#   -Z          improve BET if FOV is very small in Z (by temporarily padding end slices)
#   -F          apply to 4D FMRI data (uses -f 0.4 and dilates brain mask slightly)
#   -A          run bet2 and then betsurf to get additional skull and scalp surfaces (includes registrations)
#   -A2 <T2>    as with -A, when also feeding in non-brain-extracted T2 (includes registrations)
# 
# Miscellaneous options:
#   -v          verbose (switch on diagnostic messages)
#   -h          display this help, then exits
#   -d          debug (don't delete temporary intermediate images)
