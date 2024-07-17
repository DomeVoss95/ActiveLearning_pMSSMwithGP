#!/bin/bash

# Default input arguments
EXPROOT="/raven/u/your-user-name/path-to-tutorial-folder/tutorial_mpcdf"
OUTDIR="/raven/u/your-user-name/path-to-tutorial-folder/tutorial_mpcdf/out"
JOBTIME="1-00:00:00" # Wall clock limit (max. is 24 hours)
TLIMIT=23.6
NUMJOBS=4
# Parse input arguments
while getopts ":m:t:e:j:r" o; do
    case "${o}" in
	r)
	    EXPROOT=${OPTARG}
	    ;;
    o)
        OUTDIR=${OPTARG}
        ;;
    j)
	    NUMJOBS=${OPTARG}
	    ;;
    esac
done

# Save git version
VERSIONPATH="${OUTDIR}/version_info.txt"
date > $VERSIONPATH
echo " " >> $VERSIONPATH
echo " " >> $VERSIONPATH
git --git-dir=/raven/u/your-user-name/path-to-tutorial-folder/tutorial_mpcdf/.git log -1 >> $VERSIONPATH

# Submit all jobs
FLAG=0
for i in $(eval echo {1..${NUMJOBS}});
do
	JOBNAME="job${i}"
	if [ "$FLAG" = "0" ];
    then
		prev_id=$( sbatch --job-name=$JOBNAME --time=${JOBTIME} single_job.sbatch | awk ' { print $4 }')
		FLAG=1
	else
		prev_id=$( sbatch --job-name=$JOBNAME --time=${JOBTIME} --dependency=afterok:$prev_id single_job.sbatch | awk ' { print $4 }')
	fi
	echo "${JOBNAME} as ${prev_id}"
done
