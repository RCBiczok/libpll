#!/bin/bash
#SBATCH -o /home/hpc/pr63lo/di56sad2/alexey/myjob.%j.%N.out
#SBATCH -D /home/hpc/pr63lo/di56sad2/alexey
#SBATCH -J libpll_benchmark_sumtables
#SBATCH --clusters=mpp3
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --mail-type=end
#SBATCH --mail-user=alexey.kozlov@h-its.org
#SBATCH --export=NONE
#SBATCH --time=00:10:00
source /etc/profile.d/modules.sh
module load amplifier_xe

export TMPDIR=/home/hpc/pr63lo/di56sad2/tmp

TEST=/home/hpc/pr63lo/di56sad2/rudolf/libpll/test/obj/sumtables-aa-benchmark

amplxe-cl -target-tmp-dir="/home/hpc/pr63lo/di56sad2/tmp" -r profile_avx512 -collect advanced-hotspots $TEST avx512
