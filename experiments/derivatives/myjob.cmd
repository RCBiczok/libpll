#!/bin/bash
#SBATCH -o /home/hpc/pr63lo/di56sad2/rudolf/libpll/experiments/derivative/myjob.%j.%N.out
#SBATCH -D /home/hpc/pr63lo/di56sad2/rudolf/libpll/experiments/derivative
#SBATCH -J libpll_benchmark_derivative
#SBATCH --clusters=mpp3
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --mail-type=end
#SBATCH --mail-user=rudolf.biczok@student.kit.edu
#SBATCH --export=NONE
#SBATCH --time=01:00:00
source /etc/profile.d/modules.sh
module load amplifier_xe

export TMPDIR=/home/hpc/pr63lo/di56sad2/rudolf/libpll/experiments/derivative/tmp

/home/hpc/pr63lo/di56sad2/rudolf/libpll/experiments/benchmark.py /home/hpc/pr63lo/di56sad2/rudolf/libpll/test/obj/derivatives-aa-benchmark-simple avx2
/home/hpc/pr63lo/di56sad2/rudolf/libpll/experiments/benchmark.py /home/hpc/pr63lo/di56sad2/rudolf/libpll/test/obj/derivatives-aa-benchmark-simple avx512f

amplxe-cl -target-tmp-dir="/home/hpc/pr63lo/di56sad2/rudolf/libpll/experiments/derivative" -r profile_gcc_avx2 -collect advanced-hotspots /home/hpc/pr63lo/di56sad2/rudolf/libpll/test/obj/derivatives-aa-benchmark-simple avx2
amplxe-cl -target-tmp-dir="/home/hpc/pr63lo/di56sad2/rudolf/libpll/experiments/derivative" -r profile_gcc_avx512 -collect advanced-hotspots /home/hpc/pr63lo/di56sad2/rudolf/libpll/test/obj/derivatives-aa-benchmark-simple avx512f
