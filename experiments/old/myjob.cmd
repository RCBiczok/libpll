#!/bin/bash
#SBATCH -o /home/hpc/pr63lo/di56sad2/rudolf/myjob.%j.%N.out
#SBATCH -D /home/hpc/pr63lo/di56sad2/rudolf
#SBATCH -J libpll_benchmark_sumtables
#SBATCH --clusters=mpp3
#SBATCH --get-user-env
#SBATCH --ntasks=1
#SBATCH --mail-type=end
#SBATCH --mail-user=rudolf.biczok@student.kit.edu
#SBATCH --export=NONE
#SBATCH --time=01:00:00
source /etc/profile.d/modules.sh
module load amplifier_xe

export TMPDIR=/home/hpc/pr63lo/di56sad2/tmp

amplxe-cl -target-tmp-dir="/home/hpc/pr63lo/di56sad2/rudolf/tmp" -r profile_avx512 -collect advanced-hotspots ./libpll/test/obj/sumtables-aa-benchmark avx512f
amplxe-cl -report hotspots -source-object function=pll_core_update_sumtable_ii_20x20_avx512f -group-by basic-block,address -format csv -csv-delimiter comma -search-dir ./libpll/ -r profile_avx512 > pll_core_update_sumtable_ii_20x20_avx512f.inst.out
