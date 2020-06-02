#!/bin/bash

lattice=${1:-squ}
dir=${2:-../data}   
seed=${3:-1234567}
size=${4:-10}
temp_i=${5:-4.0}
temp_f=${6:-4.0}
dtemp=${7:-0.1}
configs=${8:-2}

codedir=`pwd`

echo "ISING: dir=" $dir ", seed=" $seed ", size=" $size \
", [Ti,Tf,dT]= [" $temp_i, $temp_f, $dtemp "], configs=" $configs

mkdir -p $dir
cd $dir
mkdir -p $lattice-"L"$size
cd $lattice-"L"$size

for temp in $(seq $temp_i $dtemp $temp_f) 
do

echo "--- making jobfile for temperature" $temp

jobfile=`printf "$seed-$temp.sh"`
echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2012
#SBATCH --time=08:30:00

module load Anaconda3

pwd
echo "--- working on temperature $temp for $seed"

python $codedir/ising_data_generate_$lattice.py $seed $size $temp $temp 0.1 $configs

echo "--- finished with temperature $temp and seed $seed"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
#(sbatch ${jobfile})

done
