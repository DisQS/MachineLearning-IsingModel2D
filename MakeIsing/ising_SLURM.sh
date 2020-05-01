#!/bin/bash

dir=${1:-../data}   
seed=${2:-1234567}
size=${3:-10}
temp_i=${4:-4.0}
temp_f=${5:-4.0}
dtemp=${6:-0.1}
configs=${7:-2}

codedir=`pwd`

echo "ISING: dir=" $dir ", seed=" $seed ", size=" $size \
", [Ti,Tf,dT]= [" $temp_i, $temp_f, $dtemp "], configs=" $configs

mkdir -p $dir
cd $dir
mkdir -p "L"$size
cd "L"$size

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

python $codedir/ising_data_generate.py $seed $size $temp $temp 0.1 $configs

echo "--- finished with temperature $temp and seed $seed"
EOD

cat ${jobfile}

chmod 755 ${jobfile}
#(sbatch -q devel ${jobfile})
(sbatch -q taskfarm ${jobfile})
#(sbatch ${jobfile})
#(sbatch ${jobfile})

done
