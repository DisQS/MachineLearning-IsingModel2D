#!/bin/bash

num_iteration=5

directory_data='/u/ptm/bcivitci/ising_model/'

size=100


num_configs_per_job=20

echo "ISING: dir=" $directory_data ", size=" $size \
", configs=" $num_configs_per_job


mkdir -p $directory_data
cd $directory_data

for ising in $(seq $num_iteration) 
do
seed=$((1 + $RANDOM % 2**63-1))

echo "--- making the " $ising " .job file with seed =" $seed " ---"

jobfile=`printf "ising_${ising}_seed_$seed.job"`
echo $jobfile

cat > ${jobfile} << EOD
#!/bin/bash
#$ -N ising_data_$ising
#$ -q seq_medium
#$ -l m_mem_free=8G
#$ -M burak.civitcioglu@cyu.fr
#$ -m esa
#$ -cwd
#$ -j y

module load python3
python3 ising_data_generate.py $seed $size $num_configs_per_job

EOD

cat ${jobfile}

chmod 755 ${jobfile}
chmod g+w ${jobfile}

(qsub ${jobfile})

done
