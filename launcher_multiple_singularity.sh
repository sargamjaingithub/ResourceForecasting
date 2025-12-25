#!/bin/sh

# variables initialization
input_directory_path="../input/configs"
output_directory_path="../output"
temporary_launcher_path="./temporary_launcher_file"

# loop by all config files of the input directory and run all
file_names=$(find $input_directory_path/* -iname "*.json")
for file_name in $file_names; do
    base_name=$(basename "$file_name")
    file_name=${base_name%.*}  # no extension
    echo "$file_name"

    # create output directory for the specific job
    mkdir -p ${output_directory_path}/"$file_name"

    # create temporary launcher file
    {
      echo "#!/bin/bash"
      echo "#SBATCH --job-name=${file_name}"
      echo "#SBATCH --qos=bsc_cs"
      echo "#SBATCH -D ./"
      echo "#SBATCH --ntasks=1"
      echo "#SBATCH --output=${output_directory_path}/${file_name}/log_%j.out"
      echo "#SBATCH --error=${output_directory_path}/${file_name}/log_%j.err"
      echo "#SBATCH --cpus-per-task=40"
      echo "#SBATCH --gres gpu:1"
      echo "#SBATCH --time=10:05:00"
      echo "module purge; module load singularity"
      echo "SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=false PYTHONPATH=. singularity exec --nv singularity_image.sif python3 ./app/train_test_model.py --config ${input_directory_path}/${file_name}.json --output ${output_directory_path}/${file_name}"
    } > "$temporary_launcher_path"

    # run job
    sbatch "$temporary_launcher_path"

done

# remove temporary launcher file
rm ${temporary_launcher_path}

