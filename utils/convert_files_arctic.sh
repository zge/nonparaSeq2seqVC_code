#!/bin/bash
#
# Change the sampling frequency from 32K to 22.05K
# Also save them in the new name inside the destination directory
#
# Zhenhao Ge, 2021-05-19

spk=$1 # dbl, slt
ROOT_DIR=/mnt/psvr/SpeechData/TTS/Arctic
FROM_DIR=${ROOT_DIR}/cmu_us_${spk}_arctic-WAVEGG/cmu_us_${spk}_arctic/orig
TO_DIR=/data/evs/Arctic/wav22/${spk}

[ ! -d ${TO_DIR} ] && mkdir -p ${TO_DIR}

sr1=32000
sr2=22050

# get audio files
wav_files=($(find ${FROM_DIR} -name "*.wav" | sort))
nwavs=${#wav_files[@]}

# print # of wav files
echo "# of wavs in ${FROM_DIR}: ${nwavs}"

for (( i=0; i<nwavs; i++ )); do

  # get the source filename
  f=${wav_files[$i]}
  # echo $f

  sr=$(soxi $f | grep "Sample Rate" | tr ":" "\n" | sed '2q;d' | sed 's/ //g')
  [ "$sr" != "$sr1" ] && echo "$f with sample rate ${sr} (should be ${sr1})"

  # create the destination filename
  f2=$(echo $f | sed "s|${FROM_DIR}|${TO_DIR}|g") # substitute path
  f2=$(dirname $f2)/${spk}-$(basename $f2)

  # create the destination dir
  dest_dir=$(dirname $f2)
  [ ! -d ${dest_dir} ] && mkdir -p ${dest_dir}

  # convert source file to destination file
  if [ ! -f ${f2} ]; then
    echo "$f ($sr) -> $f2 ($sr2)"
    sox $f -R -r ${sr2} $f2
  else
    echo "Destination file already exist: $f2"
  fi

done

# print # of wav files
echo "# of wavs: ${nwavs}"
