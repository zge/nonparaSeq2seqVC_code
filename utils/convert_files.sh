#!/bin/bash
#
# Convert audio file from .flac to .wav
# Also change the sampling frequency from 48K to 22.05K
#
# Zhenhao Ge, 2021-03-15

FROM_DIR=/data/evs/VCTK/VCTK-Corpus-0.92/wav48_silence_trimmed
TO_DIR=/data/evs/VCTK/VCTK-sie/wav22_silence_trimmed

[ ! -d ${TO_DIR} ] && mkdir -p ${TO_DIR}

# specify the source and destination file extensions
ext1=".flac"
ext2=".wav"

# specify the source and destination sampling rates
sr1=48000
sr2=22050

# get audio files
wav_files=($(find ${FROM_DIR} -name "*${ext1}" | sort))
nwavs=${#wav_files[@]}

## print # of wav files
#echo "# of wavs: ${nwavs}"
#exit 0

for (( i=0; i<nwavs; i++ )); do

  # get the source filename
  f=${wav_files[$i]}
  # echo $f

  sr=$(soxi $f | grep "Sample Rate" | tr ":" "\n" | sed '2q;d' | sed 's/ //g')
  [ "$sr" != "$sr1" ] && echo "$f with sample rate ${sr} (should be ${sr1})"

  # create the destination filename
  f2=$(echo $f | sed "s|${FROM_DIR}|${TO_DIR}|g") # substitute path
  f2=${f2%.*}${ext2} # replace ext

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
