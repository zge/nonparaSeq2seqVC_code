# preprocess the arctic database
#  - extract 1st channel of the dual-channel audio, which contains meaningful sound
#  - normalize the audio
#  - trim the silence at the beginning and at the end based on alignment info
#
# Zhenhao Ge, 2021-05-19

import os
import argparse
import glob

# change working dir if needed
import sys
sys.path.append('utils')

from audio import extract_wav_channel, audioread, audiowrite, normalize_wav

def get_start_end(labfile):
  lines = open(labfile, 'r').readlines()
  idx = [i for i, line in enumerate(lines) if ' pau' in line]

  # find start time
  cnt = 0
  while cnt < len(idx):
    if cnt+1 < len(idx) and idx[cnt+1] - idx[cnt] == 1:
      cnt += 1
    else:
      start_idx = idx[cnt] + 1
      break
  start_time = float(lines[start_idx].rstrip().split()[0])

  # find the end time
  cnt = len(idx)-1
  if lines[cnt].rstrip().split()[-1] != 'pau':
    raise Exception('not ending in pau!')
  else:
    while cnt > 0:
      if idx[cnt] - idx[cnt-1] == 1:
        cnt -= 1
      else:
        end_idx = idx[cnt]
        break
  end_time = float(lines[end_idx].rstrip().split()[0])

  return start_time, end_time

def get_start_end_simple(labfile):
  lines = open(labfile, 'r').readlines()
  idx = [i for i, line in enumerate(lines) if ' pau' in line]

  # find start time
  if idx[0] == 0:
    start_time = float(lines[idx[0]].rstrip().split()[0])
  else:
    start_time = float(lines[idx[0]+1].rstrip().split()[0])

  # find end time
  if lines[-1].rstrip().split()[-1] != 'pau':
    raise Exception('not ending in pau!')
  else:
    end_time = float(lines[-1].rstrip().split()[0])

  return start_time, end_time

def parse_args():

  usage = 'usage: preprocess the arctic database'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-di', 'in-dir', type=str, help='input dir')
  parser.add_argument('-do', 'out-dir', type=str, help='output dir')
  parser.add_argument('-sp', 'spk', type=str, help='speaker')
  parser.add_argument('-dl', 'lab_dir', type=str, help='label dir')
  parser.add_argument('-bt', 'buffer_time', type=float, help='buffer time for silence')
  return parser.parse_args()

def main():

  # # runtime mode
  # args = parse_args()

  root_dir = '/mnt/psvr/SpeechData/TTS/Arctic'

  # interactive mode (comment out before running the script)
  args = argparse.ArgumentParser()
  args.in_dir = '/data/evs/Arctic/wav22'
  # output dir contains files which are single-channel, normalized and silence trimmed
  args.out_dir = '/data/evs/Arctic/wav22_silence_trimmed'
  args.spk = 'slt' # 'dbl' or 'slt'
  args.lab_dir = os.path.join(root_dir,
    'cmu_us_{}_arctic-0.95-release/cmu_us_{}_arctic/lab'.format(args.spk, args.spk))
  args.buffer_time = 0.125

  print('input dir: {}'.format(args.in_dir))
  print('output dir: {}'.format(args.out_dir))
  print('speaker: {}'.format(args.spk))
  print('label dir: {}'.format(args.lab_dir))
  print('buffer time: {}'.format(args.buffer_time))

  assert os.path.isdir(args.lab_dir), 'label dir: {} does not exist!'

  in_dir = os.path.join(args.in_dir, args.spk)
  wavfiles = sorted(glob.glob(os.path.join(in_dir, '*.wav')))

  nwavs = len(wavfiles)
  print('# of wav files in {}: {}'.format(in_dir, nwavs))

  out_dir = os.path.join(args.out_dir, args.spk)
  os.makedirs(out_dir, exist_ok=True)

  blksize = 100
  for i in range(nwavs):

    if i % blksize == 0:
      print('processing {}/{} - {}/{} ...'.format(
        i, nwavs, min(i+blksize, nwavs), nwavs))

    # extract mono channel (must do before normalization)
    wavfile_mono = os.path.splitext(os.path.basename(wavfiles[i]))[0] + '_mic1.wav'
    wavfile_mono = os.path.join(out_dir, wavfile_mono)
    extract_wav_channel(wavfiles[i], wavfile_mono)

    # normalize wav file
    wavfile_mono_norm = os.path.splitext(wavfile_mono)[0] + '.norm.wav'
    normalize_wav(wavfile_mono, wavfile_mono_norm)

    # get the start time and duration of the trimmed wav file
    labfile = os.path.basename(wavfiles[i]).split('-')[1].replace('.wav', '.lab')
    labfile = os.path.join(args.lab_dir, labfile)
    assert os.path.isfile(labfile), '{} not exist!'.format(labfile)
    # starttime, endtime = get_start_end(labfile)
    # starttime, endtime = get_start_end_simple(labfile)
    starttime, endtime = 0.0, float('inf') # disable trimming

    # write the final processed wav files
    duration = endtime - starttime + args.buffer_time
    data, params = audioread(wavfile_mono_norm, starttime, duration)
    wavfile_out = wavfile_mono.replace('_mic1', '')
    audiowrite(wavfile_out, data, params)

  # remove the temp files
  tmpfiles = sorted(glob.glob(os.path.join(out_dir, '*_mic1*.wav')))
  for f in tmpfiles:
    os.remove(f)

