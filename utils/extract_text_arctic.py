# Extract text and phones for Arctic database
# Save in the same format as VCTK

import os
import argparse
import glob

# change working dir if needed
import sys
sys.path.append('utils')

def parse_args():

  usage = 'usage: extract text and phones for Arctic database'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-sp', 'spk', type=str, help='speaker')
  parser.add_argument('-ft', '--text-file', type=str, help='text file')
  parser.add_argument('-dp', '--phone-dir', type=str, help='phone dir')
  return parser.parse_args()

def main():

  datadir = '/data/evs/Arctic'
  textdir = '/mnt/psvr/SpeechData/TTS/Arctic'

  # runtime mode
  # args = parse_args()

  # interactive mode (comment out before running the script)
  args = argparse.ArgumentParser()
  args.spk = 'bdl' # 'bdl', or 'slt'
  args.text_file = os.path.join(textdir, 'cmu_us_{}_arctic-WAVEGG'.format(args.spk),
    'cmu_us_{}_arctic'.format(args.spk), 'etc', 'txt.done.data')
  args.phone_dir = os.path.join(textdir, 'cmu_us_{}_arctic-0.95-release'.format(args.spk),
    'cmu_us_{}_arctic'.format(args.spk), 'lab')

  assert os.path.isfile(args.text_file), 'file {} does not exsit!'.format(args.text_file)
  assert os.path.isdir(args.phone_dir), 'dir {} does not exist!'.format(args.phone_dir)

  outdir = os.path.join(datadir, 'text', args.spk)
  os.makedirs(outdir, exist_ok=True)

  # read text from text file
  lines = open(args.text_file, 'r').readlines()

  # write individual text files
  for line in lines:
    line = line.rstrip()
    parts = line.split('"')
    fid = parts[0].split()[1]
    fname = os.path.join(outdir, '{}-{}.txt'.format(args.spk, fid))
    text = parts[1]
    open(fname, 'w').writelines(text)

  # read and write phones
  phone_files = sorted(glob.glob(os.path.join(args.phone_dir, '*.lab')))
  for f in phone_files:
    # read phones from individual phone alignment files
    lines = open(f, 'r').readlines()
    phones = []
    for l in lines:
      parts = l.rstrip().split()
      if len(parts) == 3:
        phones.append(parts[2])
    # write phones to individual phone files
    fid = os.path.splitext(os.path.basename(f))[0]
    fname = os.path.join(outdir, '{}-{}.phones'.format(args.spk, fid))
    open(fname, 'w').writelines(' '.join(phones))
