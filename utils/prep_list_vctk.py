# Prepare file lists for VCTK and save in the data list directory
# columns: acoustic feature path | phones | #frames | #phones
#
# Zhenhao Ge, 2021-03-10

import os
import argparse
import glob
import numpy as np
import random

# change working dir if needed
import sys
sys.path.append('utils')

from utils import get_meta, sel_speaker
from utils import get_accent2nspk, write_flist, read_flist, exclude_mono
from utils import get_2dflist, get_uids, sel_flist

# for database exploration
from utils import get_uid2text, get_uid2ntexts, find_sid2nuids

def parse_args():

  usage = 'usage: prepare file lists for VCTK database'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-da', '--acoustic-dir', type=str,
                      help='directory of acoustic data')
  parser.add_argument('-dt', '--text-dir', type=str,
                      help='directory of text data')
  parser.add_argument('-dl', '--list-dir', type=str,
                      help='directory of list')
  parser.add_argument('-fm', '--metafile', type=str,
                      help='speaker info (meta) file')
  parser.add_argument('-d', '--delimiter', type=str, default='|',
                      help='delimiter to separate columns in file list')
  parser.add_argument('-r', '--ratio', type=str, default='x:10:20',
                      help='train/valid/test #sentences')
  parser.add_argument('-s', '--seed', type=int, default=0,
                      help=('random seed to split file list'
                            ' into training, validation and test'))
  parser.add_argument('-o', '--ordered', action='store_true',
                      help=('file list will be sorted after randomization'
                            ' if specified'))
  parser.add_argument('-g', '--gender', type=str, default='both',
                      help='gender of speakers selected')
  parser.add_argument('-ax', '--accents-excluded', nargs='+',
                      help='list of accents excluded')
  parser.add_argument('-mic', '--mic', type=str, default='dual',
                      help='microphone index, either 1, 2, or dual')
  return parser.parse_args()

def main():

  datadir = '/data/evs/VCTK'

  version = 'VCTK-Corpus-0.92' # 'VCTK-Corpus-0.92' or 'VCTK-sie'
  sr = '48k' # '22k' or '48k'

  # version = 'VCTK-sie' # 'VCTK-Corpus-0.92' or 'VCTK-sie'
  # sr = '22k' # '22k' or '48k'

  # speakers that the original authors selected (another source)
  spk_file = os.path.join(datadir, version, 'speaker-selected.txt')
  sids_sel = ''.join(open(spk_file).readlines()).split('\n')

  # # runtime mode
  # args = parse_args()

  # interactive mode (comment out before running the script)
  args = argparse.ArgumentParser()
  # args.acoustic_dir = '{}/{}/wav{}_silence_trimmed'.format(datadir, version, sr[:2])
  args.acoustic_dir = '{}/{}/spec'.format(datadir, version)
  args.text_dir = '{}/{}/text'.format(datadir, version)
  args.list_dir = '{}/{}/list'.format(datadir, version)
  args.metafile = '{}/{}/speaker-info.txt'.format(datadir, version)
  args.delimiter = '|'
  args.seed = 0
  args.ordered = True
  args.gender = 'both'
  args.accents_excluded = ['Indian']
  args.mic = 'dual'

  # print out arguments
  print('acoustic dir: {}'.format(args.acoustic_dir))
  print('text dir: {}'.format(args.text_dir))
  print('list dir: {}'.format(args.list_dir))
  print('meta file: {}'.format(args.metafile))
  print('delimiter: {}'.format(args.delimiter))
  print('list random seed: {}'.format(args.seed))
  print('sort list by duration: {}'.format(args.ordered))
  print('gender: {}'.format(args.gender))
  print('accents excluded: {}'.format(args.accents_excluded))
  print('microphone: {}'.format(args.mic))

  # get speaker info (list of dicts) from meta file
  speakers = get_meta(args.metafile)

  # accent-to-speaker distribution from speaker info
  accent2nspkr = get_accent2nspk(speakers)

  speakers = sel_speaker(speakers, gender=args.gender,
                         accents_excluded=args.accents_excluded)
  nspeakers = len(speakers)
  print('{} speakers selected by gender ({}) and accents (no {})'.format(
    nspeakers, args.gender, ' '.join(args.accents_excluded)))

  # get speaker IDs for those being selected
  sids = sorted(([spk['id'] for spk in speakers]))
  sids_intersected = sorted([s for s in sids if s in sids_sel])
  assert len(sids_intersected)==len(sids_sel), \
    'selected speakers not found in speaker pool!'
  del sids_intersected

  # get spaekers IDs for those not being selected
  sids_unsel = [spk['id'] for spk in speakers if spk['id'] not in sids_sel]
  for spk in speakers:
    if spk['id'] in sids_unsel:
      print(spk)

  ext = '.spec.npy' # '.spec.npy' or '-feats_norm.npy'
  # ext = '-feats_norm.npy'  # '.spec.npy' or '-feats_norm.npy'

  listname = 'audio-txt-nframe-nphone_no-indian.txt'
  # listname = 'wgannorm-txt-nframe-nphone_no-indian.txt'

  listpath = os.path.join(args.list_dir, listname)
  if os.path.isfile(listpath):
    flists_flat = read_flist(listpath)
  else:

    # get file list per speaker
    flists = []
    for i, sid in enumerate(sids):
      print('processing speaker {} ({}/{}) ...'.format(sid, i, nspeakers))
      flist = []
      feats = sorted(glob.glob(os.path.join(args.acoustic_dir, sid,
                                            '*{}'.format(ext))))
      # filter by microphone
      if args.mic != 'dual':
        feats = [f for f in feats if 'mic{}{}'.format(args.mic, ext) in f]
      for j, feat in enumerate(feats):
        feat_no_ext = os.path.join(os.path.dirname(feat),
                                   os.path.basename(feat).split('.')[0])
        uid = os.path.basename(feat).split('.')[0].split('_')[1]
        txtfile = '{}_{}.phones'.format(sid, uid)
        txtpath = os.path.join(args.text_dir, sid, txtfile)
        if os.path.isfile(txtpath):
          phone_str = open(txtpath).readline().strip()
          nphones = len(phone_str.split())
          log_spectrogram = np.load(feat)
          nframes = log_spectrogram.shape[0]
          flist.append([feat_no_ext, phone_str, nframes, nphones])

      # append file list for current speaker
      flists.append(flist)

    # flatten 2-layer file list (flists) to 1-layer
    flists_flat = [item for sublist in flists for item in sublist]

    # write file list
    write_flist(flists_flat, args.delimiter, listpath)

  # find sid-uid with both mics
  if args.mic == 'dual':
    flists_dual, flists_mono = exclude_mono(flists_flat)
    flists_all = flists_dual[:]
  else:
    flists_all = flists_flat[:]

  # # find common utterance ids (potentially to choose valid, test set from)
  # uid2text = get_uid2text(flists_all)
  # uid2ntexts = get_uid2ntexts(uid2text)
  # nuids_uniq_text = len([1 for (uid, ntexts) in uid2ntexts.items() if ntexts==1])
  # msg = 'uids with single text / # of uids: {} / {}'
  # print(msg.format(nuids_uniq_text, len(uid2ntexts)))
  #
  # # find the dictionary of {sid:ntexts}
  # sid2nuids = find_sid2nuids(flists_all)
  #
  # # select sid with enough utterances
  # nuids_cutoff = 600 # select sids with at least 600 utterances
  # nuids_sorted = sorted(sid2nuids.values(), reverse=True)
  # sids_selected = [sid for sid in sid2nuids.keys() if sid2nuids[sid]>=nuids_cutoff]

  # get 2-layer file lists (file list per sid)
  flists = get_2dflist(flists_all)

  # randomly select 10/20 utterances for validation/testing per speaker
  flists_train, flists_valid, flists_test = [], [], []
  for i, sid in enumerate(sids_sel):
    flists_sid = flists[sid]
    uids_sid = get_uids(flists_sid)

    # randomize utterance ids with fixed random seed
    random.seed(args.seed)
    random.shuffle(uids_sid)

    uids_valid = uids_sid[:10]
    uids_test = uids_sid[10:30]
    uids_train = uids_sid[30:]

    flist_train = sel_flist(uids_train, flists_sid)
    flist_valid = sel_flist(uids_valid, flists_sid)
    flist_test = sel_flist(uids_test, flists_sid)

    flists_train += flist_train
    flists_valid += flist_valid
    flists_test += flist_test

  # write out split file list
  # listname = 'wgannorm-txt-nframe-nphone_no-indian_train.txt'
  listname = 'audio-txt-nframe-nphone_no-indian_train.txt'
  write_flist(flists_train, args.delimiter, os.path.join(args.list_dir, listname))
  # listname = 'wgannorm-txt-nframe-nphone_no-indian_valid.txt'
  listname = 'audio-txt-nframe-nphone_no-indian_valid.txt'
  write_flist(flists_valid, args.delimiter, os.path.join(args.list_dir, listname))
  # listname = 'wgannorm-txt-nframe-nphone_no-indian_test.txt'
  listname = 'audio-txt-nframe-nphone_no-indian_test.txt'
  write_flist(flists_test, args.delimiter, os.path.join(args.list_dir, listname))

if __name__ == '__main__':
  main()
