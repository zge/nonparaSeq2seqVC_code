# Prepare file lists for Arctic and save in the data list directory
# columns: acoustic feature path | phones | #frames | #phones
#
# Zhenhao Ge, 2021-05-31

import os
import glob
import argparse
import random
import numpy as np

# change working dir if needed
import sys
sys.path.append('utils')

from utils import write_flist

def feat2entry(feats, text_dir):

  flist = []
  for i, feat in enumerate(feats):
    feat_no_ext = os.path.join(os.path.dirname(feat),
                               os.path.basename(feat).split('.')[0])
    sid = os.path.basename(os.path.dirname(feat))
    uid = feat.split('.')[0].split('_')[-1]
    txtfile = '{}-arctic_{}.phones'.format(sid, uid)
    txtpath = os.path.join(text_dir, sid, txtfile)
    if os.path.isfile(txtpath):
      phone_str = open(txtpath).readline().strip()
      nphones = len(phone_str.split())
      log_spectrogram = np.load(feat)
      nframes = log_spectrogram.shape[0]
      flist.append([feat_no_ext, phone_str, nframes, nphones])

  return flist

def intervene_list(lst1, lst2):
  assert len(lst1) == len(lst2), 'two lists are not in the same length!'
  lst = []
  for i in range(len(lst1)):
    lst.append(lst1[i])
    lst.append(lst2[i])
  return lst

def parse_args():

  usage = 'usage: prepare file list for Arctic database'
  parser = argparse.ArgumentParser(description=usage)
  parser.add_argument('-da', '--acoustic-dir', type=str,
                      help='directory of acoustic data')
  parser.add_argument('-dt', '--text-dir', type=str,
                      help='directory of text data')
  parser.add_argument('-dl', '--list-dir', type=str,
                      help='directory of list')
  parser.add_argument('-d', '--delimiter', type=str, default='|',
                      help='delimiter to separate columns in file list')
  parser.add_argument('-s', '--seed', type=int, default=0,
                      help=('random seed to split file list'
                            ' into training, validation and test'))
  return parser.parse_args()

def main():

  datadir = '/data/evs/Arctic'

  # runtime mode
  args = parse_args()

  # interactive mode (comment out before running the script)
  args = argparse.ArgumentParser()
  args.acoustic_dir = '{}/spec-wgan'.format(datadir)
  args.text_dir = '{}/text'.format(datadir)
  args.list_dir = '{}/list'.format(datadir)
  args.delimiter = '|'
  args.seed = 0

  # print out arguments
  print('acoustic dir: {}'.format(args.acoustic_dir))
  print('text dir: {}'.format(args.text_dir))
  print('list dir: {}'.format(args.list_dir))
  print('delimiter: {}'.format(args.delimiter))
  print('list random seed: {}'.format(args.seed))

  os.makedirs(args.list_dir, exist_ok=True)

  ext = '.spec.npy'
  feats = sorted(glob.glob(os.path.join(args.acoustic_dir, '**',
                                      '*{}'.format(ext)), recursive=True))
  sids = sorted(set([os.path.basename(os.path.dirname(feat)) for feat in feats]))

  uids = {sid: sorted([os.path.basename(feat).split('.')[0].split('_')[-1] \
                      for feat in feats if sid in feat]) for sid in sids}

  uids_common = [uid for uid in uids[sids[0]] if uid in uids[sids[1]]]

  # get randomized uids
  uids_random = uids_common[:]
  random.seed(args.seed)
  random.shuffle(uids_random)

  uids_valid = sorted(uids_random[:66])
  uids_test = sorted(uids_random[66:132])
  uids_train = sorted(uids_random[132:])

  feats_valid = [feat for feat in feats if feat.split('.')[0].split('_')[-1] in uids_valid]
  feats_test =  [feat for feat in feats if feat.split('.')[0].split('_')[-1] in uids_test]
  feats_train = [feat for feat in feats if feat.split('.')[0].split('_')[-1] in uids_train]

  flist_valid = feat2entry(feats_valid, args.text_dir)
  flist_test = feat2entry(feats_test, args.text_dir)
  flist_train = feat2entry(feats_train, args.text_dir)

  # generate speaker individual file list for validation set
  flist_valid_dbl = [entry for entry in flist_valid if 'bdl' in entry[0]]
  flist_valid_slt = [entry for entry in flist_valid if 'slt' in entry[0]]

  # write file list for validation set
  flist_valid_both = intervene_list(flist_valid_dbl, flist_valid_slt)
  listpath = os.path.join(args.list_dir, 'wgan-txt-nframe-nphone_bdl_slt_valid.txt')
  write_flist(flist_valid_both, args.delimiter, listpath)

  # generate speaker individual file list for testing set
  flist_test_dbl = [entry for entry in flist_test if 'bdl' in entry[0]]
  flist_test_slt = [entry for entry in flist_test if 'slt' in entry[0]]

  # write file list for test set
  flist_test_both = intervene_list(flist_test_dbl, flist_test_slt)
  listpath = os.path.join(args.list_dir, 'wgan-txt-nframe-nphone_bdl_slt_test.txt')
  write_flist(flist_test_both, args.delimiter, listpath)

  # generate speaker individual file list for training set
  flist_train_bdl = [entry for entry in flist_train if 'bdl' in entry[0]]
  flist_train_slt = [entry for entry in flist_train if 'slt' in entry[0]]

  # select the first 500 utterances
  flist_train_bdl_sel = flist_train_bdl[:500]
  flist_train_slt_sel = flist_train_slt[:500]

  # generate file list for training set (parallel)
  flist_train_both_sel = intervene_list(flist_train_bdl_sel, flist_train_slt_sel)
  listpath = os.path.join(args.list_dir, 'wgan-txt-nframe-nphone_bdl_slt_train_para.txt')
  write_flist(flist_train_both_sel, args.delimiter, listpath)

  # shuffle file list of speaker SLT
  random.seed(args.seed)
  random.shuffle(flist_train_slt_sel)

  # generate file list for training set (non-parallel)
  flist_train_both_sel2 = intervene_list(flist_train_bdl_sel, flist_train_slt_sel)
  listpath = os.path.join(args.list_dir, 'wgan-txt-nframe-nphone_bdl_slt_train_nonpara.txt')
  write_flist(flist_train_both_sel2, args.delimiter, listpath)

if __name__ == '__main__':
  main()
