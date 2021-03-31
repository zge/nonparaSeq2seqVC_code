# Collection of utility methods
#
# Zhenhao Ge, 2021-03-10

import os

def get_meta(metafile):
  """ get speaker meta data in a list of dictionary of {field: value}"""

  # read in meta file
  lines = open(metafile, 'r').readlines()

  # get fields from header
  delimiter = '  '  # double-space
  fields = lines[0].rstrip().split(delimiter)
  fields = [f.strip().lower() for f in fields]
  nfields = len(fields)

  # remove header line
  lines = lines[1:]

  delimiter = ' '  # single-space
  speakers = []  # list of dictionary of {field: value}
  for line in lines:
    elements = line.rstrip().split(delimiter) # split element line
    elements = [e for e in elements if e.strip() != ''] # remove empty element

    spk_dict = {f:e for f,e in zip(fields[:nfields-1], elements[:nfields-1])}
    spk_dict[fields[-1]] = ' '.join(elements[nfields-1:])
    speakers.append(spk_dict)

  print('info of {} speakers extracted from {}'.format(len(speakers), metafile))

  return speakers

def sel_speaker(speakers, gender='both', accents_excluded=[]):
  """select a subset of speakers based on gender and accents
     gender: 'M', 'F', or 'both'
     accents_excluded: list of excluded accents, e.g. ['Indian', 'Unknown']
  """
  if gender != 'both':
    speakers_selected = [spk for spk in speakers if spk['gender'] == gender]
  else:
    speakers_selected = speakers[:]

  for accent in accents_excluded:
    speakers_selected = [spk for spk in speakers_selected if spk['accents']!=accent]

  return speakers_selected

def get_accent2nspk(speakers, verbose=True):
  """get accent distribution (i.e. accent:nspkr)"""
  accent2nspk = {}
  for spk in speakers:
    k = spk['accents']
    if k not in accent2nspk.keys():
      accent2nspk[k] = 1
    else:
      accent2nspk[k] += 1
  kv = ['{}: {}'.format(k,v) for k,v in accent2nspk.items()]

  if verbose:
    print('accent distribution: \n{}'.format('\n'.join(kv)))

  return kv

def write_flist(flists_flat, delimiter, listpath):
  """write file list to file"""

  # convert int to str for subitem in items of file list
  # (e.g. nframes, nphones for each line)
  flists_flat_str = [[str(i) for i in item] for item in flists_flat]
  # concatenate subitems by delimiter (get ready to write out)
  flists_str = ['{}'.format(delimiter).join(item) for item in flists_flat_str]

  # write the overall file list (before splitting to train/valid/test
  open(listpath, 'w').writelines('\n'.join(flists_str))
  print('wrote file list to {} ({} lines)'.format(listpath, len(flists_str)))

def read_flist(listpath, delimiter='|'):
  """read file list from file"""
  flists_flat = []
  lines = open(listpath, 'r').readlines()
  for line in lines:
    entry = line.rstrip().split(delimiter)
    flists_flat.append(entry)
  return flists_flat

def exclude_mono(flists_flat):
  """exclude uid-sid if not both mics are available"""

  # find 'uid-sid-mic' triples
  uid_sid_mics = []
  for entry in flists_flat:
    uid_sid_mic = '_'.join(os.path.basename(entry[0]).split('_')[:3])
    uid_sid_mics.append(uid_sid_mic)

  # count 'uid-sid' tuples
  uid_sid_count = {}
  for uid_sid_mic in uid_sid_mics:
    uid_sid = '_'.join(uid_sid_mic.split('_')[:2])
    if uid_sid in uid_sid_count.keys():
      uid_sid_count[uid_sid] += 1
    else:
      uid_sid_count[uid_sid] = 1

  # split 'uid-sid' based on their counts (either 1 or 2)
  uid_sid_mono, uid_sid_dual = [], []
  for uid_sid in uid_sid_count.keys():
    if uid_sid_count[uid_sid] == 1:
      uid_sid_mono.append(uid_sid)
    elif uid_sid_count[uid_sid] == 2:
      uid_sid_dual.append(uid_sid)

  # split flist based on the counts of uid_sid
  flists_mono, flists_dual = [], []
  for entry in flists_flat:
    uid_sid_mic = '_'.join(os.path.basename(entry[0]).split('_')[:3])
    uid_sid = '_'.join(uid_sid_mic.split('_')[:2])
    if uid_sid in uid_sid_mono:
      flists_mono.append(entry)
    elif uid_sid in uid_sid_dual:
      flists_dual.append(entry)

  return flists_dual, flists_mono

def get_uid2text(flists):
  """get dictionary of {uid:{sid:text}}
     e.g. {'001': {p225: 'a b c', 'p226': 'c b a'}}
  """

  uid2text = {}
  for entry in flists:
    sid, uid = os.path.basename(entry[0]).split('_')[:2]
    text = entry[1]
    if uid not in uid2text.keys():
      uid2text[uid] = {sid:text}
    else:
      uid2text[uid][sid] = text
  return uid2text

def get_uid2ntexts(uid2text):
  """get dictionary of {uid: ntexts}
     in order to check whether or not one unique uid is corresponding to
     unique text, i.e. ntexts == 1
  """

  uids = sorted(uid2text.keys())
  uid2ntexts = {}
  for uid in uids:
    texts = sorted(uid2text[uid].values())
    uid2ntexts[uid] = len(set(texts))
    print('uid: {} --> # of unique utterances: {}'.format(uid, uid2ntexts[uid]))
  return uid2ntexts

def find_sid2nuids(flists):
  """find the dictionary of {sid:ntexts}
     in order to find how many texts are spoken by each speaker
  """

  sid2nuids = {}
  for entry in flists:
    sid = os.path.basename(entry[0]).split('_')[0]
    if sid in sid2nuids.keys():
      sid2nuids[sid] += 1
    else:
      sid2nuids[sid] = 1

  nuids = sid2nuids.values()
  print('nuids distribution: {}(mean) / {}(max) / {}(min)'.format(
    round(sum(nuids)/len(nuids)), max(nuids), min(nuids)))

  return sid2nuids

def get_2dflist(flists):
  """get flist list per speaker id"""

  flists_per_sid = {}
  for entry in flists:
    sid = os.path.basename(entry[0]).split('_')[0]
    if sid in flists_per_sid.keys():
      flists_per_sid[sid].append(entry)
    else:
      flists_per_sid[sid] = [entry]
  return flists_per_sid

def get_uids(flists):
  """get uids from file list"""
  uids = []
  for entry in flists:
    uid = os.path.basename(entry[0]).split('_')[1]
    uids.append(uid)
  uids = sorted(set(uids))
  return uids

def sel_flist(uids, flists):
  """select a subset of file list based on utterance ids"""
  flists_sel = []
  for entry in flists:
    uid = os.path.basename(entry[0]).split('_')[1]
    if uid in uids:
      flists_sel.append(entry)
  return flists_sel
