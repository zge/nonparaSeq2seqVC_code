import os
import numpy as np
import torch
import argparse

import sys
if os.path.isdir(os.path.join(os.getcwd(),'fine-tune')):
  sys.path.append('fine-tune')

from hparams import create_hparams
from model import lcm
from inference_utils import plot_data
from train import load_model
from torch.utils.data import DataLoader
from reader import TextMelIDLoader, TextMelIDCollate, id2sp

if os.path.basename(os.getcwd()) != 'fine-tune':
  os.chdir('fine-tune')
print('current dir: {}'.format(os.getcwd()))

def parse_args():

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--checkpoint_path', type=str,
                          help='directory to save checkpoints')
  parser.add_argument('--hparams', type=str,
                          required=False, help='comma separated name=value pairs')

  return parser.parse_args()

# runtime mode
args = parse_args()

# # interactive mode
# args = argparse.ArgumentParser()
# args.checkpoint_path = os.path.abspath(os.path.join(os.getcwd(),
#   '../pre-train/outdir/vctk/test_wgan_bs16/checkpoint_302000'))
# hparams = ["speaker_A=slt", "speaker_B=bdl",
#            "training_list=/data/evs/Arctic/list/wgan-txt-nframe-nphone_bdl_slt_train_nonpara.txt",
#           "SC_kernel_size=1"]
# args.hparams = ','.join(hparams)

checkpoint_path=args.checkpoint_path

hparams = create_hparams(args.hparams)

model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)
_ = model.eval()


def gen_embedding(speaker):

    training_list = hparams.training_list

    train_set_A = TextMelIDLoader(training_list, hparams.mel_mean_std,
                                  hparams.speaker_A, hparams.speaker_B,
                                  shuffle=False,pids=[speaker])
            
    collate_fn = TextMelIDCollate(lcm(hparams.n_frames_per_step_encoder,
                            hparams.n_frames_per_step_decoder))

    train_loader_A = DataLoader(train_set_A, num_workers=1, shuffle=False,
                                sampler=None,
                                batch_size=1, pin_memory=False,
                                drop_last=True, collate_fn=collate_fn)

    with torch.no_grad():

        speaker_embeddings = []

        for i,batch in enumerate(train_loader_A):
            #print i
            x, y = model.parse_batch(batch)
            text_input_padded, mel_padded, text_lengths, mel_lengths, speaker_id = x
            speaker_id, speaker_embedding = model.speaker_encoder.inference(mel_padded)

            speaker_embedding = speaker_embedding.data.cpu().numpy()
            speaker_embeddings.append(speaker_embedding)

        speaker_embeddings = np.vstack(speaker_embeddings)
        
    print(speaker_embeddings.shape)
    if not os.path.exists('outdir/embeddings'):
        print('creating outdir/embeddings ...')
        os.makedirs('outdir/embeddings')
    
    np.save('outdir/embeddings/%s.npy'%speaker, speaker_embeddings)
    plot_data([speaker_embeddings], 
        'outdir/embeddings/%s.pdf'%speaker)


print('Generating embedding of %s ...'%hparams.speaker_A)
gen_embedding(hparams.speaker_A)

print('Generating embedding of %s ...'%hparams.speaker_B)
gen_embedding(hparams.speaker_B)
