import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
from shutil import copyfile

import os
import librosa
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

import sys
if os.path.isdir(os.path.join(os.getcwd(),'fine-tune')):
    sys.path.append('fine-tune')

from reader import TextMelIDLoader, TextMelIDCollate, id2ph, id2sp
from hparams import create_hparams
from model import Parrot, lcm
from inference_utils import plot_data, levenshteinDistance, recover_wav_wgan
import scipy.io.wavfile
from train import load_model

if os.path.basename(os.getcwd()) != 'fine-tune':
  os.chdir('fine-tune')
print('current dir: {}'.format(os.getcwd()))

AA_tts, BB_tts, AB_vc, BA_vc = False, False, True, True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str,
                            help='directory to save checkpoints')
    parser.add_argument('--num', type=int, default=10,
                            required=False, help='num of samples to be generated')
    parser.add_argument('--hparams', type=str,
                            required=False, help='comma separated name=value pairs')
    return parser.parse_args()

# runtime mode
args = parse_args()

# # interactive mode
# args = argparse.ArgumentParser()
# # args.checkpoint_path = 'outdir/arctic/test_wgan_bs16/logdir_bdl_slt/checkpoint_8700'
# args.checkpoint_path = 'outdir/arctic/test_wgan_bs16_700ep/logdir_bdl_slt/checkpoint_61000'
# args.num = 10
# hparams = ["validation_list=/data/evs/Arctic/list/wgan-txt-nframe-nphone_bdl_slt_valid.txt",
#                "SC_kernel_size=1"]
# args.hparams = ','.join(hparams)

hparams = create_hparams(args.hparams)

test_list = hparams.validation_list
checkpoint_path=args.checkpoint_path
gen_num = args.num
ISMEL=(not hparams.predict_spectrogram)


model = load_model(hparams)


model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)
_ = model.eval()



train_set_A = TextMelIDLoader(hparams.training_list, hparams.mel_mean_std,
        hparams.speaker_A,hparams.speaker_B,
        shuffle=False,pids=[hparams.speaker_A])

train_set_B = TextMelIDLoader(hparams.training_list, hparams.mel_mean_std,
        hparams.speaker_A,hparams.speaker_B, 
        shuffle=False,pids=[hparams.speaker_B])

test_set_A = TextMelIDLoader(test_list, hparams.mel_mean_std, 
        hparams.speaker_A,hparams.speaker_B,
        shuffle=False,pids=[hparams.speaker_A])

test_set_B = TextMelIDLoader(test_list, hparams.mel_mean_std, 
        hparams.speaker_A,hparams.speaker_B,
        shuffle=False,pids=[hparams.speaker_B])

sample_list_A = test_set_A.file_path_list
sample_list_B = test_set_B.file_path_list

collate_fn = TextMelIDCollate(lcm(hparams.n_frames_per_step_encoder,
                        hparams.n_frames_per_step_decoder))

test_loader_A = DataLoader(test_set_A, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=1, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)
                            
test_loader_B = DataLoader(test_set_B, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=1, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)


id2sp[0] = hparams.speaker_A
id2sp[1] = hparams.speaker_B

_, mel, __, speaker_id = train_set_A[0]
reference_mel_A = speaker_id.cuda()
ref_sp_A = id2sp[speaker_id.item()]

_, mel, __, speaker_id = train_set_B[0]
reference_mel_B = speaker_id.cuda()
ref_sp_B = id2sp[speaker_id.item()]



def get_path(input_text, A, B):
    task = 'tts' if input_text else 'vc'

    path_save = os.path.join(checkpoint_path.replace('checkpoint', 'test'), task)
    
    path_save += '_%s_to_%s'%(A, B)

    subfolders = ['wav_mel', 'mel', 'hid', 'ali', 'txt', 'wav']
    for folder in subfolders:
        if not os.path.exists(os.path.join(path_save, folder)):
            os.makedirs(os.path.join(path_save, folder))

    print(path_save)
    return path_save


def generate(loader, reference_mel, beam_width, path_save, ref_sp, 
        sample_list, num=10, input_text=False):

    with torch.no_grad():
        errs = []
        totalphs = []

        for i, batch in enumerate(loader):
            if i == num:
                break
            
            #sample_id = sample_list[i].split('/')[-1][9:17+4]
            sample_id = sample_list[i].split('/')[-1]
            print(('index:%d, decoding %s ...'%(i, sample_id)))

            text_file = '{}.txt'.format(os.path.basename(sample_list[i]))
            text_path = os.path.dirname(sample_list[i]).replace('spec-wgan', 'text')
            text_path = os.path.join(text_path, text_file)
            text = open(text_path, 'r').readlines()[0].rstrip()
            print('{}: {}'.format(text_file, text))

            text_path_output = os.path.join(path_save, 'txt/Txt_{}'.format(text_file))
            copyfile(text_path, text_path_output)

            wav_file = '{}.wav'.format(os.path.basename(sample_list[i]))
            wav_path = os.path.dirname(sample_list[i]).replace('spec-wgan', 'wav22_silence_trimmed')
            wav_path = os.path.join(wav_path, wav_file)

            wav_path_output = os.path.join(path_save, 'wav/Wav_{}'.format(wav_file))
            copyfile(wav_path, wav_path_output)

            x, y = model.parse_batch(batch)
            predicted_mel, post_output, predicted_stop, alignments, \
                text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments, \
                speaker_id = model.inference(x, input_text, reference_mel, beam_width)

            post_output = post_output.data.cpu().numpy()[0]
            alignments = alignments.data.cpu().numpy()[0].T
            audio_seq2seq_alignments = audio_seq2seq_alignments.data.cpu().numpy()[0].T

            text_hidden = text_hidden.data.cpu().numpy()[0].T #-> [hidden_dim, max_text_len]
            audio_seq2seq_hidden = audio_seq2seq_hidden.data.cpu().numpy()[0].T
            audio_seq2seq_phids = audio_seq2seq_phids.data.cpu().numpy()[0] # [T + 1]
            speaker_id = speaker_id.data.cpu().numpy()[0] # scalar

            task = 'TTS' if input_text else 'VC'

            wav_path = os.path.join(path_save, 'wav_mel/Wav_%s_ref_%s_%s.wav' % (sample_id, ref_sp, task))
            # recover_wav(post_output, wav_path, hparams.mel_mean_std, ismel=ISMEL)
            recover_wav_wgan(post_output, wav_path, hparams.mel_mean_std, ismel=ISMEL,
                             n_fft=1024, win_length=1024, hop_length=256)
            
            post_output_path = os.path.join(path_save, 'mel/Mel_%s_ref_%s_%s.npy'%(sample_id, ref_sp, task))
            np.save(post_output_path, post_output.T)
                    
            plot_data([alignments, audio_seq2seq_alignments], 
                os.path.join(path_save, 'ali/Ali_%s_ref_%s_%s.pdf'%(sample_id, ref_sp, task)))
            
            plot_data([np.hstack([text_hidden, audio_seq2seq_hidden])], 
                os.path.join(path_save, 'hid/Hid_%s_ref_%s_%s.pdf'%(sample_id, ref_sp, task)))
            
            audio_seq2seq_phids = [id2ph[id] for id in audio_seq2seq_phids[:-1]]
            target_text = y[0].data.cpu().numpy()[0]
            target_text = [id2ph[id] for id in target_text[:]]

            if not input_text:
                #print 'Sounds like %s, Decoded text is '%(id2sp[speaker_id])
                print(audio_seq2seq_phids)
                print(target_text)
        
            err = levenshteinDistance(audio_seq2seq_phids, target_text)
            print(err, len(target_text))

            errs.append(err)
            totalphs.append(len(target_text))

    # save phone error rate
    per = float(sum(errs))/float(sum(totalphs))
    per_file = os.path.join(path_save, 'per.txt')
    open(per_file, 'w').writelines('{:.3f}\n'.format(per))
    open(per_file, 'a').writelines(','.join([str(e) for e in errs]) + '\n')
    open(per_file, 'a').writelines(','.join([str(l) for l in totalphs]) + '\n')

    # print float(errs)/float(totalphs)
    # return float(errs)/float(totalphs)
    return per



####### TTS A - A ############

if AA_tts:
    path_save = get_path(True, ref_sp_A, ref_sp_A)
    generate(test_loader_A, reference_mel_A, hparams.beam_width,
        path_save, ref_sp_A, sample_list_A, num=gen_num, input_text=True)

####### TTS B - B ############
if BB_tts:
    path_save = get_path(True, ref_sp_B, ref_sp_B)
    generate(test_loader_B, reference_mel_B, hparams.beam_width,
        path_save, ref_sp_B, sample_list_B, num=gen_num, input_text=True)

####### VC A - B #############
if AB_vc:
    path_save = get_path(False, ref_sp_A, ref_sp_B)
    per_AB = generate(test_loader_A, reference_mel_B, hparams.beam_width,
        path_save, ref_sp_B, sample_list_A, num=gen_num, input_text=False)
    print(('PER %s-to-%s is %.4f'%(ref_sp_A, ref_sp_B, per_AB)))

####### VC B - A #############
if BA_vc:
    path_save = get_path(False, ref_sp_B, ref_sp_A)
    per_BA = generate(test_loader_B, reference_mel_A, hparams.beam_width,
        path_save, ref_sp_A, sample_list_B, num=gen_num, input_text=False)
    print(('PER %s-to-%s is %.4f'%(ref_sp_B, ref_sp_A, per_BA)))

