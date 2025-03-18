import torch
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(0)

import numpy as np
np.random.seed(0)

# load packages
import time
import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa
from nltk.tokenize import word_tokenize
import soundfile as sf

from models import *
from utils import *
from text_utils import TextCleaner
textclenaer = TextCleaner()
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument("--restore_step", type=int, default=0)
parser.add_argument(
    "-n",
    "--exp_name",
    type=str,
    required=True,
)
parser.add_argument( 
    "--epoch", 
    type=int,
    required=True, 
)
parser.add_argument( 
    "--setting", 
    type=int,
    required=True, 
)

args = parser.parse_args()

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def compute_style(path, return_mel_len=False):
    wave, sr = librosa.load(path, sr=24000)
    mel_len = preprocess(wave).to(device).shape[-1]
    if return_mel_len:
        return _ , mel_len
    # Trying remove the audio trim
    # wave, index = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
    mel_tensor = preprocess(wave).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1), mel_len

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

config = yaml.safe_load(open("output/{}/config_grid.yml".format(args.exp_name)))
data_params = config.get('data_params', None)

# load pretrained ASR model
ASR_config = config.get('ASR_config', False)
ASR_path = config.get('ASR_path', False)
text_aligner = load_ASR_models(ASR_path, ASR_config)

# load pretrained F0 model
F0_path = config.get('F0_path', False)
pitch_extractor = load_F0_models(F0_path)

# load BERT model
from Utils.PLBERT.util import load_plbert
BERT_path = config.get('PLBERT_dir', False)
plbert = load_plbert(BERT_path)

model_params = recursive_munch(config['model_params'])
model = build_model_GRID(model_params, text_aligner, pitch_extractor, plbert)
_ = [model[key].eval() for key in model]
_ = [model[key].to(device) for key in model]

params_whole = torch.load("output/{}/ckpt/epoch_2nd_{}.pth".format(args.exp_name, args.epoch), map_location='cpu')
params = params_whole['net']

for key in model:
    if key in params:
        print('%s loaded' % key)
        try:
            model[key].load_state_dict(params[key])
        except:
            from collections import OrderedDict
            state_dict = params[key]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            # load params
            model[key].load_state_dict(new_state_dict, strict=False)
#             except:
#                 _load(params[key], model[key])
_ = [model[key].eval() for key in model]

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

sampler = DiffusionSampler(
    model.diffusion.diffusion,
    sampler=ADPM2Sampler(),
    sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
    clamp=False
)

def get_visual_feature(speaker, name):
    
    lip_path = os.path.join(
        data_params['feature_root_path'],
        "extrated_embedding_Grid_152_gray",
        "{}-face-{}.npy".format(speaker, name.split('-')[-1]),
    )
    lip_feature = torch.from_numpy(np.load(lip_path)).float().to(device)

    emotion_path = os.path.join(
            data_params['feature_root_path'],
            "VA_feature",
            "{}-feature-{}.npy".format(speaker, name.split('-')[-1]),
        )
    emotion_feature = torch.from_numpy(np.load(emotion_path)).float().to(device)

    return lip_feature, emotion_feature

def inference(text, ref_s, alpha = 0.0, beta = 0.3, diffusion_steps=5, embedding_scale=1, 
              mel_len=None, emotion_feature=None, lip_feature=None):
    text = text.replace('"', '')
    text = text.strip()
    ps = global_phonemizer.phonemize([text])
    ps = word_tokenize(ps[0])
    ps = ' '.join(ps)
    tokens = textclenaer(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        prosody_phoneme_feature = model.bert_encoder(bert_dur).transpose(-1, -2) 
        emotion_feature = emotion_feature.unsqueeze(0)
        lip_feature = lip_feature.unsqueeze(0)
        prosody_phoneme_feature_emotion = model.prosody_fusion(prosody_phoneme_feature, text_mask, length_to_mask(torch.LongTensor([emotion_feature.shape[1]])).to(device), emotion_feature) + prosody_phoneme_feature

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device), 
                                        #   embedding=bert_dur,
                                          embedding=prosody_phoneme_feature_emotion.transpose(-1, -2),
                                          embedding_scale=embedding_scale,
                                          features=ref_s, # reference from the same speaker as the embedding
                                          num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]


        duration = model.duration_predictor(prosody_phoneme_feature,
                                            lip_feature,
                                            input_lengths,
                                            text_mask,
                                            length_to_mask(torch.LongTensor([emotion_feature.shape[1]])).to(device) # Visual Mask
                                            )

        duration = torch.sigmoid(duration).sum(axis=-1)
        duration[0][0] = duration[0][0] * 1.3

        duration_sum = mel_len
        duration_logits = duration / duration.sum()
        duration = (duration_logits * duration_sum) / 2
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)
        duration_diff = (duration_sum / 2) - pred_dur.sum()
        if duration_diff > 0:
            i = 0
            while duration_diff > 1:
                if i == 0:
                    i = i + 1
                else:
                    pred_dur[i] += 1
                    duration_diff -= 1
                    i = i + 1
        else:
            pred_dur[0] += duration_diff


        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        d = model.predictor.text_encoder(prosody_phoneme_feature_emotion, 
                                         s, input_lengths, text_mask)
        # encode prosody
        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, 
                                F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    
        
    return out.squeeze().cpu().numpy()[..., :-50] # weird pulse at the end of the model, need to be fixed later


if args.setting == 1:

    save_path = 'output/{}/result_setting1_epoch{}'.format(args.exp_name, args.epoch)
    os.makedirs(save_path, exist_ok=True)

    val_path = config['val_path_setting1']
    wav_path = config['val_wav_path']

    with open(val_path, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

    noise = torch.randn(1,1,256).to(device)
    for i in tqdm(range(len(name))):

        lip_feature, emotion_feature = get_visual_feature(speaker[i], name[i])
        gt_wav_path = "{}/{}/{}.wav".format(wav_path, speaker[i], name[i])
                
        try:
            ref_s, mel_len = compute_style(gt_wav_path)
            wav = inference(raw_text[i], ref_s, embedding_scale=1, 
                            mel_len=mel_len, emotion_feature=emotion_feature, lip_feature=lip_feature)
            # wav = DFA(wav, gt_wav_path)
            sf.write('{}/wav_pred_{}.wav'.format(save_path, name[i]), wav, 24000)
        except Exception:
            continue


if args.setting == 2:

    save_path = 'output/{}/result_setting2_epoch{}'.format(args.exp_name, args.epoch)
    os.makedirs(save_path, exist_ok=False)

    val_path = config['val_path_setting2']
    wav_path = config['val_wav_path']

    with open(val_path, "r", encoding="utf-8") as f:
            name = []
            text = []
            speaker = []
            ref_name = []
            ref_text = []
            for line in f.readlines():
                n, t, s, rn, rt = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                ref_name.append(rn)
                ref_text.append(rt)

    noise = torch.randn(1,1,256).to(device)
    for i in tqdm(range(len(name))):

        lip_feature, emotion_feature = get_visual_feature(speaker[i], name[i])
          
        try:
            ref_s, _ = compute_style("{}/{}/{}.wav".format(wav_path, speaker[i], ref_name[i]))
            _, mel_len = compute_style("{}/{}/{}.wav".format(wav_path, speaker[i], name[i]), return_mel_len=True)
            with open(os.path.join("{}/{}/{}.lab".format(wav_path, speaker[i], name[i])), "r", encoding="utf-8") as f:
                raw_text = f.readlines()

            wav = inference(raw_text[0], ref_s, embedding_scale=1, 
                            mel_len=mel_len, emotion_feature=emotion_feature, lip_feature=lip_feature)
            sf.write('{}/wav_pred_{}.wav'.format(save_path, name[i]), wav, 24000)
        except Exception:
            # print("{} is not normal".format(name[i]))
            # results_P.append([1, i, 'None', 'None'])
            continue

        # ref_s, _ = compute_style("{}/{}/{}.wav".format(wav_path, speaker[i], ref_name[i]))
        # _, mel_len = compute_style("{}/{}/{}.wav".format(wav_path, speaker[i], name[i]), return_mel_len=True)
        # with open(os.path.join("{}/{}/{}.lab".format(wav_path, speaker[i], name[i])), "r", encoding="utf-8") as f:
        #     raw_text = f.readlines()

        # wav = inference(raw_text[0], ref_s, alpha=alpha, beta=beta, embedding_scale=1, 
        #                 mel_len=mel_len, emotion_feature=emotion_feature, lip_feature=lip_feature)
        # sf.write('{}/wav_pred_{}.wav'.format(save_path, name[i]), wav, 24000)


if args.setting == 3:

    save_path = 'output/{}/result_setting3_epoch{}'.format(args.exp_name, args.epoch)
    os.makedirs(save_path, exist_ok=True)

    val_path = config['val_path_setting3']
    wav_path = config['val_wav_path']

    with open(val_path, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)

    noise = torch.randn(1,1,256).to(device)
    for i in tqdm(range(len(name))):

        lip_feature, emotion_feature = get_visual_feature(speaker[i], name[i])
        gt_wav_path = "{}/{}/{}.wav".format(wav_path, speaker[i], name[i])
                
        try:
            ref_s, mel_len = compute_style(gt_wav_path)
            wav = inference(raw_text[i], ref_s, embedding_scale=1, 
                            mel_len=mel_len, emotion_feature=emotion_feature, lip_feature=lip_feature)
            # wav = DFA(wav, gt_wav_path)
            sf.write('{}/wav_pred_{}.wav'.format(save_path, name[i]), wav, 24000)
        except Exception:
            # print("{} is not normal".format(name[i]))
            # results_P.append([1, i, 'None', 'None'])
            continue
