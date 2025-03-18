#coding: utf-8
import os
import os.path as osp
import time
import random
import numpy as np
import random
import soundfile as sf
import librosa

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import pandas as pd

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                print(text)
        return indexes

np.random.seed(1)
random.seed(1)
SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
MEL_PARAMS = {
    "n_mels": 80,
}

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

class FilePathDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 feature_root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 use_random_ref=True
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path
        self.feature_root_path = feature_root_path
        self.use_random_ref = use_random_ref
        self.is_val = validation

        if 'Grid' in self.root_path:
            self.is_grid = True
        else:
            self.is_grid = False

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # data format: [path, phoneme, speaker id]
        # example: ['Toy@Buzz/Toy@Buzz_00_0315_00.wav', 
        # 'ðeɪ ɑːɹ ɐ tɛɹˈɪliəmkɑːɹbˈɑːnɪk ˈælɔɪ, ænd ˈaɪ kæn flˈaɪ. ',
        # '5']
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        mel_tensor = preprocess(wave).squeeze()
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        # ref_mel_tensor, ref_label = self._load_data(ref_data[:3])

        # get reference sample
        if self.use_random_ref==False or self.is_val:
            # # 50% chance use GT as reference 50% chance use different speech from GT speaker as reference (poor MCD/SCES but good UT-MOS)
            # if bool(random.getrandbits(1)):
            #     ref_data = data
            # else:
            #     ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            ref_data = data
        else:
            ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
        
        # get OOD text
        ps = ""
        
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)

        # get visual lip motion feature and emotion feature
        lip_feature = self._load_lip(data)
        emotion_feature = self._load_emotion(data)
        atm_feature = self._load_atm(data)

        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave, lip_feature, emotion_feature, atm_feature

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            # print(wave_path, sr)
            
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        # mel_length = mel_tensor.size(1)
        # if mel_length > self.max_mel_length:
        #     random_start = np.random.randint(0, mel_length - self.max_mel_length)
        #     mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id
    
    # Load lip motion feature for visual voice cloning
    def _load_lip(self, data):
        if self.is_grid:
            basename = data[0].split('/')[-1].split('.wav')[0]
            speaker = basename.split('-')[0]
            id = basename.split('-')[-1]
            lip_path = os.path.join(self.feature_root_path, 
                                    "extrated_embedding_Grid_152_gray")
            lip_feature = np.load(os.path.join(lip_path,
                                            "{}-face-{}.npy".format(speaker, id)))
            lip_feature= torch.from_numpy(lip_feature).float()

        else:
            basename = data[0].split('/')[-1].split('.wav')[0]
            speaker = basename.split('_00')[0]
            lip_path = os.path.join(self.feature_root_path, 
                                    "extrated_embedding_V2C_gray")
            lip_feature = np.load(os.path.join(lip_path,
                                            "{}-face-{}.npy".format(speaker, basename)))
            lip_feature= torch.from_numpy(lip_feature).float()
        return lip_feature
    
    # Load emotion feature for visual voice cloning
    def _load_emotion(self, data):
        if self.is_grid:
            basename = data[0].split('/')[-1].split('.wav')[0]
            speaker = basename.split('-')[0]
            id = basename.split('-')[-1]
            emo_path = os.path.join(self.feature_root_path, 
                                    "VA_feature")
            emotion_feature = np.load(os.path.join(emo_path,
                                            "{}-feature-{}.npy".format(speaker, id)))
            emotion_feature= torch.from_numpy(emotion_feature).float()

        else:
            basename = data[0].split('/')[-1].split('.wav')[0]
            speaker = basename.split('_00')[0]
            emo_path = os.path.join(self.feature_root_path, 
                                    "VA_feature")
            emotion_feature = np.load(os.path.join(emo_path,
                                                "{}-feature-{}.npy".format(speaker, basename)))
            emotion_feature = torch.from_numpy(emotion_feature).float()
        return emotion_feature
    
    # Load atmosphere feature for visual voice cloning
    def _load_atm(self, data):
        if self.is_grid:
            basename = data[0].split('/')[-1].split('.wav')[0]
            speaker = basename.split('-')[0]
            id = basename.split('-')[-1]
            atm_path = os.path.join(self.feature_root_path, 
                                    "emos")
            try:
                atm_feature = np.load(os.path.join(atm_path,
                                            "{}-emo-{}.npy".format(speaker, id)))
            except IOError:
                print("{} is not existed".format(os.path.join(atm_path,
                                                    "{}-emo-{}.npy".format(speaker, id))))
                atm_feature= np.zeros(256)
            
            atm_feature = torch.from_numpy(atm_feature).float()


        else:
            basename = data[0].split('/')[-1].split('.wav')[0]
            speaker = basename.split('_00')[0]
            atm_path = os.path.join(self.feature_root_path, 
                                    "emos")
            atm_feature = np.load(os.path.join(atm_path,
                                                "{}-emo-{}.npy".format(speaker, basename)))
            atm_feature = torch.from_numpy(atm_feature).float()
        return atm_feature



class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        # [speaker_id, 
        # acoustic_feature, (mel-spectrogram)
        # text_tensor, 
        # ref_text, 
        # ref_mel_tensor, 
        # ref_label, 
        # path, 
        # wave,
        # lip_feature,
        # emotion_feature,
        # atm_features]
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])
        max_rmel_length = max([b[4].shape[1] for b in batch])
        max_lip_length = max([b[8].shape[0] for b in batch])
        max_emotion_length = max([b[8].shape[0] for b in batch])

        assert max_emotion_length == max_lip_length

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()
        lip_features = torch.zeros((batch_size, max_lip_length, 512)).float()
        emotion_features = torch.zeros((batch_size, max_emotion_length, 256)).float()
        atm_features = torch.zeros((batch_size, 256)).float()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        visual_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, max_rmel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave, lip_feature, emotion_feature, atm_feature) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)

            assert lip_feature.size(0) == emotion_feature.size(0)
            visual_length = lip_feature.size(0)

            visual_lengths[bid] = visual_length
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel

            lip_features[bid, :visual_length, :] = lip_feature
            emotion_features[bid, :visual_length, :] = emotion_feature
            atm_features[bid] = atm_feature
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels, lip_features, emotion_features, atm_features, visual_lengths



class FilePathDataset_Stage1(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 root_path,
                 sr=24000,
                 data_augmentation=False,
                 validation=False,
                 OOD_data="Data/OOD_texts.txt",
                 min_length=50,
                 use_random_ref=True
                 ):

        spect_params = SPECT_PARAMS
        mel_params = MEL_PARAMS

        _data_list = [l.strip().split('|') for l in data_list]
        self.data_list = [data if len(data) == 3 else (*data, 0) for data in _data_list]
        self.text_cleaner = TextCleaner()
        self.sr = sr

        self.df = pd.DataFrame(self.data_list)

        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.data_augmentation = data_augmentation and (not validation)
        self.max_mel_length = 192
        
        self.min_length = min_length
        with open(OOD_data, 'r', encoding='utf-8') as f:
            tl = f.readlines()
        idx = 1 if '.wav' in tl[0].split('|')[0] else 0
        self.ptexts = [t.split('|')[idx] for t in tl]
        
        self.root_path = root_path
        self.use_random_ref = use_random_ref

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):        
        data = self.data_list[idx]
        path = data[0]
        
        wave, text_tensor, speaker_id = self._load_tensor(data)
        
        mel_tensor = preprocess(wave).squeeze()
        
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]
        
        # get reference sample
        if self.use_random_ref==False:
            # # 50% chance use GT as reference 50% chance use different speech from GT speaker as reference (poor MCD/SCES but good UT-MOS)
            # if bool(random.getrandbits(1)):
            #     ref_data = data
            # else:
            #     ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            ref_data = data
        else:
            ref_data = (self.df[self.df[2] == str(speaker_id)]).sample(n=1).iloc[0].tolist()
            print("random_ref")
        ref_mel_tensor, ref_label = self._load_data(ref_data[:3])
        
        # get OOD text
        
        ps = ""
        
        while len(ps) < self.min_length:
            rand_idx = np.random.randint(0, len(self.ptexts) - 1)
            ps = self.ptexts[rand_idx]
            
            text = self.text_cleaner(ps)
            text.insert(0, 0)
            text.append(0)

            ref_text = torch.LongTensor(text)
        
        return speaker_id, acoustic_feature, text_tensor, ref_text, ref_mel_tensor, ref_label, path, wave

    def _load_tensor(self, data):
        wave_path, text, speaker_id = data
        speaker_id = int(speaker_id)
        wave, sr = sf.read(osp.join(self.root_path, wave_path))
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        if sr != 24000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
            # print(wave_path, sr)
            
        wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
        
        text = self.text_cleaner(text)
        
        text.insert(0, 0)
        text.append(0)
        
        text = torch.LongTensor(text)

        return wave, text, speaker_id

    def _load_data(self, data):
        wave, text_tensor, speaker_id = self._load_tensor(data)
        mel_tensor = preprocess(wave).squeeze()

        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, speaker_id


class Collater_Stage1(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.min_mel_length = 192
        self.max_mel_length = 192
        self.return_wave = return_wave
        

    def __call__(self, batch):
        # batch[0] = wave, mel, text, f0, speakerid
        batch_size = len(batch)

        # sort by mel length
        lengths = [b[1].shape[1] for b in batch]
        batch_indexes = np.argsort(lengths)[::-1]
        batch = [batch[bid] for bid in batch_indexes]

        nmels = batch[0][1].size(0)
        max_mel_length = max([b[1].shape[1] for b in batch])
        max_text_length = max([b[2].shape[0] for b in batch])
        max_rtext_length = max([b[3].shape[0] for b in batch])

        labels = torch.zeros((batch_size)).long()
        mels = torch.zeros((batch_size, nmels, max_mel_length)).float()
        texts = torch.zeros((batch_size, max_text_length)).long()
        ref_texts = torch.zeros((batch_size, max_rtext_length)).long()

        input_lengths = torch.zeros(batch_size).long()
        ref_lengths = torch.zeros(batch_size).long()
        output_lengths = torch.zeros(batch_size).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()
        paths = ['' for _ in range(batch_size)]
        waves = [None for _ in range(batch_size)]
        
        for bid, (label, mel, text, ref_text, ref_mel, ref_label, path, wave) in enumerate(batch):
            mel_size = mel.size(1)
            text_size = text.size(0)
            rtext_size = ref_text.size(0)
            labels[bid] = label
            mels[bid, :, :mel_size] = mel
            texts[bid, :text_size] = text
            ref_texts[bid, :rtext_size] = ref_text
            input_lengths[bid] = text_size
            ref_lengths[bid] = rtext_size
            output_lengths[bid] = mel_size
            paths[bid] = path
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref_labels[bid] = ref_label
            waves[bid] = wave

        return waves, texts, input_lengths, ref_texts, ref_lengths, mels, output_lengths, ref_mels



def build_dataloader(path_list,
                     root_path,
                     feature_root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     use_random_ref=False):
    
    dataset = FilePathDataset(path_list, root_path, feature_root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config, use_random_ref=use_random_ref)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader


def build_dataloader_Stage1(path_list,
                     root_path,
                     validation=False,
                     OOD_data="Data/OOD_texts.txt",
                     min_length=50,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     dataset_config={},
                     use_random_ref=False):
    
    dataset = FilePathDataset_Stage1(path_list, root_path, OOD_data=OOD_data, min_length=min_length, validation=validation, **dataset_config, use_random_ref=use_random_ref)
    collate_fn = Collater_Stage1(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader

