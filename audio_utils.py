import re

import librosa
import numpy as np


ID_factor = {
       'fan': 0,
       'pump': 1,
       'slider': 2,
       'valve': 3,
       'ToyCar': 4,
       'ToyConveyor': 5,
}


def get_audio_info(file_path, sr=16000):
    (wav, _) = librosa.load(file_path, sr=sr, mono=True)  # 加载音频
    wav = wav[:sr * 10]  # (1, audio_length) # 截取前10s

    mel_spect = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel_spect, ref=np.max)

    # 对不同的文件路径有不同的解析策略
    machine = file_path.split('/')[-3]  # 机器类型
    id_str = re.findall('id_[0-9][0-9]', file_path)  # 机器编号
    if machine == 'ToyCar' or machine == 'ToyConveyor':
        id = int(id_str[0][-1]) - 1
    else:
        id = int(id_str[0][-1])
    label = int(ID_factor[machine] * 7 + id)

    return log_mel, wav, label


def load_audio(file_path, sr=16000):
    (wav, _) = librosa.load(file_path, sr=sr, mono=True)  # 加载音频
    wav = wav[:sr * 10]  # (1, audio_length) # 截取前10s

    mel_spect = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel_spect, ref=np.max)

    return log_mel, wav

