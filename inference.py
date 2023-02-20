import time

import onnxruntime
import numpy as np

import audio_utils


ID_factor = {
       'fan': 0,
       'pump': 1,
       'slider': 2,
       'valve': 3,
       'ToyCar': 4,
       'ToyConveyor': 5,
}

def softmax(x):
    """ softmax function """

    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行

    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


def inference(file, machine_type, machine_id):
    # wav = np.random.random((1, 1, 160000)).astype(np.float32)  # 音频波形信号
    # log_mel = np.random.random((1, 128, 313)).astype(np.float32)  # log-mel频谱
    # label = np.random.random((1, 1)).astype(np.float32)  # 机器标签

    # 加载待检测音频
    # audio_file_path = "E:/STgram_MFN/data/dataset/fan/test/anomaly_id_00_00000021.wav"
    log_mel, wav = audio_utils.load_audio(file)
    log_mel = np.expand_dims(log_mel, axis=0).astype(np.float32)
    wav = np.expand_dims(wav, axis=(0, 1)).astype(np.float32)

    label = int(ID_factor[machine_type] * 7 + int(machine_id))
    label = np.array(label).reshape(1, 1).astype(np.float32)

    # 启动Onnx推理
    ort_session = onnxruntime.InferenceSession("model_weight/SW-Wavenet_best.onnx")
    ort_inputs = {'log-mel': log_mel, 'wav': wav, 'label': label}
    t1 = time.time()
    ort_output = ort_session.run(['predict_ids'], ort_inputs)  # 一个10s的音频文件推理需要50ms左右
    t2 = time.time()
    print("Inference time : {}s".format(t2 - t1))

    # 对结果后处理
    sofmax_id = np.squeeze(-np.log(softmax(ort_output[0])))
    anomaly_score = sofmax_id[int(label)]

    print("Anomaly score: {}".format(anomaly_score))

    return anomaly_score


