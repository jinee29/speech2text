import os # dùng để truy cập đến thư mục của ổ đĩa, hoặc các việc liên quan đến hệ thống máy tính
import librosa #xử lý audio (mở, lưu, convert, preprocessing audio data)
from tqdm import tqdm #báo tiến độ xử
import numpy as np #xử lý data, kiểu dữ liệu,...
from sklearn.model_selection import train_test_split # chia data thành 2 tập dữ liệu là train vs test
from src.coremodel import Tuning


label_map = {"Excel": 0, "Word": 1, "Google":2, "Note": 3, "Power point": 4}
all_wave = []
all_label = []

for type_ in label_map.keys():
    path = "Data\\" + type_
    for name_file in tqdm(os.listdir(path)):
        path_file = path + "\\" + name_file
        samples, sample_rate = librosa.load(path_file, sr = 8000)
        samples = np.abs(librosa.stft(samples))
        samples = samples[:, :50]
        if samples.shape[1] < 50:
            shape_inves = 50 - samples.shape[1]
            samples = np.concatenate((samples, np.array([[0]*shape_inves] * 1025)), 1)
        all_wave.append(samples)
        all_label.append(label_map[type_])
all_wave = np.array(all_wave)
all_wave = all_wave.reshape(all_wave.shape[0], all_wave.shape[1], all_wave.shape[2], 1)
x_train, x_valid, y_train, y_valid = train_test_split(all_wave,
                                                      np.array(all_label),
                                                      stratify=np.array(all_label),
                                                      test_size = 0.3,
                                                      random_state=77,
                                                      shuffle=True)

train_models = Tuning(label_map)
train_models.run(x_train, y_train, x_valid, y_valid)
