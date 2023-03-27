import numpy as np
import keras
import csv
import time


model_load_time_toc = time.perf_counter()
model = keras.models.load_model(r"C:\Users\ccm51\Documents\SD_SG_GESTURE\model\model2_1.h5")
model_load_time_tic = time.perf_counter()

valid_csv = open(r"C:\Users\ccm51\Documents\SD_SG_GESTURE\validation_dataset.csv", 'r')
data = list(csv.reader(valid_csv, delimiter=","))
valid_csv.close()
np_data = np.array(data)
data_feats = []
for i in range(1, len(data)):
    data_feats.append(data[i][:-1])
np_feats = np.array(data_feats).astype(np.float16)

model_predict_time_toc = time.perf_counter()
model_results = model.predict(np_feats, batch_size=1, verbose=1)
model_predict_time_tic = time.perf_counter()

print(f"Time taken to load the model = {round(model_load_time_tic-model_load_time_toc, 5)} s\n"
      f"Time taken to predict the validation set = {round(model_predict_time_tic-model_predict_time_toc, 5)} s\n"
      f"Estimate time to predict 1 input = {round((model_predict_time_tic-model_predict_time_toc)/len(model_results), 5)} s")
