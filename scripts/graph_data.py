import matplotlib.pyplot as plt
import numpy as np
import csv

def moving_average(data, width):
        return np.convolve(data, np.ones(width), 'valid') / width

def graph_csv(filename, headers):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file, headers)
        data = {
            'steps': [],
            'RAdam': [],
            'Adam': [],
            'RMSProp': [],
            'SGD': []
        }
        header = True
        for row in reader:
            if not header:
                data['steps'].append(int(row['Step']))
                data['RAdam'].append(float(row[headers[1]]))
                data["Adam"].append(float(row[headers[4]]))
                data["RMSProp"].append(float(row[headers[7]]))
                data["SGD"].append(float(row[headers[10]]))
            else:
                header = False

    width = 15
    data_loss = int(width / 2)
    plt.plot(data["steps"][data_loss:-data_loss], moving_average(data["Adam"], width), label='Adam')
    plt.plot(data["steps"][data_loss:-data_loss], moving_average(data["RAdam"], width), label='RAdam')
    plt.plot(data["steps"][data_loss:-data_loss], moving_average(data["RMSProp"], width), label='RMSProp')
    plt.plot(data["steps"][data_loss:-data_loss], moving_average(data["SGD"], width), label='SGD')
    plt.legend()
    plt.xlim((0, data["steps"][-1]))
    plt.show()

headers = [
    "Step",
    "subsea_video2_colmap_RAdam - Train Metrics Dict/psnr",
    "subsea_video2_colmap_RAdam - Train Metrics Dict/psnr__MIN",
    "subsea_video2_colmap_RAdam - Train Metrics Dict/psnr__MAX",
    "subsea_video2_colmap_Adam - Train Metrics Dict/psnr",
    "subsea_video2_colmap_Adam - Train Metrics Dict/psnr__MIN",
    "subsea_video2_colmap_Adam - Train Metrics Dict/psnr__MAX",
    "subsea_video2_colmap_RMSProp - Train Metrics Dict/psnr",
    "subsea_video2_colmap_RMSProp - Train Metrics Dict/psnr__MIN",
    "subsea_video2_colmap_RMSProp - Train Metrics Dict/psnr__MAX",
    "subsea_video2_colmap_SGD - Train Metrics Dict/psnr",
    "subsea_video2_colmap_SGD - Train Metrics Dict/psnr__MIN",
    "subsea_video2_colmap_SGD - Train Metrics Dict/psnr__MAX"
]
filename='ExperimentData/optimiser_train_psnr.csv'
graph_csv(filename, headers)