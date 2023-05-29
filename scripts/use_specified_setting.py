import sys
import os
import torch

project_path = '/home/lanbo/RWT_wifi_code/'
sys.path.append(project_path)

from util import load_setting, update_time, get_time, get_log_path, get_result_path, get_day, write_setting
from setting import get_dataset_setting, get_model_setting, Run_config



if __name__ == '__main__':

    setting_path = "/home/lanbo/RWT_wifi_code/result/05-26/person-1/WiVioPerson-1_i-window-w-s_RWT_waveres_8_s_16_0.4_0.1/setting.json"
