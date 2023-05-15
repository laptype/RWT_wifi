from util import load_setting, update_time, get_time, get_log_path, get_result_path, get_day, write_setting
from setting import get_wivio_setting, get_rwt_setting
import os
import sys

project_path = 'D:/study/postgraduate/study_project/wavelet_wifi/server/new_lanbo'
sys.path.append(project_path)


if __name__ == '__main__':

    '''
        {backbone}_{atten}_{layer}_{scale}_{patch_size}_{dropout}_{droppath}
    '''
    model_str_list = [
        ('rwt_waveres_8_s_16_0.4_0.1', 64)
    ]

    dataset_str_list = [
        'WiVioAUG-1_i-window-w-s',
    ]

    day = get_day()

    for dataset_str in dataset_str_list:
        dataset_setting = get_wivio_setting(dataset_str)
        for model_str in model_str_list:
            model_set   = model_str[0]
            batch_size  = model_str[1]

            model_setting = get_rwt_setting(model_set)

            config = load_setting(r'D:\study\postgraduate\study_project\wavelet_wifi\server\new_lanbo\setting.json')

            config['datetime'] = get_time()

            config['path']['datasource_path'] = "/home/lanbo/dataset/wifi_violence_processed_loc_class/"
            config['path']['log_path'] = get_log_path(config, day, dataset_str, model_set)
            config['path']['result_path'] = get_result_path(config, day, dataset_str, model_set)

            config['dataset']['dataset_name'] = dataset_setting['dataset_name']
            config['dataset']['dataset_setting'] = dataset_setting

            config['model']['backbone_name'] = model_setting['backbone_name']
            config['model']['model_setting'] = model_setting

            # write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))
            write_setting(config, r'D:\study\postgraduate\study_project\wavelet_wifi\server\new_lanbo')

