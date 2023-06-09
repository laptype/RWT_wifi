import sys
import os
import torch

project_path = '/home/lanbo/RWT_wifi_code/'
sys.path.append(project_path)

from util import load_setting, update_time, get_time, get_log_path, get_result_path, get_day, write_setting
from setting import get_dataset_setting, get_model_setting, Run_config


if __name__ == '__main__':

    day = get_day()

    '''
        {backbone}_{atten}_{layer}_{scale}_{patch_size}_{dropout}_{droppath}
    '''
    model_str_list = [
        ('RWT_waveres_8_s_16_0.4_0.1', 64, 300),
    ]

    dataset_str_list = [
        'WiVioPerson-1_i-window-w-s'
        # 'WiVioLoc-1_i-window-w-s',
        # 'WiVioLoc-2_i-window-w-s',
        # 'WiVioLoc-3_i-window-w-s',
        # 'WiVioLoc-4_i-window-w-s',
        # 'WiVioLoc-5_i-window-w-s',
        # 'WiVioLoc-6_i-window-w-s',
        # 'WiVioLoc-7_i-window-w-s',
        # 'WiVioLoc-8_i-window-w-s',
        # 'WiVio_i-window-w-s'
    ]


    for dataset_str in dataset_str_list:
        dataset_setting = get_dataset_setting(dataset_str)
        for model_str in model_str_list:
            model_set   = model_str[0]
            batch_size  = model_str[1]
            epoch       = model_str[2]

            backbone_setting = get_model_setting(model_set)

            config = load_setting(r'/home/lanbo/RWT_wifi_code/basic_setting.json')

            config['datetime'] = get_time()

            config["training"]["DDP"]["enable"] = True
            config["training"]["DDP"]["devices"] = [2, 3]
            test_gpu = 3

            # TAG ===============================================================================================
            tag = f'person-{dataset_setting["person"]}-epoch-{epoch}'

            # 数据集路径 ==========================================================================================
            config['path']['datasource_path'] = "/home/lanbo/dataset/wifi_violence_processed_loc_class/"
            # config['path']['datasource_path'] = '/home/lanbo/dataset/wifi_violence_processed_loc/'

            config['path']['log_path']      = get_log_path(config, day, dataset_str, model_set, tag)
            config['path']['result_path']   = get_result_path(config, day, dataset_str, model_set, tag)
            # ===================================================================================================

            config['dataset']['dataset_name']    = dataset_setting['dataset_name']
            config['dataset']['dataset_setting'] = dataset_setting

            # model setting =====================================================================================
            config['model']['backbone_name'] = backbone_setting['backbone_name']
            config['model']['backbone_setting'] = backbone_setting

            config['model']['head_name']     = 'WiVio_cls'
            config['model']['strategy_name'] = 'ViTSpanCLS'
            config["model"]["strategy_setting"]["calc_data"] = "raw"
            # ===================================================================================================

            config['learning']['train_batch_size'] = int(batch_size)
            config['learning']['test_batch_size'] = int(batch_size)
            config['learning']['save_epoch']  = 50

            # epoch =============================================================================================
            config["learning"]["num_epoch"] = epoch
            # ===================================================================================================

            write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))

            # TRAIN =============================================================================================
            run = Run_config(config, "train")

            os.system(
                f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} -m torch.distributed.launch --nproc_per_node {run.nproc_per_node} "
                f"--master_port='29501' --use_env "
                f"{run.main_path} --is_train true --config_path {run.config_path} "
                f"> {run.log_path}"
            )
            # TEST ==============================================================================================
            run = Run_config(config, "test")

            os.system(
                f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
                f"{run.main_path} --config_path {run.config_path} "
                f"> {run.log_path}"
            )

