from util import load_setting, update_time, get_time, get_log_path, get_result_path, get_day, write_setting
from setting import get_dataset_setting, get_model_setting, Run_config
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import os
import sys
import logging
import init_util

project_path = '/home/lanbo/RWT_wifi_code/'
sys.path.append(project_path)

logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)
use_gpu = 1

def _to_var(data: dict):
    if use_gpu:
        for key, value in data.items():
            data[key] = Variable(value.cuda())
    else:
        for key, value in data.items():
            data[key] = Variable(value)
    return data

def temp_test(config):
    train_dataset, eval_dataset = init_util.init_dataset(config, is_test=True)
    train_dataset.update_config(config)
    model = init_util.init_model(config)

    train_data_loader = DataLoader(train_dataset, batch_size=config["learning"]["train_batch_size"],
                                   shuffle=True, drop_last=True)

    model.cuda()
    model.train()

    data = next(iter(train_data_loader))
    data = _to_var(data)

    loss = model(data)

    print(data['data'].shape)
    print(loss)

    write_setting(config, os.path.join(config["path"]["result_path"], 'setting.json'))

def get_ddp_config(devices:list):
    nproc_per_node = len(devices)
    master_port = '29501'
    ddp_devices = ''
    for i in range(nproc_per_node-1):
        ddp_devices += f'{devices[i]},'
    ddp_devices += f'{devices[nproc_per_node-1]}'

    return ddp_devices, master_port, nproc_per_node

if __name__ == '__main__':

    day = get_day()

    '''
        {backbone}_{atten}_{layer}_{scale}_{patch_size}_{dropout}_{droppath}
    '''
    model_str_list = [
        ('RWT_waveres_4_s_16_0.4_0.1', 64),
    ]

    dataset_str_list = [
        'WiVioLoc-1_i-window-w-s',
    ]


    for dataset_str in dataset_str_list:
        dataset_setting = get_dataset_setting(dataset_str)
        for model_str in model_str_list:
            model_set   = model_str[0]
            batch_size  = model_str[1]

            backbone_setting = get_model_setting(model_set)

            config = load_setting(r'/home/lanbo/RWT_wifi_code/basic_setting.json')

            config['datetime'] = get_time()

            config["training"]["DDP"]["enable"] = True
            config["training"]["DDP"]["devices"] = [2, 3]

            config['path']['datasource_path'] = "/home/lanbo/dataset/wifi_violence_processed_loc_class/"
            config['path']['log_path']      = get_log_path(config, day, dataset_str, model_set)
            config['path']['result_path']   = get_result_path(config, day, dataset_str, model_set)

            config['dataset']['dataset_name']    = dataset_setting['dataset_name']
            config['dataset']['dataset_setting'] = dataset_setting

            config['model']['backbone_name'] = backbone_setting['backbone_name']
            config['model']['head_name']     = 'WiVio_cls'
            config['model']['strategy_name'] = 'ViTSpanCLS'
            config["model"]["strategy_setting"]["calc_data"] = "raw"
            config['model']['backbone_setting'] = backbone_setting

            config['learning']['train_batch_size'] = int(batch_size)
            config['learning']['test_batch_size'] = int(batch_size)

            config["learning"]["num_epoch"] = 500


            write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))

            # run = Run_config(config, "train")
            #
            # os.system(
            #     f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} -m torch.distributed.launch --nproc_per_node {run.nproc_per_node} "
            #     f"--master_port='29501' --use_env "
            #     f"{run.main_path} --is_train true --config_path {run.config_path} "
            #     f"> {run.log_path}"
            # )

            run = Run_config(config, "test")

            os.system(
                f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} -m torch.distributed.launch --nproc_per_node {run.nproc_per_node} "
                f"--master_port='29501' --use_env "
                f"{run.main_path} --config_path {run.config_path} "
                f"> {run.log_path}"
            )


