
def get_wivio_setting(dataset_set:str)->dict:
    dataset_setting = {}
    dataset_name, augment, *others = dataset_set.split('_')
    dataset_setting['dataset_name'] = dataset_name
    dataset_setting['augment'] = augment
    dataset_setting['others'] = {}
    return dataset_setting