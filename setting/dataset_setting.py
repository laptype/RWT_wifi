


def get_dataset_setting(dataset_setting: str)->dict:

    if dataset_setting.startswith('WiVio'):
        if dataset_setting.startswith('WiVioLoc'):
            return get_wivio_loc_setting(dataset_setting)
        return get_wivio_setting(dataset_setting)

def get_wivio_loc_setting(dataset_set:str)->dict:
    dataset_name, augment, *others = dataset_set.split('_')
    dataset_name, loc = dataset_name.split('-')
    dataset_setting = {
        'dataset_name': dataset_name,
        'loc': int(loc),
        'augment': augment,
        'others': {}
    }
    return  dataset_setting

def get_wivio_setting(dataset_set:str)->dict:
    dataset_name, augment, *others = dataset_set.split('_')
    dataset_setting = {
        'dataset_name': dataset_name,
        'augment': augment,
        'others': {}
    }
    return dataset_setting