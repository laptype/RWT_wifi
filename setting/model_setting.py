

def get_model_setting(model_set:str)->dict:
    if model_set.startswith('rwt'):
        return get_rwt_setting(model_set)

def get_rwt_setting(model_set:str)->dict:
    '''
        {backbone}_{atten}_{layer}_{scale}_{patch_size}_{dropout}_{droppath}
    '''
    backbone_name, atten, layer, scale, patch_size, dropout, droppath, *others = model_set.split('_')

    model_setting = {
        "backbone_name": backbone_name,
        "attn_type": atten,
        "attn_type_layer": int(layer),
        "scale": scale,
        "patch_size": int(patch_size),
        "dropout": float(dropout),
        "droppath": float(droppath),
        "others": {}
    }
    return model_setting