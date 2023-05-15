

def get_rwt_setting(model_set:str)->dict:
    '''
        {backbone}_{atten}_{layer}_{scale}_{patch_size}_{dropout}_{droppath}
    '''
    model_setting = {}
    backbone_name, atten, layer, scale, patch_size, dropout, droppath, *others = model_set.split('_')

    model_setting["backbone_name"] = backbone_name
    model_setting["attn_type"] = atten
    model_setting["attn_type_layer"] = layer
    model_setting["scale"] = scale
    model_setting["patch_size"] = patch_size
    model_setting["dropout"] = dropout
    model_setting["droppath"] = droppath
    model_setting["others"] = {}

    return model_setting