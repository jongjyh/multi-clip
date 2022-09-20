from opendelta import AdapterModel

def adding_adapter_layer(text_encoder,bottleneck_dim):
    # add adapter after every tfm module
    delta = AdapterModel(backbone_model=text_encoder,modified_modules=[r'[r]layer\.[0-9]+\.output'],bottleneck_dim=bottleneck_dim)
    # freeze delta model only
    delta.freeze_module(exclude=['deltas','transformation','pre_LN'],set_state_dict=False)
    delta.log()
    return text_encoder