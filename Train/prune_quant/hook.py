activation = {}
def get_activation(name):
     def hook(model,input,output):
         activation[name] = output.detach()
     return hook



            layers = list(model.module._modules.items())
            for [name,layer] in layers:
                    layer.register_forward_hook(get_activation(name))
            outputs = model(inputs)
            for [name,layer] in layers:
                layer_output = activation[name]