def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'peggnet':
        from .ggcnn4 import GGCNN4
        return GGCNN4
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
