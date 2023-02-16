def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'peggnet':
        from .peggnet import PEGG_NET
        return PEGG_NET
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
