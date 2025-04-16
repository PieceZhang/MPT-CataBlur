import importlib
from shutil import copy2

def create_model(config):
    lib = importlib.import_module('models.trainers.{}'.format(config.trainer))
    model = lib.Model(config)
    
    return model


def create_network(config, **kwargs):
    modulename = f'models.{config.network}'

    if not isinstance(modulename, list):
        lib = importlib.import_module(modulename)
        copy2('./models/{}.py'.format(modulename.split('.')[-1]), config.LOG_DIR.offset)
    else:
        lib = importlib.import_module(modulename[0])
        for mn in modulename:
            copy2('./models/{}.py'.format(mn.split('.')[-1]), config.LOG_DIR.offset)

    if config.CL.CL:
        copy2('./models/loss.py', config.LOG_DIR.offset)

    try:
        network = eval(f'lib.{config.network}(config).cuda()')
    except TypeError:
        network = eval(f'lib.{config.network}().cuda()')
    return network