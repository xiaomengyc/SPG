from __future__ import  absolute_import

from .resnet import *
from .densenet import *
from .vgg import *
from .vggcoco import *
from .deform import *
from .google import *

# __factory = {
#     'densenet_org': densenet_org,
#
# }
#
# def create(name, *args, **kwargs):
#
#     if name not in __factory:
#         raise KeyError("Unknow model:", name)
#     return __factory[name](*args, **kwargs)