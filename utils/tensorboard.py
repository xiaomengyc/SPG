
from tensorboardX import SummaryWriter
import os


if not os.path.exists('../log'):
    os.mkdir('../log')

writer = SummaryWriter(log_dir='../log')
