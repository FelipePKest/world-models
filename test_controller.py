""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator
import torch

def test_controller():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--logdir', type=str, help='Where models are stored.')
    # args = parser.parse_args()

    ctrl_file = join("exp_dir", 'ctrl', 'best.tar')

    assert exists(ctrl_file),\
        "Controller was not trained..."

    device = torch.device('cpu')

    generator = RolloutGenerator("exp_dir", device, 1000, render_mode="human")

    with torch.no_grad():
        print(generator.rollout(None, render=True))

test_controller()