import argparse
import tqdm

def parse_args():
    parser = argparse.ArgumentParser
    parser.add_argument('--content', type=str, help='content')
    args = parser.parse_args()
    return args

def chat():
    # TODO: make this work
    pass
