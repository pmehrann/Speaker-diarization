import numpy as np
import sys
sys.path.append('ghostvlad')
sys.path.append('visualization')
import ghostvlad.toolkits as toolkits
import os
from visualization.viewer import PlotDiar
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from speakerDiarization import run_diarization, groundtruth_annotation

# ===========================================
#        Parse the argument
# ===========================================
import argparse
parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/VCTK_weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()


def main():
    # gpu configuration
    toolkits.initialize_GPU(args)

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 109,
              'sampling_rate': 16000,
              'normalize': True,
              }

    # define directories
    input_dir = 'E:/Projects/Speech_recognition/dataset/Voxconverse/audio/'
    groundtruth_dir = 'E:/Projects/Speech_recognition/dataset/Voxconverse/dev/'
    wav_name = 'abjxc.wav'
    wav_path = os.path.join(input_dir, wav_name)
    out_dir = 'E:/Projects/Speech_recognition/dataset/Voxconverse/dev_predict/'
    SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'

    # run diarization
    groundtruth = groundtruth_annotation(groundtruth_dir, wav_name)
    diarization, speakerSlice = run_diarization(params, args, SAVED_MODEL_NAME,
                                                input_dir, wav_name, out_dir,
                                                observation_dim=512, embedding_per_second=1.2, overlap_rate=0.4)

    metric = DiarizationErrorRate()
    der = metric(groundtruth, diarization)
    print(f'diarization error rate = {100 * der:.1f}%')
    p = PlotDiar(map=speakerSlice, wav=wav_path, gui=True, size=(25, 6))
    p.draw()
    p.plot.show()


if __name__ == '__main__':
    main()

