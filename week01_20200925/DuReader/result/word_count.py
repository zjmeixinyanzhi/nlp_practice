# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module will count all word.
"""

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import BRCDataset
from vocab import Vocab
import jieba

class Counter(object):
    """
    word counter
    """

    def __init__(self, train_files, dev_files, test_files):
        """
        init func
        """
        self.train_files = train_files
        self.dev_files = dev_files
        self.test_files = test_files

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on BaiduRC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    

    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--input_files', nargs='+',
                               default=['../data/demo/trainset/search.train.json'],
                               help='list of files that contain the preprocessed train data')
    
    
    return parser.parse_args()

def prepare(args):
    """
    excute word count
    """
    print(args.input_files)

    with open(args.input_files[0], 'r', encoding='utf-8') as fin:
        for line in fin:
            contents = line.strip().split()
            for item in contents:
                seg_list = jieba.cut(item)
                print("\n".join(seg_list))


def run():
    """ run """
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)



if __name__ == '__main__':
    run()