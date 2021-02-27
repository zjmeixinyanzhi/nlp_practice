import os
import json
from typing import Any

from src.xbot.util.nlu_util import NLU
from src.xbot.constants import DEFAULT_MODEL_PATH
from src.xbot.util.path import get_root_path
from src.xbot.util.download import download_from_url
from data.crosswoz.data_process.nlu.nlu_slot_dataloader import Dataloader
from data.crosswoz.data_process.nlu.nlu_slot_postprocess import recover_intent

import torch
from torch import nn
from transformers import BertModel


class ClassifiersWithBert(nn.Module):

    def _forward_unimplemented(self, *input: Any) -> None:
        pass


