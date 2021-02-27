import os
import json
import torch
import torch.nn as nn
from typing import Any
from torchcrf import CRF
from transformers import BertModel
from data.crosswoz.data_process.nlu.nlu_dataloader import Dataloader
from data.crosswoz.data_process.nlu.nlu_postprocess import recover_intent
from src.xbot.constants import DEFAULT_MODEL_PATH
from src.xbot.util.nlu_util import NLU
from src.xbot.util.path import get_root_path
from src.xbot.util.download import download_from_url

# 设置slot与Intent总分类器
class Classifier(nn.Module):
    def __init__(self, input_dim, num_labels, dropout_rate=0.):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointBERT(nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # 加载Bert模型

        self.intent_classifier = Classifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = Classifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)

                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs

class JointWithBertPredictor(NLU):
    """NLU Joint with Bert 预测器"""

    default_model_config = "nlu/crosswoz_all_context_joint_nlu.json"
    default_model_name = "pytorch-model-nlu-joint.pt"
    default_model_url = "http://xbot.bslience.cn/pytorch-joint-with-bert.pt"

    def __init__(self):
        root_path = get_root_path()
        config_file = os.path.join(
            root_path,
            "src/xbot/config/{}".format(JointWithBertPredictor.default_model_config),
        )
        config = json.load(open(config_file))
        device = config["DEVICE"]
        data_dir = os.path.join(root_path, config["data_dir"])

        intent_vocab = json.load(
            open(os.path.join(data_dir, "intent_vocab.json"), encoding="utf-8")
        )
        tag_vocab = json.load(
            open(os.path.join(data_dir, "tag_vocab.json"), encoding="utf-8")
        )
        dataloader = Dataloader(
            intent_vocab=intent_vocab,
            tag_vocab=tag_vocab,
            pretrained_weights=config["model"]["pretrained_weights"],
        )

        best_model_path = os.path.join(
            DEFAULT_MODEL_PATH, JointWithBertPredictor.default_model_name
        )
        if not os.path.exists(best_model_path):
            download_from_url(JointWithBertPredictor.default_model_url, best_model_path)

        model = JointBERT(
            config["model"], device, dataloader.tag_dim, dataloader.intent_dim
        )
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        DEFAULT_MODEL_PATH, JointWithBertPredictor.default_model_name
                    ),
                    map_location="cpu",
                )
            )
        except Exception as e:
            print(e)
        model.to(device)

        self.model = model
        self.dataloader = dataloader
        print(f"{best_model_path} loaded")

    def predict(self, utterance, context=list()):
        ori_word_seq = self.dataloader.tokenizer.tokenize(utterance)
        ori_tag_seq = ["O"] * len(ori_word_seq)
        context_seq = self.dataloader.tokenizer.encode(
            "[CLS] " + " [SEP] ".join(context[-3:])
        )
        intents = []
        da = {}

        word_seq, tag_seq, new2ori = ori_word_seq, ori_tag_seq, None
        batch_data = [
            [
                ori_word_seq,
                ori_tag_seq,
                intents,
                da,
                context_seq,
                new2ori,
                word_seq,
                self.dataloader.seq_tag2id(tag_seq),
                self.dataloader.seq_intent2id(intents),
            ]
        ]

        pad_batch = self.dataloader.pad_batch(batch_data)
        pad_batch = tuple(t.to("cpu") for t in pad_batch)
        (
            word_seq_tensor,
            tag_seq_tensor,
            intent_tensor,
            word_mask_tensor,
            tag_mask_tensor,
            context_seq_tensor,
            context_mask_tensor,
        ) = pad_batch
        slot_logits, intent_logits = self.model(
            word_seq_tensor,
            word_mask_tensor,
            context_seq_tensor=context_seq_tensor,
            context_mask_tensor=context_mask_tensor,
        )
        intent = recover_intent(
            self.dataloader,
            intent_logits[0],
            slot_logits[0],
            tag_mask_tensor[0],
            batch_data[0][0],
            batch_data[0][-4],
        )
        return intent


if __name__ == "__main__":
    nlu = JointWithBertPredictor()
    print(
        nlu.predict(
            "北京布提克精品酒店酒店是什么类型，有健身房吗？",
            ["你好，给我推荐一个评分是5分，价格在100-200元的酒店。", "推荐您去北京布提克精品酒店。"],
        )
    )
