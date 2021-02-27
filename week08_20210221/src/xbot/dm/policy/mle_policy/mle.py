import os
import json

import torch
import torch.nn as nn

from src.xbot.util.policy_util import Policy
from src.xbot.util.path import get_config_path, get_data_path
from data.crosswoz.data_process.policy.mle_preprocess import CrossWozVector

class MultiDiscretePolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim):
        super(MultiDiscretePolicy, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(s_dim, h_dim),
            nn.RReLU(),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Linear(h_dim, a_dim),
        )

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True):
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.relu(a_weights)

        # [a_dim] => [a_dim, 2]
        a_probs = torch.cat([1 - a_probs.unsqueeze(1), a_probs.unsqueeze(1)], 1)

        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)

        return a

    def get_log_prob(self, s, a):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.tanh(a_weights)

        # [b, a_dim] => [b, a_dim, 2]
        a_probs = torch.cat([1 - a_probs.unsqueeze(-1),
                             a_probs.unsqueeze(-1)], -1)

        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1).long()).squeeze(-1)
        log_prob = torch.log(trg_a_probs)

        return log_prob.sum(-1, keepdim=True)


class MLEPolicy(Policy):
    def __init__(self):
        self.model_config_name = "policy/mle/inference.json"
        self.common_config_name = "policy/mle/common.json"
        self.model_data = {
            "model.pth": "",
            "sys_da_voc.json": "data/crosswoz/database/sys_da_voc.json",
            "usr_da_voc.json": "data/crosswoz/database/usr_da_voc.json",
        }
    def init_session(self):
        pass

    def predict(self, state):
        s_vec = torch.tensor(self.vector.state_vectorize(state))
        a = self.policy.select_action(s_vec.to(device=self.model_config["device"]),
                                      sample=False).cpu().numpy()
        action = self.vector.action_devectorize(a)
        state["system_action"] = action
        return action

    def load(self):
        # 加载所需要的文件
        common_config_path = os.path.join(
            get_config_path(), self.common_config_name
        )
        common_config = json.load(open(common_config_path))
        model_config_path = os.path.join(get_config_path(), self.model_config_name)
        model_config = json.load(open(model_config_path))
        model_config.update(common_config)
        self.model_config = model_config
        self.model_config["data_path"] = os.path.join(
            get_data_path(), "crosswoz/policy_mle_data"
        )
        self.model_config["n_gpus"] = (
            0 if self.model_config["device"] == "cpu" else torch.cuda.device_count()
        )
        self.model_config["device"] = torch.device(self.model_config["device"])

        # 载入数据
        for key, data in self.model_data.items():
            dst = os.path.join(self.model_config["data_path"], key)
            file_name = (
                key.split(".")[0]
                if not key.endswith("pth")
                else "trained_model_path"
            )
            self.model_config[file_name] = dst

        self.vector = CrossWozVector(
            sys_da_voc_json=self.model_config["sys_da_voc"],
            usr_da_voc_json=self.model_config["usr_da_voc"],
        )

        policy = MultiDiscretePolicy(
            self.vector.state_dim, model_config["hidden_size"], self.vector.sys_da_dim
        )

        policy.load_state_dict(torch.load(self.model_config["trained_model_path"]))

        self.policy = policy.to(self.model_config["device"]).eval()
        print(f'>>> {self.model_config["trained_model_path"]} loaded ...')
