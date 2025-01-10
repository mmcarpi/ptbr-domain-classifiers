import json
from dataclasses import dataclass, asdict

@dataclass
class DeviceConfig:
    rank: int
    local_rank: int
    world_size: int
    device_type: str
    num_gpus_per_node: int

@dataclass
class ModelConfig:
    model_name: int
    num_labels: int
    batch_size: int
    max_length: int

    weight_decay: float
    warm_up_ratio: float
    learning_rate: float

    @classmethod
    def load_config(cls, path):
        with open(path, "r") as config_file:
            config = json.load(config_file)
        return cls(**config)

    def save_config(self, path):
        with open(path, "w") as config_file:
            json.dump(asdict(self), config_file, indent=4)

    def __str__(self):
        return "\n".join(map(lambda x: x[0] + ': ' + str(x[1]), asdict(self).items()))
