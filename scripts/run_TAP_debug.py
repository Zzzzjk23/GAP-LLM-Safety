import os
import sys

from gap_easyjailbreak.attacker.TAP_Mehrotra_2023_debug import TAP

from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.models.openai_model import OpenaiModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["OPENAI_BASE_URL"] = "https://api.tokenfactory.nebius.com/v1/"

sys.path.append(os.getcwd())
dataset_name_or_path = '../easyjailbreak/datasets/data/advbench_subset.json'
dataset = JailbreakDataset(local_file_type='json', dataset=dataset_name_or_path)


openai_key = "v1.CmQKHHN0YXRpY2tleS1lMDBxMDN4MHl3d3FlZ2Q4cDkSIXNlcnZpY2VhY2NvdW50LWUwMHdwcWpmYWcwNXp6YTRrZzIMCMrxgcoGEPyfrr0DOgwIy_SZlQcQgIyNngJAAloDZTAw.AAAAAAAAAAFZJjga8wP8uscjRXekw5wFdE2ySYvpwKLD-pH8KlRyKUYypqrhqdunKX2HlZ7svbE61JCwPV0gTI3eXy-3Z7AG"

attack_model = OpenaiModel(
    model_name="Qwen/Qwen3-Coder-480B-A35B-Instruct",
    api_keys=openai_key,
)
eval_model = OpenaiModel(
    model_name="moonshotai/Kimi-K2-Instruct",
    api_keys=openai_key,
)
target_model = OpenaiModel(
    model_name="openai/gpt-oss-20b",
    api_keys=openai_key,
)

dataset_name = 'TAP_Atk_Mis_Evl_GPT4_Tgt_GPT35_D_AdvBench_all'
# dataset_name = 'TAP_Atk_Vic_Evl_GPT4_Tgt_GPT35_D_AdvBench_all'
attacker = TAP(attack_model=attack_model,
               target_model=target_model,
               eval_model=eval_model,
               jailbreak_datasets=dataset,
               tree_width=10,
               tree_depth=10,
               root_num=1,
               branching_factor=4,
               keep_last_n=3,
               max_n_attack_attempts=5,
               template_file='../easyjailbreak/seed/seed_template.json',
               logging_filename=dataset_name)
eval_result = attacker.attack(save_path="{}_result.jsonl".format(dataset_name))
