import argparse
import csv
import os
import sys

from gap_easyjailbreak.attacker.GAP_Schwartz_2024 import GAP

from easyjailbreak.datasets import JailbreakDataset
from easyjailbreak.datasets.instance import Instance
from easyjailbreak.models.openai_model import OpenaiModel

DEFAULT_API_KEY = "v1.CmQKHHN0YXRpY2tleS1lMDBxMDN4MHl3d3FlZ2Q4cDkSIXNlcnZpY2VhY2NvdW50LWUwMHdwcWpmYWcwNXp6YTRrZzIMCMrxgcoGEPyfrr0DOgwIy_SZlQcQgIyNngJAAloDZTAw.AAAAAAAAAAFZJjga8wP8uscjRXekw5wFdE2ySYvpwKLD-pH8KlRyKUYypqrhqdunKX2HlZ7svbE61JCwPV0gTI3eXy-3Z7AG"


def load_goal_target_csv(path: str) -> JailbreakDataset:
    dataset = JailbreakDataset([])
    with open(path, mode="r", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            goal = row.get("goal", "").strip()
            target = row.get("target", "").strip()
            if not goal:
                continue
            reference_responses = [target] if target else []
            dataset.add(
                Instance(
                    query=goal,
                    reference_responses=reference_responses,
                )
            )
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAP on a goal/target CSV dataset.")
    parser.add_argument("csv_path", help="Path to the CSV file containing goal/target columns.")
    parser.add_argument(
        "--output-name",
        default="GAP_Atk_Qwen_Evl_Kimi_Tgt_GPTOSS",
        help="Base name for logging and output JSONL.",
    )
    parser.add_argument(
        "--template-file",
        default="../easyjailbreak/seed/seed_template.json",
        help="Path to the seed template JSON file.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_KEY", DEFAULT_API_KEY),
        help="Nebius Token Factory API key (defaults to OPENAI_KEY).",
    )
    parser.add_argument(
        "--attack-model",
        default="Qwen/Qwen3-Coder-480B-A35B-Instruct",
        help="Attack model name.",
    )
    parser.add_argument(
        "--eval-model",
        default="moonshotai/Kimi-K2-Instruct",
        help="Evaluator model name.",
    )
    parser.add_argument(
        "--target-model",
        default="openai/gpt-oss-20b",
        help="Target model name.",
    )
    parser.add_argument("--tree-width", type=int, default=10)
    parser.add_argument("--tree-depth", type=int, default=10)
    parser.add_argument("--root-num", type=int, default=1)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--keep-last-n", type=int, default=3)
    parser.add_argument("--max-n-attack-attempts", type=int, default=5)
    args = parser.parse_args()

    os.environ.setdefault("OPENAI_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    sys.path.append(os.getcwd())
    dataset = load_goal_target_csv(args.csv_path)

    attack_model = OpenaiModel(model_name=args.attack_model, api_keys=args.api_key)
    eval_model = OpenaiModel(model_name=args.eval_model, api_keys=args.api_key)
    target_model = OpenaiModel(model_name=args.target_model, api_keys=args.api_key)

    attacker = GAP(
        attack_model=attack_model,
        target_model=target_model,
        eval_model=eval_model,
        jailbreak_datasets=dataset,
        tree_width=args.tree_width,
        tree_depth=args.tree_depth,
        root_num=args.root_num,
        branching_factor=args.branching_factor,
        keep_last_n=args.keep_last_n,
        max_n_attack_attempts=args.max_n_attack_attempts,
        template_file=args.template_file,
        logging_filename=args.output_name,
    )
    attacker.attack(save_path=f"{args.output_name}_result.jsonl")


if __name__ == "__main__":
    main()
