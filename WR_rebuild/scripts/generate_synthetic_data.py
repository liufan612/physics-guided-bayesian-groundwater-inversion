"""Generate the study's synthetic conductivity fields and MODFLOW 6 heads."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from run_mf6_pilot_experiments import Config, generate_dataset, set_seed


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mf6', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=root / 'configs/synthetic_forward.json')
    parser.add_argument('--out-dir', type=Path, default=root / 'runs/generated_data')
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text(encoding='utf-8')))
    set_seed(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    dataset = generate_dataset(config, args.mf6.resolve(), args.out_dir)
    print(f'Generated {len(dataset["case_id"])} cases in {args.out_dir}')


if __name__ == '__main__':
    main()
