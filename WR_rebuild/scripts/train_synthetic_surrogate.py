"""Train the physics-guided forward U-Net on a synthetic dataset."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from run_mf6_pilot_experiments import Config, load_dataset_from_dir, set_seed, train_surrogate


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=root / 'WR_rebuild/outputs/mf6_highcontrast_physics_surrogate')
    parser.add_argument('--config', type=Path, default=root / 'configs/synthetic_forward.json')
    parser.add_argument('--out-dir', type=Path, default=root / 'runs/trained_forward')
    parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text(encoding='utf-8')))
    if args.epochs is not None:
        config.epochs = args.epochs
    set_seed(config.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
    dataset = load_dataset_from_dir(args.data_dir)
    _, metrics = train_surrogate(dataset, config, args.out_dir)
    (args.out_dir / 'metrics/surrogate_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
