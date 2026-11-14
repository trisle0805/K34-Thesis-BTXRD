"""Cache fixed MedCPT CLS location queries."""
import argparse
from pathlib import Path
import torch
from btxrd.data.locations import location_definition
from btxrd.knowledge.text_encoder import encode_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='kb/location_propositions_v1.csv')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    payload = location_definition(args.csv)
    payload['embeddings'] = encode_texts(payload['texts'], payload['model_name'], 64, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)


if __name__ == '__main__':
    main()