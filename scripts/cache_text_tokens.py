"""Encode image_id/text CSV into frozen MedCPT token features."""
import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from btxrd.utils.io import sha256_file

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text-csv', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--encoder', default='ncbi/MedCPT-Query-Encoder')
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.max_length < 1 or args.batch_size < 1:
        parser.error('length and batch size must be positive')
    frame = pd.read_csv(args.text_csv, usecols=['image_id', 'text'], dtype=str)
    if frame.empty or frame.isna().any().any() or frame.image_id.duplicated().any() or (frame.text.str.strip() == '').any() or (frame.image_id.str.strip() == '').any():
        raise ValueError('CSV requires unique nonempty IDs and nonempty text')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    model = AutoModel.from_pretrained(args.encoder).to(args.device).eval().requires_grad_(False)
    tokens, masks = [], []
    with torch.no_grad():
        for start in range(0, len(frame), args.batch_size):
            inputs = tokenizer(frame.text.iloc[start:start + args.batch_size].tolist(), padding='max_length',
                               truncation=True, max_length=args.max_length, return_tensors='pt').to(args.device)
            tokens.append(model(**inputs).last_hidden_state.cpu().half())
            masks.append(inputs['attention_mask'].cpu().bool())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(image_ids=frame.image_id.tolist(), tokens=torch.cat(tokens), mask=torch.cat(masks),
                    encoder_name=args.encoder, max_length=args.max_length,
                    text_csv_sha256=sha256_file(args.text_csv)), args.out)

if __name__ == '__main__':
    main()