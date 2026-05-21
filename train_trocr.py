"""
Minimal TrOCR fine-tuning script for Arabic invoice OCR using the dataset produced by
`prepare_ocr_dataset.py` (output/ocr_finetune_dataset).

Usage:
    python train_trocr.py --dataset_dir output/ocr_finetune_dataset --output_dir saved_models/trocr_arabic --epochs 10

This script is intentionally minimal — adapt hyperparameters and model choice for best results.
"""
import os
import argparse
from PIL import Image
from datasets import Dataset, DatasetDict
import random

import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
)


def load_label_file(labels_path, images_root):
    examples = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                continue
            rel_path, text = parts
            img_path = os.path.join(images_root, os.path.basename(rel_path))
            if not os.path.exists(img_path):
                continue
            examples.append({'image_path': img_path, 'text': text})
    return examples


def prepare_datasets(examples, seed=42):
    random.seed(seed)
    random.shuffle(examples)
    n = len(examples)
    if n == 0:
        raise ValueError('No examples found to train on. Run prepare_ocr_dataset.py first.')

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    ds = DatasetDict({
        'train': Dataset.from_list(train),
        'validation': Dataset.from_list(val),
        'test': Dataset.from_list(test),
    })
    return ds


def preprocess_examples(batch, processor, max_target_length=256):
    # Load image
    images = [Image.open(p).convert('RGB') for p in batch['image_path']]
    pixel_inputs = processor(images=images, return_tensors='pt').pixel_values

    # Tokenize targets
    with processor.tokenizer.as_target_tokenizer():
        labels = processor.tokenizer(batch['text'], padding='longest', truncation=True, max_length=max_target_length).input_ids

    batch['pixel_values'] = pixel_inputs
    batch['labels'] = labels
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='output/ocr_finetune_dataset')
    parser.add_argument('--output_dir', type=str, default='saved_models/trocr_arabic')
    parser.add_argument('--pretrained_model', type=str, default='microsoft/trocr-base')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    labels_path = os.path.join(args.dataset_dir, 'labels.txt')
    images_root = os.path.join(args.dataset_dir, 'images')

    examples = load_label_file(labels_path, images_root)
    ds = prepare_datasets(examples)

    print(f"Dataset sizes: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")

    print('Loading processor and model...')
    processor = TrOCRProcessor.from_pretrained(args.pretrained_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model)
    model.to(args.device)

    # Preprocess and set format
    def collate_fn(batch_list):
        # This collate uses the processor to create pixel_values and tokenizer to create labels
        images = [Image.open(x['image_path']).convert('RGB') for x in batch_list]
        pixel_values = processor(images=images, return_tensors='pt').pixel_values
        texts = [x['text'] for x in batch_list]
        with processor.tokenizer.as_target_tokenizer():
            labels = processor.tokenizer(texts, padding='longest', return_tensors='pt').input_ids
        # Replace tokenizer pad token id's by -100 to ignore in loss
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch = {
            'pixel_values': pixel_values,
            'labels': labels,
        }
        return batch

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        evaluation_strategy='steps',
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        save_total_limit=3,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    print('Starting training...')
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f'Model saved to {args.output_dir}')


if __name__ == '__main__':
    main()

