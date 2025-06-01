# OCR Fine-Tuning Guide for Arabic Invoice Generator

This guide details how to use the generated Arabic invoice dataset to fine-tune various OCR models.

## Dataset Overview

The Arabic Invoice Generator creates:
- Synthetic Arabic invoices with realistic data
- Annotations for text elements (position and content)
- Augmented variations with realistic document effects

This provides an excellent foundation for OCR model fine-tuning.

## Preparing the Dataset

Run the dataset preparation script:

```bash
python prepare_ocr_dataset.py
```

This will:
1. Split data into train/validation/test sets (80/10/10 split)
2. Extract text regions from invoices
3. Create format-specific versions for different OCR engines
4. Generate appropriate label files

## OCR Fine-tuning Options

### 1. EasyOCR Fine-tuning

EasyOCR supports fine-tuning on custom datasets.

#### Prerequisites
- Install EasyOCR and its dependencies
- CUDA-capable GPU recommended

#### Steps
1. Use the prepared dataset in `ocr_dataset/easyocr`
2. Fine-tune using EasyOCR's training script:

```bash
python -m easyocr.trainer --train --lang ar --train_annotation ocr_dataset/easyocr/train_annotations.txt --valid_annotation ocr_dataset/easyocr/val_annotations.txt --model_name arabic_invoice_recognizer --batch_size 32 --imgH 64 --imgW 256 --epochs 100
```

3. The fine-tuned model will be saved to `./saved_models/arabic_invoice_recognizer/`

### 2. Tesseract Fine-tuning

Tesseract supports adding new language models or improving existing ones.

#### Prerequisites
- Tesseract 4.1+ with training tools
- Required font files

#### Steps
1. Convert dataset to Tesseract-compatible format:

```bash
# Generate box files
for img in ocr_dataset/train/images/*.png; do
    tesseract "$img" "$(basename "$img" .png)" -l ara batch.nochop makebox
done

# Create training data
tesstrain --lang ar --linedata_only \
  --train_txt ocr_dataset/tesseract/train.txt \
  --dev_txt ocr_dataset/tesseract/val.txt \
  --output_dir ocr_dataset/tesseract/output
```

2. Start training:

```bash
lstmtraining --model_output ocr_dataset/tesseract/output/arabic_invoice \
  --traineddata ocr_dataset/tesseract/output/ar.traineddata \
  --train_listfile ocr_dataset/tesseract/output/ar.training_files.txt \
  --eval_listfile ocr_dataset/tesseract/output/ar.eval_files.txt \
  --continue_from ara.traineddata
```

### 3. Transformer-based OCR Fine-tuning

Modern OCR approaches using vision transformers can be fine-tuned on our dataset.

#### Prerequisites
- PyTorch
- Transformers library
- High-end GPU (8GB+ VRAM)

#### Steps
1. Install dependencies:

```bash
pip install torch transformers datasets
```

2. Fine-tune using a script like:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imagefolder", 
                      data_dir="ocr_dataset",
                      split={"train": "train", "validation": "val"})

# Load model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base")

# Train
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()
```

### 4. Keras-OCR / PaddleOCR

Both Keras-OCR and PaddleOCR offer Arabic support and can be fine-tuned.

#### PaddleOCR Example

```bash
# Install PaddleOCR
pip install paddleocr

# Run training 
python -m paddleocr.tools.train --config_path config/arabic_config.yml \
  --train_data_dir ocr_dataset/train/images/ \
  --train_label_file ocr_dataset/train/labels/gt.txt
```

## Evaluation

After fine-tuning, evaluate your model:

```bash
python evaluate_ocr.py --model_type easyocr --model_path saved_models/arabic_invoice_recognizer --test_data ocr_dataset/test
```

This will produce:
- Character Error Rate (CER)
- Word Error Rate (WER)
- Precision, Recall, F1-score
- Confusion matrix for Arabic characters

## Integration with your Pipeline

To use your fine-tuned model in the invoice processing pipeline:

```python
# For EasyOCR example
import easyocr

reader = easyocr.Reader(['ar'], model_storage_directory='./saved_models', 
                       user_network_directory='./saved_models', 
                       recog_network='arabic_invoice_recognizer')

# Read text from invoice
results = reader.readtext('path/to/invoice.jpg')
```

## Common Issues & Solutions

1. **Poor recognition of numerical fields**: Focus training on invoice-specific number formats
2. **Arabic ligatures issues**: Ensure dataset includes diverse Arabic typography
3. **GPU memory limitations**: Reduce batch size or use mixed precision training
4. **Overfitting**: Increase augmentation or reduce model complexity

## Best Practices

1. Start with a model pre-trained on Arabic
2. Use your augmentation pipeline to further diversify the training data
3. Pay special attention to domain-specific terminology
4. Consider ensemble approaches for critical applications
5. Keep some real invoices (if available) for final testing

## Resources

- [Arabic OCR Benchmark](https://github.com/mmmz/arabic-ocr-benchmark)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)
- [Tesseract Training Guide](https://tesseract-ocr.github.io/tessdoc/TrainingTesseract-4.00.html)
- [PaddleOCR Arabic Support](https://github.com/PaddlePaddle/PaddleOCR)