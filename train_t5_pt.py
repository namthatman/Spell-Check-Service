import tensorflow as tf
import torch
#import nlp
from datasets import Dataset
import numpy as np
import pandas as pd

import json
import os
import dataclasses
import logging
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from transformers import T5ForConditionalGeneration, T5Tokenizer, EvalPrediction, TFT5ForConditionalGeneration
from transformers import (
    HfArgumentParser,
    DataCollator,
    Trainer,
    TFTrainer,
    TrainingArguments,
    set_seed,
)



# Build T5 dataset

tokenizer = T5Tokenizer.from_pretrained('t5-small')

num_sample = 1000
input_articles = pd.read_csv("resources/train_dataset/false_input_articles_full.csv", header=None)
output_articles = pd.read_csv("resources/train_dataset/false_output_articles_full.csv", header=None)

inputs = input_articles[:num_sample]
outputs = output_articles[:num_sample]

qa_inputs = []
for text in inputs[0].values:
    qa_inputs.append("question: " + str(text) + "  context:  </s>")
    
qa_outputs = []
for text in outputs[0].values:
    qa_outputs.append(str(text) + " </s>")

#with open("resources/train_dataset/T5/inputs.txt", 'w', encoding='utf-8') as f:
#    for text in qa_inputs:
#        f.write(str(text) + "\n")

#with open("resources/train_dataset/T5/outputs.txt", 'w', encoding='utf-8') as f:
#    for text in qa_outputs:
#        f.write(str(text) + "\n")
        
        
df = pd.DataFrame({"inputs": qa_inputs, "outputs": qa_outputs})
dataset = Dataset.from_pandas(df)


# tokenize the examples
def convert_to_features(example_batch):
    input_encodings = tokenizer.batch_encode_plus(example_batch['inputs'], pad_to_max_length=True, max_length=256)
    target_encodings = tokenizer.batch_encode_plus(example_batch['outputs'], pad_to_max_length=True, max_length=256)

    encodings = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }
    return encodings

t5_dataset = dataset.map(convert_to_features, batched=True)

# set the tensor type and the columns which the dataset should return
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
#t5_dataset.set_format(type='tensorflow', columns=columns)
t5_dataset.set_format(type='torch', columns=columns)
len(t5_dataset)

torch.save(t5_dataset, "resources/train_dataset/T5/t5_dataset.pt")


# Training Script

#model.resize_token_embeddings(len(tokenizer))

logger = logging.getLogger(__name__)

@dataclass
class T2TDataCollator():
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        labels = torch.stack([example['target_ids'] for example in batch])
        labels[labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': labels, 
            'decoder_attention_mask': decoder_attention_mask
        }
    
    
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_file_path: Optional[str] = field(
        default='resources/train_dataset/T5/t5_dataset.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='resources/train_dataset/T5/t5_dataset.pt',
        metadata={"help": "Path for cached valid dataset"},
    )
    max_len: Optional[int] = field(
        default=256,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=256,
        metadata={"help": "Max input length for the target text"},
    )
    
    
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # we will load the arguments from a json file, 
    #make sure you save the arguments in at ./resources/T5/args.json
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath("resources/T5/args_pt.json"))

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = T5Tokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    print('loading data')
    train_dataset  = torch.load(data_args.train_file_path)
    valid_dataset = torch.load(data_args.valid_file_path)
    print('loading done')

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=T2TDataCollator(),
        prediction_loss_only=True
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


# Train

args_dict = {
    "num_cores": 8,
    'training_script': 'train_t5_.py',
    "model_name_or_path": 't5-small',
    "max_len": 256 ,
    "target_max_len": 256,
    "output_dir": './resources/T5/models/tpu',
    "overwrite_output_dir": True,
    "per_gpu_train_batch_size": 8,
    "per_gpu_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 3e-4,
    "tpu_num_cores": 8,
    "num_train_epochs": 4,
    "do_train": True
}

with open("resources/T5/args_pt.json", 'w') as f:
    json.dump(args_dict, f)
    
    
_mp_fn(1)


# Eval

model = TFT5ForConditionalGeneration.from_pretrained('resources/T5/models/tpu', from_pt=True)
tokenizer = T5Tokenizer.from_pretrained('resources/T5/models/tpu')

inputs = tokenizer.encode("question: anh yeu yeu em.  context:  </s>", return_tensors="tf")  # Batch size 1
generated_ids = model.generate(
    inputs,
    max_length=80,
    #repetition_penalty=2.5
)
predicted_span = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

inputs[0]
generated_ids[0]







