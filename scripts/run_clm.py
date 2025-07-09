import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder


def parse_arge():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="lm_dataset", help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--hf_token", type=str, default=HfFolder.get_token(), help="add any help statement if you want(optional parameter)"
    )
  
    parser.add_argument(
        "--epochs", type=int, default=3, help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="add any help statement if you want(optional parameter)"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="add any help statement if you want(optional parameter)",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="add any help statement if you want(optional parameter)",
    )
    args, _ = parser.parse_known_args()

    if args.hf_token:
       print("logging into hf hub with token")
       login(token=args.hf_token)

    return args



def print_trainable_parameters(model, use_4bit=False):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )



def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer


    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()


    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)


    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model


def training_function(args):

    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False
        if args.gradient_checkpointing
        else True,
        device_map="auto",
        quantization_config=bnb_config,
    )


    model = create_peft_model(
        model, gradient_checkpointing=args.gradient_checkpointing, bf16=args.bf16
    )


    output_dir = "/tmp/mixtral"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )


    trainer.train()

    sagemaker_save_dir="/opt/ml/model/"
    if args.merge_weights:

        trainer.model.save_pretrained(output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )  

        model = model.merge_and_unload()        
        model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(
            sagemaker_save_dir, safe_serialization=True
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)


def main():
    args = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()