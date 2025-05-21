import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from typing import List, Optional, Union

from copy import deepcopy as cp
from peft import PeftModel

import json
import os
import inspect

"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import (  # noqa: E402
    LoraConfig,
    BottleneckConfig,
    PrefixTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, AutoModel, set_seed  # noqa: F402

from kd_trainer import Student_Trainer

#os.environ["WANDB_DISABLED"] = "true"

def train(
        # model/data params
        teacher_base_model: str = "",  # the only required argument
        teacher_lora_weights: str = "",  # optional
        student_base_model: str = "none", #student model, required
        kd_loss: str = "kld", #reversekld, mse, none 
        kd_loss_weight: float = 0.5, 
        student_slice_layers: float = -1,
        finetune_teacher: bool = False, 
        inject_random_noise: bool = False,
        random_noise_scale: float = 0.001,
        kd_temperature: float = 1.0,
        data_path: str = "yahma/alpaca-cleaned",
        output_dir: str = "./lora-alpaca_student",
        adapter_name: str = "lora",
        load_8bit : bool = False,
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        use_gradient_checkpointing: bool = False,
        eval_step: int = 200,
        save_step: int = 200,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ['q_proj','k_proj','v_proj','up_proj','down_proj'],
        # bottleneck adapter hyperparams
        bottleneck_size: int = 256,
        non_linearity: str = "tanh",
        adapter_dropout: float = 0.0,
        use_parallel_adapter: bool = False,
        use_adapterp: bool = False,
        target_modules: List[str] = ['q_proj','k_proj','v_proj','up_proj','down_proj'],
        scaling: Union[float, str] = 1.0,
        # prefix tuning hyperparams
        num_virtual_tokens: int = 30,
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        seed: int = 6,
):
    teacher_output_dir = os.path.join(output_dir, "teacher_model")
    student_output_dir = os.path.join(output_dir, "student_model") 
    print(
        f"Finetuning model with params:\n"
        f"teacher_base_model: {teacher_base_model}\n"
        f"teacher_lora_weights: {teacher_lora_weights}\n"
        f"student_base_model: {student_base_model}\n"
        f"data_path: {data_path}\n"
        f"teacher_output_dir: {teacher_output_dir}\n"
        f"student_output_dir: {student_output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"bottleneck_size: {bottleneck_size}\n"
        f"non_linearity: {non_linearity}\n"
        f"adapter_dropout: {adapter_dropout}\n"
        f"use_parallel_adapter: {use_parallel_adapter}\n"
        f"use_adapterp: {use_adapterp}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"scaling: {scaling}\n"
        f"adapter_name: {adapter_name}\n"
        f"target_modules: {target_modules}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"seed: {seed}\n"
    )
    assert (
        teacher_base_model
    ), "Please specify a --teacher_base_model, e.g. --teacher_base_model='decapoda-research/llama-7b-hf'"

    assert not (student_base_model == 'none' and student_slice_layers == -1)

    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    args_to_log = {arg: values[arg] for arg in args if arg != 'frame'}
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "arguments_log.json")
    with open(log_file, "w") as file:
        json.dump(args_to_log, file, indent=4)

    set_seed(seed)
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if load_8bit:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            teacher_base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )

        if student_base_model != 'none':
            student_model = AutoModelForCausalLM.from_pretrained(
                student_base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=True,
            )
        else:
            student_model = cp(teacher_model)
            student_model.model.layers  = student_model.model.layers[-1*student_slice_layers:]

    else:
        teacher_model = None
        if teacher_lora_weights != "":
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
            if teacher_lora_weights != "base-model":
                teacher_model = PeftModel.from_pretrained(
                        teacher_model,
                        teacher_lora_weights,
                        torch_dtype=torch.float16,
                        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)}
                )
            else:
                print(f"No lora adapter. Using Teacher base model - {teacher_base_model}")

        if student_base_model != 'none':
            student_model = AutoModelForCausalLM.from_pretrained(
                student_base_model,
                load_in_8bit=False,
                torch_dtype=torch.float16,
                device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
                trust_remote_code=True,
            )
        else:
            #config = teacher_model.config
            #config.num_hidden_layers = student_slice_layers
            #student_model = AutoModelForCausalLM.from_config(config,
            #    torch_dtype=torch.float16,
            #    trust_remote_code=True,)
            student_model = cp(teacher_model)
            student_model.model.layers  = student_model.model.layers[-1*student_slice_layers:]

    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_base_model, trust_remote_code=True)
    if student_base_model != 'none':
        student_tokenizer = AutoTokenizer.from_pretrained(student_base_model, trust_remote_code=True)
    else:
        student_tokenizer = AutoTokenizer.from_pretrained(teacher_base_model, trust_remote_code=True)

    if not teacher_tokenizer.pad_token_id:
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id
    if not student_tokenizer.pad_token_id:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    teacher_tokenizer.padding_side = "left"  # Allow batched inference
    student_tokenizer.padding_side = "left"  # Allow batched inference

    assert teacher_tokenizer.vocab == student_tokenizer.vocab

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = teacher_tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != teacher_tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(teacher_tokenizer.eos_token_id)
            if "chatglm" not in teacher_base_model:
                result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        if "chatglm" in teacher_base_model:
            return {"input_ids": result["input_ids"], "labels": result["labels"]}
        else:
            return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)

        # for GKD we need to also pass the prompt_attention_mask 
        #   this will help extract the prompt from the input string
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        full_prompt_len = len(tokenized_full_prompt["input_ids"])
        tokenized_full_prompt["prompt_attention_mask"] = [1] * user_prompt_len + [0] * (full_prompt_len-user_prompt_len)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
        return tokenized_full_prompt

    if teacher_model:
        teacher_model = prepare_model_for_int8_training(teacher_model, use_gradient_checkpointing=use_gradient_checkpointing)
    student_model = prepare_model_for_int8_training(student_model, use_gradient_checkpointing=use_gradient_checkpointing)

    if adapter_name == "lora":
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "bottleneck":
        config = BottleneckConfig(
            bottleneck_size=bottleneck_size,
            non_linearity=non_linearity,
            adapter_dropout=adapter_dropout,
            use_parallel_adapter=use_parallel_adapter,
            use_adapterp=use_adapterp,
            target_modules=target_modules,
            scaling=scaling,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif adapter_name == "prefix-tuning":
        config = PrefixTuningConfig(
            num_virtual_tokens=num_virtual_tokens,
            task_type="CAUSAL_LM",
        )
    
    if finetune_teacher:
        teacher_model = get_peft_model(teacher_model, config)
    
    student_model = get_peft_model(student_model, config)
    if adapter_name == "prefix-tuning":
        teacher_model.to('cuda')
        student_model.to('cuda')

    if data_path.endswith(".json"):  # todo: support jsonl
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    student_model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=seed
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if finetune_teacher and not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        teacher_model.is_parallelizable = True
        teacher_model.model_parallel = True

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        student_model.is_parallelizable = True
        student_model.model_parallel = True

    if finetune_teacher:
        trainer = transformers.Trainer(
            model=teacher_model if kd_loss != '' else None,
            train_dataset=train_data,
            eval_dataset=val_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=50,
                optim="adamw_torch",
                evaluation_strategy="steps" if val_set_size > 0 else "no",
                save_strategy="steps",
                eval_steps=eval_step if val_set_size > 0 else None,
                save_steps=save_step,
                output_dir=teacher_output_dir,
                save_total_limit=3,
                load_best_model_at_end=True if val_set_size > 0 else False,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=group_by_length,
                report_to="wandb" if use_wandb else [],
                run_name=wandb_run_name if use_wandb else None,
                remove_unused_columns=False,
            ),
            data_collator=transformers.DataCollatorForSeq2Seq(
                teacher_tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        teacher_model.config.use_cache = False

        old_state_dict = teacher_model.state_dict
        teacher_model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(teacher_model, type(teacher_model))

        if torch.__version__ >= "2" and sys.platform != "win32":
            teacher_model = torch.compile(teacher_model)

        try:
            trainer.train(resume_from_checkpoint=False)
        except:
            pass
        #teacher_model.save_pretrained(teacher_output_dir)

    trainer = Student_Trainer(
        model=student_model,
        teacher_model=teacher_model if kd_loss != '' else None,
        kd_loss=kd_loss,
        kd_loss_weight=kd_loss_weight,
        inject_random_noise=inject_random_noise,
        random_noise_scale=random_noise_scale,
        kd_temperature=kd_temperature,
        max_length=cutoff_len,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=student_tokenizer,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=50,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=student_output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else [],
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            teacher_tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    student_model.config.use_cache = False

    old_state_dict = student_model.state_dict
    student_model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(student_model, type(student_model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        student_model = torch.compile(student_model)

    trainer.train(resume_from_checkpoint=False)

    student_model.save_pretrained(student_output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

def generate_prompt(data_point):
    full_prompt = ""
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    else:
        full_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}""" # noqa: E501
    return full_prompt


if __name__ == "__main__":
    fire.Fire(train)
