import copy
import json
import os
import re
import sys
import argparse

import fire

import torch
import wandb

sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, set_seed

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
        load_8bit: bool = False,
        base_model: str = "",
        student_slice_layers: int = -1, 
        lora_weights: str = "tloen/alpaca-lora-7b",
        share_gradio: bool = False,
):
    args = parse_args()
    #import pdb; pdb.set_trace()

    if args.seed:
        print("========================================================================")
        print(f"setting seed to {args.seed}")
        set_seed(int(args.seed))

    if args.report_to == "wandb":
        wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "model": args.base_model,
                    "do-sample": args.do_sample,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "num_beams": args.num_beams,
                    "max_new_tokens": args.max_new_tokens,
                }
        )

    generation_config = GenerationConfig(
        do_sample = bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        num_beams=int(args.num_beams),
    )

    def evaluate(
            instructions,
            input_text = None
    ):
        prompts = [generate_prompt(instruction, input_text) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=int(args.max_new_tokens),
                pad_token_id=tokenizer.pad_token_id,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        outputs = [o.split("### Response:")[1].strip() for o in outputs]
        #print(outputs)
        return outputs

    if args.output_dir != "":
        results_path = os.path.join(args.output_dir, args.dataset)
    elif args.lora_weights != "":
        results_path = os.path.join(args.lora_weights, args.dataset)
    else:
        raise ValueError(f'can not determine output directory')

    os.makedirs(results_path, exist_ok=True)
    save_file = f'{results_path}/seed_{args.seed}.json'
    save_files2 = f'{results_path}/seed_{args.seed}.txt'
    ff = open(save_files2, 'w')
    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)
    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            '''
            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
           '''
        ff.write(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
        ff.write("\n")
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)

        if args.report_to == "wandb":
            wandb.log({
                "instance_index": idx,
                "correct_count": correct,
                "accuracy_so_far": correct/ total,
                })

        pbar.update(1)
    pbar.close()
    if args.report_to == "wandb":
        wandb.log({"final_accuracy": correct/ total})
        wandb.finish()
    print('\n')
    print('test finished')


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input_text=None):
    if input_text and input_text != "":
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input_text}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', "LLaMA-13B",'BLOOM-7B', 'GPT-j-6B', 'Other'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--student_slice_layers', required=False, default=-1)
    parser.add_argument('--lora_weights', required=True)
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--load_8bit', action='store_true', default=False)
    parser.add_argument('--do-sample', required=False, default=True)
    parser.add_argument('--temperature', required=False, default=1.0)
    parser.add_argument('--top_p', required=False, default=1.0)
    parser.add_argument('--top_k', required=False, default=0)
    parser.add_argument('--num_beams', required=False, default=1)
    parser.add_argument('--max_new_tokens', required=False, default=256)

    parser.add_argument('--output_dir', required=False, default="")
    parser.add_argument('--report_to', required=False, default="")
    parser.add_argument('--wandb_project', required=False, default="")
    parser.add_argument('--wandb_run_name', required=False, default="")
    parser.add_argument('--seed')

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f'can not find base model name by the value: {args.model}')
    lora_weights = args.lora_weights

    load_8bit = args.load_8bit
    if "LLaMA" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        ) # fix zwq
	
        if int(args.student_slice_layers) != -1:
            model.model.layers  = model.model.layers[-1*int(args.student_slice_layers):]

        if lora_weights != "":
            try:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16,
                    device_map={"":0}
                )
            except:
                pass

    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora_weights != "":
            try:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    device_map={"": device},
                    torch_dtype=torch.float16,
                )
            except:
                pass
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if int(args.student_slice_layers) != -1:
            model.model.layers  = model.model.layers[-1*int(args.student_slice_layers):]

        if lora_weights != "":
            try:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    device_map={"": device},
                )
            except:
                pass

        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()
