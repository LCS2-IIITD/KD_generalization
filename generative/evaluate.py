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
    #import pdb; pdb.set_trace()
    args = parse_args()
    if args.seed:
        print("========================================================================")
        print(f"setting seed to {args.seed}")
        set_seed(int(args.seed))

    if args.report_to == "wandb":
        wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "model": "Qwen/Qwen2.5-0.5B",
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
            instruction,
            input_text=None,
    ):
        prompt = generate_prompt(instruction, input_text)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=int(args.max_new_tokens),
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """
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
    tokenizer, model = load_model(args)
    total = len(dataset)
    correct = 0
    is_correct = False
    miss = 0.001
    output_data = []
    pbar = tqdm(total=total)
    for idx, data in enumerate(dataset):
        instruction = data.get('instruction')
        input_txt = data.get('input')
        expexted_output = data.get('output')

        outputs = evaluate(instruction, input_txt)
        label = data.get('answer')
        flag = False
        if args.dataset.lower() in ['aqua']:
            predict = extract_answer_letter(args, outputs)
            if label == predict:
                correct += 1
                flag = True
        else:
            if isinstance(label, str):
                label = float(label)
            predict = extract_answer_number(args, outputs)
            if abs(label - predict) <= miss:
                correct += 1
                flag = True
        new_data = copy.deepcopy(data)
        new_data['output_pred'] = outputs
        new_data['pred'] = predict
        new_data['flag'] = flag
        output_data.append(new_data)
        '''
        print(' ')
        print('---------------')
        print(outputs)
        print('prediction:', predict)
        print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
        '''
        ff.write(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}')
        ff.write("\n")
        with open(save_file, 'w+') as f:
            json.dump(output_data, f, indent=4)

        if args.report_to == "wandb":
            wandb.log({
                "instance_index": idx,
                "instruction": instruction,
                "input": input_txt,
                "output": expexted_output,
                "output_pred": outputs,
                "expected_anwer": label,
                "pred_answer": predict,
                "is_correct": is_correct,
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
    ff.close()


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input_text=None):
    if input_text != None and input_text != "":
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['AddSub', 'MultiArith', 'SingleEq', 'gsm8k', 'AQuA', 'SVAMP'],
                        required=True)
    parser.add_argument('--model', choices=['LLaMA-7B', 'BLOOM-7B', 'GPT-j-6B', 'Other'], required=True)
    parser.add_argument('--adapter', choices=['LoRA', 'AdapterP', 'AdapterH', 'Parallel', 'Prefix'],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--student_slice_layers', required=False, default=-1)
    parser.add_argument('--lora_weights', required=True)
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
    #if not lora_weights:
    #    raise ValueError(f'can not find lora weight, the value is: {lora_weights}')

    load_8bit = args.load_8bit
    if args.model == 'LLaMA-7B':
        tokenizer = LlamaTokenizer.from_pretrained(base_model)

    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "cuda" in device:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        ) # fix zwq

        if int(args.student_slice_layers) != -1:
            model.model.layers  = model.model.layers[-1*int(args.student_slice_layers):]

        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map=None
            )
        model = model.to(device)
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if int(args.student_slice_layers) != -1:
            model.model.layers  = model.model.layers[-1*int(args.student_slice_layers):]

        if lora_weights != "":
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                device_map={"": device},
            )

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


def extract_answer_number(args, sentence: str) -> float:
    dataset = args.dataset.lower()
    if dataset in ["multiarith", "addsub", "singleeq", "gsm8k", "svamp"]:
        sentence = sentence.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
        if not pred:
            return float('inf')
        pred_answer = float(pred[-1])
    else:
        raise NotImplementedError(' not support dataset: {}'.format(dataset))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(args, sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''


if __name__ == "__main__":
    fire.Fire(main)
