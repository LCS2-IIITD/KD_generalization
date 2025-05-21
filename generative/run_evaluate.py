import json
import os
common_args_file = 'eval_args.json'
f = open(common_args_file)
common_args = json.load(f)
f.close()
cuda_devices_to_use = common_args.pop('cuda_devices_to_use', None)

# Only support single gpu since we use cuda_device='cuda' later which may cause problems in multi-gpu training 
assert(cuda_devices_to_use in ["0", "1", "2", "3", "4", "5", "6", "7"])
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices_to_use

eval_seed = "seed_" + str(common_args["seed"])

for dataset in ['gsm8k', 'SVAMP', 'MultiArith', 'SingleEq', 'AddSub', 'AQuA']:
    common_args["dataset"] = dataset
    model_name = common_args["base_model"]

    if "Qwen2.5-0.5B" in model_name:
        model_name = "Qwen2.5-0.5B"
    elif "Qwen2.5-1.5B" in model_name:
        model_name = "Qwen2.5-1.5B"
    elif "Qwen2.5-3B" in model_name:
        model_name = "Qwen2.5-3B"
    elif "Qwen2.5-7B" in model_name:
        model_name = "Qwen2.5-7B"
    elif "Qwen2.5-14B" in model_name:
        model_name = "Qwen2.5-14B"
    elif "Llama-3.2-1B" in model_name:
        model_name = "Llama-3.2-1B"
    elif "Llama-3.2-3B" in model_name:
        model_name = "Llama-3.2-3B"
    elif "Llama-3.1-8B" in model_name:
        model_name = "Llama-3.1-8B"


    if common_args["lora_weights"] == "":
        common_args["output_dir"] = os.path.join("math_10k/base_model_eval_results", f"model_{model_name}")
        common_args["wandb_run_name"] = os.path.join("evaluate", common_args["output_dir"], common_args["dataset"], eval_seed)
    else:
        common_args["wandb_run_name"] = os.path.join("evaluate", common_args["lora_weights"], common_args["dataset"], str(eval_seed))

    cmd_line_args = " ".join("--{} '{}'".format(key, value) for key,value in common_args.items())
    file_to_run = 'evaluate.py'
    cmd = 'python3 {} {}'.format(file_to_run, cmd_line_args)
    print(cmd)
    os.system(cmd)
