import json
import os
common_args_file = 'train_args.json'
f = open(common_args_file)
common_args = json.load(f)
f.close()
cuda_devices_to_use = common_args.pop('cuda_devices_to_use', None)

# Only support single gpu since we use cuda_device='cuda' later which may cause problems in multi-gpu training 
assert(cuda_devices_to_use in ["0", "1", "2", "3", "4", "5", "6", "7"])
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices_to_use


student_models = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"]
for student_model in student_models:
    common_args["student_base_model"] = student_model

    dataset = "math_10k" if "math_10k" in common_args["data_path"] else "cs_10k"
    if "math_10k" in common_args["data_path"]:
        common_args["num_epochs"] = 4
    else:
        common_args["num_epochs"] = 3

    method_str = "student_SFT"
    temp = ""
    if "kd_loss" in common_args.keys():
        if common_args["kd_loss"] == "kld":
            method_str = "student_KD"
            temp = f"temp_{common_args['kd_temperature']}"
        elif common_args["kd_loss"] == "reversekld":
            method_str = "student_RevKD"
            temp = f"temp_{common_args['kd_temperature']}"
        elif common_args["kd_loss"] == "gkd":
            method_str = "student_GKD"
            temp = f"temp_{common_args['kd_temperature']}"

    if common_args["teacher_lora_weights"] == "base-model":
        method_str += "_base_teacher"


    noise_scale = ""
    if "inject_random_noise" in common_args.keys():
        method_str += "_noisy"
        noise_scale = "noise_scale_"
        noise_scale += str(common_args["random_noise_scale"])

    teacher_name = common_args["teacher_base_model"]
    student_name = common_args["student_base_model"]
    if "Qwen2.5-0.5B" in teacher_name:
        teacher_name = "Qwen2.5-0.5B"
    elif "Qwen2.5-1.5B" in teacher_name:
        teacher_name = "Qwen2.5-1.5B"
    elif "Qwen2.5-3B" in teacher_name:
        teacher_name = "Qwen2.5-3B"
    elif "Qwen2.5-7B" in teacher_name:
        teacher_name = "Qwen2.5-7B"
    elif "Qwen2.5-14B" in teacher_name:
        teacher_name = "Qwen2.5-14B"
    elif "Llama-3.2-1B" in teacher_name:
        teacher_name = "Llama-3.2-1B"
    elif "Llama-3.2-3B" in teacher_name:
        teacher_name = "Llama-3.2-3B"
    elif "Llama-3.1-8B" in teacher_name:
        teacher_name = "Llama-3.1-8B"

    if "Qwen2.5-0.5B" in student_name:
        student_name = "Qwen2.5-0.5B"
    elif "Qwen2.5-1.5B" in student_name:
        student_name = "Qwen2.5-1.5B"
    elif "Qwen2.5-3B" in student_name:
        student_name = "Qwen2.5-3B"
    elif "Qwen2.5-7B" in student_name:
        student_name = "Qwen2.5-7B"
    elif "Qwen2.5-14B" in student_name:
        student_name = "Qwen2.5-14B"
    elif "Llama-3.2-1B" in student_name:
        student_name = "Llama-3.2-1B"
    elif "Llama-3.2-3B" in student_name:
        student_name = "Llama-3.2-3B"
    elif "Llama-3.1-8B" in student_name:
        student_name = "Llama-3.1-8B"

    common_args["output_dir"] = os.path.join( dataset, f"teacher_{teacher_name}", f"student_{student_name}", method_str, temp, noise_scale, f"seed_{common_args['seed']}")
    common_args["wandb_run_name"] = "train/" + common_args["output_dir"]

    cmd_line_args = " ".join("--{} '{}'".format(key, value) for key,value in common_args.items())
    file_to_run = 'finetune.py'
    cmd = 'python3 {} {}'.format(file_to_run, cmd_line_args)
    print("===============================================================")
    print(cmd)
    os.system(cmd)
