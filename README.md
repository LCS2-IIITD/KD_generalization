### Installation

1. Create a new conda environment and activate it
    ```
    conda create -n kd
    conda activate kd
    ``` 
2. Install the required libraries using pip install -r requirements.txt
3. Install the custom PEFT library using cd peft and then pip install -e .

### Training the models
5. Train Teacher model. In the train_args.json file, set the "teacher_base_model" and "student_base_model" pointing to the same model, "teacher_lora_weights" and "kd_loss" to empty string.
    ```
    python3 run_train.py
    ```
    This will save the Teacher model's LoRA weights in the at "output_dir"
6.  To train student KD models, in train_args.json file, set the "teacher_base_model" and "student_base_model" to appropriate model paths, "teacher_lora_weights" to the path in previous step. 
   For KD related arguments you can use:
   "kd_loss" : "kld" (for SeqKD) / "reversekld" (for RevKD) / "gkd" (for GKD)
    We use a "kd_loss_weight" of 1 for all experiments.
    You can use the "kd_temperature" parameter to smoothen the logit distribution before distillation. 
7. For mathematical and commonsense reasoning tasks set the parameter "data_path" to "ft-training_set/math_10k.json" and "ft-training_set/commonsense_15k.json" respectively.

### Evaluating the models
8. To evaluate the models we use eval_args.json file. You need to metnion the model name in "base_mode" parameter and provide the path to the saved LoRA weights in parameter "lora_weights"
   Use run_evaluate.py and run_cs_evaluate.py for mathematical and commonsense reasoning tasks respectively.
   ```
    python3 run_evaluate.py
    python3 run_cs_evaluate.py
    ```
   The results are saved in the same directory as the LoRA weights. A sub-directory is created for each task with two files - one showing the accuracy scores and another with the instruction text and the model's output.

### Citation
```bibtex
@inproceedings{
    kdacl2025,
    title={On the Generalization vs Fidelity Paradox in Knowledge Distillation},
    author={Suhas K Ramesh, Ayan Sengupta, Tanmoy Chakraborty},
    booktitle={The 63rd Annual Meeting of the Association for Computational Linguistics},
    year={2025},
    url={https://openreview.net/forum?id=co3zQsH7yz}
}
```
