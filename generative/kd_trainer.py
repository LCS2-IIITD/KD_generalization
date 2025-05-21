import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import GenerationConfig
import random
import wandb

def sequence_reverse_kl_divergence(student_logits, teacher_logits, temperature=1.0, epsilon=1e-9, pad_mask=None):
    """
    Compute reverse KL divergence (D_KL(q || p)) for causal LLMs with sequence-level logits.

    Args:
        student_logits (torch.Tensor): Logits from the model's distribution (shape: [batch_size, block_size, vocab_size]).
        teacher_logits (torch.Tensor): Logits from the reference distribution (shape: [batch_size, block_size, vocab_size]).
        temperature (float): Temperature scaling factor. Default is 1.0 (no scaling).
        epsilon (float): Small constant for numerical stability.
        pad_mask (torch.Tensor or None): Boolean mask to ignore padded tokens (shape: [batch_size, block_size]).

    Returns:
        torch.Tensor: Mean reverse KL divergence over the batch.
    """
    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Stabilize logits by subtracting max value along the vocab dimension
    student_logits = student_logits - student_logits.max(dim=-1, keepdim=True).values
    teacher_logits = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values

    # Compute log probabilities directly
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Compute probabilities and add epsilon
    student_probs = F.softmax(student_logits, dim=-1) + epsilon

    # Compute reverse KL: sum(q(x) * (log q(x) - log p(x))) along vocab dimension
    kl_div = torch.sum(student_probs * (student_log_probs - teacher_log_probs), dim=-1)  # Shape: [batch_size, block_size]

    # Apply padding mask, if provided
    if pad_mask is not None:
        kl_div = kl_div * pad_mask  # Mask out padded tokens (pad_mask = 1 for real tokens, 0 for padding)
        valid_tokens = pad_mask.sum(dim=-1)  # Count non-padding tokens for each sequence
    else:
        valid_tokens = student_logits.size(1)  # Block size

    # Compute the mean loss over valid tokens for each sequence, then average across batch
    sequence_loss = kl_div.sum(dim=-1) / valid_tokens
    mean_loss = sequence_loss.mean()

    # Scale the final loss by T^2
    scaled_loss = mean_loss * (temperature ** 2)
    return scaled_loss

def sequence_kl_divergence(student_logits, teacher_logits, temperature=1.0, epsilon=1e-9, pad_mask=None):
    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Stabilize logits by subtracting max value along the vocab dimension
    student_logits = student_logits - student_logits.max(dim=-1, keepdim=True).values
    teacher_logits = teacher_logits - teacher_logits.max(dim=-1, keepdim=True).values

    # Compute log probabilities directly
    student_log_probs = F.log_softmax(student_logits, dim=-1)

    # Compute probabilities and add epsilon
    teacher_probs = F.softmax(teacher_logits, dim=-1) + epsilon

    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none').sum(dim=-1)  # Sum over vocab_size

    # Apply padding mask, if provided
    if pad_mask is not None:
        kl_loss = kl_loss * pad_mask  # Mask out padded tokens (pad_mask = 1 for real tokens, 0 for padding)
        valid_tokens = pad_mask.sum(dim=-1)  # Count non-padding tokens for each sequence
    else:
        valid_tokens = student_logits.size(1)  # Block size

    # Compute the mean loss over valid tokens for each sequence, then average across batch
    sequence_loss = kl_loss.sum(dim=-1) / valid_tokens
    mean_loss = sequence_loss.mean()

    #loss_2 = kl_loss.sum() / pad_mask.sum()
    #loss_3 = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")

    # Scale the final loss by T^2
    scaled_loss = mean_loss * (temperature ** 2)
    return scaled_loss

def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
        ):

    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Compute log probabilities for student and probabilities for teacher
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Compute the log of the mixture distribution
    # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
    beta = torch.tensor(beta, dtype=student_log_probs.dtype)
    mixture_log_probs = torch.logsumexp(
            torch.stack([student_log_probs + torch.log(beta), teacher_log_probs + torch.log(1 - beta)]),
            dim=0,
            )

    # Compute KL divergences using F.kl_div
    # PyTorch differs from the standard mathematical definition, so the order of the probability distributi    ons is swapped compared to that defined in the paper.
    kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

    # Compute the Generalized Jensen-Shannon Divergence
    jsd = beta * kl_teacher + (1 - beta) * kl_student

    # Masking
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    # Apply reduction
    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd

def save_logit_stats(logits, filename):
    """Compute min, max, mean, and std of logits and save to a file."""
    stats = {
        "min": logits.min().item(),
        "max": logits.max().item(),
        "mean": logits.mean().item(),
        "std": logits.std().item(),
    }

    with open(filename, "a") as f:  # Append mode to log multiple batches
        f.write(f"{stats['min']}, {stats['max']}, {stats['mean']}, {stats['std']}\n")

class Student_Trainer(transformers.Trainer):
    def __init__(self, teacher_model, kd_loss, kd_loss_weight, \
            inject_random_noise, random_noise_scale, kd_temperature, max_length=256, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.kd_loss = kd_loss
        self.kd_loss_weight = kd_loss_weight
        self.inject_random_noise = inject_random_noise
        self.random_noise_scale = random_noise_scale
        self.temperature = float(kd_temperature)
        self.max_length = max_length
        self.processing_class = tokenizer
        self.lmbda: float = 0.5
        self.beta: float = 0.5

        if self.kd_loss == 'kld' or self.kd_loss == 'reversekld':
            self.loss_fn = nn.KLDivLoss(reduction="mean")
        elif self.kd_loss == 'mse':
            self.loss_fn = nn.MSELoss(reduction="mean")

        if self.processing_class:
            self.generation_config = GenerationConfig(
                    #max_new_tokens=args.max_new_tokens,
                    temperature=1.0,
                    do_sample=True,
                    top_k=0,
                    use_cache=True,
                    pad_token_id=self.processing_class.pad_token_id,
                )

    def generate_new_inputs(self, model, inputs, generation_config):
        model.eval()

        #extract the prompt from inputs
        prompt_mask = inputs["prompt_attention_mask"].bool()
        prompt_input_ids_split = [
                torch.masked_select(input_id, mask).tolist()
                for input_id, mask in zip(inputs["input_ids"], prompt_mask)
            ]

        # Find max length of extracted sequences
        max_prompt_len = max(len(seq) for seq in prompt_input_ids_split)

        # Pad each sequence on the left & create mask
        padded_prompt_input_ids = []
        padded_prompt_attention_mask = []
        pad_token_id = self.processing_class.pad_token_id

        for seq in prompt_input_ids_split:
            pad_len = max_prompt_len - len(seq)
            padded_seq = [pad_token_id] * pad_len + seq  # Left padding
            mask = [0] * pad_len + [1] * len(seq)  # Mask with 0 for padding, 1 for tokens
            
            padded_prompt_input_ids.append(padded_seq)
            padded_prompt_attention_mask.append(mask)

        # Convert to tensor
        padded_prompt_input_ids = torch.tensor(padded_prompt_input_ids)
        padded_prompt_attention_mask = torch.tensor(padded_prompt_attention_mask)
        padded_prompt_input_ids = padded_prompt_input_ids.to(model.device)
        padded_prompt_attention_mask = padded_prompt_attention_mask.to(model.device)


        max_new_tokens = self.max_length - max_prompt_len
        generated_ids = model.generate(input_ids=padded_prompt_input_ids, 
                                        attention_mask=padded_prompt_attention_mask,
                                        generation_config=generation_config,
                                        max_new_tokens=max_new_tokens)

        seqs = self.processing_class.batch_decode(generated_ids, skip_special_tokens=True)
        new_inputs = self.processing_class(seqs, return_tensors="pt", padding=True)
        if pad_token_id is not None:
            new_labels = new_inputs['input_ids'].clone()
            new_labels[new_labels == pad_token_id] = -100
            new_inputs['labels'] = new_labels

        new_inputs.to(model.device)
        model.train()
        return new_inputs

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        if self.teacher_model == None:
            out_student = model(**inputs)
            return (out_student.loss, out_student) if return_outputs else out_student.loss


        if self.kd_loss == "gkd":
            if random.random() <= self.lmbda:
                new_inputs = self.generate_new_inputs(
                        model, inputs, self.generation_config
                        )
                inputs = new_inputs

        # implement custom logic here
        out_student = model(**inputs)
        with torch.no_grad():
            self.teacher_model.eval()
            out_teacher = self.teacher_model(**inputs)
        
        if self.inject_random_noise:
            out_teacher.logits = out_teacher.logits + \
                torch.normal(torch.zeros_like(out_teacher.logits), 1)*self.random_noise_scale
        
        teacher_logits = out_teacher.logits
        student_logits = out_student.logits

        # fix for qwen models where the tokenizer vocab size and causal model logit layer do not match
        if teacher_logits.shape[-1] != student_logits.shape[-1]:
            common_shape = min(teacher_logits.shape[-1], student_logits.shape[-1])
            teacher_logits = teacher_logits[:,:,:common_shape]
            student_logits = student_logits[:,:,:common_shape]


        if return_outputs == False:
            if self.kd_loss == 'kld':
                kd_loss = sequence_kl_divergence(student_logits, teacher_logits, self.temperature, 1e-9, inputs["attention_mask"])
                out_student.loss = (1-self.kd_loss_weight) * out_student.loss + \
                                    self.kd_loss_weight * kd_loss 
            elif self.kd_loss == 'reversekld':
                rkd_loss = sequence_reverse_kl_divergence(student_logits, teacher_logits, self.temperature, 1e-9, inputs["attention_mask"])
                out_student.loss = (1-self.kd_loss_weight) * out_student.loss + \
                                    self.kd_loss_weight * rkd_loss 
            elif self.kd_loss == 'gkd':
                gkd_loss = generalized_jsd_loss(student_logits, teacher_logits, inputs["labels"], self.beta)
                out_student.loss = (1-self.kd_loss_weight) * out_student.loss + \
                                    self.kd_loss_weight * gkd_loss

        return (out_student.loss, out_student) if return_outputs else out_student.loss

