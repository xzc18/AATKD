import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller


base_t = 2
k=10
r = 9

def calculate_attention_temperature(
    attention_weights,
    candidate_temperatures,
):
    temperatures = candidate_temperatures.to(
        device=attention_weights.device,
        dtype=attention_weights.dtype,
    )

    inverse_temperature = (
        attention_weights / temperatures.unsqueeze(0)
    ).sum(dim=1)

    tiny = torch.finfo(
        attention_weights.dtype
    ).tiny

    effective_temperature = (
        inverse_temperature.clamp_min(tiny).reciprocal()
    )

    return effective_temperature

def fuse_temperature_distributions(
    logits,
    temperatures,
    attention_net,
):
    batch_size = logits.shape[0]

    logits_fp32 = logits.float()

    temperatures = temperatures.to(
        device=logits.device,
        dtype=torch.float32,
    ).view(1, -1, 1)

    stacked_log_probs = F.log_softmax(
        logits_fp32.unsqueeze(1) / temperatures,
        dim=-1,
    )

    stacked_probs = stacked_log_probs.exp()

    attention_input = stacked_probs.detach().reshape(batch_size,-1)

    attention_weights = attention_net(
        attention_input
    ).float()

    tiny = torch.finfo(attention_weights.dtype).tiny

    log_attention_weights = (
        attention_weights.clamp_min(tiny).log()
    )

    combined_log_probs = torch.logsumexp(
        log_attention_weights.unsqueeze(-1)
        + stacked_log_probs,
        dim=1,
    )

    return (
        combined_log_probs,
        attention_weights
    )

def normalize(logit):
    stdv = base_t*logit.std(dim=-1, keepdims=True)
    return (logit) / (1e-7 + stdv) , stdv


class AKD100(Distiller):
    def __init__(self, student, teacher, cfg):
        super(AKD100, self).__init__(student, teacher)
        
        temperatures = torch.arange(2, k + 1, dtype=torch.float32)
        
        self.register_buffer("temperatures", temperatures)
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

        num_classes = 100
        self.student_net = nn.Sequential(
            nn.Linear(len(self.temperatures) * num_classes, len(self.temperatures)),
            nn.Softmax(dim=1)
        )

    def get_learnable_parameters(self):
        parameters = list(super().get_learnable_parameters())
        parameters.extend(self.student_net.parameters())
        return parameters

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        standardized_teacher,tt = normalize(logits_teacher)
        teacher_probs = F.softmax(standardized_teacher,dim=1)

        (combined_student_log_probs,attention_weights_student) = fuse_temperature_distributions(
            logits_student,
            self.temperatures,
            self.student_net,
        )
        
        ts = calculate_attention_temperature(attention_weights_student.detach(),self.temperatures)
        ts = ts.unsqueeze(1) 
        t = (tt * ts).view(-1, 1)

        loss_kd_raw = F.kl_div(
            combined_student_log_probs,
            teacher_probs.detach(),
            reduction="none"
        )  * (r*t)

        loss_kd = loss_kd_raw.sum(1).mean()
        loss_ce = F.cross_entropy(logits_student, target)
        total_loss_ce = self.ce_loss_weight * loss_ce
        total_loss_kd = self.kd_loss_weight * loss_kd
        losses_dict = {
            "loss_ce": total_loss_ce,
            "loss_kd": total_loss_kd,
        }
        return logits_student, losses_dict

    
class AKD1000(Distiller):
    def __init__(self, student, teacher, cfg):
        super(AKD1000, self).__init__(student, teacher)
        temperatures = torch.arange(2, k + 1, dtype=torch.float32)
        self.register_buffer("temperatures", temperatures)
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

        num_classes = 1000

        self.student_net = nn.Sequential(
            nn.Linear(len(self.temperatures) * num_classes, len(self.temperatures)),
            nn.Softmax(dim=1)
        )
    def get_learnable_parameters(self):
        parameters = list(super().get_learnable_parameters())
        parameters.extend(self.student_net.parameters())
        return parameters

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        standardized_teacher,tt = normalize(logits_teacher)
        teacher_probs = F.softmax(standardized_teacher,dim=1)

        (combined_student_log_probs,attention_weights_student) = fuse_temperature_distributions(
            logits_student,
            self.temperatures,
            self.student_net,
        )
        
        ts = calculate_attention_temperature(attention_weights_student.detach(),self.temperatures)
        ts = ts.unsqueeze(1) 
        t = (tt * ts).view(-1, 1)

        loss_kd_raw = F.kl_div(
            combined_student_log_probs,
            teacher_probs.detach(),
            reduction="none"
        )  * (r*t)

        loss_kd = loss_kd_raw.sum(1).mean()
        loss_ce = F.cross_entropy(logits_student, target)
        total_loss_ce = self.ce_loss_weight * loss_ce
        total_loss_kd = self.kd_loss_weight * loss_kd
        losses_dict = {
            "loss_ce": total_loss_ce,
            "loss_kd": total_loss_kd,
        }
        return logits_student, losses_dict