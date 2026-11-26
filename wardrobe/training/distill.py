"""Knowledge distillation (M7): a small student model learns from a big teacher model and from the labels."""
import torch
import torch.nn.functional as F

from wardrobe.training.finetune import MISSING


def distillation_loss(student_logits, teacher_logits, labels, temperature: float, alpha: float) -> torch.Tensor:
    """Loss for one head: alpha * (match the teacher's soft answers) + (1 - alpha) * (match the true labels).

    The temperature softens both distributions, so the student also sees which wrong classes the teacher
    found plausible (a polo is "almost a shirt"). alpha=0 is normal training without a teacher.
    """
    student_soft = student_logits / temperature
    teacher_soft = teacher_logits / temperature
    # kl_div expects the student as log-probabilities and the teacher as plain probabilities.
    student_log_probs = F.log_softmax(student_soft, dim=1)
    teacher_probs = F.softmax(teacher_soft, dim=1)
    # * T²: soft distributions give small gradients; this keeps the balance to hard_loss when T changes.
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * temperature**2
    # No label in this batch (e.g. formality): cross_entropy would divide 0 by 0 and return nan.
    # The teacher still has an answer for every image, so the student learns from it alone.
    if not (labels != MISSING).any():
        return alpha * soft_loss
    hard_loss = F.cross_entropy(student_logits, labels)  # skips MISSING (-100) by default
    return alpha * soft_loss + (1 - alpha) * hard_loss


def multi_head_loss(student_outputs, teacher_outputs, labels, temperature: float, alpha: float) -> torch.Tensor:
    """Sum of distillation_loss over all heads; all three arguments are dicts with one entry per attribute."""
    loss = 0
    for attribute in student_outputs:
        loss += distillation_loss(
            student_outputs[attribute], teacher_outputs[attribute], labels[attribute], temperature, alpha
        )
    return loss
