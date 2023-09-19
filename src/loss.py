import numpy as np
import torch
import torch.nn.functional as F

# manual l2 regularisation
def get_l2_loss(model):
    reg = 0
    for param in model.parameters():
        reg += 0.5 * (param ** 2).sum()
    return reg

# cover areas with activations smaller or greater than a threshold  
def get_mask(attributions, mask_region, mask_threshold, device='cpu', manual_mask_region=None):

    if manual_mask_region is not None:
        mask_region = manual_mask_region

    if mask_region == 'smaller': # covers areas with small activations 
        # take abs value and sum the colour channels but keep dim
        attributions = attributions.abs().sum(dim=-3, keepdims=True).to(device) # should be (N,1,H,W)
        mask_threshold = torch.quantile(a=attributions.flatten(start_dim=1), q=1-mask_threshold, dim=1)
        mask = torch.where(torch.abs(attributions) < mask_threshold[:,None,None,None], 0, 1)
    elif mask_region == 'larger': # covers areas with strong activations
        # take abs value and sum the colour channels but keep dim
        attributions = attributions.abs().sum(dim=-3, keepdims=True).to(device) # should be (N,1,H,W)
        mask_threshold = torch.quantile(a=attributions.flatten(start_dim=1), q=mask_threshold, dim=1) # should be (N,)
        mask = torch.where(torch.abs(attributions) > mask_threshold[:,None,None,None], 0, 1)
    elif mask_region == 'less_than': # covers areas with activations less than threshold - this respects signs
        # sum the colour channels but keep dim
        attributions = attributions.sum(dim=-3, keepdims=True).to(device) # should be (N,1,H,W)
        mask_threshold = torch.quantile(a=attributions.flatten(start_dim=1), q=1-mask_threshold, dim=1)
        mask = torch.where(torch.abs(attributions) < mask_threshold[:,None,None,None], 0, 1)
    else:
        print("invalid masking choice, exiting...")
        exit(1)

    return mask

def get_reasons_loss(type, best_reason, input_gradients, mask_threshold, device, manual_mask_region):

    # taking abs then summing colour channels if there
    input_gradients = input_gradients.sum(dim=-3, keepdim=True)

    if type == 'wrong':
        loss_region = get_mask(best_reason, mask_region='larger', mask_threshold=mask_threshold, device=device, manual_mask_region=manual_mask_region)
        loss_region = loss_region * (loss_region.flatten(start_dim=1).size(dim=1) / loss_region.flatten(start_dim=1).sum(dim=1))[:,None,None,None]
        sq_norm = torch.linalg.matrix_norm(input_gradients * loss_region, dim=(-2,-1)) ** 2
        return sq_norm.mean() # batchmean
    else:
        print("invalid type for reasons loss")
        exit(1)

def get_rbr_loss(best_reason, input_gradients, influence_function, mask_threshold, device, manual_mask_region):

    # summing colour channels if there
    input_gradients = input_gradients.sum(dim=-3, keepdim=True)
    influence_function = influence_function.sum(dim=-3, keepdim=True)

    loss_region = get_mask(best_reason, mask_region='larger', mask_threshold=mask_threshold, device=device, manual_mask_region=manual_mask_region)
    
    return torch.sum((loss_region * input_gradients * influence_function) ** 2).mean()

def get_kd_loss(y_hat, y_hat_teacher, temperature):
    # https://github.com/haitongli/knowledge-distillation-pytorch/blob/master/model/net.py
    # log_softmax and softmax since that's what kl_div expects
    return F.kl_div(F.log_softmax(y_hat / temperature, dim=1), 
                        F.softmax(y_hat_teacher / temperature, dim=1),
                        reduction='batchmean')

def get_at_loss(student_acts, teacher_acts):
    loss = 0
    for student_act, teacher_act in zip(student_acts, teacher_acts):
        student_act = F.normalize(student_act.sum(dim=-3), p=2)
        teacher_act = F.normalize(teacher_act.sum(dim=-3), p=2)

        assert student_act.shape[0] == teacher_act.shape[0]
        if student_act.shape[1] != teacher_act.shape[1]:
            teacher_act = F.interpolate(teacher_act[:,None,:,:], size=student_act.shape[1])
            teacher_act = teacher_act.sum(dim=-3) # need to sum out the added channel dim

        diff = torch.linalg.matrix_norm(student_act - teacher_act, dim=(-2,-1)) # Frobenius Norm
        loss += diff.mean()
    return loss

def get_et_loss(student_explanation, teacher_explanation):
    e1 = F.normalize(student_explanation.sum(dim=-3), p=2)
    e2 = F.normalize(teacher_explanation.sum(dim=-3), p=2)
    diff = torch.linalg.matrix_norm(e1 - e2, dim=(-2,-1)) ** 2 
    return diff.mean()

def suppress_at(best_reason, student_acts, mask_threshold, device):
    assert isinstance(student_acts, tuple)
    
    loss_region = get_mask(best_reason, mask_region='larger', mask_threshold=mask_threshold, device=device, manual_mask_region=None)
    loss_region = loss_region.float()

    loss = 0
    for act in student_acts:
        # assert loss_region.shape[-1] % act.shape[-1] == 0
        kernel_size = loss_region.shape[-1] // act.shape[-1] # downscale by this factor
        interpolation_ratio = act.shape[-1] / (loss_region.shape[-1] / kernel_size)
        assert interpolation_ratio <= 1
        if kernel_size > 1:
            loss_region = F.max_pool2d(loss_region, kernel_size=3, stride=1, padding=1)
            loss_region = F.avg_pool2d(loss_region, kernel_size=kernel_size)
        if interpolation_ratio < 1:
            # print('interpolating')
            loss_region = F.interpolate(loss_region, scale_factor=interpolation_ratio)
        assert loss_region.shape[-1] == act.shape[-1]
        act_loss = loss_region * act.sum(dim=-3, keepdim=True)
        loss += torch.linalg.matrix_norm(act_loss, dim=(-2,-1)) ** 2
    loss = loss / len(student_acts)
    return loss.mean()