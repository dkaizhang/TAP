import numpy as np
import torch

from torch.autograd import grad
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_input_gradients(model, input, labels, keep_grad, device='cpu', label_specific=False):
    model.to(device)
    input = input.to(device)
    
    model.eval()

    input.requires_grad_()
    model.zero_grad()
    out = model(input)
    if isinstance(out, tuple):
        out = out[0]

    # First getting LSE'd logits 
    out = out - torch.logsumexp(out, dim=1, keepdim=True)
    if label_specific:
        out = out[torch.arange(len(labels)), labels] # indexing the entry for the true label

    # create_graph allows a higher derivative to be calc'ed
    if keep_grad:
        input_grad = grad(out, input, grad_outputs=torch.ones_like(out), create_graph=True)
    else:
        input_grad = grad(out, input, grad_outputs=torch.ones_like(out), create_graph=False)
    model.train()

    # input_grad is a tuple with the first element being gradient wrt x 
    # https://stackoverflow.com/questions/54166206/grad-outputs-in-torch-autograd-grad-crossentropyloss
    return input_grad[0]

def get_influence_function(model, input, loss, device):
    model.to(device)
    input = input.to(device)

    model.eval()

    input.requires_grad_()
    model.zero_grad()
    out = model(input)
    if isinstance(out, tuple):
        out = out[0]

    # First getting LSE'd logits 
    out = out - torch.logsumexp(out, dim=1, keepdim=True)

    loss.retain_grad()
    loss_grads_wrt_model_params_all = torch.autograd.grad(loss, model.parameters(), \
        torch.ones_like(loss), create_graph=True)
    # currently the grads are a list of every grad loss for all params wrt to the layer 
    # --> we need to get the sum of all grads
    loss_grads_wrt_model_params = torch.sum((torch.cat([t.flatten() for t in loss_grads_wrt_model_params_all])))
    loss_grads_wrt_model_params.retain_grad()

    if_grads = torch.autograd.grad(loss_grads_wrt_model_params, input, \
        torch.ones_like(loss_grads_wrt_model_params), create_graph=True)

    model.train()
    return if_grads[0]

def get_input_gradients_label_specific(model, input, labels, keep_grad, device):
    return get_input_gradients(model, input, labels, keep_grad, device, label_specific=True)

def get_random_gradients(model, input, labels, keep_grad=False, device='cpu'):
    return torch.randn_like(input)

def get_mnist_decoys(data):

    path = 'data/MNIST/decoyed_mnist.npz'
    files = np.load(path)

    # print('saving explained instances')
    # temp_file = 'explained_instances.npy'
    # np.save(temp_file, data.indices)

    # the decoy positions as explanations results in the decoys being the "most important"
    # to penalise the decoys we need to make it as if a teacher had said the decoys were the "least important"
    # 1 - decoy_position so that the positions will have 0s 

    exps = torch.Tensor(1 - files['decoy_positions_train'])[:,None,:,:]
    if hasattr(data, "indices"):
        exps = exps[data.indices,:,:,:]
    return exps

def get_pneu_text_decoys(data):

    path = 'data/pneu/chest_xray/train_text/decoy_positions.npy'
    decoys = np.load(path)
    exps = torch.Tensor(1 - decoys)[:,None,:,:]
    if hasattr(data, "indices"):
        exps = exps[data.indices,:,:,:]
    return exps

def get_pneu_text_decoys_RGB(data):

    path = 'data/pneu/chest_xray/train_text/decoy_positions.npy'
    decoys = np.load(path)
    exps = torch.Tensor(1 - decoys)[:,None,:,:].expand(-1,3,-1,-1)
    if hasattr(data, "indices"):
        exps = exps[data.indices,:,:,:]
    return exps

def get_pneu_stripe_decoys(data):

    path = 'data/pneu/chest_xray/train_stripe/decoy_positions.npy'
    decoys = np.load(path)
    exps = torch.Tensor(1 - decoys)[:,None,:,:]
    if hasattr(data, "indices"):
        exps = exps[data.indices,:,:,:]
    return exps

def get_pneu_stripe_decoys_RGB(data):

    path = 'data/pneu/chest_xray/train_stripe/decoy_positions.npy'
    decoys = np.load(path)
    exps = torch.Tensor(1 - decoys)[:,None,:,:].expand(-1,3,-1,-1)
    if hasattr(data, "indices"):
        exps = exps[data.indices,:,:,:]
    return exps

def get_knee_text_decoys_RGB(data):

    path = 'data/kneeKL224/decoy_positions.npy'
    decoys = np.load(path)
    exps = torch.Tensor(1 - decoys)[:,None,:,:].expand(-1,3,-1,-1)
    if hasattr(data, "indices"):
        exps = exps[data.indices,:,:,:]
    return exps

def get_knee_stripe_decoys_RGB(data):

    path = 'data/kneeKL224/decoy_positions_stripe.npy'
    decoys = np.load(path)
    exps = torch.Tensor(1 - decoys)[:,None,:,:].expand(-1,3,-1,-1)
    if hasattr(data, "indices"):
        exps = exps[data.indices,:,:,:]
    return exps

class Explainer():
    def __init__(self, batch_size=32, num_workers=0, method="input_gradients", device='cpu'):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        if method == 'input_gradients':
            self.method = get_input_gradients
        elif method == 'input_gradients_label_specific':
            self.method = get_input_gradients_label_specific
        elif method == 'random_gradients':
            self.method = get_random_gradients
        elif method == 'mnist_decoys':
            self.method = get_mnist_decoys
        elif method == 'pneu_text_decoys':
            self.method = get_pneu_text_decoys
        elif method == 'pneu_text_decoys_RGB':
            self.method = get_pneu_text_decoys_RGB
        elif method == 'pneu_stripe_decoys':
            self.method = get_pneu_stripe_decoys
        elif method == 'pneu_stripe_decoys_RGB':
            self.method = get_pneu_stripe_decoys_RGB
        elif method == 'knee_text_decoys_RGB':
            self.method = get_knee_text_decoys_RGB
        elif method == 'knee_stripe_decoys_RGB':
            self.method = get_knee_stripe_decoys_RGB

    def explain(self, model, data, stop_after=-1):
        if self.method == get_mnist_decoys:
            return get_mnist_decoys(data)
        if self.method == get_pneu_text_decoys:
            return get_pneu_text_decoys(data)
        if self.method == get_pneu_text_decoys_RGB:
            return get_pneu_text_decoys_RGB(data)
        if self.method == get_pneu_stripe_decoys:
            return get_pneu_stripe_decoys(data)
        if self.method == get_pneu_stripe_decoys_RGB:
            return get_pneu_stripe_decoys_RGB(data)
        if self.method == get_knee_text_decoys_RGB:
            return get_knee_text_decoys_RGB(data)
        if self.method == get_knee_stripe_decoys_RGB:
            return get_knee_stripe_decoys_RGB(data)

        loader = DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        attributions = []
        for batch_idx, data in enumerate(tqdm(loader)):
            if batch_idx == stop_after:
                break
            attribution = self.method(model, data[0], data[1], keep_grad=False, device=self.device)
            attributions.append(attribution.detach().cpu().clone())
        attributions = torch.cat(attributions, dim=0)
        return attributions
    
