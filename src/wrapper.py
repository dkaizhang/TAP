import torch
import torch.nn.functional as F
import warnings

from .explainer import get_input_gradients, get_influence_function
from .loss import get_mask, get_l2_loss, get_reasons_loss, get_kd_loss, get_at_loss, get_et_loss, get_rbr_loss, suppress_at
from .model import load_model
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
from torchmetrics import AUROC, MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import BinaryF1Score, MulticlassF1Score

class ModelWrapper():
    def __init__(self, 
                    model_name, 
                    load_from=None, 
                    pretrained=True, 
                    device='cpu',
                    task_loss='cross_entropy',
                    optimizer='SGD',
                    lr=1e-4,
                    l2=0,
                    weights=None,
                    labels=[i for i in range(10)]):
        self.model = None
        self.device = torch.device(device)
        print(f"Initialised on {self.device}")
        self.lr = lr
        self.l2 = l2
        self.weights = weights
        if self.weights is not None:
            self.weights = self.weights.to(self.device)
        self.labels = labels

        self.model = load_model(model_name, pretrained, len(labels), task_loss)

        if load_from is not None:
            print('loading from state dict...')
            self.model.load_state_dict(torch.load(load_from, map_location='cpu'))

        self.task_loss = task_loss
        print('Loss function:', task_loss)

        if optimizer == 'Adam':
            print('Adam')
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == 'SGD':
            print('SGD')
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            print("Invalid optimizer name, exiting...")
            exit(1)

        self.scheduler = StepLR(self.optimizer, step_size=75, gamma=1)

    def scheduler_step(self):
        self.scheduler.step()
        print(self.scheduler.get_last_lr())

    def success_step(self, data):
        self.model.to(self.device)
        self.model.eval()
        input, label = data
        input = input.to(self.device)
        label = label.to(self.device)
        with torch.no_grad():
            output = self.model(input)
            if isinstance(output, tuple):
                output = output[0]
            top_p, top_class = output.softmax(dim=1).topk(1, dim=1)
        return torch.eq(top_class.squeeze(), label)

    def training_step(self, data):
        self.model.to(self.device)
        self.model.train()

        x, y = data
        x = x.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()
        y_hat = self.model(x)
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]

        if self.task_loss == 'cross_entropy':
            pred = y_hat.argmax(dim=1, keepdim=True)
            task_loss = F.cross_entropy(y_hat, y, weight=self.weights)
        elif self.task_loss == 'mse':
            pred = torch.clamp(y_hat.round(), min=0, max=max(self.labels))
            task_loss = F.mse_loss(y_hat, y)
        c_m = confusion_matrix(y.cpu(), pred.cpu(), labels=self.labels)

        l2_loss = get_l2_loss(self.model)
        loss = task_loss + self.l2 * l2_loss 
        loss.backward()
        self.optimizer.step()

        loss_dict = defaultdict(lambda: 0)
        loss_dict["loss"] = loss
        loss_dict["ce"] = task_loss
        loss_dict["l2"] = self.l2 * l2_loss

        return loss_dict, c_m

    def validation_step(self, data):
        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            x, y = data
            x = x.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(x)
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]

            if len(self.labels) == 2:
                auroc = AUROC('binary').to(self.device)
                f1_scorer = BinaryF1Score().to(self.device)
                with warnings.catch_warnings():
                    # warnings.filterwarnings("ignore","No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score")
                    # warnings.filterwarnings("ignore","No negative samples in targets, false positive value should be meaningless. Returning zero tensor in false positive score")    
                    auc = auroc(y_hat[:,1], y).detach().item()
                    f1 = f1_scorer(y_hat[:,1], y).detach().item()
            else:
                auroc = AUROC('multiclass', num_classes=len(self.labels), average='weighted').to(self.device)
                f1_scorer = MulticlassF1Score(num_classes=len(self.labels)).to(self.device)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore","No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score")
                    warnings.filterwarnings("ignore","No negative samples in targets, false positive value should be meaningless. Returning zero tensor in false positive score")
                    auc = auroc(y_hat, y).detach().item()
                    f1 = f1_scorer(y_hat, y).detach().item()
            mse_scorer = MeanAbsoluteError().to(self.device)
            # mse_scorer = MeanSquaredError().to(self.device)
            mse = mse_scorer(y_hat.argmax(dim=1), y).detach().item()

            if self.task_loss == 'cross_entropy':
                pred = y_hat.argmax(dim=1, keepdim=True)
                loss = F.cross_entropy(y_hat, y, weight=self.weights)
            elif self.task_loss == 'mse':
                pred = torch.clamp(y_hat.round(), min=0, max=max(self.labels))
                loss = F.mse_loss(y_hat, y)
            c_m = confusion_matrix(y.cpu(), pred.cpu(), labels=self.labels)

            loss_dict = defaultdict(lambda: 0)
            loss_dict["loss"] = loss

            perf_dict = {}
            perf_dict["auc"] = auc
            perf_dict["f1"] = f1
            perf_dict["mse"] = mse

        return loss_dict, c_m, perf_dict

    def get_labels(self):
        return self.labels

class TAPModelWrapper(ModelWrapper):
    def __init__(self,
                    model_name, 
                    load_from=None, 
                    pretrained=True, 
                    teacher_model_name=None,
                    teacher_load_from=None,
                    teacher_pretrained=False,
                    kd_lambda=1e-4,
                    temperature=1.0, 
                    device='cpu', 
                    task_loss='cross_entropy',
                    optimizer='SGD', 
                    lr=1e-4,
                    penal_loss='rrr',
                    penal_lambda=0,
                    tap_lambda=0,
                    l2=0,
                    weights=None,
                    mask_threshold_wr=0.1, 
                    manual_mask_region=None, 
                    labels=[i for i in range(10)]):
        super().__init__(model_name, load_from=load_from, pretrained=pretrained, device=device, task_loss=task_loss, optimizer=optimizer, lr=lr, l2=l2, weights=weights, labels=labels)
        self.penal_loss = penal_loss
        self.penal_lambda = penal_lambda
        self.tap_lambda = tap_lambda
        self.mask_threshold_wr = mask_threshold_wr
        self.manual_mask_region = manual_mask_region
        if teacher_model_name is not None:
            self.teacher_model = load_model(teacher_model_name, teacher_pretrained, len(labels), task_loss)
        if teacher_load_from is not None:
            self.teacher_model.load_state_dict(torch.load(teacher_load_from, map_location='cpu'))
        self.kd_lambda = kd_lambda
        self.temperature = temperature

        if self.penal_lambda or self.tap_lambda:
            print(self.penal_loss, self.penal_lambda, mask_threshold_wr, f"tap: {self.tap_lambda}")

        if self.kd_lambda:
            print("KD", self.kd_lambda, self.temperature)

    def reaction_step(self, data):
        self.model.to(self.device)
        self.model.eval()
        input, _, attributions = data
        input = input.to(self.device)
        attributions = attributions.to(self.device)
        
        # mask regions of strong activation
        masked_input = get_mask(attributions, mask_region='larger', mask_threshold=self.mask_threshold_wr, device=self.device, manual_mask_region=self.manual_mask_region) * input

        # getting the reaction strength wrt original prediction 
        with torch.no_grad():
            output = self.model(input)
            if isinstance(output, tuple):
                output = output[0]
            top_p, top_class = output.softmax(dim=1).topk(1, dim=1)
            masked_output = self.model(masked_input)
            if isinstance(masked_output, tuple):
                masked_output = masked_output[0]
            strengths = top_p - masked_output.softmax(dim=1).gather(1, top_class)
        return strengths

    def training_step(self, data):
        self.model.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        x, y, best_reason = data
        x = x.to(self.device)
        y = y.to(self.device)
        best_reason = best_reason.to(self.device)
        
        y_hat = 0
        student_acts = 0
        out = self.model(x)
        if isinstance(out, tuple):
            y_hat = out[0]
            student_acts = out[1]
        else:
            y_hat = out


        if self.task_loss == 'cross_entropy':
            pred = y_hat.argmax(dim=1, keepdim=True)
            task_loss = F.cross_entropy(y_hat, y, weight=self.weights)
        elif self.task_loss == 'mse':
            pred = torch.clamp(y_hat.round(), min=0, max=max(self.labels))
            task_loss = F.mse_loss(y_hat, y)
        c_m = confusion_matrix(y.detach().cpu(), pred.detach().cpu(), labels=self.labels)

        # first penalising IGs in irrelevant regions
        wrong_reasons = 0
        if self.penal_lambda:
            input_gradients = get_input_gradients(self.model, x, y, keep_grad=True, device=self.device)

            if self.penal_loss == 'rrr':
                wrong_reasons = get_reasons_loss('wrong', best_reason, input_gradients, self.mask_threshold_wr, self.device, self.manual_mask_region)
            elif self.penal_loss == 'rbr':
                influence_function = get_influence_function(self.model, x, task_loss, self.device)
                wrong_reasons = get_rbr_loss(best_reason, input_gradients, influence_function, self.mask_threshold_wr, self.device, self.manual_mask_region)
            else:
                print('invalid penal_loss')
                exit(1)
        # print(task_loss)
        # print(wrong_reasons)

        tap_loss = 0
        if self.tap_lambda:
            tap_loss = suppress_at(best_reason, student_acts, self.mask_threshold_wr, self.device)
        # print(tap_loss)

        # manual l2 regularisation
        l2_loss = 0
        if self.l2:
            l2_loss = get_l2_loss(self.model)

        # kd loss 
        kd_loss = 0
        if self.kd_lambda:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
            with torch.no_grad():
                y_hat_teacher = self.teacher_model(x)
            kd_loss = get_kd_loss(y_hat, y_hat_teacher, self.temperature) * self.temperature * self.temperature

        loss = task_loss \
                + self.penal_lambda * wrong_reasons \
                + self.tap_lambda * tap_loss \
                + self.l2 * l2_loss \
                + self.kd_lambda * kd_loss

        loss.backward()
        self.optimizer.step()

        loss_dict = defaultdict(lambda: 0)
        loss_dict["loss"] = loss.detach()
        loss_dict["ce"] = task_loss.detach() 
        loss_dict["wr"] = self.penal_lambda * wrong_reasons.detach() if self.penal_lambda else 0
        loss_dict["tap"] = self.tap_lambda * tap_loss.detach() if self.tap_lambda else 0
        loss_dict["l2"] = self.l2 * l2_loss.detach() if self.l2 else 0
        loss_dict["kd"] = self.kd_lambda * kd_loss.detach() if self.kd_lambda else 0

        return loss_dict, c_m
    

class TransferModelWrapper(ModelWrapper):
    def __init__(self,
                    model_name, 
                    load_from=None, 
                    pretrained=True, 
                    teacher_model_name=None,
                    teacher_load_from=None,
                    teacher_pretrained=False,
                    kd_lambda=1e-4,
                    temperature=1.0, 
                    device='cpu', 
                    task_loss='cross_entropy',
                    optimizer='SGD', 
                    lr=1e-4,
                    et_lambda=0,
                    at_lambda=0,
                    l2=0,
                    weights=None,
                    labels=[i for i in range(10)]):
        super().__init__(model_name, load_from=load_from, pretrained=pretrained, device=device, task_loss=task_loss, optimizer=optimizer, lr=lr, l2=l2, weights=weights, labels=labels)
        self.et_lambda = et_lambda
        self.at_lambda = at_lambda
        if teacher_model_name is not None:
            self.teacher_model = load_model(teacher_model_name, teacher_pretrained, len(labels), task_loss)
        if teacher_load_from is not None:
            self.teacher_model.load_state_dict(torch.load(teacher_load_from, map_location='cpu'))
        self.kd_lambda = kd_lambda
        self.temperature = temperature
        self.scheduler = StepLR(self.optimizer, step_size=50, gamma=1)

    def training_step(self, data):
        self.model.to(self.device)
        self.model.train()

        x, y, best_reason = data
        x = x.to(self.device)
        y = y.to(self.device)
        best_reason = best_reason.to(self.device)
        
        y_hat, student_acts = self.model(x)

        if self.task_loss == 'cross_entropy':
            pred = y_hat.argmax(dim=1, keepdim=True)
            task_loss = F.cross_entropy(y_hat, y, weight=self.weights)
        elif self.task_loss == 'mse':
            pred = torch.clamp(y_hat.round(), min=0, max=max(self.labels))
            task_loss = F.mse_loss(y_hat, y)
        c_m = confusion_matrix(y.detach().cpu(), pred.detach().cpu(), labels=self.labels)

        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        with torch.no_grad():
            y_hat_teacher, teacher_acts = self.teacher_model(x)

        # get AT loss
        at_loss = 0
        if self.at_lambda:
            at_loss = get_at_loss(student_acts, teacher_acts)

        # get ET loss
        et_loss = 0
        if self.et_lambda:
            ig_student = get_input_gradients(self.model, x, y, keep_grad=True, device=self.device, label_specific=True)
            et_loss = get_et_loss(ig_student, best_reason)

        # manual l2 regularisation
        l2_loss = 0
        if self.l2:
            l2_loss = get_l2_loss(self.model)

        # kd loss 
        kd_loss = 0
        if self.kd_lambda:
            kd_loss = get_kd_loss(y_hat, y_hat_teacher, self.temperature) * self.temperature * self.temperature

        loss = task_loss \
                + self.et_lambda * et_loss \
                + self.at_lambda * at_loss \
                + self.l2 * l2_loss \
                + self.kd_lambda * kd_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        etat_loss = 0
        if self.et_lambda:
            etat_loss += self.et_lambda * et_loss.detach()
        if self.at_lambda:
            etat_loss += self.at_lambda * at_loss.detach()
        loss_dict = defaultdict(lambda: 0)
        loss_dict["loss"] = loss.detach()
        loss_dict["ce"] = task_loss.detach()
        loss_dict["wr"] = etat_loss 
        loss_dict["l2"] = self.l2 * l2_loss.detach() if self.l2 else 0
        loss_dict["kd"] = self.kd_lambda * kd_loss.detach() if self.kd_lambda else 0

        return loss_dict, c_m
