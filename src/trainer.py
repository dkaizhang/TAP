import copy
import numpy as np
import os
import torch

from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer():
    def __init__(self, batch_size=32, epochs=0, num_workers=0, writer=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.writer = writer

    def fit(self, model, train_data, val_data, silent=False, save_every=0):
        train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
        best_val_loss = 100000
        step = 0
        for i in range(self.epochs):
            train_loss = 0
            wr_loss = 0
            tap_loss = 0
            kd_loss = 0
            label_len = len(model.get_labels())
            train_c_m = np.zeros((label_len, label_len), dtype=int)
            val_loss = 0
            val_c_m = np.zeros((label_len, label_len), dtype=int)
            val_f1 = 0
            val_auc = 0
            val_mse = 0

            for batch_idx, data in enumerate(tqdm(train_loader)):
                batch_loss_dict, c_m = model.training_step(data)
                train_loss += batch_loss_dict["loss"]
                wr_loss += batch_loss_dict["wr"]
                tap_loss += batch_loss_dict["tap"]
                kd_loss += batch_loss_dict["kd"]
                train_c_m += c_m

                self.writer.add_scalar("Loss - train", batch_loss_dict["loss"], step)
                self.writer.add_scalar("Loss - wrong_reasons", batch_loss_dict["wr"], step)
                self.writer.add_scalar("Loss - TAP", batch_loss_dict["tap"], step)
                self.writer.add_scalar("Loss - KD", batch_loss_dict["kd"], step)
                step += 1
            model.scheduler_step()

            for batch_idx, data in enumerate(tqdm(val_loader)):    
                batch_loss_dict, c_m, batch_perf_dict = model.validation_step(data)
                val_loss += batch_loss_dict["loss"]
                val_c_m += c_m
                val_f1 += batch_perf_dict["f1"]
                val_auc += batch_perf_dict["auc"]
                val_mse += batch_perf_dict["mse"]

                self.writer.add_scalar("Loss - val", batch_loss_dict["loss"], step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_at = os.path.join(self.writer.log_dir, 'best-model.pt')
                # torch.save(model.model.state_dict(), save_at)

                # self.writer.add_scalar("Saving best model", 1, step)

            if save_every > 0 and i > 0 and i % save_every == 0:
                save_at = os.path.join(self.writer.log_dir, f'model-{i}.pt')
                torch.save(model.model.state_dict(), save_at)

            self.writer.add_scalar("Acc - train", train_c_m.diagonal().sum() / train_c_m.sum(), step)
            self.writer.add_scalar("Acc - val", val_c_m.diagonal().sum() / val_c_m.sum(), step)
            self.writer.add_scalar("AUC - val", val_auc / len(val_loader), step)
            self.writer.add_scalar("F1 - val", val_f1 / len(val_loader), step)
            self.writer.add_scalar("MSE - val", val_mse / len(val_loader), step)
            if not silent:
                print(f"Epoch: {i}")
                print(f"Train loss: {train_loss / len(train_loader)} \t \t WR loss: {wr_loss / len(train_loader)} \t \t TAP loss: {tap_loss / len(train_loader)} \t \t KD loss: {kd_loss / len(train_loader)} \t \t Val loss: {val_loss / len(val_loader)}")
                print(f"Train acc: {train_c_m.diagonal().sum() / train_c_m.sum()} \t \t Val acc: {val_c_m.diagonal().sum() / val_c_m.sum()}")
                print(f"Val AUC: {val_auc / len(val_loader)} \t \t F1: {val_f1 / len(val_loader)} \t \t MSE: {val_mse / len(val_loader)}")
            
            if torch.isnan(train_loss):
                break

        save_at = os.path.join(self.writer.log_dir, 'last-model.pt')
        torch.save(model.model.state_dict(), save_at)

        loss_dict = defaultdict(lambda: 0)
        loss_dict["train_loss"] = train_loss.item() / len(train_loader)
        loss_dict["val_loss"] = val_loss.item() / len(val_loader)
        if wr_loss: loss_dict["wr"] = wr_loss.item() / len(train_loader)
        if tap_loss: loss_dict["tap"] = tap_loss.item() / len(train_loader)
        if kd_loss: loss_dict["kd"] = kd_loss.item() / len(train_loader)

        return loss_dict

    def test(self, model, data, silent=False):

        dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        
        test_loss = 0
        label_len = len(model.get_labels())
        
        confusion_matrix = np.zeros((label_len, label_len), dtype=int)
        test_f1 = 0
        test_auc = 0
        test_mse = 0

        for batch_idx, data in enumerate(tqdm(dataloader)):
            batch_loss_dict, c_m, batch_perf_dict = model.validation_step(data) # add test loss log
            test_loss += batch_loss_dict["loss"]
            confusion_matrix += c_m
            test_f1 += batch_perf_dict["f1"]
            test_auc += batch_perf_dict["auc"]
            test_mse += batch_perf_dict["mse"]

        acc = confusion_matrix.diagonal().sum()/confusion_matrix.sum()        
        test_f1 = test_f1 / len(dataloader)
        test_auc = test_auc / len(dataloader)
        test_mse = test_mse / len(dataloader)
        perf_dict = {}
        perf_dict["f1"] = test_f1
        perf_dict["auc"] = test_auc
        perf_dict["mse"] = test_mse

        if not silent:
            print(f"Test loss: {test_loss / len(dataloader)} \t \t Test acc: {acc} \t \t Test F1: {test_f1}")
            print(f"Test AUC: {test_auc}")
            print(f"Test MSE: {test_mse}")

        return test_loss / len(dataloader), confusion_matrix, perf_dict

    def react(self, model, data):
        dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        reactions = []

        for batch_idx, data in enumerate(tqdm(dataloader)):
            reaction = model.reaction_step(data)
            reaction_copy = copy.deepcopy(reaction)
            del reaction
            reactions.append(reaction_copy)
        
        reactions = torch.cat(reactions).cpu().numpy()

        return reactions

    def success(self, model, data):
        dataloader = DataLoader(data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

        successes = []

        for batch_idx, data in enumerate(tqdm(dataloader)):
            success = model.success_step(data)
            success_copy = copy.deepcopy(success)
            del success
            successes.append(success_copy)

        successes = torch.cat(successes).cpu().numpy()

        return successes
