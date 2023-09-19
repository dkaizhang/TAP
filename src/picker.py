import numpy as np
import os

class Picker():
    def __init__(self, config):
        
        self.config = config
    
        explanations = []
        for id, _ in enumerate(self.config['teacher_load_froms']):
            file = os.path.join('experiments', self.config['experiment'], f"teacher_{id}", f"explain_{self.config['split']}.npy")
            explanations.append(np.load(file))
        self.explanations = np.stack(explanations)
        print('Explanations loaded by picker (peers, N, C, H, W)', self.explanations.shape)     

        self.picks = np.zeros(len(explanations[0]), dtype=int)

    def get_winners(self):
        if len(self.explanations[0]) != len(self.picks):
            print(f"Mismatch error: {len(self.explanations[0])} explanations with {len(self.picks)} picks")
            exit(1)
        winning_exp = self.explanations[self.picks, range(len(self.picks)), :, :]
        file = os.path.join('experiments', self.config['experiment'], "winning_exp.npy")
        np.save(file, winning_exp)

        return winning_exp



