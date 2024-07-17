import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import torch
from PIL import Image
import tqdm
import torchsort
import cv2

class SaliencyOptimization:
    def __init__(self, img, number_of_points=500):
        self.img = img
        self.number_of_points = number_of_points
    
        reshaped = self.img.reshape(self.img.shape[0] * self.img.shape[1], -1)

        sampled_indices = np.random.choice(np.arange(len(reshaped)), size=self.number_of_points, replace=False)
        self.indices = [i for i in sampled_indices if reshaped[i, :].max(axis=-1) > 0]
        print("self.indices", len(self.indices))
        coreset = reshaped[self.indices, :]
        cosine = pairwise_distances(reshaped, coreset, metric='cosine')
        chebyshev = pairwise_distances(reshaped, coreset, metric='chebyshev')

        cosine = cosine.argsort().argsort()
        chebyshev = chebyshev.argsort().argsort()

        self.img_mask = self.img.max(axis = -1) > 0
        self.input_max_rank = torch.from_numpy(np.maximum(chebyshev, cosine))

        if torch.cuda.is_available():
            self.input_max_rank = self.input_max_rank.cuda()
        input_cosine = torch.from_numpy(cosine)
        
        if torch.cuda.is_available():
            input_cosine = input_cosine.cuda()

        self.visualization = torch.rand(size=(reshaped.shape[0], 3)) * 10

        if torch.cuda.is_available():
            self.visualization = self.visualization.cuda()
        self.visualization.requires_grad = True
        self.optim = torch.optim.Adam([self.visualization], lr = 1.0)
        delta = 0.1 * len(self.indices)
        self.loss_saliency = torch.nn.MarginRankingLoss(margin = -delta, reduction='none')
        N = len(self.indices)
        self.rank_squares = self.input_max_rank ** 2
        self.mask_np = reshaped.max(axis = -1) > 0
        self.mask = torch.from_numpy(self.mask_np).float().cuda()

    def compute_epoch(self):
        x = self.visualization
        d = torch.cdist(x, x[self.indices])
        output = torchsort.soft_rank(d, regularization_strength=0.01)
        
        saliency = self.loss_saliency(output, self.input_max_rank, torch.ones_like(output))
        saliency = (saliency * self.mask[:, None] * self.rank_squares).sum() / (self.mask[:, None] * self.rank_squares).sum()
        self.optim.zero_grad()
        loss = saliency
        loss.backward()
        self.optim.step()
        
        x = self.visualization.detach().cpu().numpy()
        x[self.mask_np == 0] = 0
        x = x.reshape((self.img.shape[0], self.img.shape[1], 3))
            
        for i in range(3):
            x[:, :, i] = x[:, :, i] - np.percentile(x[:, :, i], 0.01)
            x[:, :, i][x[:, :, i] < 0] = 0 
            x[:, :, i] = x[:, :, i] / np.percentile(x[:, :, i], 99.99)
            x[:, :, i][x[:, :, i] > 1] = 1
        
        x[self.img_mask == 0] = 0
        x = np.uint8(255 * x)
        x = cv2.cvtColor(x, cv2.COLOR_LAB2RGB)
        return x

        