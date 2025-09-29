
import os
import cv2
import torch
import numpy as np
from modules import ml
from torchvision import transforms
from PIL import Image


class Classifier:
    def __init__(self,
        model_config
        ) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = model_config
        self.model = getattr(ml, self.config['network']['model'])(**self.config['network']['arguments']).to(self.device)
        self.optimizer = getattr(torch.optim, self.config['optimizer']['model'])(self.model.parameters(), **self.config['optimizer']['arguments'])
        self.objective = getattr(torch.nn, self.config['objective']['model'])(**self.config['objective']['arguments'])
        self.input_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def train(self, x, y, model_path=None):
        # x = torch.Tensor(x).to(self.device)
        y = torch.Tensor(y).to(self.device)
        batch_size = self.config['batchsize']
        dataset_size = len(x)
        n_batch = np.ceil(dataset_size / batch_size).astype(int)
        acc = []
        n_div = self.config['epochsize'] // 10
        for epoch in range(1, self.config['epochsize']+1):
            shuffle = np.random.permutation(dataset_size)
            x_shuffle = x[shuffle]
            y_shuffle = y[shuffle]
            for idx in range(n_batch):
                s = idx * batch_size
                e = s + batch_size
                _x = x_shuffle[s:e]
                _y = y_shuffle[s:e]
                _x = torch.cat([self.input_preprocess(Image.fromarray(img.astype(np.uint8).transpose(1,2,0))).float().unsqueeze_(0) for img in _x], 0).to(self.device)
                pred = self.model(_x)

                loss = self.objective(pred, _y)
                acc += (pred.argmax(1) == _y.argmax(1)).tolist()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('Epoch {} : {:5.2f}'.format(epoch, np.mean(acc) * 100))
            if model_path and epoch % n_div == 0:
                self.save_model(model_path, 'model-{}.ckpt'.format(epoch))


    def save_model(self, save_path, file_name='model.ckpt'):
        torch.save(self.model.state_dict(), os.path.join(save_path, file_name))


    def load_model(self, model_path, cpu=False):
        # if cpu:
        #     # self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        #     self.model.load_state_dict(
        #         torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        # else:
        #     self.model.load_state_dict(
        #         torch.load(model_path, map_location=torch.device('cuda'), weights_only=True))
            # self.model.load_state_dict(torch.load(model_path))
        pass


    def predict(self, image):
        x = torch.cat([self.input_preprocess(Image.fromarray(img.astype(np.uint8).transpose(1,2,0))).float().unsqueeze_(0) for img in image], 0).to(self.device)
        # x = torch.Tensor(image).to(self.device)
        pred = self.model(x)
        return pred

