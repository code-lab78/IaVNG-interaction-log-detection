import json
import numpy as np
import os

import torch
import cv2

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import check_img_size, non_max_suppression, \
    scale_coords, xywh2xyxy
from yolov5.utils.torch_utils import select_device
from yolov5.utils.metrics import ap_per_class, ConfusionMatrix
from yolov5.val import process_batch
from yolov5.utils.plots import colors, plot_one_box


class ModelManager():
    def __init__(self, setupfile='model_args.json', **kwargs):
        self.load_model(setupfile, kwargs)
        self.classlist = None
        self.classlists = None
        self.n = None


    def load_model(self, filename='model_args.json', kwargs_dict=None):
        with open(filename) as f:
            self.model_args = json.load(f)

        for key, value in kwargs_dict.items():
            if key in self.model_args.keys():
                self.model_args[key] = value

        weights = self.model_args['weights']
        imgsz = self.model_args['imgsz']
        self.conf_thres = self.model_args['conf_thres']
        self.iou_thres = self.model_args['iou_thres']
        device = self.model_args['device']
        half = bool(self.model_args['half'])
        self.agnostic_nms = bool(self.model_args['agnostic_nms'])
        self.augment = bool(self.model_args['augment'])
        self.visualize = bool(self.model_args['visualize'])

        

        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        pt = weights.endswith('.pt')
        assert pt, 'weight file must be end with .pt'

        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        self.imgsz = check_img_size(imgsz, s=stride)

        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        model.eval()
        self.model = model
        self.device = device
        self.half = half


    def _preprocessimg(self, img):
        im = cv2.resize(img, (self.imgsz, self.imgsz),
                        interpolation=cv2.INTER_LINEAR)
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        return im


    def set_labels(self, classlist):
        self.classlist = classlist
        self.classlists = [[c] for c in classlist]
        self.n = len(classlist)


    def get_labelnames(self):
        names = [self.names[i] for i in self.classlist]
        return names


    def detect_all(self, img0):
        img = self._preprocessimg(img0)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred = self.model(img, augment=self.augment, visualize=self.visualize)[0]
        classes = None
        pred_t = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, self.agnostic_nms, max_det=10)

        preds = []
        unique_class = []
        for p in pred_t[0]:
            if p[5] not in unique_class:
                unique_class.append(p[5])
                preds.append(p)
        if len(preds) != 0:
            det = torch.stack(preds)
        else:
            det = pred_t

        detn = det.clone()      # detn : scaled for img0 (288x288)
        if len(det):
            detn[:, :4] = scale_coords(img.shape[2:], detn[:, :4],
                                      img0.shape).round()

        return detn, det


    def detect(self, img0, state):
        img = self._preprocessimg(img0)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        pred = self.model(img, augment=self.augment, visualize=self.visualize)[0]

        # NMS
        preds = []
        for i in range(self.n):
            if not state[i]:
               continue
            classes = self.classlists[i]
            pred_t = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                         classes, self.agnostic_nms,
                                         max_det=1)[0]
            if pred_t.shape[0] != 0:
                preds.append(pred_t)
        if len(preds) != 0:
            det = torch.cat(preds, 0)
        elif np.sum(state) == 0:
            det = torch.tensor([])
        else:
            det = pred_t

        detn = det.clone()      # detn : scaled for img0 (288x288)
        if len(det):
            detn[:, :4] = scale_coords(img.shape[2:], detn[:, :4],
                                      img0.shape).round()

        return detn, det


    def start_validation(self):
        # iou vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)
        self.niou = self.iouv.numel()
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.nc = len(self.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc)
        self.stats = []
        self.seen = 0


    def run_validation_batch(self, dets, targets):
        for pred, labels in zip(dets, targets):
            if 157 in pred[:, 5] and 129 in np.array(labels)[:, 0] :
                if isinstance(pred, torch.Tensor) :
                    # synd_pred = np.where(pred[:, 5].cpu() == 157)[0].item()
                    synd_pred = [i for i, val in enumerate(pred[:, 5].cpu()) if val not in np.array(labels)
                                 [:, [0]]][0]
                else:
                    synd_pred = [i for i, val in enumerate(pred[:, 5]) if val not in np.array(labels)
                                 [:, [0]]][0]
                    pred = torch.tensor(pred)
                synd_lab = np.where(np.array(labels)[:, 0] == 129)[0].item()
                
                # Remove the corresponding prediction
                pred = torch.cat((pred[:synd_pred], pred[synd_pred + 1:]), dim=0)
                # Remove the corresponding label
                labels = np.delete(labels, synd_lab, axis=0)
            elif 129 in np.array(labels)[:, 0] :
                synd_lab = np.where(np.array(labels)[:, 0] == 129)[0].item()
                # Remove the corresponding label
                labels = np.delete(labels, synd_lab, axis=0)
            nl = len(labels)
            labels = torch.Tensor(labels).to(self.device)
            tcls = labels[:, 0].tolist() if nl else []
            self.seen += 1
            # print("self.seen", self.seen)
            if 129 in np.array(labels.cpu())[:, 0] :
                print("Syndra is in")
            if 134 in np.array(labels.cpu())[:, 0] :
                print("Teemo is in")
            if len(pred) == 0:
                if nl:
                    self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords((1,1), tbox, (self.imgsz, self.imgsz))
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(pred, labelsn, self.iouv)
                self.confusion_matrix.process_batch(pred, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
        # print(self.seen, len(self.stats))



    def end_validation(self, save_dir):
        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(self.stats) and self.stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*self.stats, plot=True, save_dir=save_dir, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(self.stats[3].astype(np.int64), minlength=self.nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        self.confusion_matrix.plot(save_dir=save_dir, names=self.names)
        self.p, self.r, self.ap50, self.ap, self.ap_class = p, r, ap50, ap, ap_class
        self.nt, self.mp, self.mr, self.map50, self.map = nt, mp, mr, map50, map

        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        pf = '%20s' + '%11i' * 2 + '%11.2g' * 4  # print format
        # s = ('%s\t' + '%s\t' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        # pf = '%s\t' + '%i\t' * 2 + '%.3g\t' * 4  # print format
        
        with open(save_dir + '/mAPs.txt', 'w') as f:
            f.write(s + '\n')
            f.write(pf % ('all', self.seen, nt.sum(), mp, mr, map50, map) + '\n')
            if self.nc > 1 and len(self.stats):
                for i, c in enumerate(ap_class):
                    f.write(pf % (self.names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]) + '\n')
        
        self.plot_map_horizontal(save_dir, self.ap50, self.ap_class, self.names)

    def plot_map_horizontal(self, save_dir, ap50, ap_class, names):
        import matplotlib.pyplot as plt
        """
        Plot horizontal bar chart of mAP@.5 for each class and save the plot.
        """
        # if not hasattr( 'ap50') or not hasattr( 'ap_class'):
        #     raise ValueError("Validation must be completed before plotting mAP.")
        
        # Get mAP@.5 values and corresponding class names
        ap50 = ap50
        ap_class = ap_class
        class_names = [names[c] for c in ap_class]
        
        # Create the plot
        plt.figure(figsize=(10, len(class_names) * 0.15))  # Adjust height based on number of classes
        bars = plt.barh(class_names, ap50, color='skyblue')
        plt.xlabel('mAP@.5', fontsize=12)
        plt.ylabel('Champions (Classes)', fontsize=12)
        plt.title('mAP@.5 by Champion', fontsize=14)
        plt.tight_layout()

        # Add values on the bars
        for bar, value in zip(bars, ap50):
            plt.text(value, bar.get_y() + bar.get_height() / 2,
                    f'{value:.2f}', va='center', ha='left', fontsize=10, color='black')

        # Save the plot
        plot_path = os.path.join(save_dir, 'map50_plot_horizontal.png')
        plt.savefig(plot_path)
        plt.show()

        print(f"Horizontal mAP@.5 plot saved to {plot_path}")

    def plot_map(self, save_dir, ap50, ap_class, names):
        import matplotlib.pyplot as plt

        """
        Plot mAP@.5 for each class and save the plot.
        """
        # if not hasattr( 'ap50') or not hasattr( 'ap_class'):
        #     raise ValueError("Validation must be completed before plotting mAP.")
        
        # Get mAP@.5 values and corresponding class names
        ap50 = ap50
        ap_class = ap_class
        class_names = [names[c] for c in ap_class]
        
        # Create the plot
        plt.figure(figsize=(20, 12))
        bars = plt.bar(class_names, ap50, color='skyblue')
        plt.xlabel('Champions (Classes)', fontsize=12)
        plt.ylabel('mAP@.5', fontsize=12)
        plt.title('mAP@.5 by Champion', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        
        for bar, value in zip(bars, ap50):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{value:.2f}', ha='center', va='bottom', fontsize=10, color='black')

        # Save the plot
        plot_path = os.path.join(save_dir, 'map50_plot.png')
        plt.savefig(plot_path)
        plt.show()

        print(f"mAP@.5 plot saved to {plot_path}")
        
    def save_validation_img(self, detns, targets, imgs, img_prefix, save_dir,
                            line_thickness=1):
        save_img_dir = os.path.join(save_dir, 'imgs')
        os.makedirs(save_img_dir, exist_ok=True)
        for i, (predn, labels, img) in enumerate(zip(detns, targets, imgs)):
            if 157 in predn[:, 5] and 129 in np.array(labels)[:, 0] :
                if isinstance(predn, torch.Tensor) :
                    synd_pred = [i for i, val in enumerate(predn[:, 5].cpu()) if val not in np.array(labels)
                                 [:, [0]]][0] # finding syndra predicted
                else:
                    synd_pred = [i for i, val in enumerate(predn[:, 5]) if val not in np.array(labels)
                                 [:, [0]]][0] # finding syndra predicted
                    predn = torch.tensor(predn)
                synd_lab = np.where(np.array(labels)[:, 0] == 129)[0].item()
                
                # Remove the corresponding prediction
                
                predn = torch.cat((predn[:synd_pred], predn[synd_pred + 1:]), dim=0)
                # Remove the corresponding label
                labels = np.delete(labels, synd_lab, axis=0)
            elif 129 in np.array(labels)[:, 0] :
                synd_lab = np.where(np.array(labels)[:, 0] == 129)[0].item()
                # Remove the corresponding label
                labels = np.delete(labels, synd_lab, axis=0)
            
            # Original
            save_path = save_img_dir + '/{}_{}.png'.format(img_prefix, i)
            cv2.imwrite(save_path, img)

            # Predicted
            im0 = img.copy()
            save_path = save_img_dir + '/{}_{}_pred.png'.format(img_prefix, i)
            for pred in predn:
                xyxy, conf, c = pred[:4], pred[4], pred[5]
                label = f'{self.names[int(c)]} {conf:.2f}' 
                # self.names will be different champ list if the version is different
                # label = f'{api_champs[int(c)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
            cv2.imwrite(save_path, im0)

            # Target
            im0 = img.copy()
            save_path = save_img_dir + '/{}_{}_label.png'.format(img_prefix, i)
            labels = torch.Tensor(labels)
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords((1,1), tbox, (im0.shape[0], im0.shape[1]))
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            for pred in labelsn:
                c, xyxy = pred[0], pred[1:5]
                plot_one_box(xyxy, im0, label=self.names[int(c)], color=colors(c, True), line_thickness=line_thickness)
            cv2.imwrite(save_path, im0)

    def presave_val_img(self, targets, imgs, img_prefix, save_dir,
                            line_thickness=1):
        save_img_dir = os.path.join(save_dir, 'imgs')
        os.makedirs(save_img_dir, exist_ok=True)
        for i, (labels, img) in enumerate(zip(targets, imgs)):
            # Original
            save_path = save_img_dir + '/{}_{}.png'.format(img_prefix, i)
            cv2.imwrite(save_path, img)
            
            # Target
            im0 = img.copy()
            save_path = save_img_dir + '/{}_{}_label.png'.format(img_prefix, i)
            labels = torch.Tensor(labels)
            tbox = xywh2xyxy(labels[:, 1:5])
            scale_coords((1,1), tbox, (im0.shape[0], im0.shape[1]))
            labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            for pred in labelsn:
                c, xyxy = pred[0], pred[1:5]
                plot_one_box(xyxy, im0, label=self.names[int(c)], color=colors(c, True), line_thickness=line_thickness)
            cv2.imwrite(save_path, im0)


if __name__ == '__main__':
    mm = ModelManager(r'yolov5\resources\model_args.json')
    