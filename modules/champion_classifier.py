import json
import numpy as np

from modules.NeuralNetwork.CNN import Classifier
from modules.image_process import crop, resize


class ChampionClassifier():
    def __init__(self):
        self._load_model()


    def _load_model(self):
        with open('resources/cnn_for_event_detect.json') as f:
            config = json.load(f)
        self.model = Classifier(config)
        self.model.load_model('models/' + 'model_cnn.ckpt', cpu=True)
        # self.model.load_model('models/' + 'model_cnn_159.ckpt', cpu=True)
        self.model.model.eval()

        self.input_w = self.model.config['input_size']
        self.input_h = self.model.config['input_size']

        self.target_map = {}
        with open('resources/boxes_for_event_detect2.json') as f:
            target_boxes = json.load(f)
        for key, value in target_boxes.items():
            self.target_map[key] = value["tlbr"]

    
    def inference_frame(self, frame, champ_only=True):
        # import matplotlib
        # matplotlib.use("TkAgg")
        # import matplotlib.pyplot as plt
        h, w = frame.shape[:2]
        pred_map = {}
        # if key:
        #     top, left, bottom, right = self.target_map[key]
        #     x1 = int(left * w)
        #     y1 = int(top * h)
        #     x2 = int(right * w)
        #     y2 = int(bottom * h)
        #     cropped = crop(frame, y1, y2, x1, x2)
        #     cropped = resize(cropped, self.input_w, self.input_h)
        #     pred = self.model.predict([cropped.transpose(2, 1, 0)]).detach().cpu().numpy().argmax(1)[0]
        #     pred_map[key] = pred
        # else:
        import cv2
        cropped_list = []
        cr_list = []
        for key in self.target_map:
            top, left, bottom, right = self.target_map[key]
            x1 = int(left * w)
            y1 = int(top * h)
            x2 = int(right * w)
            y2 = int(bottom * h)
            cropped = crop(frame, y1, y2, x1, x2)
            cr_list.append(cropped)
            temp_cr = resize(cropped, self.input_w, self.input_h)
            cropped_list.append(temp_cr)

        cropped_list = np.array(cropped_list).transpose(0, 3, 2, 1)
        preds = self.model.predict(cropped_list).detach().cpu().numpy()
        probs = preds.max(1)
        champs = preds.argmax(1)
        champs = self._mislabel_correction(champs)
        for idx, key in enumerate(self.target_map):
            pred_map[key] = (champs[idx], probs[idx])

        if champ_only:
            class_list = [v[0] for k, v in pred_map.items()]
            return class_list[:20]
        
        return pred_map

    def champ_onmap(self, frame) :
        # import matplotlib
        # matplotlib.use("TkAgg")
        # import matplotlib.pyplot as plt
        h, w = frame.shape[:2]
        import cv2
        cr_list = []
        for key in self.target_map:
            top, left, bottom, right = self.target_map[key]
            x1 = int(left * w)
            y1 = int(top * h)
            x2 = int(right * w)
            y2 = int(bottom * h)
            cropped = crop(frame, y1, y2, x1, x2)
            cr_list.append(cropped)
        base = np.full((288, 288, 3), 0, np.uint8)
        for ch_r in range(2) : #champ_row
            n_xc, n_yc = 0, 0
            xc, yc = 18, 18
            for ch_c in range(5) : #champ_column
                
                ch_wh = 21
                rad_add = 1
                out_line = 2
                temp_ch = cr_list[(ch_r * 5) + ch_c]
                temp_ch = resize(temp_ch, ch_wh, ch_wh)
                radius = ch_wh // 2
                champ_w = temp_ch.shape[0]
                champ_h = temp_ch.shape[1]
                n_xc = (ch_r * 150) + xc +10 # 18, 18, 18, ..., 118, 118
                n_yc += (yc * 2) # 36, 72, 108, 144 
                
                rect0 = (n_xc - radius, n_yc - radius)
                rect1 = (n_xc + radius, n_yc + radius)
                
                mask0 = np.zeros_like(base)
                
                mask0 = cv2.circle(mask0, (n_xc, n_yc), radius+rad_add, (1, 1, 1), -1)
                # base[rect0[0]:rect1[0], rect0[1]:rect1[1], :]
                
                mask2 = cv2.multiply(mask0[rect0[1]:rect0[1] + champ_w,
                                            rect0[0]:rect0[0] + champ_h, :], temp_ch)
                
                # base1 = np.full((70, 70, 3), 0, np.uint8)
                if ch_r == 0 :
                    ol_color = (220, 150, 0)
                else :
                    ol_color = (61, 61, 232)

                mask11 = np.zeros_like(base)
                mask11 = cv2.circle(mask11, (n_xc, n_yc), radius + rad_add + out_line, ol_color, -1)
                mask12 = np.zeros_like(base)
                mask12 = cv2.circle(mask12, (n_xc, n_yc), radius + rad_add, ol_color, -1)
                mask13 = cv2.subtract(mask11, mask12)


                base += mask13
                base[rect0[1]:rect0[1] + champ_w,
                                            rect0[0]:rect0[0] + champ_h, :] += mask2
                n_yc += 15
        return base
    
    def champ_onmap_resize(self, frame):
        """
        Draws circular champion icons with precise outlines on a 576x576 minimap,
        then downsamples to 288x288.

        Returns:
            np.ndarray: 288x288 map image with team-colored champ icons.
        """
        import cv2
        import numpy as np

        ch_wh = 43
        icon_radius = ch_wh // 2
        outline_thickness = 2

        # Load minimap background as base
        minimap_bg = cv2.imread("./map11.png")  # make this path configurable if needed
        base = cv2.resize(minimap_bg, (576, 576))

        # Extract champion crops from frame
        h, w = frame.shape[:2]
        cr_list = []
        for key in self.target_map:
            top, left, bottom, right = self.target_map[key]
            x1 = int(left * w)
            y1 = int(top * h)
            x2 = int(right * w)
            y2 = int(bottom * h)
            cropped = crop(frame, y1, y2, x1, x2)
            cr_list.append(cropped)

        # Place icons in 2-row layout
        for ch_r in range(2):
            for ch_c in range(5):
                index = ch_r * 5 + ch_c
                icon = resize(cr_list[index], ch_wh, ch_wh)

                # Create circular mask
                mask = np.zeros((ch_wh, ch_wh), dtype=np.uint8)
                cv2.circle(mask, (icon_radius, icon_radius), icon_radius, 255, -1)
                icon_circ = cv2.bitwise_and(icon, icon, mask=mask)

                # Compute placement position
                offset = 50
                n_xc = 36 + ch_c * 72 + 80
                n_yc = 36 + ch_r * 288 + 40
                rect0 = (n_xc - icon_radius, n_yc - icon_radius)
                x1_, x2_ = rect0[0], rect0[0] + ch_wh
                y1_, y2_ = rect0[1], rect0[1] + ch_wh

                # Paste circular icon
                roi = base[y1_:y2_, x1_:x2_]
                mask3c = cv2.merge([mask] * 3)
                np.copyto(roi, icon_circ, where=mask3c.astype(bool))

                # Draw outline
                ol_color = (220, 150, 0) if ch_r == 0 else (61, 61, 232)
                cv2.circle(base, (n_xc, n_yc), icon_radius, ol_color, thickness=outline_thickness)

        return cv2.resize(base, (288, 288), interpolation=cv2.INTER_AREA)

    def majority_voting(self, infer_results):
        class_list = []
        infer_results = np.array(infer_results).reshape(-1,10)
        for i in range(10):
            infer_by_player = infer_results[:,i]
            bins = np.bincount(infer_by_player)
            class_list.append(np.argmax(bins))
        return class_list


    def start_validation(self):
        # self.stats = []
        self.val_class = {}
    

    def run_validation_batch(self, classlist, cids_od, match_id):
        classlist_i = [int(x) for x in classlist]
        cids_od_i = [int(x) for x in cids_od]
        # self.stats.append((classlist, cids_od_i, [match_id]))
        self.val_class[match_id] = {
            'pred': classlist_i,
            'gt': cids_od_i
        }
    
    def end_validation(self, save_dir):
        # self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        # if len(self.stats) and self.stats[0].any():


        # print('t')
        with open(save_dir + '/classses.json', 'w') as f:
            json.dump(self.val_class, f)

        pass

    def _mislabel_correction(self, cids):
        '''
        DrMundo <--> Draven
        '''
        res = cids.copy()
        for i, cid in enumerate(cids):
            if cid == 23:
                res[i] = 24
            elif cid == 24:
                res[i] = 23
        return res
