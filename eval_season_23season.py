import sys
import os
import datetime
import json
import numpy as np
import time
import shutil

# from modules.TraceExtractor.position_data_manager import PositionDataManager

from modules.champion_classifier import ChampionClassifier

sys.path.append('./yolov5')
from modules.model_manager import ModelManager
import cv2
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("TkAgg")
from collections import defaultdict
import copy
def make_directory(func_name):
    result_dir = os.path.join('data','results')
    func_dir = os.path.join(result_dir, func_name)
    save_dir = os.path.join(func_dir, '{}'.format(
            datetime.datetime.now().strftime('%y%m%d_%H%M')
        ))
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(func_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_metadata(save_dir, func_name, test_args, vlist, vm, mm, cc):
    metadata = {
        'evaluation': func_name,
        'test_args': test_args,
        'vodlist': vlist,
        'image_mode': vm.im.mode,
        'image_args': vm.im.img_args,
        'model_args': mm.model_args,
        'champion_classifier_args': cc.model.config
    }
    with open(save_dir + '/metadata.json', 'w') as f:
        json.dump(metadata, f)


def save_time(save_dir, cf_time, dd_time, detect_time, kf_time=None):
    time_dict = {
        'class_filter': np.sum(cf_time),
        'death_discriminator': np.sum(dd_time),
        'inference': np.sum(detect_time),
        'n_detect': len(detect_time)
    }
    if kf_time is not None:
        time_dict.update(
            {
                'kalman_filter': np.sum(kf_time)
            }
        )
    with open(save_dir + '/times.json', 'w') as f:
        json.dump(time_dict, f)


def extend_tstamp(tstamp, start_sec, end_ex_sec=5):
    extend_ts = []
    tstamp = [0] + tstamp
    alpha = np.array(tstamp) % 1000

    cur_sec = start_sec
    idx = 0
    while cur_sec < tstamp[-1]//1000 + end_ex_sec:
        # target_millisec = (cur_sec+1)*1000 + alpha[idx]
        # if target_millisec-1000 < tstamp[-1]:
        #     if target_millisec >= tstamp[idx+1]:
        #         idx += 1
        cur_millisec = cur_sec*1000 + alpha[idx]
        if idx < len(alpha) - 1:
            if cur_millisec + 1000 >= tstamp[idx+1]:
                idx += 1
        cur_millisec = cur_sec*1000 + alpha[idx]
        extend_ts.append(cur_millisec)
        cur_sec += 1

    return extend_ts

def get_10champs(det_result, thrs = .7) :
    
    import torch
    detns_all = torch.concat(det_result[:])
    np_detns = np.array(detns_all[detns_all[:,-2].argsort(descending=True)].cpu())
    champs_10, detns = [], []
    for detn in np_detns :
        if detn[-1] not in champs_10 and detn[-2] > thrs :
            champs_10.append(int(detn[-1]))    
            detns.append(detn.copy())
        if len(champs_10) == 10 :
            break
    return champs_10, detns

def compute_iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_10champs(det_result, thrs=0.7, iou_thrs=0.5):
    detns_all = torch.concat(det_result[:])
    np_detns = np.array(detns_all[detns_all[:, -2].argsort(descending=True)].cpu())
    
    champs_10 = []
    detns = []

    for detn in np_detns:
        conf = detn[-2]
        cls = int(detn[-1])
        box = detn[:4]

        if conf < thrs:
            continue

        # Check for unique class
        if cls in champs_10:
            continue

        # Check for bounding box overlap with existing ones
        too_close = False
        for existing_detn in detns:
            existing_box = existing_detn[:4]
            if compute_iou(box, existing_box) > iou_thrs:
                too_close = True
                break
        if too_close:
            continue

        champs_10.append(cls)
        detns.append(detn.copy())

        if len(champs_10) == 10:
            break

    return champs_10, detns

def get_leaderboard_boxes() :
    blue_boxes = []
    red_boxes = []

    start_y = 881
    end_y = 886
    row_stride = 44

    for i in range(5):
        y1 = start_y + i * row_stride
        y2 = end_y + i * row_stride

        blue_boxes.append((y1, y2, 900, 920))   # Blue team (left column)
        red_boxes.append((y1, y2, 1020, 1040))  # Red team (right column)
    
    return blue_boxes, red_boxes
def get_valid_popup_boxes_1(frame):
    # Each pair = (killer, victim) as (y1, y2, x1, x2)
    popup_rows = [
        [(709-3, 749-3, 1609-3, 1649-3), (709-3, 749-3, 1692-3, 1732-3)],
        [(643-3, 683-3, 1609-3, 1649-3), (643-3, 683-3, 1692-3, 1732-3)],
        [(577-3, 617-3, 1609-3, 1649-3), (577-3, 617-3, 1692-3, 1732-3)],
        [(511-3, 551-3, 1609-3, 1649-3), (511-3, 551-3, 1692-3, 1732-3)],
    ]

    valid_boxes = []

    for row in popup_rows:

        valid_boxes.extend(row)

    return valid_boxes

def eval_detect(test_args, weights=None):
    
    func_name = sys._getframe().f_code.co_name
    save_dir = make_directory(func_name)
    
    cc = ChampionClassifier()
    if test_args['class_filter']:
        cc.start_validation()
        
    # if test_args['Death_discriminator']:
    #     dd = DeathDiscriminator()
    cf_time = []
    dd_time = []
    detect_time = []

    mm = ModelManager('./resources/model_args.json', weights=weights)
    mm.start_validation()
    mm1 = ModelManager('./resources/model_args.json', weights="models/epoch-146_250630.pt")
    mm1.start_validation()
    # mm2 = ModelManager('./resources/model_args.json', weights="models/epoch-97_250720_14_20_1.pt")
    # mm2.start_validation()
    
    # vod_dir="/home/ari/Documents/ari_server/ari/_hdd_storage_24tb/_ari/video/2023summer_split"
    
    # vod_dir = "/home/ari/_hdd_storage_24tb/_ari/video/2023summer"
    vod_dir = "./data/23_summer"
    
    mp4_files = []

    for root, dirs, files in os.walk(vod_dir):
        for file in files:
            if file.endswith(".mp4"):
                full_path = os.path.join(root, file)
                mp4_files.append(full_path)

    print(f"Found {len(mp4_files)} MP4 files.")
    
    vlist = mp4_files

    # vod_dir = './data/2023spring/'
    # vlist = os.listdir(vod_dir)
    # mp4_files = vlist
    # import matplotlib
    # matplotlib.use("TkAgg")
    import pickle
    # with open("./ch_list_13_24_1_ch_objects.pkl", "rb") as po :
    #     ch_new_list = pickle.load(po)
    # with open("./ch_list.pkl", "rb") as po :
    #     ch_new_list_14 = pickle.load(po)
    def extract_minimap(frame, minimap_crop):   
        y1, y2, x1, x2 = minimap_crop
        return frame[y1:y2, x1:x2]
    ch_new_list = mm.names
    # ch_new_list_14 = mm1.names
    minimap_crop = (792, 1080, 1632, 1920)
    from collections import defaultdict
    frame_val = 10
    
    # with open("./data/results/eval_detect/250730_2239"
    #         "/jax_vods_236.json", "r", encoding="utf-8") as f:
    #     jax_vods = json.load(f)
    # num_jaxvods = [idjax[0] for idjax in jax_vods]

    # with open("./scan_frames/23_spring_scan_frames.json", "r") as f :
    #     event_fnum = json.load(f) 
    
    # with open("./2023spring_poses.json", "r") as f :
    #     pos_data = json.load(f) 
    poses_log = {}
    j = 0
    save_dir += f"_23season_{frame_val}frame"
    print(save_dir)
    for i, vodf in enumerate(mp4_files):
        # if i not in num_jaxvods :
        #     continue
        # j += 1
        # if i < 142 :
        #     continue
        # if vodf.split(".mp4")[0].split("/")[-1] != "BRO vs DK - HLE vs T1  2023 LCK 서머 스플릿_1" :
        #     continue
        print('[{} / {}]'.format(i, len(vlist)))
        match_id = vodf.split('.')[0].split('_')[-1]
        print('processing : match id {} {}'.format(match_id, vodf))

        # Extract VOD name without extension (e.g., "LCK_2023_Summer_Game1.mp4" → "LCK_2023_Summer_Game1")
        vod_name = os.path.splitext(vodf)[0]

        # Create a subfolder named after the vod
        # event_save_dir = os.path.join(save_dir, f'event_frames_{weights}', vod_name)
        event_save_dir = os.path.join(save_dir, f'event_frames_{weights.split("/")[-1].split(".")[0]}'\
                                      f'_per_{frame_val}', 
                                      vod_name.split("/")[-1])
        os.makedirs(event_save_dir, exist_ok=True)

        map_template = cv2.imread("./map11.png")
        map_template = cv2.resize(map_template, (288, 288))
        map_template_gray = cv2.cvtColor(map_template, cv2.COLOR_BGR2GRAY)
        # video_path = os.path.join(vod_dir, vodf)
        video_path = vodf
        cap = cv2.VideoCapture(video_path)
        ch_detected = False
        frame_ch = 0
        ch_maps, detns, dets = [], [], []
        ch_maps1, detns1, dets1 = [], [], []

        blue_score = 0
        red_score = 0
        frame_idx = 0
        victim_kill_timestamps = defaultdict(lambda: -float('inf'))  # victim_id: last_killed_time
        viego_override = False
        victim_viego = ""
        viego_kill = -float('inf')
        tracking_kills = {}  # key: (killer_name, victim_name, pair_idx), value: last_good_frame
        tracking_log = {}
        blue_boxes, red_boxes = get_leaderboard_boxes()
        last_global_dragon_kill_time = -float('inf')  # in milliseconds
        dragon_cooldown_ms = 300000  # 5 minutes
        event_log = {}
        event_log["metadata"] = {}
        
        poses_log[vodf.split("/")[-1]] = {}
        # success = vm.load_vod(vod_dir + vodf)
        # vname = vodf.split("/")[-1]
        # try :
        #     event_intervals = event_fnum["sources"]\
        #         [vodf.split(".mp4")[0].split("/")[-1] + "_event_log.json"]\
        #         ['intervals']
        # except :
        #     continue
        # poses = None
        # try : 
        #     # poses = pos_data[vodf.split("/")[-1]]["poses"]
        #     # combined_labels = pos_data[vodf.split("/")[-1]]["combined"]
        #     ch_names = [mm.names[int(i)] for i in poses] 
        #     ch_detected = True
        #     mm.set_labels(combined_labels)
        #     print("from defined positions found position!")
        #     print("detected champs ", ch_names) 

        #     object_keywords = ["Dragon", "Baron"]
        #     object_indices = [i for i, name in enumerate(ch_new_list) if any(obj in name for obj in object_keywords)]

        #     model1_work = False
        #     source_files = ["eval_season.py", "eval_season_24_summer.py",
        #                     "eval_season_23season.py", "eval_season_24_summer_23season.py",
        #                     "eval_season_23season_fnum.py"]

        #     for src_file in source_files:
        #         if os.path.exists(src_file):
        #             shutil.copy(src_file, os.path.join(save_dir, os.path.basename(src_file)))
        #             print(f"[Copied] {src_file} → {save_dir}")
        #         else:
        #             print(f"[Warning] File not found: {src_file}")
        # except :
        #     continue
        
        # success = vm.load_vod(vod_dir + vodf)
        while True :
            
            frame_idx += 1
            ret, frame = cap.read()
            # current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
            
            if not ret:
                break

            minimap = extract_minimap(frame, minimap_crop)
            minimap_resized = cv2.resize(minimap, (288, 288))
            minimap_gray = cv2.cvtColor(minimap_resized, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(minimap_gray, map_template_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            m_val = str(max_val).replace(".", "_")
            corner = np.mean(minimap[0:20, 0:20])
            if max_val > 0.39 and not ch_detected :
                # center, area = detect_viewport_center_rgb_mask(minimap)

                frame_ch += 1
                # ui class listing
                if test_args['class_filter'] :
                    tic = time.time()
                    # infer_results = []
                    # for f in range(100):
                        # _, fim = vm.get_img_byframe(f)
                    # infer_result = cc.inference_frame(frame, champ_only=True)
                    
                    # ch_on_map = cc.champ_onmap(frame)
                    ch_on_map = cc.champ_onmap_resize(frame)
                    detn, det = mm.detect_all(ch_on_map)
                    # detn1, det1 = mm1.detect_all(ch_on_map)

                    # infer_results.append(infer_result)

                    ch_maps.append(ch_on_map)
                    detns.append(detn)
                    dets.append(det)

                    # ch_maps1.append(ch_on_map)
                    # detns1.append(detn1)
                    # dets1.append(det1)

                    # classlist = cc.majority_voting(infer_results)
                    # cc.run_validation_batch(classlist, cids_od, match_id)
                    # mm.set_labels(classlist)

                    toc = time.time()
                    cf_time.append(toc - tic)
                    
                if frame_ch == 100 :
                    
                    # if match_id[0:2] == '53' :
                    #     mm = ModelManager('./resources/model_args.json', weights="models/epoch-66.pt")
                    #     mm.start_validation()
                    #     mm.set_labels(classlist)
                    #     ch_names = [mm.names[int(i)] for i in classlist]
                    #     print(ch_names)
                    # else :
                    ch_thrs = .98
                    champs_10, det10 = get_10champs(detns, ch_thrs)[:10] 
                    if len(champs_10) < 10 : 
                        while len(champs_10) != 10 : 
                            ch_thrs = np.maximum(ch_thrs - 0.05, 0.02)  
                            champs_10, det10 = get_10champs(detns, ch_thrs)[:10] 
                            if ch_thrs == .02 :
                                break
                    
                    if ch_thrs < 0.1 :
                        break
                    ch_names = [mm.names[int(i)] for i in champs_10] 
                    det10 = torch.stack([torch.tensor(det) for det in det10])
                    
                    print("detected champs ", ch_names, ch_thrs) 
                    poses = assign_champ_positions(det10)
                    object_keywords = ["Dragon", "Baron"]
                    object_indices = [i for i, name in enumerate(ch_new_list) if any(obj in name for obj in object_keywords)]
                    combined_labels = list(sorted(set(champs_10 + object_indices)))
                    ch_detected = True
                    # mm.set_labels(champs_10)
                    mm.set_labels(combined_labels)
                    
                    # for another model 14.20
                    # model1_work = True
                    model1_work = False
                    # ch_thrs = .8
                    # champs_10_1, det10_1 = get_10champs(detns1, ch_thrs)[:10] 
                    # if len(champs_10_1) < 10 : 
                    #     while len(champs_10_1) != 10 : 
                    #         ch_thrs = np.maximum(ch_thrs - 0.05, 0.02)  
                    #         champs_10_1, det10_1 = get_10champs(detns1, ch_thrs)[:10] 
                    #         if ch_thrs == .02 :
                    #             break
                    
                    # if ch_thrs < 0.1 :
                    #     model1_work = False
                    # else :
                    #     ch_names1 = [mm1.names[int(i)] for i in champs_10_1] 
                    #     det10_1 = torch.stack([torch.tensor(det1) for det1 in det10_1])
                        
                    #     print("detected champs ", ch_names1, ch_thrs) 
                    #     poses1 = assign_champ_positions(det10_1)

                    #     object_keywords1 = ["Dragon", "Baron"]
                    #     object_indices1 = [i for i, name in enumerate(ch_new_list_14) if any(obj in name for obj in object_keywords1)]
                    #     combined_labels1 = list(sorted(set(champs_10_1 + object_indices1)))
                        
                    #     mm1.set_labels(combined_labels1)
                    
                    # # if set(ch_names) == set(ch_names1) :
                    # #     model1_work = True
                    # print(f"model1 work? {model1_work}")

                    # Save copies of key Python files for reproducibility
                    source_files = ["eval_season.py", "eval_season_24_summer.py",
                                    "eval_season_23season.py", "eval_season_24_summer_23season.py",
                                    "eval_season_23season_fnum.py"]

                    for src_file in source_files:
                        if os.path.exists(src_file):
                            shutil.copy(src_file, os.path.join(save_dir, os.path.basename(src_file)))
                            print(f"[Copied] {src_file} → {save_dir}")
                        else:
                            print(f"[Warning] File not found: {src_file}")
                    # gt_champs = [mm.names[int(i)] for i in np.array(gt[2]).astype(int)]

                popup_rows = [
                    [(709-5, 749, 1609-5, 1649), (709-5, 749, 1692-5, 1732)],
                    [(643-5, 683, 1609-5, 1649), (643-5, 683, 1692-5, 1732)],
                    [(577-5, 617, 1609-5, 1649), (577-5, 617, 1692-5, 1732)],
                    [(511-5, 551, 1609-5, 1649), (511-5, 551, 1692-5, 1732)],
                ]
                tagged_boxes = []
                for row in popup_rows:
                    tagged_boxes.append((*row[0], "left"))
                    tagged_boxes.append((*row[1], "right"))

            # # if frame_idx == 33428 : 
            # if 117600 < frame_idx < 117700 : 
            #     print("tt") 
            
            
            # # Example: Inside your main video loop
            # if frame_idx % 15 == 0 :
            #     timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)  # current time in ms

            #     # Check if timestamp_ms is close to a 500ms multiple (e.g., 500, 1000, 1500, ...)
            #     rounded = round(timestamp_ms / 500) * 500
            #     if abs(timestamp_ms - rounded) < 10:  # 100ms tolerance
            #         # This is a valid 0.5s frame
            #         process_frame = True
            #         print(frame_idx)
            #     else:
            #         process_frame = False
            # event_in = [ev for ev in event_intervals if frame_idx in range(ev[0], ev[1])]
            if ch_detected and frame_idx % frame_val == 0 and max_val > 0.34 : 
            # if ch_detected and process_frame and max_val > 0.37 and corner < 10 : 
                # Popup region(s) — can be expanded if multiple icons shown
                # print(frame_idx)
                popup_boxes = get_valid_popup_boxes(frame)
                # popup_boxes1 = get_valid_popup_boxes_1(frame)
                
                
                # if frame_idx < 57000 :
                #     continue
                # cv2.imshow("Current Frame", frame)
                # key = cv2.waitKey(100)  # 1 ms delay (use 0 if you want it to wait for keypress)                
                # if key == ord('q'):
                #     break  # Optional: press 'q' to quit early

                hp_ratios = get_hp(frame)
                zero_hp = len([hp for hp in hp_ratios if hp < 0.05])
                current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
                # if frame_idx % 1000 == 0 :
                #     print(current_time_sec // 1000)

                if viego_override and current_time_sec - viego_kill >= 10000.0:
                    viego_override = False
                    viego_kill = -float('inf')
                # if popup_boxes and zero_hp > 0 :
                if popup_boxes :
                
                    # event_map = champ_from_eventlog_to_map(frame, cc, popup_boxes)
                    # event_map = champ_from_eventlog_to_map(frame, cc, popup_boxes, minimap_bg=map_template)

                    detn_with_index = []
                    detected_pairs = []
                    eligible_pairs = []
                    deleting_keys = []
                    for k, v in tracking_log.items() :
                        tracking_log[k] = [None, None]
                        # print(frame_idx)
                    for pair_start in range(0, len(popup_boxes), 2):
                        pair_idx = pair_start // 2  # which logical pair we’re on (0, 1, 2...)

                        # Skip this pair if the previous one was not detected like [True, False]
                        if pair_idx > 1:
                            if len(detected_pairs) < pair_idx or not detected_pairs[pair_idx - 1]:
                                continue

                        this_pair_detected = False
                        this_pair_eligible = False

                        for i in [pair_start, pair_start + 1]:
                            if i >= len(popup_boxes):
                                continue
                            box = popup_boxes[i]
                            # box1 = popup_boxes1[i]
                            single_box_map = champ_from_eventlog_to_map(frame, cc, [box], minimap_bg=map_template)
                            detn_single, _ = mm.detect(single_box_map, [True] * len(combined_labels))
                            
                            threshold = .9
                            
                            # Keep only the detection with the highest confidence
                            if detn_single is not None and len(detn_single) > 0:
                                # Sort by confidence score (descending), take top-1
                                detn_single = detn_single[detn_single[:, 4].argsort(descending=True)[:1]]
                                if detn_single[-1][-1] in poses :
                                    threshold /= max(pair_idx, 1.0)
                                # else :
                                #     threshold = .8
                                if len(detn_with_index) > 0 and detn_with_index[-1][0] + 1 == i and detn_single[-1][4] < 0.9 \
                                    and detn_single[-1][-1] in poses and model1_work and len(detn_single) > 0 and \
                                    detn_single[-1][-1] != 60 :
                                    # detn_single1, _ = mm1.detect(single_box_map, [True] * len(combined_labels))

                                    if detn_single1 is not None and len(detn_single1) > 0 :
                                        # Sort by confidence score (descending), take top-1
                                        detn_single1 = detn_single1[detn_single1[:, 4].argsort(descending=True)[:1]]
                                        if mm1.names[int(detn_single1[0, 5].item())] in ch_new_list : 
                                            detn_single1[0, 5] = ch_new_list.index(mm1.names[int(detn_single1[0, 5].item())])
                                        else :
                                            detn_single1 = []
                                
                                
                                    detns = torch.concat([detn_single, detn_single1])
                                    detn_single = detns[detns[:, 4].argsort(descending=True)[:1]]
                            if detn_single is not None and len(detn_single) > 0 and detn_single[0, 4] >= threshold :
                                
                                for d in detn_single:
                                    this_pair_detected = True
                                    detn_with_index.append((i, d))
                        
                        # if len(detn_with_index) == (pair_idx + 1) * 2 and \
                        #     detn_with_index[-1][1][-1] != detn_with_index[-2][1][-1]:
                        if len(detn_with_index) >= 2 and len(detn_with_index)  > 0 and \
                            detn_with_index[-1][0] == detn_with_index[-2][0] + 1 and \
                            detn_with_index[-1][1][-1] != detn_with_index[-2][1][-1] and \
                            detn_with_index[-2][0] % 2 == 0 :
                            
                            killer = detn_with_index[-2][1][-1].item()
                            victim = detn_with_index[-1][1][-1].item()
                            hp_ratios = get_hp(frame)
                            poses_hp_map = list(zip(poses, hp_ratios))
                            
                            
                            killer_class_idx = int(killer)
                            killer_name = ch_new_list[killer_class_idx]

                            # Get victim class index and name
                            victim_class_idx = int(victim)
                            victim_name = ch_new_list[victim_class_idx]

                            original_kill_time = victim_kill_timestamps[victim_name]
                            
                            if viego_override :
                                if killer_name == "Viego" and victim in poses and victim_viego != victim_name:
                                    viego_kill = current_time_sec
                            
                            # if killer_name == "Viego" and victim in poses and not viego_override :
                            if killer_name == "Viego" and victim in poses and not viego_override :
                                # Temporarily override to force time condition pass
                                victim_kill_timestamps[victim_name] = -float('inf')
                                viego_override = True
                                victim_viego = victim_name
                                viego_kill = current_time_sec


                            if killer_name != "Viego" and victim_name == victim_viego and pair_idx == 0 and current_time_sec - viego_kill < 10000.0 :
                                victim_kill_timestamps[victim_name] = -float('inf')

                            # Time since last kill
                            kname = ch_new_list[int(detn_with_index[-2][1][-1].item())]
                            vname = ch_new_list[int(detn_with_index[-1][1][-1].item())]
                            key = (kname, vname)

                            time_since_last_kill = current_time_sec - victim_kill_timestamps[victim_name]
                            # Check if it's a champion and find HP
                            # print("viego_overide", viego_override)
                            for i, det in detn_with_index:
                                confidence = det[4].item()
                                class_idx = int(det[-1].item())
                                class_name = ch_new_list[class_idx]

                            if victim_class_idx in poses and killer_class_idx in poses :
                                v_pos = poses.index(victim_class_idx)
                                player_name, player_hp = poses_hp_map[v_pos]
                                
                                leader_board_hp = []
                                for idx, (y1, y2, x1, x2) in enumerate(blue_boxes + red_boxes):
                                    
                                    red_region = frame[y1:y2, x1:x2, 2]  # Red channel
                                    avg_red = np.mean(red_region)
                                    leader_board_hp.append(avg_red)
                                    # print(f"Player {idx+1} - Avg Red: {avg_red:.2f}")

                                if player_hp < 0.05 or (leader_board_hp[v_pos] > 40 and leader_board_hp[v_pos] < 60) :
                                    this_pair_eligible = time_since_last_kill >= 10000.0
                                    
                                    if this_pair_eligible:
                                        
                                        if killer_name != "Viego" and viego_override and victim_name == victim_viego and pair_idx == 0 and current_time_sec - viego_kill < 10000.0 :
                                            viego_override = False
                                            victim_class_idx = ch_new_list.index("Viego")
                                            v_pos = poses.index(victim_class_idx)
                                            viego_kill = -float('inf')
                                        
                                        victim_kill_timestamps[victim_name] = current_time_sec
                                        
                                        # Restore original time if Viego override was used
                                        victim_team = 'blue' if v_pos < 5 else 'red'
                                        print(f"✓ Champion kill: {killer_name} victim {victim_name} at {current_time_sec:.2f}s")
                                        if victim_team == 'red':
                                            blue_score += 1
                                        else:
                                            red_score += 1
                                        # snapshot_detected = copy.deepcopy(this_pair_detected)
                                        # snapshot_eligible = copy.deepcopy(this_pair_eligible)
                                        # tracking_log[key] = (snapshot_detected, snapshot_eligible)
                                        tracking_log[key] = [copy.deepcopy(this_pair_detected), copy.deepcopy(this_pair_eligible)]
                                        tracking_kills[key] = {
                                            "frame": frame.copy(),
                                            "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                                            "score": (blue_score, red_score),
                                            "confidence" : [detn_with_index[-2][1][4].item(), detn_with_index[-1][1][4].item()]
                                            # "confidence": [d[1].tolist() if isinstance(d[1], torch.Tensor) else d[1] for d in detn_with_index] 
                                        }

                            else :
                                # Object case
                                if "Dragon" in victim_name :
                                    time_since_last_kill = current_time_sec - victim_kill_timestamps[victim_name]
                                    dragon_recently_killed = current_time_sec - last_global_dragon_kill_time < dragon_cooldown_ms

                                    if time_since_last_kill >= 300000.0 and not dragon_recently_killed:
                                    # if time_since_last_kill >= 30000.0:
                                        killer_dragon = len([obj for obj in object_keywords if obj in killer_name])

                                        if killer_dragon == 0 :
                                            victim_kill_timestamps[victim_name] = current_time_sec
                                            last_global_dragon_kill_time = current_time_sec  # 🟡 GLOBAL timestamp
                                            this_pair_eligible = True
                                            
                                elif "Baron" in victim_name :
                                    time_since_last_kill = current_time_sec - victim_kill_timestamps[victim_name]
                                    if time_since_last_kill >= 360000.0 :

                                        victim_kill_timestamps[victim_name] = current_time_sec
                                        this_pair_eligible = True

                                        # print("baron")
                                elif "Dragon" in killer_name :
                                    pass
                            
                            if key in list(tracking_kills.keys()) : # for saving frame again again
                                tracking_log[key] = [this_pair_detected, this_pair_eligible]
                            #     tracking_kills[key]["frame"] = frame.copy()
                            # if key not in list(tracking_kills.keys()) :   
                            #     print("weird")    
                            # else :
                                
                            
                        detected_pairs.append(this_pair_detected)
                        eligible_pairs.append(this_pair_eligible)
                        # print(tracking_log)
                    for k, v in tracking_log.items() :
                        if v == [None, None] :
                            # tracking_log.pop(k)
                            info = tracking_kills[k]
                            frame_to_save = info["frame"]
                            timestamp_sec = round(info["timestamp"] / 1000.0, 2)
                            bscore, rscore = info["score"]
                            confidence = info["confidence"]
                            kname, vname = k

                            name_str = f"{kname}_{vname}".replace(" ", "")[:80]
                            filename = f"frame_at_{timestamp_sec:.2f}s_B{bscore}_R{rscore}_{name_str}.png"
                            save_path = os.path.join(event_save_dir, filename)
                            cv2.imwrite(save_path, frame_to_save)
                            print(f"[Saved] Log disappeared → {save_path}")
                            deleting_keys.append(k)
                            # print("tt")
                            
                            # 🟩 Save event to event_log
                            event_log[timestamp_sec] = {
                                "frame_idx": frame_idx,
                                "event_type": "champion_kill",
                                "killer": kname,
                                "victim": vname,
                                "confidence": confidence  # You can update this if you store confidence elsewhere
                            }

                    for k in deleting_keys :
                        tracking_kills.pop(k)
                        tracking_log.pop(k)

                    champ_indices = []
                    object_indices = []

                    for pair_idx, eligible in enumerate(eligible_pairs):
                        if not eligible:
                            continue

                        start = pair_idx * 2
                        end = start + 2
                        if end > len(detn_with_index):
                            continue  # safety check

                        det_pair = detn_with_index[start:end]
                        if len(det_pair) != 2:
                            continue  # must be a full pair

                        for i, d in det_pair:
                            class_idx = int(d[-1].item())
                            class_name = ch_new_list[class_idx]

                            if any(obj in class_name for obj in ["Dragon", "Baron"]):
                                object_indices.append(i)
                            else:
                                champ_indices.append(i)

                    # Group detections by popup pair
                    detections_by_pair = defaultdict(list)
                    # Now validate each pair
                    valid_detn_with_index = []

                    for pair_id, entries in detections_by_pair.items():
                        if len(entries) < 2:
                            # Only one detection — keep it for now
                            valid_detn_with_index.extend(entries)
                            continue

                        # Extract class names
                        name0 = ch_new_list[int(entries[0][1][-1].item())]
                        name1 = ch_new_list[int(entries[1][1][-1].item())]

                        if name0 != name1:
                            valid_detn_with_index.extend(entries)
                        else:
                            print(f"[Warning] Same class in pair {pair_id}: {name0}. Ignoring both.")
                    
                    for i, d in valid_detn_with_index:
                        pair_id = i // 2
                        detections_by_pair[pair_id].append((i, d))

                    # Filter detn_with_index to only eligible detections
                    filtered_detn_with_index = []
                    for pair_idx, eligible in enumerate(eligible_pairs):
                        if eligible:
                            start = pair_idx * 2
                            end = start + 2
                            if end <= len(detn_with_index):  # safe slice
                                filtered_detn_with_index.extend(detn_with_index[start:end])

                    valid_conf = (
                        filtered_detn_with_index and
                        len(filtered_detn_with_index) % 2 == 0 and
                        len(filtered_detn_with_index) > 1 and
                        all(d[4] >= .9 for _, d in filtered_detn_with_index)
                    )

                    objects_after_champs = (
                        not object_indices or not champ_indices or
                        min(object_indices) >= max(champ_indices)
                    )


                    valid_eligible_pairs = sum(eligible_pairs)
                    expected_pairs = len(popup_boxes) // 2

                    if valid_conf and objects_after_champs and valid_eligible_pairs > 0:

                        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        timestamp_sec = round(timestamp_ms / 1000.0, 2)

                        # Get detected names
                        detected_names = [ch_new_list[int(d[-1].item())] for _, d in filtered_detn_with_index]
                        
                        # Sanitize and shorten names for filename
                        name_str = "_".join(detected_names)
                        name_str = name_str.replace(" ", "")[:80]  # Remove spaces, limit filename length

                        if "Dragon" in name_str or "Baron" in name_str or "Void" in name_str:
                            if "Dragon" in name_str :
                                event_type = "dragon"
                            elif "Baron" in name_str :
                                event_type = "baron"
                            elif "Void" in name_str :
                                event_type = "voidgrub"
                            # event_type = "dragon" if "Dragon" in name_str else "baron"

                            # Ensure even-numbered grouping
                            for i in range(0, len(filtered_detn_with_index), 2):
                                pair = filtered_detn_with_index[i:i+2]
                                if len(pair) != 2:
                                    continue  # skip incomplete pairs

                                # Get names
                                pair_names = [ch_new_list[int(d[1][-1].item())] for d in pair]
                                pair_confidences = [d[1][4].item() for d in pair]

                                # File name (index appended to avoid overwrite)
                                filename = f"frame_{frame_idx}_at_{timestamp_sec:.2f}s_B{blue_score}_R{red_score}_{event_type}_{pair_names[0]}_{pair_names[1]}_{i//2}.png"
                                filename = filename.replace(" ", "")[:100]

                                # Save event
                                event_log[f"{timestamp_sec}_{i//2}"] = {
                                    "frame_idx": frame_idx,
                                    "event_type": event_type,
                                    "killer": pair_names[0],
                                    "victim": pair_names[1] if len(pair_names) > 1 else "Unknown",
                                    "confidences": pair_confidences
                                }

                                # Save frame
                                save_path = os.path.join(event_save_dir, filename)
                                cv2.imwrite(save_path, frame)
                                print(f"[Saved] {event_type.capitalize()} event → {save_path}")

        # poses_log[vodf.split("/")[-1]]["poses"] = poses
        # poses_log[vodf.split("/")[-1]]["ch_names"] = [mm.names[p] for p in poses]
        # poses_log[vodf.split("/")[-1]]["combined"] = combined_labels
        # poses_log[vodf.split("/")[-1]]["combined_names"] = [mm.names[p] for p in combined_labels]

        with open(os.path.join(event_save_dir, f'{event_save_dir.split("/")[-1]}_event_log.json'), 'w') as f:
            json.dump(event_log, f, indent=2)

    # with open(os.path.join(save_dir, f'{vod_dir.split("/")[-1]}_poses.json'), 'w') as f:
    #     json.dump(poses_log, f, indent=2)

    # compute statistics
    # mm.end_validation(save_dir)

    # save
    
    # if test_args['class_filter']:
    #     cc.end_validation(save_dir)
    # save_metadata(save_dir, func_name, test_args, vlist, vm, mm, cc)
    # save_time(save_dir, cf_time, dd_time, detect_time)


    print('Evaluation Done')
    return
def get_manual_champion_boxes():
    """
    Returns the 10 fixed champion scoreboard icon boxes:
    5 for blue side (left), 5 for red side (right).
    """
    boxes = []

    # Blue side (left column)
    for i in range(5):
        y1 = 159 + i * 104
        y2 = 199 + i * 104
        x1 = 32
        x2 = 72
        boxes.append((y1, y2, x1, x2))

    # Red side (right column)
    x_offset = 1814
    for i in range(5):
        y1 = 159 + i * 104
        y2 = 199 + i * 104
        x1 = 32 + x_offset
        x2 = 72 + x_offset
        boxes.append((y1, y2, x1, x2))

    return boxes

def champ_onmap_resize(frame, boxes, minimap_bg_path="./map11.png"):
    import cv2
    import numpy as np

    ch_wh = 43
    icon_radius = ch_wh // 2
    outline_thickness = 2

    # Load minimap background as base
    minimap_bg = cv2.imread(minimap_bg_path)
    base = cv2.resize(minimap_bg, (576, 576))

    for idx, (y1, y2, x1, x2) in enumerate(boxes):
        icon = resize(frame[y1:y2, x1:x2], ch_wh, ch_wh)

        # Create circular mask
        mask = np.zeros((ch_wh, ch_wh), dtype=np.uint8)
        cv2.circle(mask, (icon_radius, icon_radius), icon_radius, 255, -1)
        icon_circ = cv2.bitwise_and(icon, icon, mask=mask)

        # Grid placement: 2 rows × 5 cols
        row = 0 if idx < 5 else 1
        col = idx % 5
        n_xc = 60 + col * 50
        n_yc = 40 + row * 30
        rect0 = (n_xc - icon_radius, n_yc - icon_radius)

        # Paste icon
        y1_, y2_ = rect0[1], rect0[1] + ch_wh
        x1_, x2_ = rect0[0], rect0[0] + ch_wh
        roi = base[y1_:y2_, x1_:x2_]
        mask3c = cv2.merge([mask] * 3)
        np.copyto(roi, icon_circ, where=mask3c.astype(bool))

        # Draw team-colored outline
        color = (61, 61, 232) if idx < 5 else (220, 150, 0)  # Blue first, then red
        cv2.circle(base, (n_xc, n_yc), icon_radius, color, thickness=outline_thickness)

    return cv2.resize(base, (288, 288), interpolation=cv2.INTER_AREA)
from modules.image_process import crop, resize
import torch
def is_in_box(x, y, box):
    y1, y2, x1, x2 = box[:4]
    return x1 <= x <= x2 and y1 <= y <= y2
def get_hp(frame) :

    # Define HP bar y positions for the five DRX players    
    xcoords = [(32, 70), (1852, 1890)]
    y_positions = [210, 314, 418, 520, 622]

    # Calculate green channel HP ratios
    hp_ratios = []
    for xc in xcoords :
        for y in y_positions:
            green_values = frame[y, xc[0]:xc[1], 1]  # Green channel
            green_hp = green_values > 40  # Threshold for greenish pixels
            hp_ratio = np.sum(green_hp) / len(green_values)
            hp_ratios.append(hp_ratio)
    return hp_ratios

def is_event_column_order_valid_with_pairs(detn):
    """
    Validates that detection appears in full killer-victim pairs (top and bottom row) per column,
    and columns appear in left-to-right continuity (starting from column 0).

    Returns:
        bool: True if full pairs appear in order; False otherwise.
    """
    if detn is None or len(detn) == 0:
        return False

    col_centers = [50, 80, 110, 140]
    row_centers = [50, 100]
    icon_radius = 21

    column_row_hits = {i: set() for i in range(len(col_centers))}

    for box in detn:
        x1, y1, x2, y2 = [b.item() for b in box[:4]]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        # Find matching column
        col_idx = None
        for i, cx in enumerate(col_centers):
            if abs(x_center - cx) <= icon_radius:
                col_idx = i
                break
        if col_idx is None:
            continue

        # Find matching row
        row_idx = None
        for j, ry in enumerate(row_centers):
            if abs(y_center - ry) <= icon_radius:
                row_idx = j
                break
        if row_idx is None:
            continue

        column_row_hits[col_idx].add(row_idx)

    # Get only fully detected (killer + victim) columns
    fully_detected_cols = sorted(
        [col for col, rows in column_row_hits.items() if rows == {0, 1}]
    )

    if len(fully_detected_cols) == 0:
        return False

    for i in range(len(fully_detected_cols)):
        if fully_detected_cols[i] != i:
            return False

    return True


def is_event_column_order_valid(detn):
    """
    Ensures detections only appear in columns left-to-right starting from column 0.

    Specifically:
    - Column 0 must exist for any detection to be valid
    - Column 1 can appear only if column 0 exists
    - Column 2 only if 0 and 1 exist
    - Column 3 only if 0, 1, 2 exist

    Returns:
        bool: True if detections are logically continuous from the left.
    """
    if detn is None or len(detn) == 0:
        return False

    # Reference column centers (from 288x288 event_map resized from 576x576)
    column_centers = [18, 55, 89, 122]
    icon_half_width = 21  # From your layout: ch_wh // 2

    detected_cols = set()
    for box in detn:
        x1, x2 = box[0].item(), box[2].item()
        x_center = (x1 + x2) / 2
        for i, col_center in enumerate(column_centers):
            if abs(x_center - col_center) <= icon_half_width:
                detected_cols.add(i)

    detected_cols = sorted(detected_cols)

    # Enforce continuity from column 0
    for i in range(len(detected_cols)):
        if detected_cols[i] != i:
            return False

    return True

def get_valid_popup_boxes(frame):
    # Each pair = (killer, victim) as (y1, y2, x1, x2)
    popup_rows = [
        [(709-5, 749, 1609-5, 1649), (709-5, 749, 1692-5, 1732)],
        [(643-5, 683, 1609-5, 1649), (643-5, 683, 1692-5, 1732)],
        [(577-5, 617, 1609-5, 1649), (577-5, 617, 1692-5, 1732)],
        [(511-5, 551, 1609-5, 1649), (511-5, 551, 1692-5, 1732)],
    ]

    valid_boxes = []

    for row in popup_rows:
        y1, y2, x1, x2 = row[0]  # just check left champ
        region = frame[y1:y2, x1:x2]
        if np.mean(region) < 15:  # very dark = empty
            break  # stop at first missing row
        valid_boxes.extend(row)

    return valid_boxes

def champ_from_eventlog_to_map(frame, classifier, boxes, minimap_bg=None):
    """
    Generate a 288x288 image placing circular champion icons with precisely matching outlines
    on a minimap background.

    Args:
        frame (np.ndarray): Full video frame.
        classifier (ChampionClassifier): CNN classifier for champ icon.
        boxes (list): List of (y1, y2, x1, x2) popup champ boxes.
        minimap_bg (np.ndarray): Optional 288x288 minimap background.

    Returns:
        np.ndarray: Final 288x288 image with icons and outlines.
    """
    import cv2
    import numpy as np

    ch_wh = 43
    outline_thickness = 2
    icon_radius = ch_wh // 2
    start_x, start_y = 100, 100
    col_spacing, row_spacing = 60, 100

    if minimap_bg is not None:
        base = cv2.resize(minimap_bg, (576, 576))
    else:
        base = np.full((576, 576, 3), 0, np.uint8)

    for i, (y1, y2, x1, x2) in enumerate(boxes):
        # Crop champ icon
        cropped = frame[y1:y2, x1:x2]
        resized = resize(cropped, classifier.input_w, classifier.input_h)
        input_arr = resized.transpose(2, 1, 0)[None]
        # with torch.no_grad():
        #     preds = classifier.model.predict(input_arr).detach().cpu().numpy()
        # pred_id = preds.argmax(1)[0]

        icon = resize(cropped, ch_wh, ch_wh)
        champ_h, champ_w = icon.shape[:2]

        # Create circular mask for the icon
        mask = np.zeros((champ_h, champ_w), dtype=np.uint8)
        cv2.circle(mask, (icon_radius, icon_radius), icon_radius, 255, -1)
        icon_circ = cv2.bitwise_and(icon, icon, mask=mask)

        # Determine position
        col = i // 2
        row = i % 2
        n_xc = start_x + col * col_spacing
        n_yc = start_y + row * row_spacing
        rect0 = (n_xc - icon_radius, n_yc - icon_radius)

        # Paste masked champ icon
        y1_, y2_ = rect0[1], rect0[1] + champ_h
        x1_, x2_ = rect0[0], rect0[0] + champ_w
        roi = base[y1_:y2_, x1_:x2_]
        mask3c = cv2.merge([mask] * 3)
        np.copyto(roi, icon_circ, where=mask3c.astype(bool))

        # Draw exact-outline circle
        color = (220, 150, 0) if row == 0 else (61, 61, 232)
        cv2.circle(base, (n_xc, n_yc), icon_radius, color, thickness=outline_thickness)

    return cv2.resize(base, (288, 288), interpolation=cv2.INTER_AREA)

def champ_icon_grid(frame, box, minimap_bg=None):
    """
    Render 10 identical champ icons in 2 rows: 5 blue, 5 red.
    Args:
        frame (np.ndarray): Full video frame.
        box (tuple): A single (y1, y2, x1, x2) popup box.
        minimap_bg (np.ndarray): Optional minimap background.

    Returns:
        np.ndarray: 288x288 image with icons in grid.
    """
    import cv2
    import numpy as np

    ch_wh = 43
    icon_radius = ch_wh // 2
    outline_thickness = 2
    col_spacing = 80
    row_spacing = 100
    start_x = 80
    start_y = 100

    if minimap_bg is not None:
        base = cv2.resize(minimap_bg, (576, 576))
    else:
        base = np.full((576, 576, 3), 0, np.uint8)

    # Extract champ icon
    y1, y2, x1, x2 = box
    cropped = frame[y1:y2, x1:x2]
    icon = cv2.resize(cropped, (ch_wh, ch_wh))
    champ_h, champ_w = icon.shape[:2]

    # Create circular mask once
    mask = np.zeros((champ_h, champ_w), dtype=np.uint8)
    cv2.circle(mask, (icon_radius, icon_radius), icon_radius, 255, -1)
    icon_circ = cv2.bitwise_and(icon, icon, mask=mask)

    for i in range(10):
        row = 0 if i < 5 else 1  # top or bottom
        col = i if i < 5 else i - 5
        n_xc = start_x + col * col_spacing
        n_yc = start_y + row * row_spacing

        x1_ = n_xc - icon_radius
        x2_ = x1_ + champ_w
        y1_ = n_yc - icon_radius
        y2_ = y1_ + champ_h

        roi = base[y1_:y2_, x1_:x2_]
        mask3c = cv2.merge([mask] * 3)
        np.copyto(roi, icon_circ, where=mask3c.astype(bool))

        color = (61, 61, 232) if row == 0 else (220, 150, 0)  # blue or red
        cv2.circle(base, (n_xc, n_yc), icon_radius, color, thickness=outline_thickness)

    return cv2.resize(base, (288, 288), interpolation=cv2.INTER_AREA)

def classify_event_champion(frame, classifier, top=709, bottom=749, left=1609, right=1649):
    """
    Crop event popup area and classify the champion.

    Args:
        frame (np.ndarray): Full frame from the video.
        classifier (ChampionClassifier): Classifier instance.
        top, bottom, left, right (int): Bounding box of event champ icon.

    Returns:
        (int, float): Predicted champion ID and confidence.
    """
    cropped = frame[top:bottom, left:right]  # crop 40x40 area
    resized = resize(cropped, classifier.input_w, classifier.input_h)  # resize to model input
    input_arr = resized.transpose(2, 1, 0)[None]  # shape (1, C, H, W)
    with torch.no_grad():
        preds = classifier.model.predict(input_arr).detach().cpu().numpy()
    pred_id = preds.argmax(1)[0]
    confidence = preds.max(1)[0]

    return pred_id, confidence

def data_set_test():
    test_args = {
        'overlay_r': 5,
        'class_filter': True,
        'Death_discriminator': True
    }
    
    eval_detect(test_args, weights='models/epoch-121_250730_13.pt') # minion + dragon + baron

    pass

def assign_champ_positions(detn):
    """
    Map detected bounding boxes to 10 standard champion roles.

    Top row → blue team: [Top, Jungle, Mid, ADC, Support]
    Bottom row → red team: [Top, Jungle, Mid, ADC, Support]
    
    Args:
        detn (Tensor): shape [N, 6], including x1, y1, x2, y2, conf, class_id

    Returns:
        List[int]: class_ids ordered by [blue_T, J, M, A, S, red_T, J, M, A, S]
    """
    if detn is None or len(detn) < 10:
        raise ValueError("Expected at least 10 champ detections.")

    # Separate by row
    top_row = []
    bottom_row = []

    for box in detn:
        x1, y1, x2, y2, conf, class_id = box.tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        entry = (x_center, int(class_id))
        if y_center < 70:
            top_row.append(entry)
        else:
            bottom_row.append(entry)

    if len(top_row) != 5 or len(bottom_row) != 5:
        raise ValueError("Did not find 5 champions per team.")

    # Sort each row by x_center (left to right)
    top_row_sorted = sorted(top_row, key=lambda x: x[0])
    bottom_row_sorted = sorted(bottom_row, key=lambda x: x[0])

    # Combine in order: blue team first
    role_ordered_ids = [cid for _, cid in top_row_sorted + bottom_row_sorted]
    return role_ordered_ids

def print_best_epoch():
    import csv

    w_box = 0.05
    w_obj = 5.0
    w_cls = 0.5


    root_dir = '//192.168.137.100/11_Archive/GameNAS_Personal/lol_resources/trace/'
    computer_dir = 'Exp28_backup/'
    print('t')
    dir_list = [
        '220715',
        '220803',
        '220907'
    ]

    for exp_folder in dir_list:
        exp_dir = root_dir + computer_dir + exp_folder
        exp_lists = [d for d in os.listdir(exp_dir) if d[:3] == 'exp']
        if len(exp_lists) == 1:
            exp_dir += '/' + exp_lists[0]

            # find best epoch

    print('t')

import pandas as pd
def create_position_table_v3(pos, match_id, labels, save_dir, mm, confidence_threshold=0.3):
    """
    Create a table of champion positions where each column represents `x y` per second.

    Args:
        pos (numpy.ndarray): Array of shape (10, 1369, 5) containing positions and other features.
        match_id (list): List of match_id (in ms).
        labels (list): Champion labels (length 10).
        save_dir (str): Directory to save the table as CSV.
        confidence_threshold (float): Minimum confidence to consider the detection valid.

    Returns:
        pd.DataFrame: DataFrame of champion positions per second.
    """
    # Prepare column names for each second
    columns = []
    for timestamp in range(pos.shape[1]):
        # time_in_seconds = int(timestamp / 1000)  # Convert ms to seconds
        columns.append(f"{timestamp}_x")
        columns.append(f"{timestamp}_y")
    
    # Prepare data for the table
    table_data = {}
    for champ_idx, champ_label in enumerate(labels):
        row = []
        for t_idx in range(pos.shape[1]):  # Iterate over time steps
            if pos[champ_idx, t_idx, -1] >= confidence_threshold:  # Check confidence
                x, y = int(pos[champ_idx, t_idx, 0]), int(pos[champ_idx, t_idx, 1])
            else:
                x, y = None, None  # Mark invalid positions
            row.extend([x, y])  # Append x and y for this timestamp
        table_data[mm.names[champ_label]] = row
    # return table_data

    # Create DataFrame
    df = pd.DataFrame.from_dict(table_data, orient='index', columns=columns)

    # Save to CSV
    csv_path = os.path.join(save_dir, str(match_id) + '_champion_positions_v3.csv')
    df.to_csv(csv_path)
    print(f"Champion positions table saved to {csv_path}")

    # Return the DataFrame
    return df



if __name__ == '__main__':
    test_args = {
        'overlay_r': 5,
        'class_filter': True,
        'Death_discriminator': True
    }
    
    # eval_detect(test_args)
    # eval_track(test_args)
    # # eval_track('./data/results/eval_track/220718_1715/dets')

    # eval_detect(test_args, weights='models/epoch-121.pt')   # data 50k

    # test_args = {
    #     'overlay_r': 1,
    #     'class_filter': True,
    #     'Death_discriminator': True
    # }
    # eval_detect(test_args)

    # test_args = {
    #     'overlay_r': 5,
    #     'class_filter': True,
    #     'Death_discriminator': False
    # }
    # eval_detect(test_args)

    # test_args = {
    #     'overlay_r': 5,
    #     'class_filter': False,
    #     'Death_discriminator': False
    # }
    # eval_detect(test_args)
    
    
    # ablation_test()
    data_set_test()

    
    # print_best_epoch()



