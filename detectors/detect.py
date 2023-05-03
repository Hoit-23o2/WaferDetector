#!/usr/bin/env python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys, os
from NaiveDetector import *

dataset_path = "../dataset/"

naive = NaiveDetector()

for fid, wafer_name in enumerate(os.listdir(dataset_path)):
    naive.load_img(os.path.join(dataset_path, wafer_name))
    orig_img = cv2.imread(os.path.join(dataset_path, wafer_name))
    
    pre_res = naive.pre_process(naive.img, True)
    edges, eroded_edges, dilated_edges, lines, line_lengths, selected_wafers, res = pre_res
    
    naive.load_img(os.path.join(naive.pre_dir, "pre-" + wafer_name))
    mid_res = naive.process(naive.img, True)
    edges, lines, line_lengths, selected_wafers, wafer_res = mid_res

    wafers = []
    wafers.append(wafer_res)
    further_process = True
    tmp_img = cv2.imread(os.path.join(dataset_path, wafer_name))

    while further_process:
        post_res = naive.post_process(tmp_img, wafer_res=wafer_res)
        img, further_process = post_res

        print("further_process: ", further_process)
        
        if not further_process:
            for wafer_id, wafer_res in enumerate(wafers):
                naive.artist_draw_wafer(orig_img, wafer_res)
                naive.store_wafer_res(os.path.join(naive.fnl_dir, wafer_name + "wafer-res-" + str(wafer_id)), wafer_res)
            naive.store_img(os.path.join(naive.fnl_dir, wafer_name), orig_img)
        else:
            mid_res = naive.process(img, True, threshold = 0.3, error = 100, further_process = True)
            edges, lines, line_lengths, selected_wafers, wafer_res = mid_res
            wafers.append(wafer_res)

# let's display the results
fig, ax = plt.subplots(5, 4, figsize = (14, 12), dpi = 300)
titles = ["(a) original", "(b) pre-processed", "(c) mid-processed", "(d) final"]
for fid, wafer_name in enumerate(os.listdir(dataset_path)):
    # display original image
    ax[fid, 0].imshow(cv2.imread(os.path.join(dataset_path, wafer_name)))
    # display pre-processed image
    ax[fid, 1].imshow(cv2.imread(os.path.join(naive.pre_dir, "pre-" + wafer_name)))
    # display mid image
    ax[fid, 2].imshow(cv2.imread(os.path.join(naive.mid_dir, "pre-" + wafer_name)))
    # display final image
    ax[fid, 3].imshow(cv2.imread(os.path.join(naive.fnl_dir, wafer_name)))
    
    # remove ticks for every
    # for i in range(4):
    #     ax[fid, i].set_xticks([])
    #     ax[fid, i].set_yticks([])
    
    if fid == 4:
        for i in range(4):
            ax[fid, i].set_xlabel(titles[i])


plt.tight_layout()
# plt.show()
plt.savefig("results.png")