import argparse
import numpy as np
import math
from ap_util import find_bounds

def find_bounds(arr):
    """
    arr: attention pattern
    return: end-points of non-zero block
    """
    flatten = np.sum(arr, axis = (0,1))
    zero_flatten = flatten != 0
    d1, d2 = zero_flatten.shape

    d1_bound = 0
    for i in range(0, d1):
        if np.sum(zero_flatten[i:, :]) == 0:
            break
    d1_bound = i

    d2_bound = 0
    for i in range(0, d2):
        if np.sum(zero_flatten[:, i:]) == 0:
            break
    d2_bound = i

    return d1_bound,  d2_bound


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn_path', type=str)
    parser.add_argument('--block_path', type=str)
    parser.add_argument('--fine_mask_path', type=str)
    parser.add_argument('--coarse_mask_path', type=str)
    parser.add_argument('--percentage', type=str)
    parser.add_argument('--block_size', type=int)
    args = parser.parse_args()
    percentage=float(args.percentage)

    # -------------------------------- setup
    block_size = args.block_size
    attn_pattern = np.load(args.attn_path)
    n_layers, _, n_head, w, h = attn_pattern.shape
    block_w, block_h = math.ceil(w / block_size), math.ceil(h / block_size)
    block_attn_pattern = np.zeros((n_layers, 1, n_head, block_w, block_h))
    fine_prune_mask = []
    coarse_prune_mask = np.zeros(attn_pattern.shape)

    # -------------------------------- find tau for percentage and create prune mask
    for layer_attn_pattern in attn_pattern:
        d1_bound, d2_bound = find_bounds(layer_attn_pattern)
        non_zero_attn = layer_attn_pattern[:,:,0:d1_bound, 0:d2_bound]
        target_percentile = percentage * 100
        threshold = np.percentile(non_zero_attn, target_percentile, interpolation='nearest')
        fine_prune_mask.append( layer_attn_pattern  >  threshold) #0 if we want to prune
    fine_prune_mask = np.array(fine_prune_mask)
    np.save(args.fine_mask_path, fine_prune_mask)

    # -------------------------------- find block pattern
    for i in range(0, block_w):
        for j in range(0, block_h):
            w_min = i * block_size
            w_max = (i+1) * block_size
            h_min = j * block_size
            h_max = (j+1) * block_size

            value = np.sum(fine_prune_mask[:,:,:,w_min:w_max, h_min:h_max], axis=(3,4)) != 0

            block_attn_pattern[:,:,:,i,j] = value
            coarse_prune_mask[:,:,:,w_min:w_max,h_min:h_max] = value[:,:,:,np.newaxis, np.newaxis]

    np.save(args.block_path, block_attn_pattern)
    np.save(args.coarse_mask_path, coarse_prune_mask)
