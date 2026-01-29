""" Transformation of eff_medsam2_infer_CT_esion_npz_recist_local.py to 
be compatible with our data structure for NIFTI files """ 

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' # run on CPU. cannot delete
import torch
from efficient_track_anything.build_efficienttam import (
    build_efficienttam_video_predictor_npz,
)

import re
from glob import glob
from tqdm import tqdm
import os
from os.path import join, basename
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd
import numpy as np
import random
import argparse
from datetime import datetime
import time
from pathlib import Path
from skimage.draw import line
from evaluate import Evaluator
import seaborn as sns
import matplotlib.colors as mcolors 
import matplotlib.patches as mpatches
import math

from joblib import Parallel, delayed
from PIL import Image
import SimpleITK as sitk
import torch
import torch.multiprocessing as mp
#from huggingface_hub import hf_hub_download
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint',
    type=str,
    default="small",
    help='small EfficientTAM model or tiny EfficientTAM model',
)
parser.add_argument(
    '--cfg',
    type=str,
    default="efficient_track_anything/configs",
    help='model config root',
)
# Unused now 
parser.add_argument(
    '-i',
    '--imgs_path',
    type=str,
    default='./data/validation_public_npz',
    help='imgs path',
)
#Unused now
parser.add_argument(
    '--gts_path',
    default=None,
    help='gts path',
)

parser.add_argument(
    '--index_csv', 
    type = str, 
    help = 'Path to the index csv file containing the image, mask, and RECIST annotation information.'
)

parser.add_argument(
    '--disease_loc', 
    type = str, 
    help = 'Where the disease is located (e.g. Abdomen, Lung, MultiSite, etc.) corresponding to the current file structure.'
)
# Unused now
parser.add_argument(
    '-o',
    '--pred_save_dir',
    type=str,
    default="./data/segs_test",
    help='segs path',
)
parser.add_argument(
    '--shift',
    type=int,
    default=0,
)
parser.add_argument(
    '--sample_points', # used if propagate_with_box is False
    help='how to sample points for propagation',
    choices=['from_box', 'from_recist_n', 'from_recist_center', 'from_recist_3'],
    type=str,
    default="from_recist_n", # 'from_box', 'from_recist_n' (n=5), 'from_recist_center', 'from_recist_3'
)
# add option to propagate with either box or mask
parser.add_argument(
    '--propagate_with_box',
    default=True,
    action='store_true',
    help='whether to propagate with box'
)
parser.add_argument(
    '--save_nifti',
    default=False,
    action='store_true',
    help='whether to save nifti'
)
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=2,
)

args = parser.parse_args()
checkpoint = args.checkpoint
model_cfg = args.cfg
# make into absolute path
script_directory = os.path.dirname(os.path.abspath(__file__))
if checkpoint == 'small':
    model_cfg = '/' + join(script_directory, model_cfg, 'efficienttam_s_512x512.yaml')
    ckpt_path = join(script_directory, 'checkpoints', 'eff_medsam2_small_FLARE25_RECIST_baseline.pt')
else:
    model_cfg = '/' + join(script_directory, model_cfg, 'efficienttam_ti_512x512.yaml')
    ckpt_path = join(script_directory, 'checkpoints', 'eff_medsam2_tiny_FLARE25_RECIST_baseline.pt')
imgs_path = args.imgs_path
gts_path = args.gts_path
shift = args.shift
pred_save_dir = args.pred_save_dir
save_nifti = args.save_nifti
nifti_path = join(args.pred_save_dir, 'segs_nifti')
save_overlay = args.save_overlay
png_save_dir = join(args.pred_save_dir, 'png_overlay')
num_workers = args.num_workers
propagate_with_box = args.propagate_with_box
index_csv = args.index_csv
disease_location = args.disease_loc

predictor = build_efficienttam_video_predictor_npz(model_cfg, ckpt_path, device='cpu')

os.makedirs(pred_save_dir, exist_ok=True)
if save_overlay:
    os.makedirs(png_save_dir, exist_ok=True)
if save_nifti:
    os.makedirs(nifti_path, exist_ok=True)

def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     


def resize_grayscale_to_rgb_and_resize(array, image_size):
    """
    Resize a 3D grayscale NumPy array to an RGB image and then resize it.
    
    Parameters:
        array (np.ndarray): Input array of shape (d, h, w).
        image_size (int): Desired size for the width and height.
    
    Returns:
        np.ndarray: Resized array of shape (d, 3, image_size, image_size).
    """
    d, h, w = array.shape
    resized_array = np.zeros((d, 3, image_size, image_size))
    
    for i in range(d):
        img_pil = Image.fromarray(array[i].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        img_array = np.array(img_resized).transpose(2, 0, 1)  # (3, image_size, image_size)
        resized_array[i] = img_array
    
    return resized_array


def sample_points_in_bbox_grid(bbox: np.ndarray, n: int) -> np.ndarray:
    """
    Uniformly sample n grid-aligned (x, y) points inside the bbox.

    Args:
        bbox (np.ndarray): [x_min, y_min, x_max, y_max]
        n (int): Number of points to sample

    Returns:
        np.ndarray: shape (n, 2), each row is [x, y]
    """
    x_min, y_min, x_max, y_max = bbox
    grid_size = int(np.ceil(np.sqrt(n)))

    x_vals = np.linspace(x_min, x_max, grid_size, dtype=int)
    y_vals = np.linspace(y_min, y_max, grid_size, dtype=int)

    xv, yv = np.meshgrid(x_vals, y_vals)
    coords = np.stack([xv.ravel(), yv.ravel()], axis=1)

    return coords[:n]

def get_center_from_recist(recist_per_lab):
    H, W = recist_per_lab.shape
    # get line coordinates
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    # Get endpoints
    p1 = coords[0]
    p2 = coords[-1]

    # Compute midpoint and line length
    center = ((p1 + p2) / 2).astype(np.float32)
    return np.array([[center[0], center[1]]])


def get_n_points_from_recist(recist_per_lab, n=5):
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    if len(coords) < n:
        raise ValueError(f"Cannot sample {n} points; RECIST line only has {len(coords)} pixels.")

    sampled_indices = np.random.choice(len(coords), size=n, replace=False)
    sampled_points = coords[sampled_indices]
    return sampled_points  # shape: (n, 2)

def get_center_and_endpoints_from_recist(recist_per_lab):
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    if len(coords) < 2:
        raise ValueError("RECIST line must contain at least two points")

    # Endpoints
    p1 = coords[0].astype(np.float32)
    p2 = coords[-1].astype(np.float32)

    # Center (midpoint between endpoints)
    center = ((p1 + p2) / 2).astype(np.float32)

    return np.array([center, p1, p2])  # each is shape (2,)

def get_diameter_bbox(recist_per_lab, shift=0):
    H, W = recist_per_lab.shape
    # get line coordinates
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    # Get endpoints
    p1 = coords[0]
    p2 = coords[-1]

    # Compute midpoint and line length
    center = ((p1 + p2) / 2).astype(int)
    diameter = np.linalg.norm(p1 - p2)
    half_side = int((diameter) / 2)

    # Get bounding box corners
    x_min = center[0] - half_side
    x_max = center[0] + half_side
    y_min = center[1] - half_side
    y_max = center[1] + half_side

    # clamp to image bounds
    x_min = max(0, x_min - shift)
    y_min = max(0, y_min - shift)
    x_max = min(W - 1, x_max + shift)
    y_max = min(H - 1, y_max + shift)

    return np.array([x_min, y_min, x_max, y_max])


def get_diameter(recist_per_lab):
    H, W = recist_per_lab.shape
    # get line coordinates
    ys, xs = np.where(recist_per_lab > 0)
    coords = np.stack([xs, ys], axis=1)

    # Get endpoints
    p1 = coords[0]
    p2 = coords[-1]

    # Compute midpoint and line length
    #center = ((p1 + p2) / 2).astype(int)
    diameter = np.linalg.norm(p1 - p2)
    return diameter

def apply_windowing(img_array: np.ndarray,
                    window_level: int, 
                    window_width: int
                    ) -> np.ndarray:
    '''
    Window an image based on a window width (width of range of values to use) and a window level (where to center a window level). Otherwise known as clipping or clamping in image processing.
    
    Parameters
    ----------
    img_array: np.ndarray, 
        The image to be windowed 
    window_level: int
        Where to center the range defined in window_width
    window_width: int 
        How wide the of a range to include, centered on the level.  

    Returns 
    ----------
    windowed_img: np.ndarray
        The processed image with values clamped at the upper and lower value
    '''
    #Calculate upper and lower clamp values
    upper_val = window_level + window_width / 2 
    lower_val = window_level - window_width / 2 

    #Window image
    windowed_img = np.clip(img_array, lower_val, upper_val)

    return windowed_img 

def choose_windowing(disease_location: str): 
    '''  
    Determine which window level and width to use based on 
    the current dataset being used. 

    Parameters
    ----------
    disease_location: str
        The location of the disease corresponding to the window to be applied (e.g. Lung, Abodmen, Mediastinum, etc.)
    
    Returns 
    ----------
    window_level: int 
        The centering value of the window 
    window_width: int
        The width of the window
    '''
    match disease_location: 
        case 'abdomen': 
            window_level = 50
            window_width = 400
        case 'headneck': 
            window_level = 50
            window_width = 400
        case 'lung': 
            window_level = -600
            window_width = 1500
        case 'mediastinum': 
            window_level = 40
            window_width = 400
        case _: 
            raise ValueError(f"Invalid window name: {disease_location}. Please check spelling or add to this function with the correct window and level")

    return window_level, window_width

def get_line_from_recist(recist_coords: np.array, 
                         slice_number: int, 
                         img_size: np.array):
    '''
    From the RECIST measurement coordinates, generate a line connecting both coordinates on the correct slice and return an np.ndarray the same shape as the image.
    Output to be compatible with the ['recist'] array of the .npz files needed for MedSAM2-RECIST.

    Parameters
    ----------
    recist_coords: array
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement 
    slice_number: int
        The slice that the measurement was taken on
    img_size: np.array
        The x, y, and z size of the image in [z_space, x_space, y_space] format
    
    Returns
    ----------
    recist_arr: np.ndarray
        A binary array of the same shape as the image with the pixels of the line = 1
    '''
    # Check to see if RECIST coordinates are in string form and if so, convert to list 
    if type(recist_coords) == str: 
        just_coords = recist_coords.strip("[]")
        recist_coords = list(map(float, just_coords.split()))
        print(recist_coords)
        
    #Generate an array in the same size as the image filled with all zeros 
    recist_arr = np.zeros((img_size[0], img_size[1], img_size[2]), dtype = int)
    
    #Round the coordinate values to their nearest integers 
    coords_round = np.rint(recist_coords).astype(int)

    #Draw line using coordinates 
    rr, cc = line(coords_round[0], coords_round[1], coords_round[2], coords_round[3])

    #Put line into the correct slice in the RECIST array of all zeros 
    recist_arr[slice_number][cc, rr] = 1

    return recist_arr

def find_first_last_slice(mask: np.ndarray): 
    '''
    Based on a 3D mask array, get the first and last slice within the array that has masked values. 
    Assumes (z, x, y) coordinate order.

    Parameters
    ----------
    mask: np.ndarray
        3D mask array 
    
    Returns 
    ----------
    first_slice: int 
        The index where the first slice of the mask is 
    last_slice: int 
        The index where the last slice of the mask is 
    '''
    axes = tuple([i for i in range(mask.ndim) if i != 0])

    slices = mask.any(axis = axes) 

    nonzero_indices = np.where(slices)[0]

    first_slice = np.amin(nonzero_indices)
    last_slice = np.amax(nonzero_indices) 

    return first_slice, last_slice

def slice_visual(image, 
                     mask_preds, 
                     gt_masks, 
                     slice_idx: int, 
                     full_savepath: Path, 
                     text_prompts: dict = None): 
    '''
    Adjusted visualization from the inference_example_3D.ipynb example notebook that is in the BiomedParse repo. Saves a figure 
    showing the middle slice of the original image, the ground truth mask overlayed, and the predicted mask overlayed along 
    with the text prompt used to create the mask as the legend. 

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    text_prompts: dict = None
        Contains all of the prompts used to create the predicted masks (for now only one text prompt in dict) 
    slice_idx: int 
        The slice index to view
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    '''
    slice_id = slice_idx
    slice_image = image[slice_id]
    slice_mask = mask_preds[slice_id]
    slice_gt   = gt_masks[slice_id]

    # 1) Compute the mapping
    unique_ids = np.unique(np.concatenate((slice_mask, slice_gt)))
    id_map = {orig_id: new_i for new_i, orig_id in enumerate(unique_ids)}

    slice_mask_mapped = np.vectorize(id_map.get)(slice_mask)
    slice_gt_mapped   = np.vectorize(id_map.get)(slice_gt)

    slice_mask_mapped = np.ma.masked_where(slice_mask_mapped == 0, slice_mask_mapped)
    slice_gt_mapped = np.ma.masked_where(slice_gt_mapped == 0, slice_gt_mapped)

    # 2) Which IDs to show (drop background=0)
    mask_ids = unique_ids[1:]

    # 3) Labels for legend (only if there is a text prompt)
    if text_prompts is not None:
        legends = [text_prompts[str(i)] for i in mask_ids if str(i) in text_prompts]
        mask_ids = [i for i in mask_ids if str(i) in text_prompts]

    # 4) Build a *discrete* colormap of size len(unique_ids)
    #    so that cmap(k) gives exactly the k-th color.
    cmap = plt.get_cmap('tab20', len(unique_ids)-1)

    # 5) Create handles using integer lookup into the discrete cmap
    if text_prompts is not None: 
        handles = [
            mpatches.Patch(color=cmap(id_map[i]-1), label=txt)
            for i, txt in zip(mask_ids, legends)
        ]

    # 6) Plot
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    axes[0].imshow(slice_image, cmap="gray")
    axes[0].set_title("Original Image Slice")
    axes[0].axis("off")

    axes[1].imshow(slice_image, cmap='gray')
    axes[1].imshow(slice_gt_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[1].set_title("Ground Truth Masks")
    axes[1].axis("off")

    axes[2].imshow(slice_image, cmap='gray')
    axes[2].imshow(slice_mask_mapped, cmap=cmap, interpolation='nearest', alpha = 0.6)
    axes[2].set_title("Predicted Masks")
    axes[2].axis("off")

    # 7) Shared legend below, single column
    if text_prompts is not None: 
        fig.legend(handles, legends, loc='lower center', bbox_to_anchor=(0.29, -0.15), ncol=1, frameon=False, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0)
    
    fig.savefig(full_savepath, bbox_inches = 'tight')

def pos_neg_true_visual(image, 
                        mask_preds, 
                        gt_masks, 
                        full_savepath: Path): 
    '''
    Visualization of the selected slices based on the ground truth, showing the true positive, false positive, and false
    negative areas within these slices.

    Parameters
    ----------
    image: 
        The array containing the original image data (same shape as mask_preds and gt_masks)
    mask_preds: 
        The array containing the predicted mask values (same shape as the image and gt_masks) 
    gt_masks: 
        The array containing the ground truth mask values (same shape as the image and mask_preds) 
    full_savepath: Path
        Should contain where to save the path and what to call the file outputted
    '''
    # Make the predicted mask a different number to represent a different colour 
    mask_alt = mask_preds * 2 

    # Add masks together so that false negative is 1, false positive is 2, and true positive is 3 
    comb_masks = mask_alt + gt_masks
    comb_masks = np.ma.masked_where(comb_masks == 0, comb_masks)

    # Find the first and last slices that have mask in them 
    gt_min, gt_max = find_first_last_slice(gt_masks)

    num_nonzero_slices = gt_max - gt_min + 1 # need to add one to get true number. e.g. slices 0 - 5 have non zero (inclusive), true answer is 6 slices, but subtraction only will yield 5
    # Check to see if there are more than 5 slices within the ground truth mask and adjust the subplot information accordingly 
    if num_nonzero_slices < 5: 
        subplot_slices = num_nonzero_slices
        slices_to_plot = range(gt_min, gt_max + 1)
    else: 
        subplot_slices = 5
        slices_to_plot = [gt_min, gt_min + math.floor(num_nonzero_slices/4), gt_min + math.floor(num_nonzero_slices/2), gt_min + math.floor(num_nonzero_slices * 3 / 4), gt_max]
    
    print(slices_to_plot)
    fig, axes = plt.subplots(1, subplot_slices, figsize = (15, 3)) 

    # Create colour map for mask 
    colours = ['red', 'green', 'blue']
    boundaries = [1, 2, 3, 4]
    cmap = mcolors.ListedColormap(colours)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    # Make legend info 
    legend_elem = [mpatches.Patch(color = 'red', label = 'False Negative'), 
                mpatches.Patch(color = 'green', label = 'False Positive'), 
                mpatches.Patch(color = 'blue', label = 'True Positive')]
    counter = 0
    for i in slices_to_plot:
    
        axes[counter].imshow(image[i], cmap = 'gray') 
        axes[counter].imshow(comb_masks[i], cmap = cmap, norm = norm, interpolation = 'nearest', alpha = 0.6)
        axes[counter].axis("off")
        axes[counter].text(0.5, -0.1, f"Slice {i}", size = 11, ha ="center", transform = axes[counter].transAxes)
        counter += 1

        plt.tight_layout()

    fig.legend(handles = legend_elem, loc = 'lower right', bbox_to_anchor=(0.67, -0.15), ncol=3, frameon=False, fontsize=11)
    fig.savefig(full_savepath, bbox_inches = 'tight')

## Metrics and Visualization ##
def list_nonzero_seg_slices(seg: np.ndarray): 
    '''  
    From a given 3D segmentation array, list the slices that have nonzero values (mask) in them.

    Parameters
    ----------
    seg: np.ndarray
        A 3D array containing a mask (ground truth, predicted, etc.)
    
    Returns
    ----------
    nonzero_slices: list 
        Contains all of the slice numbers where there are nonzero values
    '''
    nonzero_slices = []
    for slice_idx in range(seg.shape[0]): 
        if np.count_nonzero(seg[slice_idx]) > 0: 
            nonzero_slices.append(slice_idx)
    return nonzero_slices

def get_hist_data(seg: np.ndarray): 
    '''  
    Get the nonzero pixel counts for each slice into dictionary form. Counts to be used for histogram plot.

    Parameters
    ----------
    seg: np.ndarray 
        A 3D array containing a mask (ground truth, predicted, etc.) 
    
    Returns 
    ----------
    pix_slice_dict: dict
        A dictionary with the slice number as the keys and the nonzero pixel count as the corresponding values
    '''
    pix_slice_dict = dict() 
    for slice_idx in range(seg.shape[0]): 
        pix_count = np.count_nonzero(seg[slice_idx]) 
        pix_slice_dict[slice_idx] = pix_count

    return pix_slice_dict

def get_hist_data_df(seg): 
    '''  
    Get the nonzero pixel counts for each slice into dataframe form. To be used for the density plot to be 
    compatible with seaborn. 

    Parameters
    ----------
    seg: np.ndarray
        A 3D array containing a mask (ground truth, predicted, etc.)

    Returns
    ----------
    pix_slice_df: pd.DataFrame
        A dataframe containing the slice number and the corresponding count in their respective columns
    '''
    pix_slice_dict = {
        'slice_num': [],
        'pix_count': []
    }
    for slice_idx in range(seg.shape[0]): 
        pix_count = np.count_nonzero(seg[slice_idx]) 
        pix_slice_dict['slice_num'].append(slice_idx) 
        pix_slice_dict['pix_count'].append(pix_count) 

    pix_slice_df = pd.DataFrame(pix_slice_dict)

    return pix_slice_df

def plot_hist(gt_mask: np.ndarray, 
              pred_mask: np.ndarray, 
              full_savepath: Path, 
              prompt: str): 
    '''  
    Plot a histogram of the number of mask pixels in each of the slices. To give a quick
    view of where the model is segmenting vs. where the ground truth mask is. For
    a visual check to see how well the coordinates localize the segmentation to a 
    specific point. 

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    prompt: str
        The prompt (text, RECIST, etc.) used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get histogram data of number of pixels in each slice 
    pred_hist_data = get_hist_data(pred_mask)
    gt_hist_data = get_hist_data(gt_mask)

    # Get count data and slice data in a form that is compatible with histogram
    pred_val, pred_weight = zip(*[(key, val) for key, val in pred_hist_data.items()])
    gt_val, gt_weight = zip(*[(key, val) for key, val in gt_hist_data.items()])

    # Create figure 
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    ax[0].hist(pred_val, weights = pred_weight, bins = gt_mask.shape[0]-1) 
    ax[0].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[0].set_title('Predicted Mask')
    ax[0].set_ylabel('Pixel Count')
    ax[0].set_xlabel('Slice Number')
    ax[1].hist(gt_val, weights = gt_weight, bins = gt_mask.shape[0]-1)
    ax[1].set_ylim(0, max(max(gt_hist_data.values()), max(pred_hist_data.values())))
    ax[1].set_title('Ground Truth Mask')
    ax[1].set_xlabel('Slice Number')
    plt.figtext(0.5, -0.05, "Prompt: " + prompt, ha='center', va='top')

    # Save figure 
    fig.savefig(full_savepath, bbox_inches = 'tight')

def plot_density(gt_mask: np.ndarray, 
              pred_mask: np.ndarray,
              prompt: str, 
              full_savepath: Path): 
    '''  
    Make a density plot to showcase where most of the segmented pixels
    are located. Similar to the histogram, but this shows the density
    information of the ground truth and the predicted mask overlayed 
    on the same plot.

    Parameters
    ----------
    gt_mask: np.ndarray
        A 3D array containing the ground truth mask segmentation 
    pred_mask: np.ndarray 
        A 3D array containing the predicted mask segmentation 
    text_prompt: str
        The text prompt used to generate the predicted mask 
    full_savepath: Path
        A path containing both the location for saving and the 
        name of the file to be saved.
    '''
    # Get data into a dataframe to be compatible with seaborn 
    gt_data = get_hist_data_df(gt_mask)
    pred_data = get_hist_data_df(pred_mask)

    # Add labels to each dataframe to identify which are ground truth 
    # and which are predicted
    gt_data["Mask Type"] = "Ground Truth"
    pred_data["Mask Type"] = "Predicted"

    # Combine dataframes into one for plotting
    all_data = pd.concat([gt_data, pred_data], axis = 0).reset_index(drop = True)
    # Check if the data has NaNs and if so, return and don't make the graph (it'll return an error otherwise)
    if all_data.isnull().values.any(): 
        print(f"NaNs present in the histogram data dataframe for: {full_savepath}. Cannot create density plot.")
        return 0

    # Create figure 
    try:
        plot = sns.displot(data = all_data, 
                x = "slice_num", 
                weights = "pix_count", 
                hue = "Mask Type",
                kind = "kde", 
                fill = True
                )
        plot.set(xlim=(0, gt_mask.shape[0]), xlabel = "Slice Number")

        plt.figtext(0.5, -0.05, "Prompt: " + str(prompt), ha='center', va='top')

        # Save figure
        plt.savefig(full_savepath, bbox_inches = 'tight')
    except ValueError:
        print(f"NaNs present during the calculation of density for: {full_savepath}. Cannot create density plot.")
        return 0

def calc_metrics(pred_mask: np.ndarray, 
                 gt_mask: np.ndarray, 
                 spacing: np.ndarray, 
                 filename: str): 
    '''
    Calculate performance metrics based on the predicted and ground truth masks and save into a dataframe. 

    Parameters
    ----------
    pred_mask: np.ndarray
        The mask that was predicted by the model. 
    gt_mask: np.ndarray
        The ground truth segmentation array. 
    spacing: np.ndarray
        The spacing associated with the ground truth mask. 
    filename: str 
        The name of the npz file that is being evaluated.
    
    Returns
    ----------
    metric_df: pd.DataFrame
        Contains the evaluation performance. 
    '''
    #Initialize evaluator 
    metric_eval = Evaluator() 
    metric_dict = metric_eval(preds = pred_mask, 
                              targets = gt_mask, 
                              spacing = spacing 
                              )
    
    metric_df = pd.DataFrame(metric_dict, index = [0])

    #Add columns for the range of segmentation values (both ground truth and predicted)
    first_gts, last_gts = find_first_last_slice(gt_mask)
    try: #If no mask was predicted, this will throw an error
        first_pred, last_pred = find_first_last_slice(pred_mask) 
    except ValueError: 
        print(f"Empty predicted segmentation for file: {filename}.")
        first_pred = 0
        last_pred = 0

    # Get the list of all slices that have segmentation in them for each mask 
    mask_pred_list = list_nonzero_seg_slices(pred_mask)
    gt_list = list_nonzero_seg_slices(gt_mask) 

    metric_df['GTSliceList'] = [gt_list]
    metric_df['PredSliceList'] = [mask_pred_list]

    gts_range = [first_gts, last_gts] 
    pred_range = [first_pred, last_pred] 

    metric_df['GTSliceRange'] = [gts_range]
    metric_df['PredSliceRange'] = [pred_range]
    metric_df['filename'] = filename # To ensure we can map the results back to the segmentations 

    # Get slice interval IoU 
    metric_df['SliceIoU'] = len(list(set.intersection(set(gt_list), set(mask_pred_list)))) / len(list(set.union(set(gt_list), set(mask_pred_list))))
    metric_df['MaskUniqueSlice'] = [list(set(mask_pred_list) - set(gt_list))] # Only slices that are in predicted mask and are not in ground truth mask
    metric_df['GTUniqueSlice'] = [list(set(gt_list) - set(mask_pred_list))] # Opposite of the line above
    
    return metric_df

@torch.inference_mode()
def infer_3d(img_savepath: Path, 
             img_array: np.ndarray,
             gts_array: np.ndarray, 
             spacing: np.ndarray, 
             recist_coords: np.ndarray, 
             slice_num: int):
    """ 
    Modification of the original infer_3d but instead will take in individual NIFTI
    formatted files and create all the arrays accessed in the npz file on the fly. 

    Parameters
    ----------
    img_savepath: Path
        The path to save the visualizations to.  
    img_array: np.ndarray
        Contains the imaging data after preprocessing has been completed
    gts_array: np.ndarray
        Contains the ground truth segmentation data after preprocessing has been complete 
    spacing: np.ndarray
        The 3D spacing information for the image/segmentation 
    recist_coords: np.ndarray 
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement
    slice_num: int 
        The slice that the recist_coords are taken on (after preprocessing)

    Returns 
    -----------
    pred_array: np.ndarray
        Contains the predicted segmentation
    duration: 
        How long inference took for this sample
    """
    start_time = time.time()
    img_3D_ori = img_array
    gts_3D_ori = gts_array
    assert np.max(img_3D_ori) < 256, f'input data should be in range [0, 255]'
    img_size = img_3D_ori.shape
    recist = get_line_from_recist(recist_coords = recist_coords, 
                                  slice_number = slice_num, 
                                  img_size = img_size)
    segs_3D = np.zeros(img_3D_ori.shape, dtype=np.uint8)
    unique_labs =np.unique(recist)
    unique_labs = unique_labs[unique_labs != 0]
    print(f'unique_labs: {unique_labs}')

    video_height = img_3D_ori.shape[1]
    video_width = img_3D_ori.shape[2]

    if video_height != 512 or video_width != 512:
        img_resized = resize_grayscale_to_rgb_and_resize(img_3D_ori, 512)  #d, 3, 512, 512
    else:
        img_resized = img_3D_ori[:,None].repeat(3, axis=1) # d, 3, 512, 512
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized)
    ### NOTE: Ask Jun Ma about why this is hardcoded ###
    img_mean=(0.485, 0.456, 0.406)
    img_std=(0.229, 0.224, 0.225)
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    img_resized -= img_mean
    img_resized /= img_std

    boxes_3D_ori = []
    for j, ulab in enumerate(unique_labs):
        gts = (gts_3D_ori == ulab)*ulab
        recist_per_lab = (recist == ulab)*ulab
        if len(np.unique(recist_per_lab)) == 0:
            print(f'no recist for label {ulab} for current sample, skipping...')
            continue
       
        z_mids = []
        print(f'shape of image: {img_3D_ori.shape}')
        idx = ulab
        z_indices = np.where(recist == ulab)[0]
        z_indices = np.unique(z_indices)
        assert len(z_indices) == 1, f'expected only one z index for recist=1, but got {z_indices}'
        z_mid = z_indices[0]
        diameter = get_diameter(recist_per_lab[z_mid])
        spacing_z = spacing[0]
        spacing_xy = (spacing[1] + spacing[2]) / 2
        multiplier = spacing_xy / spacing_z 
        diameter /= multiplier
        z_min_per_lab = max(0, z_mid - int(diameter / 2))
        z_max_per_lab = min(img_3D_ori.shape[0] - 1, z_mid + int(diameter / 2))
        gts = gts[z_min_per_lab:z_max_per_lab+1]
        recist_per_lab = recist_per_lab[z_min_per_lab:z_max_per_lab+1]
        img_resized_cropped = img_resized[z_min_per_lab:z_max_per_lab+1]

        z_mid_orig = z_mid
        z_mid = z_mid_orig - z_min_per_lab

        z_mids.append(z_mid_orig)
        with torch.inference_mode():
            inference_state = predictor.init_state(img_resized_cropped, video_height, video_width)
            if propagate_with_box:
                print(f'propagate with box')
                box_2d = get_diameter_bbox(recist_per_lab[z_mid], shift=shift)
                boxes_3D_ori.append([box_2d[0], box_2d[1], z_mid_orig, box_2d[2], box_2d[3], z_mid_orig])
                #box_2d = box3d[[0,1,3,4]]
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z_mid,
                                                    obj_id=1,
                                                    box=box_2d,
                                                )
                mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)
                
            else: 
                print('propagate with point')
                if args.sample_points == 'from_box':
                    box_2d = get_diameter_bbox(recist_per_lab[z_mid], shift=0)
                    boxes_3D_ori.append([box_2d[0], box_2d[1], z_mid, box_2d[2], box_2d[3], z_mid])
                    # sample points in the box
                    points = sample_points_in_bbox_grid(box_2d, n=9)
                elif args.sample_points == 'from_recist_n':
                    points = get_n_points_from_recist(recist_per_lab[z_mid], n=5)
                elif args.sample_points == 'from_recist_center':
                    points = get_center_from_recist(recist_per_lab[z_mid])
                elif args.sample_points == 'from_recist_3':
                    points = get_center_and_endpoints_from_recist(recist_per_lab[z_mid])
                else:
                    raise ValueError(f'unknown sample_points: {args.sample_points}')
                
                labels = np.ones(len(points))
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                                    inference_state=inference_state,
                                                    frame_idx=z_mid,
                                                    obj_id=1,
                                                    points=points,
                                                    labels=labels,
                                                )
                mask_prompt = (out_mask_logits[0] > 0.0).squeeze(0).cpu().numpy().astype(np.uint8)   

            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)
            segs_3D[z_mid_orig, ((masks[0] > 0.0).cpu().numpy())[0]] = idx
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=z_mid, reverse=False):
                segs_3D[(z_min_per_lab + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx

            # reverse process, delete old memory and initialize new predictor
            predictor.reset_state(inference_state)
            inference_state = predictor.init_state(img_resized_cropped, video_height, video_width)
            frame_idx, object_ids, masks = predictor.add_new_mask(inference_state, frame_idx=z_mid, obj_id=1, mask=mask_prompt)

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=z_mid, reverse=True):
                segs_3D[(z_min_per_lab + out_frame_idx), (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = idx

            predictor.reset_state(inference_state)
    # np.savez_compressed(join(pred_save_dir, npz_name), 
    #                     segs=segs_3D,
    #                     gts=gts_3D_ori,
    #                     boxes=np.stack(boxes_3D_ori), # num_boxes, 6
    #                     spacing=spacing,
    #                     )  
    end_time = time.time()
    duration = end_time - start_time
    print(f'Finished processing current sample in {duration:.2f} seconds')

    ## Commenting out for now since this is going to return and we can save after the fact. 
    # if save_nifti:
    #     sitk_image = sitk.GetImageFromArray(img_3D_ori)
    #     # add spacing
    #     sitk_image.SetSpacing(spacing)
    #     sitk.WriteImage(sitk_image, os.path.join(nifti_path, npz_name.replace('.npz', '_imgs.nii.gz')))
    #     sitk_gt = sitk.GetImageFromArray(gts_3D_ori)
    #     sitk_gt.SetSpacing(spacing)
    #     sitk.WriteImage(sitk_gt, os.path.join(nifti_path, npz_name.replace('.npz', '_gts.nii.gz')))
    #     sitk_seg = sitk.GetImageFromArray(segs_3D)
    #     sitk_seg.SetSpacing(spacing)
    #     sitk.WriteImage(sitk_seg, os.path.join(nifti_path, npz_name.replace('.npz', '_segs.nii.gz')))

    if save_overlay:
        idx = random.sample(z_mids,1)[0] 
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D_ori[idx], cmap='gray')
        ax[1].imshow(img_3D_ori[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("SAM2 Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for box_idx, label_id in enumerate(unique_labs):
            if np.sum(segs_3D[idx]==label_id) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = boxes_3D_ori[box_idx]
                box_viz = np.array([x_min, y_min, x_max, y_max])
                if idx >= z_min and idx <= z_max:
                    show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs_3D[idx]==label_id, ax[1], mask_color=color)
                if idx >= z_min and idx <= z_max:
                    show_box(box_viz, ax[0], edgecolor=color)
                show_mask(gts_3D_ori[idx]==label_id, ax[0], mask_color=color)
            else:
                print(f'no mask for file current file {label_id=} {box_idx=}')

        plt.tight_layout()
        plt.savefig(join(png_save_dir, str(img_savepath).removesuffix(".nii.gz") + '_overlay' + str(idx)+  '.png'), dpi=300)
        plt.close()

    return segs_3D, duration

def run_infer_metric_vis(img_path: Path, 
             gts_path: Path,
             disease_loc: str, 
             lesion_location: str,
             recist_coords: np.ndarray, 
             slice_num: int,
             visualization: bool = True): 
    """  
    Run inference, calculate metrics, and produce visualizations for one sample. 

    Parameters 
    ----------
    img_path: Path
        Where the CT image is located
    gts_path: Path 
        Where the ground truth segmentation is located
    lesion_location: str 
        Where the lesion is located (to be used for systematic windowing)
    recist_coords: np.ndarray 
        A list of coordinates in [x1, y1, x2, y2] format that defines the RECIST measurement
    slice_num: int 
        The slice that the recist_coords are taken on (after preprocessing)
    visualization: bool
        Whether or not to produce visualizations. Default is True. 
    Returns 
    ----------
    durations: 
        The time it took to complete one inference. 
    metric_df: 
        The evaluation results from the current run.
    """
    ## Load image and ground truth segmentation ## 
    img = sitk.ReadImage(Path("data/procdata") / disease_loc / img_path) 
    gts = sitk.ReadImage(Path("data/procdata") / disease_loc / gts_path) 

    img_array = sitk.GetArrayFromImage(img) 
    gts_array = sitk.GetArrayFromImage(gts)

    ## Window image ## 
    win_lvl, win_width = choose_windowing(disease_location = lesion_location)

    print(f"Window level chosen: {win_lvl}. Window width: {win_width}. Disease location: {lesion_location}")
    img_win = apply_windowing(img_array = img_array, 
                              window_level = win_lvl, 
                              window_width = win_width)
    
    # Get appropriate save path for the images and visualizations (if applicable)
    base_savepath = Path("data/results") / disease_loc / "/".join(gts_path.split("/")[:-1]).replace("images", "predictions_MS2R")
    mask_name = gts_path.split("/")[-1].replace(".nii.gz", "_pred.nii.gz")
    image_savepath = base_savepath / mask_name
    visual_savepath = base_savepath / 'visualization'

    # Run inference 
    pred_seg, infer_dur = infer_3d(img_savepath = image_savepath, 
                                   img_array = img_win, 
                                   gts_array = gts_array,
                                   spacing = img.GetSpacing(), 
                                   recist_coords = recist_coords, 
                                   slice_num = slice_num)

    # Save predicted segmentation 
    pred_seg_img = sitk.GetImageFromArray(pred_seg)
    pred_seg_img.SetSpacing(img.GetSpacing())
    pred_seg_img.SetOrigin(img.GetOrigin())
    pred_seg_img.SetDirection(img.GetDirection())

    sitk.WriteImage(pred_seg_img, image_savepath) 
    
    # Calculate metrics 
    metrics_df = calc_metrics(pred_mask = pred_seg, 
                              gt_mask = gts_array, 
                              spacing = img.GetSpacing(), 
                              filename = base_savepath)
    durations = pd.DataFrame({'image': str(gts_path), 
                            'duration': infer_dur}, index = [0])
    
    # Export visualizations (if applicable) 
    if visualization: 
        if not visual_savepath.exists(): 
            visual_savepath.mkdir(parents = True, exist_ok = True)
        
        # Largest slice plot 
        large_slice_savepath = visual_savepath / mask_name.replace(".nii.gz", "_largeslice.png")

        slice_visual(image = img_win, 
                    mask_preds = pred_seg, 
                    gt_masks = gts_array, 
                    slice_idx = slice_num, 
                    full_savepath = large_slice_savepath)
        
        # True positive, false positive, false negative plot 
        pos_neg_savepath = visual_savepath / mask_name.replace(".nii.gz", "_posneg.png") 

        pos_neg_true_visual(image = img_win, 
                            mask_preds = pred_seg, 
                            gt_masks = gts_array, 
                            full_savepath = pos_neg_savepath)
        
        # Pixel count histogram 
        pix_hist_savepath = visual_savepath / mask_name.replace(".nii.gz", "_pixhist.png")

        plot_hist(gt_mask = gts_array, 
                  pred_mask = pred_seg, 
                  prompt = str(recist_coords), 
                  full_savepath = pix_hist_savepath)
        
        # Pixel-slice density plot 
        pix_dens_savepath = visual_savepath / mask_name.replace(".nii.gz", "_pixdens.png") 

        plot_density(gt_mask = gts_array, 
                     pred_mask = pred_seg, 
                     prompt = str(recist_coords), 
                     full_savepath = pix_dens_savepath)
    
    return metrics_df, durations

if __name__ == '__main__':
    # img_npz_files = sorted(glob(join(imgs_path, '*.npz'), recursive=True))
    # img_npz_files = sorted(img_npz_files)
    # print(f'number of files to process: {len(img_npz_files)}')

    # dice_dict = OrderedDict()
    # dice_dict['image'] = []

    ## Load in index csv 
    index_df = pd.read_csv(index_csv)

    # Inference, evaluation, and visuals
    pred_results, durations = zip(*Parallel(n_jobs = num_workers)(delayed(run_infer_metric_vis)(img_path = row['image_path'], 
             gts_path = row['mask_path'],
             disease_loc = disease_location, 
             lesion_location = row['lesion_location'],
             recist_coords = row['annotation_coords'], 
             slice_num = row['largest_slice_index'],
             ) for _, row in tqdm(index_df.iterrows(), total = index_df.shape[0])))

    # Save all evaluation results 
    out_path = Path("data/results") / disease_location / "/".join(index_df['image_path'].iloc[0].replace("images", "predictions_MS2R").split("/")[:3])
    if not out_path.exists(): 
        out_path.mkdir(parents = True, exist_ok = True)

    for evaluate in pred_results: 
        if 'all_metrics_df' not in locals(): 
            all_metrics_df = evaluate
        else: 
            all_metrics_df = pd.concat([all_metrics_df, evaluate], ignore_index = True).reset_index(drop = True)

    for dur in durations: 
        if 'duration_df' not in locals(): 
            duration_df = dur
        else: 
            duration_df = pd.concat([duration_df, dur], ignore_index = True).reset_index(drop = True)

    all_metrics_df.to_csv(Path(out_path) / 'metric_eval.csv', index = False)
    duration_df.to_csv(Path(out_path) / 'inference_times.csv', index = False)