import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd


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



def pad_bbox(box:np.array,
             mask:np.array, 
             padding:int,
             spacing:np.array = None
             ) -> np.array:
    # Get full image dimensions to keep padding within image size
    # D, H, W (z, y, x)
    mask_shape = mask.shape

    if spacing is not None: # Use the actual image spacing to calculate the padding
        # Check that spacing can be applied to this mask's bounding box dimensions
        if len(spacing) < len(mask_shape):
            spacing = spacing[0, len(mask_shape)]
        elif len(spacing) > len(mask_shape):
            message = "Spacing for padding has more dimensions than the mask image."
            raise ValueError(message)

        # calculate the number of voxels to pad based on the actual image spacing
        padding = np.round(padding / spacing)
    else:
        # Convert padding into an array with the same length as image dimensions
        # This matches the behaviour of the spacing option
        padding = padding * np.ones(len(mask_shape))
    
    pad_x_min = max(0, box[0] - padding[0])
    pad_y_min = max(0, box[1] - padding[1])
    # Handling 2D bounding box
    if len(box) == 4:
        mask_H, mask_W = mask_shape[0], mask_shape[1]
        pad_x_max = min(mask_W, box[2] + padding[0])
        pad_y_max = min(mask_H, box[3] + padding[1])

        padded_box = np.array([pad_x_min, pad_y_min,
                               pad_x_max, pad_y_max])

    # Handling 3D bounding box
    if len(box) == 6:
        mask_D, mask_H, mask_W = mask_shape[0], mask_shape[1], mask_shape[2]
        pad_z_min = max(0, box[2] - padding[2])
        pad_x_max = min(mask_W, box[3] + padding[0])
        pad_y_max = min(mask_H, box[4] + padding[1]) 
        pad_z_max = min(mask_D, box[5] + padding[2])
        
        padded_box = np.array([pad_x_min, pad_y_min, pad_z_min,
                               pad_x_max, pad_y_max, pad_z_max])

    return padded_box.astype(int)



def mask2D_to_bbox(gt2D:np.array, 
                   mask_path:Path, 
                   padding:int | None = None,
                   spacing:np.array = None
                   ) -> np.array:
    try:
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        boxes = np.array([x_min, y_min, x_max, y_max])

        if padding:
            boxes = pad_bbox(box = boxes,
                             mask = gt2D,
                             padding = padding,
                             spacing = spacing)
        
        return boxes.astype(int)
    except Exception as e:
        raise Exception(f'error {e} with file {mask_path} and sum of gts is {gt2D.sum()}')



def mask3D_to_bbox(gt3D:np.array, 
                   mask_path:Path, 
                   padding:int | None = None,
                   spacing:np.array = None
                   ) -> np.array:
    try:
         # Find slices in the mask that have label voxels to get z boundaries
        z_indices, _, _ = np.where(gt3D > 0)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
    except Exception as e:
        raise Exception(f'error {e} with file {mask_path} and sum of gts is {gt3D.sum()}')
   
    # Find the centre slice of the mask
    z_mid = np.median(z_indices).astype(int)
    # Select out this slice from the array
    gt_mid = gt3D[z_mid]

    # Get the x and y mask boundaries using the 2D function
    box_2d = mask2D_to_bbox(gt_mid, mask_path)
    x_min, y_min, x_max, y_max = box_2d
    boxes3D = np.array([x_min, y_min, z_min, x_max, y_max, z_max])

    if padding:
        # Apply padding to bounding box if requested
        boxes3D = pad_bbox(box = boxes3D,
                           mask = gt3D,
                           padding = padding,
                           spacing = spacing)
    
    return boxes3D.astype(int)



def nifti_to_medsam_npz(image_path: Path,
                        mask_path: Path,
                        npz_path: Path,
                        # recist_path: Path | None = None,
                        window_level: int = 40,
                        window_width: int = 400) -> None:
    '''
    Convert a NIfTI image to a NumPy NPZ file after applying windowing.

    Parameters
    ----------
    image_path: Path
        Path to the image NIfTI file (e.g. CT or MRI).
    mask_path: Path
        Path to the mask NIfTI file. (e.g. RTSTRUCT or SEG)
    npz_path: Path
        Path to save the output NPZ file.
    recist_path: Path, optional
        Path to the RECIST annotations file (if available). If not provided, will generate RECIST annotation from mask.
    window_level: int, optional
        Window level for image windowing (default is 40).
    window_width: int, optional
        Window width for image windowing (default is 400).
    '''
    # Load NIfTI image using SimpleITK
    nifti_image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(nifti_image)

    # Apply windowing to the image only
    windowed_img = apply_windowing(image_array, window_level, window_width)

    # Load NIfTI mask using SimpleITK
    nifti_mask = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(nifti_mask)

    # Make RECIST annotation part of the input 
    # First make a 3D bounding box
    x_min, y_min, z_min, x_max, y_max, z_max = mask3D_to_bbox(mask_array, mask_path)
    # Get the axial bounding box from this
    bbox_2d = [x_min, y_min, x_max, y_max]
    # Get the z-axis coordinate for the middle slice
    z_mid = (z_min + z_max) // 2

    # TODO: draw RECIST line on the mid slice and save as part of the NPZ file





    # Save as NPZ file
    np.savez_compressed(npz_path, image=windowed_img)



def insert_SampleID(dataset_index:pd.DataFrame) -> pd.DataFrame:
    """Combine the PatientID and SampleNumber columns in an index to generate a SampleID
       SampleNumber is padded with 0s to make a length of four.
    """
    if "SampleID" in dataset_index.columns:
        print("SampleID column already exists in this dataset_index.")
        return dataset_index
    
    if "PatientID" not in dataset_index.columns:
        message = "PatientID column is missing in this dataset_index. Cannot make SampleID."
        raise KeyError(message)
    
    if "SampleNumber" not in dataset_index.columns:
        message = "SampleNumber column is missing in this dataset_index. Cannot make SampleID."
        raise KeyError(message)

    sample_id_series = dataset_index['PatientID'].astype(str) + "_" + dataset_index['SampleNumber'].astype(str).str.zfill(4)
    dataset_index.insert(0, "SampleID", sample_id_series)

    return dataset_index



def process_mit_images(image_path: Path,
                       window_level: int = 40,
                       window_width: int = 400) -> None:
    # load mit index file
    mit_index_df = pd.read_csv(image_path / f"{image_path.stem}_index-simple.csv")

    if 'SampleID' not in mit_index_df.columns:
        mit_index_df = insert_SampleID(mit_index_df)

    sample_ids = sorted(mit_index_df['SampleID'].tolist())

    for sample_id in sample_ids:
        images_metadata = mit_index_df[mit_index_df['SampleID'] == sample_id]

        image = sitk.Read


    return None


if __name__ == "__main__":
    process_mit_images(Path("data/rawdata/TCIA_RADCURE/images/mit_RADCURE_OCSCC"))