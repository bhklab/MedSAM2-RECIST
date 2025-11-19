import SimpleITK as sitk
import numpy as np
from pathlib import Path
import pandas as pd
from skimage.draw import line
from joblib import Parallel, delayed
import click
from tqdm import tqdm


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
    #Generate an array in the same size as the image filled with all zeros 
    recist_arr = np.zeros((img_size[0], img_size[1], img_size[2]), dtype = int)
    
    #Round the coordinate values to their nearest integers 
    coords_round = np.rint(recist_coords).astype(int)

    #Draw line using coordinates 
    rr, cc = line(coords_round[0], coords_round[1], coords_round[2], coords_round[3])

    #Put line into the correct slice in the RECIST array of all zeros 
    recist_arr[slice_number][cc, rr] = 1

    return recist_arr


def nifti_to_medsam_npz(image_path: Path,
                        mask_path: Path,
                        npz_path: Path,
                        window_level: int = 0,
                        window_width: int = -1) -> None:
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
    nifti_image = sitk.ReadImage(image_path, outputPixelType=sitk.sitkInt16)
    image_array = sitk.GetArrayFromImage(nifti_image)

    # Apply windowing to the image only
    if window_width != -1:
        image_array = apply_windowing(image_array, window_level, window_width)
    
    spacing = nifti_image.GetSpacing()
    direction = nifti_image.GetDirection()
    origin = nifti_image.GetOrigin()

    # Load NIfTI mask using SimpleITK
    nifti_mask = sitk.ReadImage(mask_path, outputPixelType=sitk.sitkUInt8)
    mask_array = sitk.GetArrayFromImage(nifti_mask)

    # Make RECIST annotation part of the input 
    # First make a 3D bounding box
    x_min, y_min, z_min, x_max, y_max, z_max = mask3D_to_bbox(mask_array, mask_path)
    # Get the z-axis coordinate for the middle slice
    z_mid = (z_min + z_max) // 2

    # make an empty 2D array for the mid slice
    rerecist_arr = get_line_from_recist(np.array([x_min, y_min, x_max, y_max]),
                                        slice_number=z_mid,
                                        img_size=image_array.shape)

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # Save as NPZ file
    np.savez_compressed(npz_path, 
                        imgs=image_array,
                        gts=mask_array,
                        recist=rerecist_arr,
                        spacing=spacing,
                        direction=direction,
                        origin=origin)

    return rerecist_arr



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


def process_one_sample(sample_metadata: str,
                       image_path: Path,
                       out_path: Path,
                       window_level: int = 0,
                       window_width: int = -1
                       ):
    sample_id = sample_metadata['SampleID'].iloc[0]
    
    img_metadata = sample_metadata[sample_metadata['class'] == 'Scan']
    if img_metadata.empty:
        print(f"{sample_id} has no Scan metadata. Skipping sample.")
        return
    
    mask_metadata = sample_metadata[sample_metadata['class'] == 'Mask']
    if mask_metadata.empty:
        print(f"{sample_id} has no Mask metadata. Skipping sample.")
        return
    mask_path = image_path / mask_metadata['filepath'].values[0]
    rtstruct_id = Path(mask_path).parent.stem
    print(rtstruct_id)

    nifti_to_medsam_npz(image_path = image_path / img_metadata['filepath'].values[0],
                        mask_path = mask_path,
                        npz_path = out_path / f"{sample_id}_{rtstruct_id}.npz",
                        window_level = window_level,
                        window_width = window_width)



@click.command()
@click.argument('image_path', type=click.Path(exists=True, readable=True))
@click.argument('out_path', type=click.Path())
@click.option('--window_level', type=click.INT, default = 0, help="Level to use if windowing is to be applied to the images.")
@click.option('--window_width', type=click.INT, default = -1, help="Width to use if windowing is to be applied to the images. If set to -1, no windowing will be applied.")
@click.option('--parallel', type=click.BOOL, default = False, help="Whether to run the conversion in parallel.")
def process_mit_images(image_path: Path,
                       out_path: Path,
                       window_level: int = 0,
                       window_width: int = -1,
                       parallel: bool = False):
    """Process images in a Med-ImageTools autopipeline index-simple file to convert to npz
    Parameters
    ----------
    image_path: Path
        Path to directory containing MIT index and sample directories 
    out_path: Path
        Path to save the npz files to.
    window_level: int = 0
        Level to use if windowing is to be applied to the images.
    window_width: int = -1
        Width to use if windowing is to be applied to the images. If set to -1, no windowing will be applied.
    parallel: bool = False
        Whether to run the image conversion in parallel.
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)

    if isinstance(out_path, str):
        out_path = Path(out_path)

    # load mit index file
    mit_index_df = pd.read_csv(image_path / f"{image_path.stem}_index-simple.csv")

    if 'SampleID' not in mit_index_df.columns:
        mit_index_df = insert_SampleID(mit_index_df)

    sample_ids = sorted(mit_index_df['SampleID'].unique())

    if parallel:
        Parallel()(
            delayed(process_one_sample)(
                sample_metadata=mit_index_df[mit_index_df['SampleID'] == sample_id],
                image_path=image_path,
                out_path=out_path,
                window_level=window_level,
                window_width=window_width
            )
            for sample_id in tqdm(
                sample_ids,
                desc="Converting nifti to npz for MedSAM2-RECIST inference",
                total=len(sample_ids)
            )
        )

    else:
        for sample_id in tqdm(
            sample_ids,
            desc="Converting nifti to npz for MedSAM2-RECIST inference",
            total=len(sample_ids)
        ):
            process_one_sample(sample_metadata=mit_index_df[mit_index_df['SampleID'] == sample_id],
                               image_path=image_path,
                               out_path=out_path,
                               window_level=window_level,
                               window_width=window_width)



if __name__ == "__main__":
    process_mit_images()