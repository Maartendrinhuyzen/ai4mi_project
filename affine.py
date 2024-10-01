import os
import argparse
import SimpleITK as sitk
import numpy as np

# Function to isolate the heart region (class 2) from the ground truth segmentation image
def isolate_heart(gt_image_path, heart_label=2):
    """
    Isolate the heart region from the ground truth segmentation image.
    
    Args:
    - gt_image_path: Path to the ground truth segmentation image.
    - heart_label: The label value that corresponds to the heart (default is 2).
    
    Returns:
    - isolated_heart: The binary image containing only the heart region.
    """
    # Load the ground truth segmentation image
    gt_image = sitk.ReadImage(gt_image_path)
    
    # Use a threshold filter to isolate the heart (label = heart_label)
    threshold_filter = sitk.BinaryThresholdImageFilter()
    threshold_filter.SetLowerThreshold(heart_label)
    threshold_filter.SetUpperThreshold(heart_label)
    threshold_filter.SetInsideValue(1)  # Set the heart region to 1
    threshold_filter.SetOutsideValue(0)  # Set the rest to 0
    
    # Apply the threshold to create a binary mask of the heart
    isolated_heart = threshold_filter.Execute(gt_image)
    
    return isolated_heart

# Function to perform affine registration
def register_affine(fixed_image_path, moving_image_path):
    """
    Perform affine registration between two images using SimpleITK.
    
    Args:
    - fixed_image_path: Path to the fixed image (reference image).
    - moving_image_path: Path to the moving image (image to be registered).
    
    Returns:
    - final_transform: The resulting affine transformation (SimpleITK Transform object).
    """
    # Read the fixed and moving images
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()

    # Set up the similarity metric
    registration_method.SetMetricAsMeanSquares()  # Use MeanSquares similarity metric

    # Set up the interpolator
    registration_method.SetInterpolator(sitk.sitkLinear)  # Linear interpolation for registration

    # Initialize the transformation (Affine)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,
        moving_image,
        sitk.AffineTransform(fixed_image.GetDimension()),  # Use affine transformation
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Set the initial transform
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Set up the optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-6,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Execute the registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Output optimizer and metric information
    print("Optimizer's stopping condition: {0}".format(registration_method.GetOptimizerStopConditionDescription()))
    print("Final metric value: {0}".format(registration_method.GetMetricValue()))

    # Return the final transform (affine transformation)
    return final_transform

# Function to load affine transformation from a file
def load_affine_transform(transform_file):
    """
    Loads an affine transformation from a file.

    Args:
    - transform_file: Path to the file containing the affine transformation.

    Returns:
    - transform: SimpleITK AffineTransform object.
    """
    transform = sitk.ReadTransform(transform_file)
    return transform

# Function to apply affine transformation to the heart (class 2)
def apply_affine_transform_to_heart(input_image_path, output_image_path, transform, heart_label=2):
    """
    Applies an affine transformation only to the heart (class 2) in a segmentation image.

    Args:
    - input_image_path: Path to the input segmentation image (GT.nii.gz).
    - output_image_path: Path to save the transformed image.
    - transform: Affine transformation object (SimpleITK Transform).
    - heart_label: Label corresponding to the heart class (default is 2).

    Returns:
    - None
    """
    # Read the input image (GT.nii.gz)
    input_image = sitk.ReadImage(input_image_path)
    
    # Convert the input image to a numpy array
    input_array = sitk.GetArrayFromImage(input_image)
    
    # Isolate the heart (class 2) by creating a binary mask
    heart_mask = (input_array == heart_label).astype(np.uint8)
    
    # Convert the heart mask to a SimpleITK image
    heart_image = sitk.GetImageFromArray(heart_mask)
    heart_image.CopyInformation(input_image)
    
    # Apply the affine transformation only to the heart
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(heart_image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor to preserve label values
    
    transformed_heart_image = resampler.Execute(heart_image)
    
    # Convert the transformed heart back to a numpy array
    transformed_heart_array = sitk.GetArrayFromImage(transformed_heart_image)
    
    # Reintegrate the transformed heart back into the original segmentation
    output_array = input_array.copy()
    output_array[transformed_heart_array == 1] = heart_label
    
    # Convert the output array back to a SimpleITK image
    output_image = sitk.GetImageFromArray(output_array)
    output_image.CopyInformation(input_image)
    
    # Write the transformed image to the output file
    sitk.WriteImage(output_image, output_image_path)
    print(f"Transformed heart saved at: {output_image_path}")

# Loop through all patient directories and apply the transformation
def process_patients(data_dir, transform):
    """
    Applies affine transformation to the heart for all patient segmentation images.

    Args:
    - data_dir: Path to the root directory containing patient folders.
    - transform: Affine transformation object (SimpleITK Transform).

    Returns:
    - None
    """
    for patient_folder in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_folder)
        
        # Check if the directory contains the GT.nii.gz file
        gt_path = os.path.join(patient_path, "GT.nii.gz")
        
        if os.path.isfile(gt_path):
            # Define the output path for the transformed image
            output_path = os.path.join(patient_path, "GT_transformed.nii.gz")
            
            # Apply the transformation to the heart and save the result
            apply_affine_transform_to_heart(gt_path, output_path, transform)
        else:
            print(f"GT.nii.gz not found for {patient_folder}")

def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description="Affine registration and transformation on segmentation images")
    parser.add_argument('--data_dir', required=True, type=str, help="Directory containing patient folders")
    parser.add_argument('--fixed_image', required=True, type=str, help="Fixed image for affine registration")
    parser.add_argument('--moving_image', required=True, type=str, help="Moving image for affine registration")
    parser.add_argument('--transform_file', required=True, type=str, help="Path to the affine transform file")
    args = parser.parse_args()

    # Isolate heart from both fixed and moving images
    isolated_heart_fixed = isolate_heart(args.fixed_image)
    isolated_heart_moving = isolate_heart(args.moving_image)

    # Save the isolated hearts
    isolated_heart_fixed_output = os.path.splitext(args.fixed_image)[0] + "_isolated.nii.gz"
    isolated_heart_moving_output = os.path.splitext(args.moving_image)[0] + "_isolated.nii.gz"
    sitk.WriteImage(isolated_heart_fixed, isolated_heart_fixed_output)
    sitk.WriteImage(isolated_heart_moving, isolated_heart_moving_output)

    # Perform affine registration and get the transformation
    affine_transform = register_affine(isolated_heart_fixed_output, isolated_heart_moving_output)
    
    # Save the affine transformation to a file
    sitk.WriteTransform(affine_transform, args.transform_file)
    
    # Process all patients in the provided directory using the derived transformation
    process_patients(args.data_dir, affine_transform)

if __name__ == '__main__':
    main()
