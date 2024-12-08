from sklearn.utils import resample
import random
import pandas as pd
import numpy as np
from scipy.ndimage import rotate

# Bias Mitigation using Over Sampling method

def convert_pixels_to_array(df):
    """
    Convert pixel data from string format to numpy arrays.
    Args:
        df (pd.DataFrame): Dataframe containing 'pixels' column with pixel data as space-separated strings.
    Returns:
        pd.DataFrame: Dataframe with 'pixels' column converted to numpy arrays.
    """
    df['pixels'] = df['pixels'].apply(lambda x: np.array(list(map(int, x.split()))))
    return df

def identify_underrepresented_classes(df, column='emotion'):
    """
    Identify underrepresented classes based on the minimum class count.
    Args:
        df (pd.DataFrame): Dataframe containing the column with classes to balance.
        column (str): Column name to check for underrepresented classes.
    Returns:
        list: List of underrepresented class labels.
    """
    emotion_counts = df[column].value_counts()
    min_count = emotion_counts.min()
    return emotion_counts[emotion_counts < min_count * 1.5].index.tolist()

def add_noise(image, noise_level=5):
    """Add random noise to an image array."""
    noise = np.random.randint(-noise_level, noise_level, image.shape)
    return np.clip(image + noise, 0, 255)  # Ensure pixel values remain within bounds

def adjust_brightness(image, factor=1.2):
    """Adjust brightness by multiplying pixels with a factor."""
    return np.clip(image * factor, 0, 255)  # Ensure pixel values remain within bounds

def rotate_image(image, angle=10):
    """Rotate a flattened 1D image array by a specified angle after reshaping it to 2D."""
    
    
    # Reshape to 2D assuming the image is 48x48
    image_2d = image.reshape(48, 48)
    
    # Rotate the image and flatten back to 1D
    rotated_image = rotate(image_2d, angle=angle, reshape=False, mode='nearest')
    return rotated_image.flatten()

def augment_samples(samples):
    """
    Augment samples by applying multiple transformations.
    Args:
        samples (pd.DataFrame): Dataframe containing rows to augment.
    Returns:
        pd.DataFrame: Augmented samples with transformed pixel data.
    """
    samples = samples.copy()
    samples['pixels'] = samples['pixels'].apply(lambda x: np.flip(x))  # Original flip
    samples['pixels'] = samples['pixels'].apply(lambda x: add_noise(x))  # Add random noise
    samples['pixels'] = samples['pixels'].apply(lambda x: adjust_brightness(x, factor=random.uniform(0.8, 1.2)))  # Adjust brightness
    samples['pixels'] = samples['pixels'].apply(lambda x: rotate_image(x, angle=random.uniform(-15, 15)))  # Random rotation
    return samples


def oversample_underrepresented_classes(df, column='emotion'):
    """
    Oversample underrepresented classes using augmentation.
    Args:
        df (pd.DataFrame): Original dataframe.
        column (str): Column name for class labels.
    Returns:
        pd.DataFrame: Dataframe with oversampled data.
    """
    # Identify underrepresented classes
    underrepresented_classes = identify_underrepresented_classes(df, column)
    
    # Initialize list to store augmented data
    augmented_data = []

    # Get the minimum count for balancing
    min_count = df[column].value_counts().min()
    
    # Augment each underrepresented class
    for emotion in underrepresented_classes:
        # Filter samples of the underrepresented class
        emotion_samples = df[df[column] == emotion]
        
        # Resample with replacement and augment
        resampled_samples = resample(
            emotion_samples, 
            replace=True, 
            n_samples=int(2 * min_count),  # Resampling target
            random_state=42
        )
        augmented_samples = augment_samples(resampled_samples)
        
        # Collect augmented samples
        augmented_data.append(augmented_samples)

    # Combine original and augmented data
    df_augmented = pd.concat([df] + augmented_data, ignore_index=True)
    return df_augmented

def oversample_data(df):
    """
    Main function to convert pixel data, oversample underrepresented classes,
    and return the balanced dataset.
    Args:
        df (pd.DataFrame): Original dataframe with 'emotion' and 'pixels' columns.
    Returns:
        pd.DataFrame: Dataframe with oversampled underrepresented classes.
    """
    # Step 1: Convert pixel data to numpy arrays
    df = convert_pixels_to_array(df)
    
    # Step 2: Oversample underrepresented classes
    df_balanced = oversample_underrepresented_classes(df, column='emotion')

    # Step 3: Convert array back to pixels string format in df_balanced
    df_balanced['pixels'] = df_balanced['pixels'].apply(lambda x: ' '.join(map(str, x)))
    
    return df_balanced
