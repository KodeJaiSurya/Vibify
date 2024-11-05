from sklearn.utils import resample
import pandas as pd
import numpy as np

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

def augment_samples(samples):
    """
    Augment samples by applying a simple pixel flip.
    Args:
        samples (pd.DataFrame): Dataframe containing rows to augment.
    Returns:
        pd.DataFrame: Augmented samples with pixel data flipped.
    """
    samples = samples.copy()
    samples['pixels'] = samples['pixels'].apply(lambda x: np.flip(x))
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
            n_samples=int(1.5 * min_count),  # Resampling target
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
    return df_balanced

