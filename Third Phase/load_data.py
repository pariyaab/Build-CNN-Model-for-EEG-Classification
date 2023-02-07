import numpy
import pyedflib
import numpy as np
import pickle

from scipy.ndimage import rotate

final_data = np.empty(shape=(0, 1280, 2))
final_label = np.empty(shape=(0, 1))
start_indexes = [2, 258, 514]


def load_data(file_name):
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    return sigbufs


def split_by_batch_size(arr, batch_size):
    return np.array_split(arr, (arr.shape[0] / batch_size) + 1)


def augment_vector(vector, label):
    global final_data, final_label
    # Random Rotation
    # specify a range of rotation angles in degrees
    angles = np.arange(-30, 30, 5)
    # select a random angle from the range
    selected_angle = np.random.choice(angles)
    rotated_vector = rotate(vector, selected_angle, axes=(1, 2), reshape=False)
    final_data = np.vstack([final_data, rotated_vector])
    final_label = np.vstack([final_label, numpy.array([label])])
    # Scaling:
    factors = np.arange(0.8, 1.2, 0.1)
    # select a random factor from the range
    selected_factor = np.random.choice(factors)
    scaled_vector = selected_factor * vector
    final_data = np.vstack([final_data, scaled_vector])
    final_label = np.vstack([final_label, numpy.array([label])])
    # Flipping:
    flipped_vector = np.fliplr(vector)
    final_data = np.vstack([final_data, flipped_vector])
    final_label = np.vstack([final_label, numpy.array([label])])
    # Noise injection:
    noise = np.random.rand(*vector.shape)
    # specify a range of noise levels
    levels = np.arange(0.05, 0.15, 0.01)
    # select a random noise level from the range
    selected_level = np.random.choice(levels)
    # add noise to the vector
    noisy_vector = vector + selected_level * noise
    final_data = np.vstack([final_data, noisy_vector])
    final_label = np.vstack([final_label, numpy.array([label])])


def cut_seizure_range(file_name, start, end, channel, index):
    seizures_range_array = split_by_batch_size(
        load_data(file_name)[channel][start: end], 5)
    splitted_array = seizures_range_array[start_indexes[index]: start_indexes[index] + 256]
    seizures_vector = np.concatenate(splitted_array)
    reshaped = seizures_vector.reshape(1, 1280)
    return reshaped


def cut_normal_range(file_name, start, channel, index):
    normal_array = split_by_batch_size(load_data(file_name)[channel][start:], 5)[
                   start_indexes[index]: start_indexes[index] + 256]
    normal_vector = np.concatenate(normal_array)
    reshaped = normal_vector.reshape(1, 1280)
    return reshaped


def build_vectors_with_seizures(file_names, seizures_starts, seizures_ends, folder, folder_number=1):
    global final_data, final_label
    for file_index in range(0, len(file_names)):
        file_name = "Dataset/" + folder + "/" + folder + "_" + file_names[file_index] + ".edf"
        for i in range(0, 3):
            seizure_vector_channel_17 = cut_seizure_range(file_name, seizures_starts[file_index] * 256,
                                                          seizures_ends[file_index] * 256, 17, i)
            seizure_vector_channel_18 = cut_seizure_range(file_name, seizures_starts[file_index] * 256,
                                                          seizures_ends[file_index] * 256, 18, i)
            seizures_vector = np.stack([seizure_vector_channel_17, seizure_vector_channel_18], axis=-1)
            # add label and data to final array
            final_data = np.vstack([final_data, seizures_vector])
            final_label = np.vstack([final_label, numpy.array([1.0])])
            # data augmentation for seizure vectors
            augment_vector(seizures_vector, 1.0)
            if folder_number == 1:
                # find normal vectors from seizure files
                normal_vector_channel_17 = cut_normal_range(file_name, seizures_ends[file_index] * 256, 17, i)
                normal_vector_channel_18 = cut_normal_range(file_name, seizures_ends[file_index] * 256, 18, i)
                normal_vector = np.stack([normal_vector_channel_17, normal_vector_channel_18], axis=-1)
                final_data = np.vstack([final_data, normal_vector])
                final_label = np.vstack([final_label, numpy.array([0.0])])
                augment_vector(normal_vector, 0.0)


def build_vectors_without_seizures(file_names, folder):
    global final_data, final_label
    for file_index in range(0, len(file_names)):
        file_name = "Dataset/" + folder + "/" + folder + "_" + file_names[file_index] + ".edf"
        for i in range(0, 3):
            normal_vector_channel_17 = split_by_batch_size(load_data(file_name)[17], 5)[
                                       start_indexes[i]:start_indexes[i] + 256]
            reshaped_17 = np.concatenate(normal_vector_channel_17).reshape(1, 1280)
            normal_vector_channel_18 = split_by_batch_size(load_data(file_name)[18], 5)[
                                       start_indexes[i]:start_indexes[i] + 256]
            reshaped_18 = np.concatenate(normal_vector_channel_18).reshape(1, 1280)
            normal_vector = np.stack([reshaped_17, reshaped_18], axis=-1)
            final_data = np.vstack([final_data, normal_vector])
            final_label = np.vstack([final_label, numpy.array([0.0])])
            augment_vector(normal_vector, 0.0)


def load_from_category_2():
    file_names_with_seizures = ["16", "16+"]
    # Channel 17: FZ-CZ
    # Channel 18: CZ-PZ
    seizures_starts = [130, 2972]
    seizures_ends = [212, 3053]
    build_vectors_with_seizures(file_names_with_seizures, seizures_starts, seizures_ends, "chb02")
    file_names_without_seizures = ["08", "14"]
    build_vectors_without_seizures(file_names=file_names_without_seizures, folder="chb02")


def load_from_category_1():
    file_names_with_seizures = ["03", "04", "15", "16", "18", "21", "26"]
    # Channel 17: FZ-CZ
    # Channel 18: CZ-PZ
    seizures_starts = [2996, 1467, 1732, 1015]
    seizures_ends = [3036, 1494, 1772, 1066]
    build_vectors_with_seizures(file_names_with_seizures[:4], seizures_starts, seizures_ends, "chb01", folder_number=1)
    build_vectors_with_seizures(file_names_with_seizures[4:], seizures_starts, seizures_ends, "chb01", folder_number=2)
    file_names_without_seizures = ["01", "02"]
    build_vectors_without_seizures(file_names=file_names_without_seizures, folder="chb01")


def load_from_category_3():
    file_names_with_seizures = ["01"]
    seizures_starts = [362]
    seizures_ends = [414]
    build_vectors_with_seizures(file_names_with_seizures, seizures_starts, seizures_ends, "chb03", folder_number=3)


# load_from_category_1()
# load_from_category_2()
# load_from_category_3()
# print(final_data.shape)
# print(final_label.shape)
# pickle.dump(final_data, open('x.pkl', 'wb'))
# pickle.dump(final_label, open('y.pkl', 'wb'))
x = pickle.load(open('x.pkl', 'rb'))
y = pickle.load(open('y.pkl', 'rb'))
print(y)