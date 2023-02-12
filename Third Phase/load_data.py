import numpy
import pyedflib
import numpy as np
import pickle

from scipy.ndimage import rotate

final_data = np.empty(shape=(0, 1280, 2))
final_label = np.empty(shape=(0, 1))
start_indexes = [2, 258, 514, 770, 1026, 1282]


def split_by_batch_size(arr, batch_size):
    return np.array_split(arr, (arr.shape[0] / batch_size) + 1)


def load_data(file_name):
    f = pyedflib.EdfReader(file_name)
    n = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    return sigbufs


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
        for i in range(0, 6):
            try:
                seizure_vector_channel_17 = cut_seizure_range(file_name, seizures_starts[file_index] * 256,
                                                              seizures_ends[file_index] * 256, 17, i)
                seizure_vector_channel_18 = cut_seizure_range(file_name, seizures_starts[file_index] * 256,
                                                              seizures_ends[file_index] * 256, 18, i)
                seizures_vector = np.stack([seizure_vector_channel_17, seizure_vector_channel_18], axis=-1)
                # add label and data to final array
                final_data = np.vstack([final_data, seizures_vector])
                final_label = np.vstack([final_label, numpy.array([1.0])])
            except:
                print("skip")
            if folder_number == 1:
                # find normal vectors from seizure files
                normal_vector_channel_17 = cut_normal_range(file_name, seizures_ends[file_index] * 256, 17, i)
                normal_vector_channel_18 = cut_normal_range(file_name, seizures_ends[file_index] * 256, 18, i)
                normal_vector = np.stack([normal_vector_channel_17, normal_vector_channel_18], axis=-1)
                final_data = np.vstack([final_data, normal_vector])
                final_label = np.vstack([final_label, numpy.array([0.0])])


def build_vectors_without_seizures(file_names, folder):
    global final_data, final_label
    for file_index in range(0, len(file_names)):
        file_name = "Dataset/" + folder + "/" + folder + "_" + file_names[file_index] + ".edf"
        splitted_array_17 = split_by_batch_size(load_data(file_name)[17], 5)
        splitted_array_18 = split_by_batch_size(load_data(file_name)[18], 5)
        index = 0
        start_index = 2
        while index <= 50:
            end_index = start_index + 256
            normal_vector_channel_17 = splitted_array_17[start_index: end_index]
            reshaped_17 = np.concatenate(normal_vector_channel_17).reshape(1, 1280)
            normal_vector_channel_18 = splitted_array_18[start_index: end_index]
            reshaped_18 = np.concatenate(normal_vector_channel_18).reshape(1, 1280)
            normal_vector = np.stack([reshaped_17, reshaped_18], axis=-1)
            final_data = np.vstack([final_data, normal_vector])
            final_label = np.vstack([final_label, numpy.array([0.0])])
            index += 1
            start_index = end_index


def load_from_category_2():
    file_names_with_seizures = ["16", "16+"]
    # Channel 17: FZ-CZ
    # Channel 18: CZ-PZ
    seizures_starts = [130, 2972]
    seizures_ends = [212, 3053]
    build_vectors_with_seizures(file_names_with_seizures, seizures_starts, seizures_ends, "chb02")
    file_names_without_seizures = ["08", "14", "15"]
    build_vectors_without_seizures(file_names=file_names_without_seizures, folder="chb02")


def load_from_category_1():
    file_names_with_seizures = ["03", "04", "15", "16", "18", "21", "26"]
    # Channel 17: FZ-CZ
    # Channel 18: CZ-PZ
    seizures_starts = [2996, 1467, 1732, 1015]
    seizures_ends = [3036, 1494, 1772, 1066]
    build_vectors_with_seizures(file_names_with_seizures[:4], seizures_starts, seizures_ends, "chb01", folder_number=1)
    build_vectors_with_seizures(file_names_with_seizures[4:], seizures_starts, seizures_ends, "chb01", folder_number=2)
    file_names_without_seizures = ["01", "02", "06", "07", "08"]
    build_vectors_without_seizures(file_names=file_names_without_seizures, folder="chb01")


def load_from_category_3():
    file_names_with_seizures = ["01", "02", "03"]
    seizures_starts = [362, 731, 432]
    seizures_ends = [414, 796, 501]
    build_vectors_with_seizures(file_names_with_seizures, seizures_starts, seizures_ends, "chb03", folder_number=3)


load_from_category_1()
load_from_category_2()
load_from_category_3()
print(final_data.shape)
print(final_label.shape)
pickle.dump(final_data, open('new_2_x.pkl', 'wb'))
pickle.dump(final_label, open('new_2_y.pkl', 'wb'))
