import numpy
import pyedflib
import numpy as np

final_data = np.empty(shape=(0, 1280))
final_label = np.empty(shape=(0, 1))


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
    augmented_vector = []
    # Add random noise to the vector
    noise = np.random.normal(0, 0.1, vector.shape)
    final_data = np.vstack([final_data, vector + noise])
    final_label = np.vstack([final_label, numpy.array([label])])
    # Multiply the vector by a random scalar
    scaled_vector = vector * np.random.uniform(0.5, 1.5)
    final_data = np.vstack([final_data, scaled_vector])
    final_label = np.vstack([final_label, numpy.array([label])])
    # Permute the elements of the vector
    permuted_vector = np.random.permutation(vector)
    final_data = np.vstack([final_data, permuted_vector])
    final_label = np.vstack([final_label, numpy.array([label])])
    smoothed_vector = np.convolve(vector, np.ones(5) / 5, mode='same')
    final_data = np.vstack([final_data, smoothed_vector])
    final_label = np.vstack([final_label, numpy.array([label])])
    return augmented_vector


def build_vectors_with_seizures(file_names, seizures_starts, seizures_ends, folder, folder_number=1):
    global final_data, final_label
    for label in range(17, 19):
        for file_index in range(0, len(file_names)):
            file_name = "Dataset/" + folder + "/" + folder + "_" + file_names[file_index] + ".edf"
            seizures_range_array = split_by_batch_size(
                load_data(file_name)[label][seizures_starts[file_index] * 256: seizures_ends[file_index] * 256], 5)
            splitted_array = seizures_range_array[2: 2 + 256]
            seizures_vector = np.concatenate(splitted_array)
            # add label and data to final array
            final_data = np.vstack([final_data, seizures_vector])
            final_label = np.vstack([final_label, numpy.array([1.0])])
            # data augmentation for seizure vectors
            augment_vector(seizures_vector, 1.0)

            if folder_number == 1:
                # find normal vectors from seizure files
                normal_array = split_by_batch_size(load_data(file_name)[label][seizures_ends[file_index] * 256:], 5)[
                               2: 2 + 256]
                normal_vector = np.concatenate(normal_array)
                final_data = np.vstack([final_data, normal_vector])
                final_label = np.vstack([final_label, numpy.array([0.0])])
                augment_vector(normal_vector, 0.0)


def build_vectors_without_seizures(file_names, folder):
    global final_data, final_label
    for label in range(17, 19):
        for file_index in range(0, len(file_names)):
            file_name = "Dataset/" + folder + "/" + folder + "_" + file_names[file_index] + ".edf"
            array_normal_splitted = split_by_batch_size(load_data(file_name)[label], 5)[2:2 + 256]
            normal_vector = np.concatenate(array_normal_splitted)
            final_data = np.vstack([final_data, normal_vector])
            final_label = np.vstack([final_label, numpy.array([0.0])])
            augment_vector(normal_vector, 0.0)


def load_from_category_2():
    file_names_with_seizures = ["16", "19"]
    # Channel 17: FZ-CZ
    # Channel 18: CZ-PZ
    seizures_starts = [130, 3369]
    seizures_ends = [212, 3378]
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


load_from_category_1()
load_from_category_2()
load_from_category_3()
print(final_data.shape)
print(final_label.shape)
