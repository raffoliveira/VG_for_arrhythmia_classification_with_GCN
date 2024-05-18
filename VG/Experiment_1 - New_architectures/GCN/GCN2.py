import os
from helpers.aux_gcn import GCNFunctions
from helpers.synthetic_dataset import SyntheticDataset

if __name__ == "__main__":

    gcn_functions = GCNFunctions()

    PATH = "../../../Data"
    files = os.listdir(os.path.join(PATH, "Train"))
    files_val = ["109.mat", "114.mat", "207.mat", "223.mat"]
    files_train = list(set(files)-set(files_val))

    print("segmentating...")
    train_signals_V1, train_signals_II = gcn_functions.segmentation_signals_v1_ii(
        path=PATH, list_ecgs=files_train, size_beat_before=100, size_beat_after=180, set_name="Train")
    val_signals_V1, val_signals_II = gcn_functions.segmentation_signals_v1_ii(
        path=PATH, list_ecgs=files_val, size_beat_before=100, size_beat_after=180, set_name="Train")

    print("sampling...")
    train_signals_V1 = gcn_functions.sampling_windows_beats(signals=train_signals_V1)
    train_signals_II = gcn_functions.sampling_windows_beats(signals=train_signals_II)
    val_signals_V1 = gcn_functions.sampling_windows_beats(signals=val_signals_V1)
    val_signals_II = gcn_functions.sampling_windows_beats(signals=val_signals_II)

    print("extracting attributes...")
    train_features = gcn_functions.get_beats_features(signals_v1=train_signals_V1, signals_ii=train_signals_II)
    val_features = gcn_functions.get_beats_features(signals_v1=val_signals_V1, signals_ii=val_signals_II)

    print("converting beats into graphs...")
    train_edges, train_properties = gcn_functions.convert_beats_in_graphs(signals=train_signals_II)
    val_edges, val_properties = gcn_functions.convert_beats_in_graphs(signals=val_signals_II)

    print("creating dataset...")
    set_train = SyntheticDataset(
        attr_edges=train_edges, attr_properties=train_properties, attr_features=train_features)
    set_val = SyntheticDataset(attr_edges=val_edges, attr_properties=val_properties, attr_features=val_features)

    print("training...")
    kwargs = {
        "epochs": 150,
        "nodes_hidden_layer": 20,
        "n_features": 3,
        "type_gcn": "gcn2",
        "path": "./VG/Experiment_1 - New_architectures/GCN/Images"
    }
    gcn_functions.training_and_testing(dataset_train=set_train, dataset_val=set_val, **kwargs)
