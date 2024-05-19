import os
import sys
from helpers.synthetic_dataset import SyntheticDataset
from helpers.aux_gcn import GCNFunctions


if __name__ == "__main__":

    gcn_functions = GCNFunctions()

    MODE = sys.argv[1]
    PATH = "../../../Data"
    files_test = os.listdir(os.path.join(PATH, "Test"))
    files_train = os.listdir(os.path.join(PATH, "Train"))

    if MODE == 'Train':
        print("segmentating...")
        train_signals_v1, train_signals_ii = (
            gcn_functions.segmentation_signals_v1_ii(
                path=PATH,
                list_ecgs=files_train,
                size_beat_before=100,
                size_beat_after=180,
                set_name="Train"
            )
        )

        print("sampling...")
        train_signals_v1 = gcn_functions.sampling_windows_beats(signals=train_signals_v1)
        train_signals_ii = gcn_functions.sampling_windows_beats(signals=train_signals_ii)

        print("extracting attributes...")
        train_features = gcn_functions.get_beats_features(
            signals_v1=train_signals_v1,
            signals_ii=train_signals_ii
        )

        print("converting beats into graphs...")
        train_edges, train_properties = gcn_functions.convert_beats_in_graphs(signals=train_signals_ii)

        print("creating dataset...")
        set_train = SyntheticDataset(
            attr_edges=train_edges,
            attr_properties=train_properties,
            attr_features=train_features
        )

        print("training...")
        kwargs = {
            "epochs": 150,
            "nodes_hidden_layer": 20,
            "n_features": 3,
            "type_gcn": "gcn7",
            "path": "./VVG/Experiment_3 - Aggregation/Images7"
        }
        gcn_functions.training(dataset_train=set_train, model_name="model7_ii_v1", **kwargs)
    else:
        test_signals_v1, test_signals_ii = (
            gcn_functions.segmentation_signals_v1_ii(
                path=PATH,
                list_ecgs=files_test,
                size_beat_before=100,
                size_beat_after=180,
                set_name="Train"
            )
        )

        test_signals_v1 = gcn_functions.sampling_windows_beats(signals=test_signals_v1)
        test_signals_ii = gcn_functions.sampling_windows_beats(signals=test_signals_ii)

        test_features = gcn_functions.get_beats_features(
            signals_v1=test_signals_v1,
            signals_ii=test_signals_ii
        )

        val_edges, val_properties = gcn_functions.convert_beats_in_graphs(signals=test_signals_ii)

        set_val = SyntheticDataset(
            attr_edges=val_edges,
            attr_properties=val_properties,
            attr_features=test_features
        )

        print("testing...")
        kwargs = {
            "nodes_hidden_layer": 20,
            "n_features": 3,
            "type_gcn": "gcn7",
            "path": "./VVG/Experiment_3 - Aggregation/Images7"
        }
        gcn_functions.testing(dataset_val=set_val, model_name="model7_ii_v1", **kwargs)
