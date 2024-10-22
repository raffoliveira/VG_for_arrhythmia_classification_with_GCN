import os
import sys

from helpers.synthetic_dataset import SyntheticDataset
from helpers.aux_gcn import GCNFunctions


if __name__ == "__main__":

    gcn_functions = GCNFunctions()

    MODE = sys.argv[1]
    PATH = "../../Data"
    files_test = os.listdir(os.path.join(PATH, "Test"))
    files_train = os.listdir(os.path.join(PATH, "Train"))

    print("segmentating...")
    train_signals_v1, train_signals_ii, train_rr_interval_pos, train_rr_interval_pre = (
        gcn_functions.segmentation_signals_v1_ii_rr(
            path=PATH,
            list_ecgs=files_train,
            size_beat_before=100,
            size_beat_after=180,
            set_name="Train"
        )
    )
    test_signals_v1, test_signals_ii, test_rr_interval_pos, test_rr_interval_pre = (
        gcn_functions.segmentation_signals_v1_ii_rr(
            path=PATH,
            list_ecgs=files_test,
            size_beat_before=100,
            size_beat_after=180,
            set_name="Test"
        )
    )

    print("resampling_intra_patient")
    train_signals_V1, test_signals_V1 = gcn_functions.resampling_intra_patient(
        dataset_train=train_signals_v1,
        dataset_test=test_signals_v1
    )
    train_signals_ii, test_signals_ii = gcn_functions.resampling_intra_patient(
        dataset_train=train_signals_ii,
        dataset_test=test_signals_ii
    )
    train_rr_interval_pos, test_rr_interval_pos = gcn_functions.resampling_intra_patient(
        dataset_train=train_rr_interval_pos,
        dataset_test=test_rr_interval_pos
    )
    train_rr_interval_pre, test_rr_interval_pre = gcn_functions.resampling_intra_patient(
        dataset_train=train_rr_interval_pre,
        dataset_test=test_rr_interval_pre
    )

    if MODE == 'Train':
        print("sampling...")
        train_signals_v1, train_signals_ii, train_rr_interval_pos, train_rr_interval_pre = (
            gcn_functions.sampling_windows_beats_signals(
                signals_v1=train_signals_v1,
                signals_ii=train_signals_ii,
                rr_interval_pos_signals=train_rr_interval_pos,
                rr_interval_pre_signals=train_rr_interval_pre
            )
        )

        print("extracting attributes...")
        train_features = gcn_functions.get_beats_features_stats(
            signals_v1=train_signals_v1,
            signals_ii=train_signals_ii,
            rr_interval_pos_signals=train_rr_interval_pos,
            rr_interval_pre_signals=train_rr_interval_pre
        )

        print("converting beats into graphs...")
        train_edges, train_properties = gcn_functions.convert_beats_in_graphs_vvg(
            signals_v1=train_signals_v1,
            signals_ii=train_signals_ii
        )

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
            "n_features": 22,
            "type_gcn": "gcn2",
            "path": "./VVG/Experiment_5_Intra_patient/Images2",
            "arch_type": "stats"
        }
        gcn_functions.training(dataset_train=set_train, model_name="model2_stats", **kwargs)
    else:
        print("sampling...")
        test_signals_v1, test_signals_ii, test_rr_interval_pos, test_rr_interval_pre = (
            gcn_functions.sampling_windows_beats_signals(
                signals_v1=test_signals_v1,
                signals_ii=test_signals_ii,
                rr_interval_pos_signals=test_rr_interval_pos,
                rr_interval_pre_signals=test_rr_interval_pre
            )
        )

        print("extracting attributes...")
        test_features = gcn_functions.get_beats_features_stats(
            signals_v1=test_signals_v1,
            signals_ii=test_signals_ii,
            rr_interval_pos_signals=test_rr_interval_pos,
            rr_interval_pre_signals=test_rr_interval_pre
        )

        print("converting beats into graphs...")
        val_edges, val_properties = gcn_functions.convert_beats_in_graphs_vvg(
            signals_v1=test_signals_v1,
            signals_ii=test_signals_ii
        )

        print("creating dataset...")
        set_val = SyntheticDataset(
            attr_edges=val_edges,
            attr_properties=val_properties,
            attr_features=test_features
        )

        print("testing...")
        kwargs = {
            "nodes_hidden_layer": 20,
            "n_features": 22,
            "type_gcn": "gcn2",
            "path": "./VVG/Experiment_5_Intra_patient/Images2",
            "arch_type": "stats"
        }
        gcn_functions.testing(dataset_val=set_val, model_name="model2_stats", **kwargs)
