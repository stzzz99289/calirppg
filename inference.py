import argparse
import os
import pandas as pd
import numpy as np
from rppg_methods.unsupervised.unsupervised_predictor import unsupervised_predict, sample_experiment
from rppg_methods.supervised.supervised_predictor import supervised_predict, get_supervised_model_dict
from evaluation.confidence_model import StatistcsConfidenceModel
from config import get_config_from_file, get_default_config
from utils import get_dataloader
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help="yaml config file path")
    args = parser.parse_args()

    # get configurations from file
    config = get_config_from_file(config_file=args.config)
    # print('Configuration:')
    # print(config, end='\n\n')

    # get dataloader
    print(f"====== Inference on {config.DATALOADER.DATASET} dataset ======")
    dataset = get_dataloader(config)
    dataset.load_preprocessed_data_info()
    dataloader = DataLoader(
        dataset=dataset,
        num_workers=2,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
    )

    if config.INFERENCE.METHOD_NAME == 'unsupervised':
        # load confidence model
        confidence_model = StatistcsConfidenceModel(get_default_config())
        confidence_model.load(load_path="results/confidence_model/statistics")

        # inference experiment for on sample
        sample_experiment(config, dataloader, confidence_model=confidence_model, test_batch_num=5)

        # inference
        # print(f"\n\n=== {dataset.dataset_name} dataset ===")
        # np.seterr(invalid='ignore')
        # saving_dir = f'results/{dataset.dataset_name}_unsupervised_inference'
        # global_metrics_lst = []
        # unsupervised_method_names = ['POS', 'CHROM', 'ICA', 'GREEN', 'LGI', 'PBV', 'OMIT']
        # for method_name in unsupervised_method_names:
        #     print(f"{method_name} inference...")
        #     global_metrics = unsupervised_predict(config, dataloader, 
        #                                             method_name=method_name, 
        #                                             confidence_model=confidence_model, 
        #                                             saving_dir=saving_dir
        #                                             )
        #     global_metrics_lst.append(global_metrics)

        # save global metrics
        # global_metrics_df = pd.DataFrame(global_metrics_lst)
        # global_metrics_df.to_csv(f'{saving_dir}/global_metrics.csv', index=False)

    else:
        # load confidence model
        confidence_model = StatistcsConfidenceModel(get_default_config())
        confidence_model.load(load_path="results/confidence_model/statistics")

        # load model path dict
        method_name = config.INFERENCE.METHOD_NAME
        supervised_method_dict = get_supervised_model_dict(method_name)
        saving_dir = f'results/supervised_inference/{dataset.dataset_name}_{method_name}_inference'

        # inference
        global_metrics_lst = []
        for checkpoint_name, model_path in supervised_method_dict.items():
            print(f"{checkpoint_name} inference...")
            global_metrics = supervised_predict(config, dataloader, 
                                                checkpoint_name, model_path, 
                                                confidence_model=confidence_model, 
                                                saving_dir=saving_dir,
                                                num_batches=None
                                                )
            global_metrics_lst.append(global_metrics)

        # save global metrics
        global_metrics_df = pd.DataFrame(global_metrics_lst)
        global_metrics_df.to_csv(f'{saving_dir}/{method_name}_global_metrics.csv', index=False)

