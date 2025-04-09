import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.post_process import post_processing, signal_to_psd, psd_to_hr
from evaluation.visualization import create_psd_figure, plot_psd, save_figure, plot_psd_statistics
from evaluation.metric import compute_metric_avg_std, compute_mae, compute_rmse, compute_mape, compute_pearson, compute_macc, compute_snr
import matplotlib.pyplot as plt
import torch.nn.functional as F

def get_supervised_model_dict(method_name):
    if method_name == 'deepphys':
        supervised_method_dict = {
            "deepphys-ubfc-rppg": "rppg_methods/supervised/weights/UBFC-rPPG_DeepPhys.pth",
            "deepphys-pure": "rppg_methods/supervised/weights/PURE_DeepPhys.pth",
            # "deepphys-ma-ubfc": "rppg_methods/supervised/weights/MA-UBFC_deepphys.pth",
            # "deepphys-scamps": "rppg_methods/supervised/weights/SCAMPS_DeepPhys.pth",
        }
    elif method_name == 'physnet':
        supervised_method_dict = {
            "physnet-ubfc-rppg": "rppg_methods/supervised/weights/UBFC-rPPG_PhysNet_DiffNormalized.pth",
            "physnet-pure": "rppg_methods/supervised/weights/PURE_PhysNet_DiffNormalized.pth",
            # "physnet-ma-ubfc": "rppg_methods/supervised/weights/MA-UBFC_physnet.pth",
            # "physnet-scamps": "rppg_methods/supervised/weights/SCAMPS_PhysNet_DiffNormalized.pth",
        }
    elif method_name == 'tscan':
        supervised_method_dict = {
            "tscan-ubfc-rppg": "rppg_methods/supervised/weights/UBFC-rPPG_TSCAN.pth",
            "tscan-pure": "rppg_methods/supervised/weights/PURE_TSCAN.pth",
            # "tscan-ma-ubfc": "rppg_methods/supervised/weights/MA-UBFC_tscan.pth",
            # "tscan-scamps": "rppg_methods/supervised/weights/SCAMPS_TSCAN.pth",
        }
    elif method_name == 'efficientphys':
        supervised_method_dict = {
            "efficientphys-ubfc-rppg": "rppg_methods/supervised/weights/UBFC-rPPG_EfficientPhys.pth",
            "efficientphys-pure": "rppg_methods/supervised/weights/PURE_EfficientPhys.pth",
            # "efficientphys-ma-ubfc": "rppg_methods/supervised/weights/MA-UBFC_efficientphys.pth",
            # "efficientphys-scamps": "rppg_methods/supervised/weights/SCAMPS_EfficientPhys.pth",
        }
    elif method_name == 'physformer':
        supervised_method_dict = {
            "physformer-ubfc-rppg": "rppg_methods/supervised/weights/UBFC-rPPG_PhysFormer_DiffNormalized.pth",
            "physformer-pure": "rppg_methods/supervised/weights/PURE_PhysFormer_DiffNormalized.pth",
            # "physformer-scamps": "rppg_methods/supervised/weights/SCAMPS_PhysFormer_DiffNormalized.pth",
        }
    elif method_name == 'ibvpnet':
        supervised_method_dict = {
            "ibvpnet-pure": "rppg_methods/supervised/weights/PURE_iBVPNet.pth",
        }
    elif method_name == 'rhythmformer':
        supervised_method_dict = {
            "rhythmformer-ubfc-rppg": "rppg_methods/supervised/weights/UBFC-rPPG_RhythmFormer.pth",
            "rhythmformer-pure": "rppg_methods/supervised/weights/PURE_RhythmFormer.pth",
        }
    elif method_name == 'factorizephys':
        supervised_method_dict = {
            "factorizephys-ubfc-rppg": "rppg_methods/supervised/weights/UBFC-rPPG_FactorizePhys_FSAM_Res.pth",
            "factorizephys-pure": "rppg_methods/supervised/weights/PURE_FactorizePhys_FSAM_Res.pth",
            "factorizephys-ibvp": "rppg_methods/supervised/weights/iBVP_FactorizePhys_FSAM_Res.pth",
        }
    else:
        raise ValueError(f"Invalid inference method: {method_name}")
    
    return supervised_method_dict

def load_supervised_model(config, model_path):
    device = config.INFERENCE.DEVICE
    base_model_name = config.INFERENCE.METHOD_NAME
    input_frame_size = config.INFERENCE.INPUT_FRAME_SIZE
    input_frames_num = config.INFERENCE.INPUT_FRAMES_NUM

    # initialize model
    if base_model_name == 'deepphys':
        from rppg_methods.supervised.models.DeepPhys import DeepPhys
        model = DeepPhys(img_size=input_frame_size[0]).to(device)
        model = torch.nn.DataParallel(model, device_ids=[0])
    elif base_model_name == 'physnet':
        from rppg_methods.supervised.models.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        model = PhysNet_padding_Encoder_Decoder_MAX(frames=input_frames_num).to(device)
    elif base_model_name == 'tscan':
        from rppg_methods.supervised.models.TS_CAN import TSCAN
        model = TSCAN(frame_depth=10, img_size=input_frame_size[0]).to(device)
        model = torch.nn.DataParallel(model, device_ids=[0])
    elif base_model_name == 'efficientphys':
        from rppg_methods.supervised.models.EfficientPhys import EfficientPhys
        model = EfficientPhys(frame_depth=10, img_size=input_frame_size[0]).to(device)
        model = torch.nn.DataParallel(model, device_ids=[0])
    elif base_model_name == 'physformer':
        from rppg_methods.supervised.models.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        physformer_patch_size = 4
        physformer_dim = 96
        physformer_ff_dim = 144
        physformer_num_heads = 4
        physformer_num_layers = 12
        physformer_theta = 0.7
        physformer_drop_rate = 0.1
        model = ViT_ST_ST_Compact3_TDC_gra_sharp(
                image_size=(input_frames_num, input_frame_size[0], input_frame_size[1]), 
                patches=(physformer_patch_size,) * 3, dim=physformer_dim, ff_dim=physformer_ff_dim, 
                num_heads=physformer_num_heads, num_layers=physformer_num_layers, 
                dropout_rate=physformer_drop_rate, theta=physformer_theta).to(device)
        model = torch.nn.DataParallel(model, device_ids=[0])
    elif base_model_name == 'ibvpnet':
        from rppg_methods.supervised.models.iBVPNet import iBVPNet
        model = iBVPNet(frames=input_frames_num, in_channels=3).to(device)
    elif base_model_name == 'rhythmformer':
        from rppg_methods.supervised.models.RhythmFormer import RhythmFormer
        model = RhythmFormer().to(device)
        model = torch.nn.DataParallel(model, device_ids=[0])
    elif base_model_name == 'factorizephys':
        from rppg_methods.supervised.models.FactorizePhys.FactorizePhys import FactorizePhys
        md_config = {}
        md_config["FRAME_NUM"] = input_frames_num
        md_config["MD_TYPE"] = "NMF"
        md_config["MD_FSAM"] = True
        md_config["MD_TRANSFORM"] = "T_KAB"
        md_config["MD_S"] = 1
        md_config["MD_R"] = 1
        md_config["MD_STEPS"] = 4
        md_config["MD_INFERENCE"] = True
        md_config["MD_RESIDUAL"] = True
        model = FactorizePhys(frames=input_frames_num, md_config=md_config, in_channels=3,
                                    dropout=0.1, device=device)
        model = torch.nn.DataParallel(model, device_ids=[0])
    else:
        raise ValueError(f"Model {base_model_name} not supported")
    print(f"{base_model_name} model initialized...")
    
    # load pre-trained weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} not found")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"loaded pre-trained weights from {model_path}...")
    return model

def prepare_data(config, input, label):
    device = config.INFERENCE.DEVICE
    input_type = config.INFERENCE.INPUT_TYPE
    label_type = config.INFERENCE.LABEL_TYPE
    input_format = config.INFERENCE.INPUT_FORMAT

    # make input
    assert input.shape[-1] == 9, "original input must have 9 channels"
    input_dict = {
        "raw": input[:, :, :, :, 0:3],
        "standardized": input[:, :, :, :, 3:6],
        "diffnormalized": input[:, :, :, :, 6:9],
    }
    input = list()
    for type in input_type:
        input.append(input_dict[type])
    input = torch.cat(input, dim=-1)

    # make label
    assert label.shape[-1] == 3, "original label must have 3 channels"
    label_dict = {
        "raw": label[:, :, 0],
        "standardized": label[:, :, 1],
        "diffnormalized": label[:, :, 2],
    }
    label = label_dict[label_type]

    # transpose input fromat
    if input_format == 'NDCHW':
        input = input.permute(0, 1, 4, 2, 3)
    elif input_format == 'NCDHW':
        input = input.permute(0, 4, 1, 2, 3)
    elif input_format == 'NDHWC':
        pass
    else:
        raise ValueError('Unsupported Data Format!')

    return input.to(device), label.to(device)
    
def frames2rPPG_supervised(config, frames_batch, supervised_model):
    base_model_name = config.INFERENCE.METHOD_NAME
    input_frame_size = config.INFERENCE.INPUT_FRAME_SIZE
    input_frames_num = config.INFERENCE.INPUT_FRAMES_NUM

    if base_model_name == 'deepphys':
        # input shape: (batch_size, chunk_length, 6, h, w)
        N, D, C, H, W = frames_batch.shape
        assert C == 6, "deepphys input must have 6 channels with first 3 diffnormalized, last 3 standardized"
        assert (H, W) == input_frame_size
        frames_flattened = frames_batch.view(N * D, C, H, W)
        rppg_flattened = supervised_model(frames_flattened)
        rppg_batch = rppg_flattened.view(N, D, -1)
    elif base_model_name == 'physnet':
        # input shape: (batch_size, 3, chunck_length, h, w)
        N, C, D, H, W = frames_batch.shape
        assert C == 3, "physnet input must have 3 channels with diffnormalized frames"
        assert D == input_frames_num
        assert (H, W) == input_frame_size
        rppg_batch, _, _, _ = supervised_model(frames_batch)
    elif base_model_name == 'tscan':
        N, D, C, H, W = frames_batch.shape
        assert C == 6, "tscan input must have 6 channels with first 3 diffnormalized, last 3 standardized"
        assert (H, W) == input_frame_size
        frames_flattened = frames_batch.view(N * D, C, H, W)
        rppg_flattened = supervised_model(frames_flattened)
        rppg_batch = rppg_flattened.view(N, D, -1)
    elif base_model_name == 'efficientphys':
        N, D, C, H, W = frames_batch.shape
        assert C == 3, "efficientphys input must have 3 channels with standardized frames"
        assert (H, W) == input_frame_size
        frames_batch = frames_batch.view(N * D, C, H, W)
        # Add one more frame for EfficientPhys since it does torch.diff for the input
        last_frame = torch.unsqueeze(frames_batch[-1, :, :, :], 0).repeat(1, 1, 1, 1)
        frames_batch = torch.cat((frames_batch, last_frame), 0)
        rppg_flattened = supervised_model(frames_batch)
        rppg_batch = rppg_flattened.view(N, D, -1)
    elif base_model_name == 'physformer':
        N, C, D, H, W = frames_batch.shape
        assert C == 3, "physformer input must have 3 channels with diffnormalized frames"
        assert D == input_frames_num
        assert (H, W) == input_frame_size
        gra_sharp = 2.0
        rppg_batch, _, _, _ = supervised_model(frames_batch, gra_sharp)
        rppg_batch = rppg_batch.view(N, D, -1)
    elif base_model_name == 'ibvpnet':
        N, C, D, H, W = frames_batch.shape
        assert C == 3, "ibvpnet input must have 3 channels with raw frames"
        assert D == input_frames_num
        assert (H, W) == input_frame_size
        # Add one more frame for iBVPNet since it does torch.diff for the input
        last_frame = torch.unsqueeze(frames_batch[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
        frames_batch = torch.cat((frames_batch, last_frame), 2)
        rppg_batch = supervised_model(frames_batch)
        rppg_batch = rppg_batch.view(N, D, -1)
    elif base_model_name == 'rhythmformer':
        N, D, C, H, W = frames_batch.shape
        assert C == 3, "rhythmformer input must have 3 channels with standardized frames"
        assert (H, W) == input_frame_size
        rppg_batch = supervised_model(frames_batch)
        rppg_batch = (rppg_batch-torch.mean(rppg_batch, axis=-1).view(-1, 1))/torch.std(rppg_batch, axis=-1).view(-1, 1) # normalize
        rppg_batch = rppg_batch.view(N, D, -1)
    elif base_model_name == 'factorizephys':
        N, C, D, H, W = frames_batch.shape
        assert C == 3, "factorizephys input must have 3 channels with raw frames"
        assert D == input_frames_num
        assert (H, W) == input_frame_size
        last_frame = torch.unsqueeze(frames_batch[:, :, -1, :, :], 2).repeat(1, 1, 1, 1, 1)
        frames_batch = torch.cat((frames_batch, last_frame), 2)
        rppg_batch, vox_embed, factorized_embed, appx_error = supervised_model(frames_batch)
        rppg_batch = (rppg_batch - torch.mean(rppg_batch)) / torch.std(rppg_batch) # normalize
        rppg_batch = rppg_batch.view(N, D, -1)
    else:
        raise ValueError(f"Supervised method {base_model_name} inference not supported")

    assert rppg_batch.shape == (N, D, 1), f"rppg_batch shape must be {(N, D, 1)}, but got {rppg_batch.shape}"
    return rppg_batch

def supervised_predict(config, dataloader, checkpoint_name, model_path, confidence_model=None, saving_dir=None, num_batches=None):
    # check if already done inference for this method on the dataset
    if saving_dir is not None:
        os.makedirs(saving_dir, exist_ok=True)
        if os.path.exists(f"{saving_dir}/{checkpoint_name}_metrics.csv"):
            print(f"{saving_dir}/{checkpoint_name}_metrics.csv already exists")
            return None
                
    # bandpass for inference (calculate hr)
    use_bandpass = config.POST_PROCESSING.BANDPASS
    inference_low_pass = config.POST_PROCESSING.BANDPASS_LOW_FREQ
    inference_high_pass = config.POST_PROCESSING.BANDPASS_HIGH_FREQ
    inference_freq_resolution = config.POST_PROCESSING.FREQ_RESOLUTION

    # bandpass for confidence model (we need wider range to capture second harmonic information)
    confidence_low_pass = config.CONFIDENCE_MODEL.BANDPASS_LOW_FREQ
    confidence_high_pass = config.CONFIDENCE_MODEL.BANDPASS_HIGH_FREQ
    confidence_freq_resolution = config.CONFIDENCE_MODEL.FREQ_RESOLUTION

    # load supervised neural model
    supervised_model = load_supervised_model(config, model_path)

    # iterate over batch
    hr_pred_lst = []
    hr_label_lst = []
    snr_lst = []
    macc_lst = []
    local_metrics = []
    with torch.no_grad():
        sbar = tqdm(dataloader, total=len(dataloader))
        for batch_idx, batch in enumerate(sbar):
            if num_batches is not None and batch_idx >= num_batches:
                break

            # raw input shape: (batch_size, chunk_length, h, w, 9) (NDHWC)
            # raw label shape: (batch_size, chunk_length, 3
            input, label, filename, chunk_idx_local = batch

            # supervised inference in batch (frames -> rPPG)
            frames_batch, ppg_batch = prepare_data(config, input, label)
            rppg_batch = frames2rPPG_supervised(config, frames_batch, supervised_model) # (batch_size, chunk_length, 1)

            # evaluation
            batch_size = frames_batch.shape[0]
            for sample_idx in range(batch_size):
                # extract sample
                frames_sample = frames_batch[sample_idx].cpu().numpy()
                ppg_sample = ppg_batch[sample_idx].cpu().numpy()
                rppg_sample = rppg_batch[sample_idx].cpu().numpy()
                filename_sample, chunk_idx_local_sample = filename[sample_idx], int(chunk_idx_local[sample_idx])

                # post processing
                diff_flag = config.INFERENCE.LABEL_TYPE == 'diffnormalized'
                rppg = post_processing(rppg_sample, config.DATALOADER.FPS, diff_flag=diff_flag, use_bandpass=use_bandpass, low_pass=inference_low_pass, high_pass=inference_high_pass)
                rppg_confidence = post_processing(rppg_sample, config.DATALOADER.FPS, diff_flag=diff_flag, use_bandpass=use_bandpass, low_pass=confidence_low_pass, high_pass=confidence_high_pass)
                ppg_sample = post_processing(ppg_sample, config.DATALOADER.FPS, diff_flag=diff_flag, use_bandpass=use_bandpass, low_pass=inference_low_pass, high_pass=inference_high_pass)

                # PPG to PSD
                freq_rppg, psd_rppg = signal_to_psd(rppg, config.DATALOADER.FPS, freq_resolution=inference_freq_resolution, low_pass=inference_low_pass, high_pass=inference_high_pass, interpolation=True)
                freq_rppg_confidence, psd_rppg_confidence = signal_to_psd(rppg_confidence, config.DATALOADER.FPS, freq_resolution=confidence_freq_resolution, low_pass=confidence_low_pass, high_pass=confidence_high_pass, interpolation=True)
                freq_ppg, psd_ppg = signal_to_psd(ppg_sample, config.DATALOADER.FPS, freq_resolution=inference_freq_resolution, low_pass=inference_low_pass, high_pass=inference_high_pass, interpolation=True)

                # PSD to HR
                hr_pred = psd_to_hr(freq_rppg, psd_rppg)
                hr_label = psd_to_hr(freq_ppg, psd_ppg)

                # calculate confidence score
                if confidence_model is not None:
                    confidence_distance = confidence_model.predict(hr_pred, freq_rppg_confidence, psd_rppg_confidence, confidence_type="distance")
                    confidence_pvalue = confidence_model.predict(hr_pred, freq_rppg_confidence, psd_rppg_confidence, confidence_type="pvalue")
                    confidence_percentile = confidence_model.predict(hr_pred, freq_rppg_confidence, psd_rppg_confidence, confidence_type="percentile")
                else:
                    confidence_distance = None
                    confidence_pvalue = None
                    confidence_percentile = None

                # evaluation (rPPG vs PPG label)
                macc = compute_macc(rppg, ppg_sample)
                snr = compute_snr(rppg, hr_label)
                hr_pred_lst.append(hr_pred)
                hr_label_lst.append(hr_label)
                snr_lst.append(snr)
                macc_lst.append(macc)
                local_metrics.append({
                    'filename': filename_sample, 
                    'chunk_idx_local': chunk_idx_local_sample, 
                    'hr_pred': hr_pred, 
                    'hr_label': hr_label, 
                    'snr': snr, 
                    'macc': macc,
                    'confidence_distance': confidence_distance,
                    'confidence_pvalue': confidence_pvalue,
                    'confidence_percentile': confidence_percentile,
                    'abs_hr_error': np.abs(hr_pred - hr_label)
                    })
                
                """
                inference&rppg&confidence plot
                """
                # dataset_name = config.DATALOADER.DATASET
                # preprocess_name = config.DATALOADER.CACHED_PATH.split("/")[-1]
                # saving_dir = f"results/supervised_samples/{dataset_name}/{preprocess_name}"
                # os.makedirs(f"{saving_dir}/batch_{batch_idx:02d}", exist_ok=True)

                # create_psd_figure(title=f"gt: {hr_label} bpm", xlim=[inference_low_pass, inference_high_pass], xticks=np.arange(inference_low_pass, inference_high_pass+1, inference_freq_resolution*2))
                # plot_psd(freq_ppg, psd_ppg, label="gt ppg")
                # plt.axvline(hr_label, linestyle='--', color='red', alpha=0.8)
                # plot_psd(freq_rppg, psd_rppg, label=f"{method_name}", alpha=0.3)
                # save_figure(f"{saving_dir}/batch_{batch_idx:02d}/gt_psd_plot_{hr_label}_{method_name}.png")

                # plt.figure()
                # plt.plot(rppg_sample, label="rppg")
                # plt.plot(rppg, label="rppg_postprocessed")
                # plt.legend()
                # plt.savefig(f"{saving_dir}/batch_{batch_idx:02d}/rppg_plot_{hr_label}_{method_name}.png")
                # plt.close()

                # if confidence_model is not None:
                #     create_psd_figure(title=f"gt: {hr_label} bpm", xlim=[confidence_low_pass, confidence_high_pass], xticks=np.arange(confidence_low_pass, confidence_high_pass+1, confidence_freq_resolution*2))
                #     plt.axvline(hr_label, linestyle='--', color='red', alpha=0.8)
                #     confidence_distance = confidence_model.predict(hr_pred, freq_rppg_confidence, psd_rppg_confidence, confidence_type="distance")
                #     confidence_pvalue = confidence_model.predict(hr_pred, freq_rppg_confidence, psd_rppg_confidence, confidence_type="pvalue")
                #     confidence_percentile = confidence_model.predict(hr_pred, freq_rppg_confidence, psd_rppg_confidence, confidence_type="percentile")
                #     plot_psd(freq_rppg_confidence, psd_rppg_confidence, label=f"rppg, hr_pred={hr_pred:.0f} bpm")
                #     rppg_hr_idx = np.argmin(np.abs(confidence_model.hr_values - hr_pred))
                #     ref_psd_mean = confidence_model.psd_statistics["psd_means"][rppg_hr_idx]
                #     ref_psd_covariance = confidence_model.psd_statistics["psd_covariances"][rppg_hr_idx]
                #     plot_psd_statistics(freq_rppg_confidence, ref_psd_mean, ref_psd_covariance, alpha=0.5)
                #     plt.title(f"distance: {confidence_distance*100:.0f}%, hr_std={confidence_model.confidence_to_std(confidence_distance):.1f} bpm\n \
                #             pvalue: {confidence_pvalue*100:.0f}%, hr_std={confidence_model.confidence_to_std(confidence_pvalue):.1f} bpm\n \
                #             percentile: {confidence_percentile*100:.0f}%, hr_std={confidence_model.confidence_to_std(confidence_percentile):.1f} bpm")
                #     plt.tight_layout()
                #     save_figure(f"{saving_dir}/batch_{batch_idx:02d}/confidence_plot_{hr_label}_{method_name}.png")
                
    hr_pred_lst = np.array(hr_pred_lst)
    hr_label_lst = np.array(hr_label_lst)
    snr_lst = np.array(snr_lst)
    macc_lst = np.array(macc_lst)

    # save local metrics
    if saving_dir is not None:
        local_metrics_df = pd.DataFrame(local_metrics)
        local_metrics_df.to_csv(f'{saving_dir}/{checkpoint_name}_metrics.csv', index=False)

    # return global metrics
    snr_mean, snr_std = compute_metric_avg_std(snr_lst)
    macc_mean, macc_std = compute_metric_avg_std(macc_lst)
    mae_mean, mae_std = compute_mae(hr_pred_lst, hr_label_lst)
    rmse_mean, rmse_std = compute_rmse(hr_pred_lst, hr_label_lst)
    mape_mean, mape_std = compute_mape(hr_pred_lst, hr_label_lst)
    pearson_mean, pearson_std = compute_pearson(hr_pred_lst, hr_label_lst)
    global_metrics = {
        'dataset_name': config.DATALOADER.DATASET, 'checkpoint_name': checkpoint_name,
        'snr_mean': snr_mean, 'snr_std': snr_std,
        'macc_mean': macc_mean, 'macc_std': macc_std,
        'mae_mean': mae_mean, 'mae_std': mae_std,
        'rmse_mean': rmse_mean, 'rmse_std': rmse_std,
        'mape_mean': mape_mean, 'mape_std': mape_std,
        'pearson_mean': pearson_mean, 'pearson_std': pearson_std
    }
    # print(f'Dataset: {dataset_name}\nMethod: {method_name}\nGlobal metrics: {global_metrics}')
    return global_metrics
