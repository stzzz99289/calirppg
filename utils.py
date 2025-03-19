import dataloader

def get_dataloader(config):
    dataset_name = config.DATALOADER.DATASET
    if dataset_name == "iBVP":
        return dataloader.ibvp.iBVPLoader(config_data=config)
    elif dataset_name == "PURE":
        return dataloader.pure.PURELoader(config_data=config)
    elif dataset_name == "UBFC-rPPG":
        return dataloader.ubfc_rppg.UBFCrPPGLoader(config_data=config)
    elif dataset_name == "UBFC-Phys":
        return dataloader.ubfc_phys.UBFCPhysLoader(config_data=config)
    elif dataset_name == "MAHNOB":
        return dataloader.mahnob.MAHNOBLoader(config_data=config)
    elif dataset_name == "Selfies-and-Videos":
        return dataloader.selfies_and_videos.SelfiesAndVideosLoader(config_data=config)
    else:
        raise ValueError(f"dataset name {dataset_name} not supported!")