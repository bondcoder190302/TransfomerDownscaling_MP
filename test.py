import logging
import torch
from os import path as osp
import sys

# --- Kaggle Compatibility Patch ---
# Prevents ImportError: cannot import name 'functional_tensor' from 'torchvision.transforms'
try:
    import torchvision.transforms.functional as TF
    sys.modules['torchvision.transforms.functional_tensor'] = TF
except ImportError:
    pass
# ----------------------------------

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options

def test_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt = parse_options(root_path, is_train=False)

    # make experiments and log dirs
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # build dataloaders
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # build model
    model = build_model(opt)

    for dataloader in test_loaders:
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = opt['val'].get('metrics') is not None
        
        # Check if saving visuals or npy is enabled in your YAML
        save_img = opt['val'].get('save_img', False)
        
        if opt['dist']:
            model.dist_validation(dataloader, 0, None, save_img)
        else:
            model.nondist_validation(dataloader, 0, None, save_img)

        if with_metrics:
            log_str = f'Validation {dataset_name}\n'
            for metric, value in model.get_current_metrics().items():
                log_str += f'\t # {metric}: {value:.4f}\n'
            logger.info(log_str)

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    test_pipeline(root_path)