import os
import os.path as osp
import hashlib
import json
import random
import sys
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.optim import Adagrad
from tqdm import tqdm

from model.MSHNet import MSHNet
from model.loss import AverageMeter, SLSIoULoss
from utils.data import IRSTD_Dataset
from utils.metric import FixedThresholdMetrics


MODEL_INPUT_MULTIPLE = 16


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise ArgumentTypeError('Boolean value expected.')


def parse_args(default_mode=None, argv=None):
    parser = ArgumentParser(description='MSHNet training and testing')

    parser.add_argument('--dataset-dir', type=str, default='datasets/IRSTD-1K')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=str2bool, default=False)
    parser.add_argument('--if-checkpoint', type=str2bool, default=False)
    parser.add_argument('--resume-path', type=str, default='')
    parser.add_argument('--save-dir', type=str, default='repro_runs')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--train-split-file',
        type=str,
        default='',
        help='Explicit training manifest (relative to dataset-dir or absolute)',
    )
    parser.add_argument(
        '--test-split-file',
        type=str,
        default='',
        help='Explicit final-test manifest (relative to dataset-dir or absolute)',
    )
    parser.add_argument(
        '--checkpoint-selection',
        choices=['fixed_last', 'legacy_test_iou'],
        default='fixed_last',
        help=(
            'fixed_last does not read test labels during training (paper default); '
            'legacy_test_iou explicitly reproduces the historical per-epoch '
            'test-set diagnostic selection and is not leakage-safe'
        ),
    )
    parser.add_argument(
        '--allow-legacy-resume',
        type=str2bool,
        default=False,
        help='allow diagnostic resume from checkpoints without a complete RNG/config contract',
    )
    parser.add_argument(
        '--allow-unsafe-legacy-resume',
        type=str2bool,
        default=False,
        help=(
            'allow unrestricted pickle loading only for a locally trusted legacy '
            'resume checkpoint; never enable this for downloaded/untrusted files'
        ),
    )

    parser.add_argument('--mode', type=str, default=default_mode or 'train', choices=['train', 'test'])
    parser.add_argument('--weight-path', type=str, default='')
    parser.add_argument(
        '--inference-warm-flag',
        type=str,
        default='auto',
        choices=['auto', 'true', 'false'],
        help='auto reads new weight metadata; legacy raw weights default to true',
    )
    parser.add_argument(
        '--test-manifest',
        type=str,
        default='auto',
        help='test manifest path; auto writes beside the weight, none disables it',
    )
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument(
        '--matching-rule',
        choices=['overlap', 'centroid'],
        default='overlap',
    )
    parser.add_argument('--centroid-distance', type=float, default=3.0)
    parser.add_argument('--connectivity', type=int, choices=[1, 2], default=2)
    parser.add_argument('--min-component-area', type=int, default=1)

    return parser.parse_args(argv)


def validate_input_size(name, value, multiple=MODEL_INPUT_MULTIPLE):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError('{} must be an integer'.format(name))
    if value <= 0:
        raise ValueError('{} must be positive'.format(name))
    if value % multiple != 0:
        raise ValueError(
            '{}={} must be divisible by {} for the four MSHNet downsampling stages'.format(
                name, value, multiple
            )
        )


def validate_args(args):
    validate_input_size('base_size', args.base_size)
    validate_input_size('crop_size', args.crop_size)
    if args.batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if args.epochs < 0:
        raise ValueError('epochs must be non-negative')
    if args.warm_epoch < -1:
        raise ValueError('warm_epoch must be at least -1')
    if args.num_workers < 0:
        raise ValueError('num_workers must be non-negative')
    if args.lr <= 0:
        raise ValueError('lr must be positive')
    if args.checkpoint_selection not in {'fixed_last', 'legacy_test_iou'}:
        raise ValueError(
            'checkpoint_selection must be fixed_last or legacy_test_iou'
        )
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError('score_threshold must be in [0, 1]')
    if args.centroid_distance < 0.0:
        raise ValueError('centroid_distance must be non-negative')
    if args.min_component_area <= 0:
        raise ValueError('min_component_area must be positive')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(device_arg):
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested, but torch.cuda.is_available() is False')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def torch_load_compat(
    path,
    map_location,
    *,
    full_checkpoint,
    allow_unsafe_legacy=False,
):
    """Load through PyTorch's restricted unpickler before any legacy fallback.

    New full training checkpoints are restricted-loader compatible.  Historical
    checkpoints containing unsupported pickle globals can be loaded only when
    the caller explicitly marks the resume artifact as locally trusted.
    """

    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError as exc:
        if 'weights_only' not in str(exc):
            raise
        if not (full_checkpoint and allow_unsafe_legacy):
            raise RuntimeError(
                'Safe checkpoint loading requires a PyTorch version with '
                'weights_only support. Use --allow-unsafe-legacy-resume true '
                'only for a locally trusted legacy resume artifact.'
            ) from exc
        warnings.warn(
            'Loading a locally trusted legacy resume checkpoint with the '
            'unrestricted pickle loader; arbitrary code may execute.',
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location)
    except Exception:
        if not (full_checkpoint and allow_unsafe_legacy):
            raise
        warnings.warn(
            'Loading a locally trusted legacy resume checkpoint with '
            'weights_only=False; arbitrary code may execute.',
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError('The weight file must contain a state-dict mapping')

    state_dict = checkpoint
    for key in ('state_dict', 'net', 'model_state_dict'):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state_dict = candidate
            break

    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        raise ValueError('Could not find a valid model state_dict in the weight file')

    # Always load into the unwrapped module.  This makes CPU, one-GPU and
    # DataParallel checkpoints mutually portable.
    return {
        (key[len('module.'):] if key.startswith('module.') else key): value
        for key, value in state_dict.items()
    }


def load_model_state_dict(model, checkpoint, *, strict=True):
    state_dict = extract_model_state_dict(checkpoint)
    return unwrap_model(model).load_state_dict(state_dict, strict=strict)


def canonical_model_state_dict(model):
    return unwrap_model(model).state_dict()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    folder = osp.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def atomic_torch_save(path, payload):
    folder = osp.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temporary = path + '.tmp'
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        'schema_version': 2,
        'python': random.getstate(),
        # NumPy's native tuple embeds an ndarray whose pickle globals are
        # rejected by torch.load(weights_only=True). Keep equivalent primitive
        # metadata plus a tensor so newly written resume files stay safe-loadable.
        'numpy': {
            'bit_generator': str(numpy_state[0]),
            'state': torch.from_numpy(numpy_state[1].copy()),
            'position': int(numpy_state[2]),
            'has_gauss': int(numpy_state[3]),
            'cached_gaussian': float(numpy_state[4]),
        },
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [],
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    if not isinstance(state, Mapping):
        raise TypeError('rng_state must be a mapping')
    required = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    missing = required.difference(state)
    if missing:
        raise ValueError('rng_state is missing: ' + ', '.join(sorted(missing)))
    random.setstate(state['python'])
    numpy_state = state['numpy']
    if isinstance(numpy_state, Mapping):
        numpy_tensor = numpy_state.get('state')
        if not isinstance(numpy_tensor, torch.Tensor):
            raise TypeError('rng_state.numpy.state must be a tensor')
        np.random.set_state(
            (
                str(numpy_state.get('bit_generator', 'MT19937')),
                numpy_tensor.detach().to(device='cpu', dtype=torch.uint32).numpy(),
                int(numpy_state.get('position', 0)),
                int(numpy_state.get('has_gauss', 0)),
                float(numpy_state.get('cached_gaussian', 0.0)),
            )
        )
    else:
        # Reached only after the caller explicitly opted into a trusted legacy
        # pickle; retain compatibility with the old NumPy RNG tuple.
        np.random.set_state(numpy_state)
    torch.set_rng_state(state['torch_cpu'].detach().to(device='cpu'))
    cuda_states = state['torch_cuda']
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError('checkpoint contains CUDA RNG state but CUDA is unavailable')
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                'CUDA topology changed across resume: saved={}, current={}'.format(
                    len(cuda_states), torch.cuda.device_count()
                )
            )
        torch.cuda.set_rng_state_all(
            [item.detach().to(device='cpu') for item in cuda_states]
        )


def dataset_split_contract(dataset):
    """Return the immutable ordered-ID contract for one dataset split."""

    split_path = osp.abspath(dataset.list_dir)
    encoded_ids = json.dumps(
        list(dataset.image_ids),
        ensure_ascii=False,
        separators=(',', ':'),
    ).encode('utf-8')
    with open(split_path, 'rb') as handle:
        split_sha256 = hashlib.sha256(handle.read()).hexdigest()
    return {
        'loader_mode': str(dataset.mode),
        'role': str(dataset.split_role),
        'path': split_path,
        'sha256': split_sha256,
        'ordered_ids_sha256': hashlib.sha256(encoded_ids).hexdigest(),
        'num_samples': int(len(dataset)),
    }


def resolve_inference_warm_flag(requested, checkpoint):
    if requested not in ('auto', 'true', 'false'):
        raise ValueError('requested inference warm flag must be auto, true, or false')
    if requested != 'auto':
        return requested == 'true'
    if isinstance(checkpoint, Mapping) and 'warm_flag' in checkpoint:
        return bool(checkpoint['warm_flag'])
    # Official/legacy raw state dicts normally represent fully trained models.
    return True


class Trainer(object):
    def __init__(self, args):
        validate_args(args)
        self.args = args
        self.start_epoch = 0
        self.mode = args.mode
        self.device = select_device(args.device)
        self.inference_warm_flag = True
        self.loaded_epoch = None
        self.loaded_weight_metadata = {}
        self.train_loader = None
        self.test_loader = None
        self.train_generator = None
        self.train_split_contract = None
        self.test_split_contract = None

        if args.mode == 'train':
            trainset = IRSTD_Dataset(args, mode='train')
            # Resolve the frozen test manifest for a split-overlap audit only.
            # The default fixed-last path never opens test images or masks.
            testset = IRSTD_Dataset(args, mode='test')
            overlap = sorted(set(trainset.image_ids).intersection(testset.image_ids))
            if overlap:
                raise ValueError(
                    'train/test split leakage: ' + ', '.join(overlap[:10])
                )
            self.train_split_contract = dataset_split_contract(trainset)
            self.test_split_contract = dataset_split_contract(testset)
            self.train_generator = torch.Generator()
            self.train_generator.manual_seed(int(args.seed))
            self.train_loader = Data.DataLoader(
                trainset,
                args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=self.device.type == 'cuda',
                persistent_workers=False,
                generator=self.train_generator,
            )
            if args.checkpoint_selection == 'legacy_test_iou':
                self.test_loader = self._build_test_loader(testset)
        else:
            testset = IRSTD_Dataset(args, mode='test')
            self.test_split_contract = dataset_split_contract(testset)
            self.test_loader = self._build_test_loader(testset)

        model = MSHNet(3)
        if args.multi_gpus and self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            print('use ' + str(torch.cuda.device_count()) + ' gpus')
            model = nn.DataParallel(model)
        model.to(self.device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.best_iou = float('-inf')
        self.warm_epoch = args.warm_epoch
        self.config = self._training_config() if args.mode == 'train' else None

        if args.mode == 'train':
            if args.if_checkpoint:
                self._resume_checkpoint(args.resume_path or args.weight_path)
            else:
                self.save_folder = self._new_save_folder(args.save_dir)
                atomic_json(osp.join(self.save_folder, 'config.json'), self.config)

        if args.mode == 'test':
            if not args.weight_path:
                raise ValueError('--weight-path is required in test mode')
            self._load_weight(args.weight_path)

    def _build_test_loader(self, dataset):
        return Data.DataLoader(
            dataset,
            1,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
            pin_memory=self.device.type == 'cuda',
            persistent_workers=False,
        )

    @staticmethod
    def _new_save_folder(save_dir):
        root = osp.abspath(save_dir)
        os.makedirs(root, exist_ok=True)
        stamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        candidate = osp.join(root, 'MSHNet-' + stamp)
        suffix = 1
        while osp.exists(candidate):
            candidate = osp.join(root, 'MSHNet-{}-{}'.format(stamp, suffix))
            suffix += 1
        os.makedirs(candidate)
        return candidate

    def _training_config(self):
        config = dict(vars(self.args))
        config['dataset_dir'] = osp.abspath(self.args.dataset_dir)
        config['split_contracts'] = {
            'train': self.train_split_contract,
            'test': self.test_split_contract,
        }
        config['selection_uses_test_labels'] = bool(
            self.args.checkpoint_selection == 'legacy_test_iou'
        )
        return config

    def _validate_resume_config(self, checkpoint):
        saved = checkpoint.get('config') if isinstance(checkpoint, Mapping) else None
        if not isinstance(saved, Mapping):
            if self.args.allow_legacy_resume:
                warnings.warn(
                    'legacy checkpoint has no config contract; continuation is diagnostic only',
                    RuntimeWarning,
                )
                return False
            raise ValueError(
                'resume checkpoint has no config contract; pass '
                '--allow-legacy-resume true only for a diagnostic continuation'
            )
        keys = (
            'dataset_dir',
            'batch_size',
            'lr',
            'warm_epoch',
            'base_size',
            'crop_size',
            'num_workers',
            'seed',
            'multi_gpus',
            'checkpoint_selection',
            'split_contracts',
        )
        mismatches = []
        for key in keys:
            if saved.get(key) != self.config.get(key):
                mismatches.append(
                    '{}: saved={!r}, current={!r}'.format(
                        key, saved.get(key, '<missing>'), self.config.get(key, '<missing>')
                    )
                )
        if mismatches:
            raise ValueError(
                'resume config mismatch; start a new run instead:\n- '
                + '\n- '.join(mismatches)
            )
        return True

    @staticmethod
    def _select_device(device_arg):
        # Kept for compatibility with callers of the previous helper.
        return select_device(device_arg)

    @staticmethod
    def _normalise_state_dict(checkpoint):
        # Kept for compatibility with callers of the previous helper.
        return extract_model_state_dict(checkpoint)

    def _load_weight(self, path):
        checkpoint = torch_load_compat(
            path,
            map_location=self.device,
            full_checkpoint=False,
        )
        load_model_state_dict(self.model, checkpoint)
        if isinstance(checkpoint, Mapping):
            if 'epoch' in checkpoint:
                self.loaded_epoch = int(checkpoint['epoch'])
            self.loaded_weight_metadata = {
                key: checkpoint.get(key)
                for key in (
                    'selection_rule',
                    'checkpoint_selection',
                    'inference_head',
                    'train_loss',
                )
                if key in checkpoint
            }
        self.inference_warm_flag = resolve_inference_warm_flag(
            self.args.inference_warm_flag,
            checkpoint,
        )

    def _resume_checkpoint(self, path):
        if not path:
            raise ValueError('--resume-path or --weight-path is required when --if-checkpoint true')

        checkpoint = torch_load_compat(
            path,
            map_location=self.device,
            full_checkpoint=True,
            allow_unsafe_legacy=self.args.allow_unsafe_legacy_resume,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError('resume checkpoint must be a mapping')
        contract_verified = self._validate_resume_config(checkpoint)
        load_model_state_dict(self.model, checkpoint)
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        elif not self.args.allow_legacy_resume:
            raise ValueError('resume checkpoint has no optimizer state')
        self.start_epoch = int(checkpoint.get('epoch', -1)) + 1
        self.best_iou = float(checkpoint.get('best_test_iou', float('-inf')))

        rng_state = checkpoint.get('rng_state')
        generator_state = checkpoint.get('train_generator_state')
        reproducible = isinstance(rng_state, Mapping) and isinstance(
            generator_state, torch.Tensor
        )
        if reproducible:
            restore_rng_state(rng_state)
            self.train_generator.set_state(generator_state.detach().to(device='cpu'))
        elif not self.args.allow_legacy_resume:
            raise ValueError(
                'resume checkpoint lacks RNG/DataLoader generator state; '
                'use --allow-legacy-resume true only for diagnostics'
            )
        else:
            warnings.warn(
                'legacy resume lacks RNG/DataLoader state; continuation is not reproducible',
                RuntimeWarning,
            )

        self.save_folder = osp.dirname(path) or self.args.save_dir
        os.makedirs(self.save_folder, exist_ok=True)
        self.resume_reproducibility = (
            'epoch_boundary_rng_restored'
            if reproducible and contract_verified
            else 'legacy_unverified'
        )

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        tag = epoch > self.warm_epoch

        for _, (data, mask) in enumerate(tbar):
            data = data.to(self.device, non_blocking=True)
            labels = mask.to(self.device, non_blocking=True)

            masks, pred = self.model(data, tag)
            loss = self.loss_fun(pred, labels, self.warm_epoch, epoch)

            scaled_labels = labels
            for j in range(len(masks)):
                if j > 0:
                    scaled_labels = self.down(scaled_labels)
                loss = loss + self.loss_fun(masks[j], scaled_labels, self.warm_epoch, epoch)

            loss = loss / (len(masks) + 1)

            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss at epoch {}'.format(epoch))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float('inf'),
                error_if_nonfinite=True,
            )
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
        return {'loss_total': float(losses.avg)}

    def test(self, epoch):
        if self.test_loader is None:
            raise RuntimeError(
                'test loader is unavailable: fixed_last training does not read test data'
            )
        self.model.eval()
        evaluator = FixedThresholdMetrics(
            self.args.score_threshold,
            matching_rule=self.args.matching_rule,
            centroid_distance=self.args.centroid_distance,
            connectivity=self.args.connectivity,
            min_component_area=self.args.min_component_area,
        )
        tbar = tqdm(self.test_loader)
        tag = self.inference_warm_flag if self.mode == 'test' else epoch > self.warm_epoch

        with torch.no_grad():
            for _, (data, mask) in enumerate(tbar):
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                _, pred = self.model(data, tag)
                evaluator.update(torch.sigmoid(pred), mask)
                current_iou = (
                    evaluator.intersection / evaluator.union
                    if evaluator.union
                    else 0.0
                )
                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, current_iou))

        metrics = evaluator.get()
        inference_head = 'multiscale_fusion' if tag else 'single_output_0'
        metrics['inference_head'] = inference_head
        if self.mode == 'test':
            print('mIoU: ' + str(metrics['mIoU']))
            print('Pd: ' + str(metrics['Pd']))
            print('Fa pixels / million: ' + str(metrics['Fa_per_million_pixels']))
            print(
                'Fa components / megapixel: '
                + str(metrics['Fa_components_per_megapixel'])
            )
            print('Inference head: ' + inference_head)
            self._write_test_manifest(epoch=epoch, metrics=metrics)
        return metrics

    def save_epoch(self, epoch, train_metrics, test_metrics=None):
        warm_flag = bool(epoch > self.warm_epoch)
        state_dict = canonical_model_state_dict(self.model)
        inference_head = 'multiscale_fusion' if warm_flag else 'single_output_0'
        base_payload = {
            'state_dict': state_dict,
            'epoch': int(epoch),
            'warm_flag': warm_flag,
            'inference_head': inference_head,
            'train_loss': float(train_metrics['loss_total']),
            'checkpoint_selection': self.args.checkpoint_selection,
            'config': self.config,
        }
        atomic_torch_save(
            osp.join(self.save_folder, 'last.pkl'),
            {**base_payload, 'selection_rule': 'last_complete_epoch'},
        )

        selected = self.args.checkpoint_selection == 'fixed_last'
        selection_rule = 'fixed_last_complete_epoch'
        if self.args.checkpoint_selection == 'legacy_test_iou':
            if test_metrics is None:
                raise ValueError('legacy_test_iou requires per-epoch test metrics')
            selected = float(test_metrics['mIoU']) > self.best_iou
            if selected:
                self.best_iou = float(test_metrics['mIoU'])
            selection_rule = 'maximum_test_iou_diagnostic_test_labels_used'
        if selected:
            atomic_torch_save(
                osp.join(self.save_folder, 'weight.pkl'),
                {
                    **base_payload,
                    'selection_rule': selection_rule,
                    'selection_uses_test_labels': bool(
                        self.args.checkpoint_selection == 'legacy_test_iou'
                    ),
                    'selection_test_metrics': test_metrics,
                },
            )

        epoch_record = {
            'epoch': int(epoch),
            'train': dict(train_metrics),
            'test_diagnostic': test_metrics,
            'checkpoint_selection': self.args.checkpoint_selection,
            'selection_uses_test_labels': bool(
                self.args.checkpoint_selection == 'legacy_test_iou'
            ),
            'warm_flag': warm_flag,
            'inference_head': inference_head,
        }
        checkpoint = {
            'net': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'epoch': int(epoch),
            'best_test_iou': float(self.best_iou),
            'warm_flag': warm_flag,
            'inference_head': inference_head,
            'checkpoint_selection': self.args.checkpoint_selection,
            'config': self.config,
            'last_metrics': epoch_record,
            'rng_state': capture_rng_state(),
            'train_generator_state': self.train_generator.get_state().clone(),
            'resume_reproducibility': getattr(
                self, 'resume_reproducibility', 'fresh_run'
            ),
        }
        atomic_torch_save(osp.join(self.save_folder, 'checkpoint.pkl'), checkpoint)
        with open(
            osp.join(self.save_folder, 'metrics.jsonl'), 'a', encoding='utf-8'
        ) as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False, sort_keys=True) + '\n')

    def _write_test_manifest(
        self,
        *,
        epoch,
        metrics,
    ):
        requested = self.args.test_manifest
        if requested.lower() == 'none':
            return
        if requested == 'auto':
            folder = osp.dirname(osp.abspath(self.args.weight_path))
            path = osp.join(folder, 'test_manifest.json')
        else:
            path = osp.abspath(requested)
        folder = osp.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        manifest = {
            'schema_version': 2,
            'dataset_dir': osp.abspath(self.args.dataset_dir),
            'weight_path': osp.abspath(self.args.weight_path),
            'weight_sha256': file_sha256(osp.abspath(self.args.weight_path)),
            'weight_metadata': self.loaded_weight_metadata,
            'test_split': self.test_split_contract,
            'loaded_epoch': (
                int(self.loaded_epoch)
                if self.loaded_epoch is not None
                else None
            ),
            'display_epoch': int(epoch),
            'inference_warm_flag': bool(self.inference_warm_flag),
            'inference_head': metrics['inference_head'],
            'base_size': int(self.args.base_size),
            'spatial_protocol': 'fixed_resize_diagnostic',
            'formal_native_resolution_metric_eligible': False,
            'metric_protocol': {
                'score_threshold': float(self.args.score_threshold),
                'score_comparator': '>=',
                'mask_binarization': '>0',
                'matching_rule': self.args.matching_rule,
                'centroid_distance': float(self.args.centroid_distance),
                'connectivity': int(self.args.connectivity),
                'min_component_area': int(self.args.min_component_area),
                'pixel_fa_definition': 'prediction AND NOT ground_truth',
            },
            'metrics': {
                key: value
                for key, value in metrics.items()
                if key != 'inference_head'
            },
            'test_labels_used': True,
            'command': list(sys.argv),
        }
        atomic_json(path, manifest)
        print('Test manifest: ' + path)


def main(default_mode=None):
    args = parse_args(default_mode)
    validate_args(args)
    seed_everything(args.seed)

    trainer = Trainer(args)

    if trainer.mode == 'train':
        for epoch in range(trainer.start_epoch, args.epochs):
            train_metrics = trainer.train(epoch)
            test_metrics = None
            if args.checkpoint_selection == 'legacy_test_iou':
                test_metrics = trainer.test(epoch)
            trainer.save_epoch(epoch, train_metrics, test_metrics)
    else:
        display_epoch = trainer.loaded_epoch if trainer.loaded_epoch is not None else 0
        trainer.test(display_epoch)


if __name__ == '__main__':
    main()
