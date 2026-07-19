import torch
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import random

from data_ext.split_utils import (
    ensure_unique_sample_ids,
    read_split_file,
    resolve_split_file,
)
from data_ext.mask_alignment import align_mask_to_image


_RASTER_SUFFIXES = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
_RESAMPLING = getattr(Image, 'Resampling', Image)


class IRSTD_Dataset(Data.Dataset):
    def __init__(self, args, mode='train'):
        dataset_dir = args.dataset_dir
        split_attributes = {
            'train': 'train_split_file',
            # This repository historically calls its test pass ``val``.
            # Preserve that API alias while recording the true split role.
            'val': 'test_split_file',
            'test': 'test_split_file',
        }
        if mode not in split_attributes:
            raise ValueError('Unknown dataset mode: {}'.format(mode))
        explicit_split = getattr(args, split_attributes[mode], None) or None
        self.list_dir = self._find_split_file(
            dataset_dir,
            mode,
            split_file=explicit_split,
        )
        self.imgs_dir = osp.join(dataset_dir, 'images')
        self.label_dir = osp.join(dataset_dir, 'masks')

        self.names = read_split_file(self.list_dir)
        self.image_ids = ensure_unique_sample_ids(self.names)

        self.mode = mode
        self.split_role = 'train' if mode == 'train' else 'test'
        self.crop_size = int(args.crop_size)
        self.base_size = int(args.base_size)
        if self.crop_size <= 0 or self.base_size <= 0:
            raise ValueError('crop_size and base_size must be positive')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = osp.splitext(self.names[i])[0]
        img_path = self._resolve_image_path(self.imgs_dir, name, is_mask=False)
        label_path = self._resolve_image_path(self.label_dir, name, is_mask=True)

        with Image.open(img_path) as opened_image:
            img = opened_image.convert('RGB')
        with Image.open(label_path) as opened_mask:
            mask = opened_mask.convert('L')
        mask, _ = align_mask_to_image(mask, img, name)

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode in {'val', 'test'}:
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unknown self.mode")

        img = self.transform(img)
        # Use the same binary-label convention as score export and formal
        # evaluation.  A few local masks contain anti-aliased non-zero pixels.
        mask = (transforms.ToTensor()(mask) > 0).to(dtype=torch.float32)
        return img, mask

    def __len__(self):
        return len(self.names)

    @staticmethod
    def _find_split_file(dataset_dir, mode, split_file=None):
        if mode not in {'train', 'val', 'test'}:
            raise ValueError('Unknown dataset mode: {}'.format(mode))
        split = 'test' if mode == 'val' else mode
        return str(
            resolve_split_file(
                dataset_dir,
                split_file,
                split=split,
            )
        )

    @staticmethod
    def _resolve_image_path(root, name, is_mask=False):
        """Resolve a raster without ever accepting XML or other sidecar files.

        Some NUAA-style layouts use ``<image_id>_pixels0.png`` next to XML
        annotations.  The old ``name.*`` fallback could silently select XML.
        Exact image ids remain preferred for the datasets in this repository.
        """

        stems = [name]
        if is_mask:
            stems.append(name + '_pixels0')

        for stem in stems:
            matches = [
                osp.join(root, stem + suffix)
                for suffix in _RASTER_SUFFIXES
                if osp.isfile(osp.join(root, stem + suffix))
            ]
            if len(matches) > 1:
                raise ValueError(
                    'Multiple raster files found for "{}" under {}: {}'.format(
                        name, root, ', '.join(osp.basename(path) for path in matches)
                    )
                )
            if matches:
                return matches[0]

        expected = [stem + '<raster-ext>' for stem in stems]
        raise FileNotFoundError(
            'Cannot find image/mask for "{}" under {}. Expected {}'.format(
                name, root, ' or '.join(expected)
            )
        )

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), _RESAMPLING.BILINEAR)
        mask = mask.resize((ow, oh), _RESAMPLING.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask


    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), _RESAMPLING.BILINEAR)
        mask = mask.resize((base_size, base_size), _RESAMPLING.NEAREST)

        return img, mask
