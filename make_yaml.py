import argparse
import pathlib
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(
        description='Make a .yaml file for YOLOv5 training.'
    )
    parser.add_argument('path', help='path to the dataset root dir')
    parser.add_argument('-o', '--output', help='output file name')
    parser.add_argument('-l', '--labels', help='path to labels file')
    args = parser.parse_args()
    if args.output is None:
        path = pathlib.Path(args.path)
        name = path.stem
        args.output = str(path / (name + '.yaml'))
    if not args.output.endswith('.yaml'):
        args.output += '.yaml'
    return args

def cfg_dataset(cfg):
    path = pathlib.Path(cfg['path'])
    texts = list(path.glob('*.txt'))

    if texts:
        for text in texts:
            cfg_dataset_internal(text, cfg, text=True)
    else:
        path_images = path / 'images'
        for p in path_images.iterdir():
            if p.is_dir():
                cfg_dataset_internal(p, cfg, text=False)

    assert any([name in cfg for name in ['train', 'val', 'test']])

def cfg_dataset_internal(path: pathlib.Path, cfg: dict, text: bool) -> None:
    names = ['train', 'val', 'test']
    matched= False
    for name in names:
        if path.name.startswith(name):
            matched = True
            break
    if not matched:
        return 
    if name in cfg:
        raise RuntimeError(f'multiple candidates exist for a {name} {"text file" if text else "directory"}: {cfg[name]} and {path}')
    cfg.update({name: str(path)})

def cfg_classes(cfg, path_labels):
    classes = []
    with open(path_labels) as f:
        for line in f:
            line = line.rstrip()
            if line not in ['__ignore__', '_background_']:
                classes.append(line)
    cfg.update({'nc': len(classes), 'names': classes})

def write_cfg(cfg, output):
    comments = dict(
        path='# dataset root dir',
        train='# train images',
        val='# val images',
        test='# test images',
        nc='# number of classes',
        names='# class names'
    )
    with open(output, 'w') as f:
        f.write('# Train/val/test sets\n')
        for k, v in cfg.items():
            if k == 'nc':
                f.write('\n# Classes\n')
            f.write(f'{k}: {v} {comments[k]}\n')

def main():
    args = parse_args()
    cfg = OrderedDict(path=args.path)
    cfg_dataset(cfg)
    cfg_classes(cfg, args.labels)
    write_cfg(cfg, args.output)

if __name__ == '__main__':
    main()
