import argparse
from pathlib import Path
from utils import get_classes
import shutil
import sys
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'root', type=Path,
        help='dataset root dir'
    )
    parser.add_argument(
        '-p', '--permutation', nargs='+', type=int, required=True,
        help='The i-th element of this list corresponds to the new class label numbered permutation[i]. '
             'This does not have to be a permutation in the mathematical sense.'
    )
    parser.add_argument(
        '-n', '--names', nargs='+',
        help='list of new class labels into which the old labels will be transformed'
    )
    parser.add_argument(
        '-b', '--backup', action='store_true',
        help='if specified, a copy of the original labels will be made beforehand.'
    )
    args = parser.parse_args()
    return args

def unique(lst):
    ret = []
    for elem in lst:
        if elem not in ret:
            ret.append(elem)
    return ret # list(set(lst)) does not always preserve the order

def show_permutation(permutation, classes_old, classes_new):
    message = 'Replacement\n' + '\n'.join([f'  {classes_old[i]} --> {classes_new[new]}' for i, new in enumerate(permutation)])
    print(message)

def make_backup(path, root):
    # for [image_name].txt files
    backup_dir = root / '__backup__'
    backup_path = backup_dir / path.relative_to(root)
    # print(f'path={path}, root={root}, backup_dir={backup_dir}')
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, backup_path)
    # for the labels.txt file
    labels_txt = root / 'labels.txt'
    if labels_txt.exists():
        labels_txt_backup = backup_dir / labels_txt.relative_to(root)
        # print(f'labels_txt={labels_txt}, labels_txt_backup={labels_txt_backup}')
        labels_txt_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(labels_txt, labels_txt_backup)

def rewrite_label(path, permutation):
    with open(path) as f:
        lines = []
        for line in f.readlines():
            idx = line.find(' ')
            i_class_old = int(line[:idx])
            i_class_new = permutation[i_class_old]
            line = f'{i_class_new} {line[idx+1:]}'
            lines.append(line)
    with open(path, 'w') as f:
        f.writelines(lines)

def make_labels_txt(root, classes):
    with open(root / 'labels.txt', 'w') as f:
        for class_name in classes:
            f.write(class_name + '\n')

def main(*, root, permutation, names, backup):
    classes_old = get_classes(root / 'labels.txt')
    n_class_old = len(classes_old)
    classes_new = names
    assert len(classes_old) == len(permutation), 'label permutation is not fully specified'
    show_permutation(permutation=permutation, classes_old=classes_old, classes_new=classes_new)
    while True:
        ans = input('Are you ok? (y/n) --> ')
        if ans not in ['y', 'yes', 'n', 'no']:
            print('Answer with y (yes) or n (no).')
            continue
        if ans.startswith('n'):
            sys.exit(1)
        break
    
    if backup:
        for path in tqdm((root / 'labels').rglob('*.txt'), desc='Makeing backups...'):
            make_backup(path, root)
    for path in tqdm((root / 'labels').rglob('*.txt'), desc='Rewriteing labels...'):
        rewrite_label(path, permutation)
    make_labels_txt(root, classes_new)
    
    print('done')

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))