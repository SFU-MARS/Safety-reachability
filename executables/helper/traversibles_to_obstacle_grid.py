import argparse
import pickle
from pathlib import Path
import numpy as np


def transform(input_file, output_file):
    with open(input_file, 'rb') as f:
        pkl_data = pickle.load(f)

    traversible = pkl_data['traversible'].T

    grid = np.ones(traversible.shape, dtype=float)
    grid[traversible == True] = 1
    grid[traversible == False] = -1

    if output_file.is_file():
        grid_old = np.load(output_file)
        same = np.all((grid - grid_old) == 0)
        print(output_file, 'exists, same:', same)

    output_file.parents[0].mkdir(exist_ok=True, parents=True)
    np.save(output_file, grid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i','--input_file',
        type=Path,
        default='Database/LB_WayPtNav_Data/stanford_building_parser_dataset/traversibles/area3/data.pkl',
    )
    parser.add_argument(
        '-o','--output_file', type=Path, default='obstacle_grid_2d.npy'
    )
    args = parser.parse_args()

    transform(**args.__dict__)
