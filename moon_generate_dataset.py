"""
Generate 1000 pairs of points as train set and another 2000 pairs as test set. Each pair of points are placed respectively in region A and region B. Both A and B are moon-shaped, defined by:
   r, the radius
   w, the width
Distance between their bottoms is d.
For more information, see "Neural Network and Machine Learning by Simon Haykin, Chinese version, Page 36.
"""


import numpy as np
import math
import random

NAME_PATTERN = 'moon_dataset_pairs{}_r{}_w{}_d{}'
TRAIN_PLUS_TEST_SIZE = 3000
RADIUS = 10
WIDTH = 6
D = -4

def random_point_on_half_ring(radius, width):
    upper_r = radius + width/2
    lower_r = radius - width/2
    upper_r_square = upper_r * upper_r
    lower_r_square = lower_r * lower_r
    while True:
        x = random.uniform(-upper_r, upper_r)
        y = random.uniform(0, upper_r)
        square_mag = x*x + y*y
        if square_mag > upper_r_square or square_mag < lower_r_square:
            continue
        break
    return (x, y)

def transform_A_to_B(point, d, r):
    (x, y) = point
    y = -y
    y -= d
    x += r
    return (x, y)

def test():
    for i in range(1000):
        print(random_point_on_half_ring(10, 6))

def generate_dataset(train_plus_test_size, radius, width, d):
    ds = np.zeros([train_plus_test_size*2, 3])
    for i in range(train_plus_test_size):
        ds[i*2] = random_point_on_half_ring(radius, width) + (0,)
        ds[i*2+1] = transform_A_to_B(random_point_on_half_ring(radius, width),
                                    d, radius) + (1,)
    return ds

def main():
    ds = generate_dataset(TRAIN_PLUS_TEST_SIZE, RADIUS, WIDTH, D)
    ds_file_name = NAME_PATTERN.format(TRAIN_PLUS_TEST_SIZE, RADIUS, WIDTH, D)
    np.save(ds_file_name, ds)
    print('Successfully saved to: ', ds_file_name)
    input('Press enter to continue..')

main()
