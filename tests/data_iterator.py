#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

from nisse import CubingAugmentor, LazyPipeIterator, NoiseAugmentor, SquaringAugmentor

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """


def test1(data):
    x = SquaringAugmentor(data)
    x = CubingAugmentor(x)
    x = NoiseAugmentor(x)

    for i in x:
        print(i)


def test2(data):
    x = LazyPipeIterator.compose(SquaringAugmentor, CubingAugmentor, NoiseAugmentor)

    for i in x(data):
        print(i)


def test3(data):
    x = LazyPipeIterator.compile(
        data, SquaringAugmentor, CubingAugmentor, NoiseAugmentor
    )

    for i in x:
        print(i)


def test4(data):
    data = NoiseAugmentor(CubingAugmentor(SquaringAugmentor(data)))

    for i in data.sample(sess="sess"):
        print(i)


def test5(data):
    pipeline = NoiseAugmentor(SquaringAugmentor(CubingAugmentor()))
    a = pipeline.apply(data)

    for i in a:
        print(i)


if __name__ == "__main__":

    data = numpy.ones((2, 2))
    data += data
    data2 = numpy.ones((5, 3))
    data3 = numpy.ones((1, 4))

    sample_dat = [(data, "label_x", 2), (data2, "label_y"), (data3, "label_z", 4)]

    # test1(sample_dat)
    # test2(sample_dat)
    # test3(sample_dat)
    test4(sample_dat)
    # test5(sample_dat)
