#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from functools import reduce

__author__ = "cnheider"

from abc import abstractmethod
from collections import Iterable
from itertools import count
from typing import Iterator, Sized, Tuple
from warg import NamedOrderedDictionary as NOD

import numpy as np


class LazyPipeIterator(Iterable):
    def __init__(
        self, input=None, *, satellite_data: NOD = NOD, auto_eval=False, **kwargs
    ):

        if input is not None:
            self._input = input

            if isinstance(self._input, LazyPipeIterator):
                self._parent = self._input

        self._satellite_data = satellite_data
        self._auto_eval = auto_eval

    def __iter__(self) -> Iterator:
        return self.sample(set_name="training")

    def __len__(self):
        if hasattr(self.root, "_input"):
            if isinstance(self.root._input, Sized):
                return len(self.root._input)
        return 0

    def __next__(self):
        return self.sample()

    def __call__(self, data, **kwargs):
        return self.apply(data, **kwargs)

    def __str__(self):
        if self._auto_eval:
            return str(self.eval())

        if self._input is None:
            return super().__str__()

        return "\n".join([super().__str__(), str(self._input)])

    @property
    def parent(self):
        if hasattr(self, "_parent"):
            return self._parent

    @property
    def root(self):
        if hasattr(self, "_parent"):
            if isinstance(self._parent, LazyPipeIterator):
                return self._parent.root
            else:
                return self._parent
        else:
            return self

    @property
    def satellite_data(self):
        return self._satellite_data

    # @abstractmethod
    def func(self, data_iterator, *args, **kwargs):
        return data_iterator

    def _build(self, **kwargs):
        return None

    def apply(self, data, **kwargs):
        if self.parent is not None:
            for data in self.parent.apply(data, **kwargs):
                return self.func(data, **kwargs)
        return self.func(data, **kwargs)

    def sample(self, **kwargs):
        if self.parent is not None:
            for data, info in self.parent.sample(**kwargs):
                data = self.func(data, info=info, **kwargs)
                yield NOD.dict_of(data, info)
        else:
            for data, info in self.random_sampler(**kwargs):
                data = self.func(data, info=info, **kwargs)
                yield NOD.dict_of(data, info)

    def batch_sample(self, batch_size=64, **kwargs):
        s = []
        for _ in range(batch_size):
            s.append(self.sample(**kwargs).__next__())

        return s

    def random_sampler(self, **kwargs):
        return self._input

    def build(self, **kwargs):
        if self.parent:
            self.parent.build(**kwargs)
        self._build(**kwargs)

    @staticmethod
    def compose(*pipeline, **kwargs):
        return lambda x: reduce(lambda f, g: g(f, **kwargs), list(pipeline), x)

    @staticmethod
    def compile(*pipeline, **kwargs):
        return reduce(lambda x, y: y(x, **kwargs), list(pipeline))

    def eval(self, **kwargs):
        return [a for a in self.sample(**kwargs)]


class SquaringAugmentor(LazyPipeIterator):
    def func(self, data, *args, **kwargs):
        return data ** 2


class CubingAugmentor(LazyPipeIterator):
    def func(self, data, *args, **kwargs):
        return data ** 3


class SqueezeAugmentor(LazyPipeIterator):
    def func(self, data, *args, **kwargs):
        return [data]


class CountingAugmentor(LazyPipeIterator):
    counter = count()

    def func(self, data, *args, **kwargs):
        return data + self.counter.__next__()


class NoiseAugmentor(LazyPipeIterator):
    def func(self, data, *args, **kwargs):
        return data + np.random.rand(*(data.shape))


class ConstantAugmentor(LazyPipeIterator):
    def __init__(self, *args, constant=6, **kwargs):
        super().__init__(*args, **kwargs)
        self._constant = constant

    def func(self, data, *args, **kwargs):
        return self._constant


if __name__ == "__main__":

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
        x = NoiseAugmentor(SquaringAugmentor(CubingAugmentor()))
        a = x.apply(data)

        for i in a:
            print(i)

    data = np.ones((2, 2))
    data += data
    data2 = np.ones((5, 3))
    data3 = np.ones((1, 4))

    sample_dat = [(data, "label_x", 2), (data2, "label_y"), (data3, "label_z", 4)]

    # test1(sample_dat)
    # test2(sample_dat)
    # test3(sample_dat)
    # test4(sample_dat)
    test5(data)
