#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import wraps

__author__ = 'cnheider'

from abc import abstractmethod
from collections import Iterable
from typing import Iterator
from itertools import count
import numpy as np

class DataIterator(Iterable):
  def __init__(self, data_iterator: Iterable = count(), meta_data:Iterable = count()):
    self._data_iterator = data_iterator
    self._meta_data = meta_data

  def __iter__(self) -> Iterator:
    '''if type(self._data_iterator) is DataIterator:
      return self.entry_point(self._data_iterator.__iter__()).__iter__()
    '''
    data = self._data_iterator.__iter__()
    generator = self.loop_entry(data)
    return generator

  @abstractmethod
  def entry_point(self, data_iterator):
    raise NotImplemented

  def loop_entry(self, data_iterator: Iterable) -> Iterator:
    for a in data_iterator:
      if isinstance(a, Iterable):
        gen = self.loop_entry(a)
        yield [b for b in gen]
      else:
        yield self.entry_point(a)


  def __getattr__(self, attr):
    return getattr(self._data_iterator, attr)

  def __evaluate__(self):
    return [a for a in self.__iter__()]

  def __getitem__(self, index):
    return self.__evaluate__()[index]

  def __len__(self):
    return len(self.__evaluate__())

class SquaringAugmentor(DataIterator):

  def entry_point(self, data) -> Iterator:
    return data ** 2

class CubingAugmentor(DataIterator):

  def entry_point(self, data) -> Iterator:
    return data ** 3

class SqueezeAugmentor(DataIterator):

  def entry_point(self, data) -> Iterator:
    return [data]

class NoiseAugmentor(DataIterator):

  def entry_point(self, data) -> Iterator:
    return data ** np.random.rand()


if __name__ == '__main__':

  data = np.ones((4, 8))
  data += data

  squared = SquaringAugmentor(data)
  expanded = SqueezeAugmentor(squared)
  cubed = CubingAugmentor(expanded)
  noised = NoiseAugmentor(cubed)

  for a,o in zip(noised, data):
    print(a)
    print(o)

  print(noised[0][0])

  print(len(noised))

  print(noised.size)
