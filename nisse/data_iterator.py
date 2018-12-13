#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'

from abc import abstractmethod
from collections import Iterable
from itertools import count
from typing import Iterator

import numpy as np


class DataIterator(Iterable):
  def __init__(self,
               iterable: Iterable = count(),
               *,
               satellite_data: Iterable = count(),
               auto_inner_loop=False,
               **kwargs):
    self._iterable = iterable
    self._satellite_data = satellite_data
    self._auto_inner_loop = auto_inner_loop

  def __iter__(self) -> Iterator:
    '''if type(self._data_iterator) is DataIterator:
      return self.entry_point(self._data_iterator.__iter__()).__iter__()
    '''
    data = self._iterable.__iter__()
    generator = self.loop_entry(data)
    return generator

  @abstractmethod
  def entry_point(self, data_iterator):
    raise NotImplemented

  def loop_entry(self, data_iterator: Iterable) -> Iterator:
    for a in data_iterator:
      if self._auto_inner_loop:
        #if isinstance(a, Image.Image):
        #  yield self.entry_point(a)
        #elif isinstance(a, Iterable):
        if isinstance(a, Iterable):
          gen = self.loop_entry(a)
          yield [b for b in gen]
        else:
          yield self.entry_point(a)
      else:
        yield self.entry_point(a)

  def __getattr__(self, attr):
    return getattr(self._iterable, attr)

  def as_list(self):
    return [a for a in self.__iter__()]

  def as_dict(self):
    return {k:v for k,v in self.__iter__()}

  def __getitem__(self, index):
    if isinstance(index,str):
      return self.as_dict()[index]
    else:
      return self.as_list()[index]

  def __len__(self):
    return len(self.as_list())

  def get_satellite_data(self):
    return self._satellite_data

  def __str__(self):
    return str(self.as_list())

  def __contains__(self, item):
    return item in self.as_dict()

class SquaringAugmentor(DataIterator):

  def entry_point(self, data) :
    return data ** 2


class CubingAugmentor(DataIterator):

  def entry_point(self, data) :
    return data ** 3


class SqueezeAugmentor(DataIterator):

  def entry_point(self, data):
    return [data]


class NoiseAugmentor(DataIterator):

  def entry_point(self, data) :
    return data ** np.random.rand()


class ConstantAugmentor(DataIterator):
  def __init__(self, iterable: Iterable = count(), *, satellite_data: Iterable = count(), constant=6,**kwargs):
    super().__init__(iterable, satellite_data=satellite_data,**kwargs)
    self._constant= constant

  def entry_point(self, data):
    return self._constant


if __name__ == '__main__':

  data = np.ones((4, 8))
  data += data

  squared = SquaringAugmentor(data,auto_inner_loop=True)
  expanded = SqueezeAugmentor(squared,auto_inner_loop=True)
  cubed = CubingAugmentor(expanded,auto_inner_loop=True)
  noised = NoiseAugmentor(cubed,auto_inner_loop=True)
  constants = ConstantAugmentor(noised,constant=23,auto_inner_loop=True)

  for a, o in zip(noised, data):
    print(a)
    print(o)

  print(constants)

  print(noised[0][0])

  print(len(noised))

  print(noised.size)
