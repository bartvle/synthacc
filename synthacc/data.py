"""
The 'data' module.
"""


from abc import ABC
import random

from .apy import (Object, is_pos_integer, is_string)


class DataRecord(ABC, Object):
    """
    A data record from a database.
    """

    def __init__(self, key, validate=True):
        """
        key: integer
        """
        if validate is True:
            assert(is_pos_integer(key))

        self._key = key

    @property
    def key(self):
        """
        """
        return self._key


class DataBase(ABC, Object):
    """
    A database of data records.
    """

    def __init__(self, records, rec_cls, validate=True):
        """
        """
        if validate is True:
            assert(type(records) is list and issubclass(rec_cls, DataRecord))

            ## Check if records are of type rec_cls and keys are unique
            keys = []
            for r in records:
                assert(type(r) is rec_cls)
                keys.append(r.key)
            assert(len(set(keys)) == len(records))

        self._records = records
        self._rec_cls = rec_cls

    def __len__(self):
        """
        """
        return len(self._records)

    def __iter__(self):
        """
        """
        for r in self._records:
            yield r

    @property
    def keys(self):
        """
        """
        return [r.key for r in self]

    def get(self, key, validate=True):
        """
        """
        if validate is True:
            assert(is_pos_integer(key))

        for r in self:
            if r.key == key:
                return r

        raise LookupError('No record with key %i' % key)


class LogicTreeLeaf(Object):
    """
    End point of logic tree.
    """

    def __init__(self, path, prob, validate=True):
        """
        """
        if validate is True:
            pass

        self._path = path
        self._prob = prob

    @property
    def path(self):
        """
        """
        return self._path

    @property
    def prob(self):
        """
        """
        return self._prob

    def get(self, name):
        """
        """
        for l, _, v in self._path:
            if l == name:
                return v
        raise


class LogicTreeBranch(Object):
    """
    """

    def __init__(self, name, value, validate=True):
        """
        """
        if validate is True:
            assert(is_string(name))

        self._name = name
        self._value = value
        self._next = None

    @property
    def name(self):
        """
        """
        return self._name

    @property
    def value(self):
        """
        """
        return self._value

    @property
    def next(self):
        """
        """
        return self._next


class LogicTreeLevel(Object):
    """
    """

    def __init__(self, name, branches, validate=True):
        """
        """
        if validate is True:
            assert(is_string(name))
            assert(type(branches) is list)
            assert(len(branches) > 1)
            for b in branches:
                assert(len(b) == 2)

        branches = [LogicTreeBranch(*b, validate=validate) for b in branches]
    
        self._name = name
        self._branches = branches

    def __str__(self, i=0):
        """
        """
        s = ''
        ind = ''
        if i != 0:
            ind = '  ' * i
        for b in self._branches:
            s += (ind + self._name + ' - ' + str(b.name) + '\n')
            if b.next is not None:
                s += b.next.__str__(i+1)
        return s

    def __len__(self):
        """
        """
        return len(self.get_branches())

    def get_branch(self, name, validate=True):
        """
        """
        if validate:
            assert(is_string(name))

        for b in self._branches:
            if b.name == name:
                return b

        raise LookupError

    def get_branches(self, of=None):
        """
        """
        branches = []
        for b in self._branches:
            if b.next is None:
                if of is None:
                    branches.append(b)
                else:
                    if self._name in of and b.value in of[self._name]:
                        branches.append(b)
            else:
                branches.extend(b.next.get_branches(of=of))
        return branches

    def sample(self, sub=None, validate=True):
        """
        """
        if validate is True:
            assert(sub is None or type(sub) is dict)

        if sub is not None and self._name in sub:
            b = self.get_branch(sub[self._name])
        else:
            b = random.choice(self._branches)
        path = [(self._name, b.name, b.value)]
        prob = 1 / len(self._branches)
        if b.next is not None:
            l = b.next.sample(sub)
            path.extend(l.path)
            prob *= l.prob
        return LogicTreeLeaf(path, prob)


class LogicTree(Object):
    """
    """

    def __init__(self):
        """
        """
        self._structure = None

    def __str__(self):
        return self._structure.__str__(i=0)

    def __len__(self):
        """
        """
        return len(self._structure)

    def get_branches(self, of=None):
        """
        """
        return self._structure.get_branches(of=of)

    def add_level(self, name, branches, to=None):
        """
        """
        if self._structure is None:
            assert(to is None)
            self._structure = LogicTreeLevel(name, branches)
        else:
            for b in self.get_branches(of=to):
                assert(b._next is None)
                b._next = LogicTreeLevel(name, branches)

    def sample(self, sub=None, validate=True):
        """
        """
        if validate is True:
            assert(sub is None or type(sub) is dict)

        return self._structure.sample(sub, validate=False)
