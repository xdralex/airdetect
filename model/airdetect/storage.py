from typing import List

from wheel5.storage import LMDBDict, encode_list, decode_list, decode_ndarray, encode_ndarray
import numpy as np

from wheel5.tasks.detection import BoundingBox


class HeatmapLMDBDict(object):
    def __init__(self, lmdb_dict: LMDBDict):
        self.lmdb_dict = lmdb_dict

    def __contains__(self, key: str) -> bool:
        return key in self.lmdb_dict

    def __getitem__(self, key: str) -> np.ndarray:
        data = self.lmdb_dict[key]
        return decode_ndarray(data)

    def __setitem__(self, key: str, value: np.ndarray):
        data = encode_ndarray(value)
        self.lmdb_dict[key] = data

    def __delitem__(self, key: str):
        del self.lmdb_dict[key]


class BoundingBoxesLMDBDict(object):
    def __init__(self, lmdb_dict: LMDBDict):
        self.lmdb_dict = lmdb_dict

    def __contains__(self, key: str) -> bool:
        return key in self.lmdb_dict

    def __getitem__(self, key: str) -> List[BoundingBox]:
        data = self.lmdb_dict[key]
        return [BoundingBox.decode(b) for b in decode_list(data, size=BoundingBox.byte_size())]

    def __setitem__(self, key: str, value: List[BoundingBox]):
        data = encode_list([bbox.encode() for bbox in value], size=BoundingBox.byte_size())
        self.lmdb_dict[key] = data

    def __delitem__(self, key: str):
        del self.lmdb_dict[key]
