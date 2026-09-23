# coding=utf-8


class PathInfo(object):
    """Store dataset path information."""
    def __init__(self, parser):
        data_dir = parser.get('Data', 'data_dir')
        self.__dataset_path = data_dir + '/sFCN.h5'

    @property
    def dataset_path(self):
        return self.__dataset_path
