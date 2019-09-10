'''
    Created on Thu Apr 11 17:18 2019

    Author           : Yu Du
    Email            : yuduseu@gmail.com
    Last edit date   : Thu Sep 5 14:43 2019

South East University Automation College
Vision Cognition Laboratory, 211189 Nanjing China
'''

import os
import cv2
__all__ = ['Loader']

class Loader():
    """
    This class is designed for loading all the images in our dataset.
    """

    def __init__(self, dir):
        """
        Args:
            dir: Directory of dataset
        """
        self.dir = dir
        self.dataList = []
        self.__traverse__()
        print('All %d files are loaded successfully!' % len(self))

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, item):
        """
        Args:
            item: Index
            return: (nd.array) RGB Image
        """
        # if item >= len(self.dataList):
        #     print('Index out of range!')
        return self.dataList[item]

    def __traverse__(self):
        last = 0
        for root, dirs, files in os.walk(self.dir):
            for file in files:
                if file[-4:] == '.jpg' and file[0] != '.':
                    new = os.path.join(root, root.split('\\')[-1] + '.jpg')
                    old = os.path.join(root, file)
                    if new != old:
                        os.rename(old, new)
                    self.dataList.append(new)
                if len(self) % 100 == 0 and len(self) != last:
                    print('%d files have been loaded' % len(self))
                    last = len(self)


if __name__ == '__main__':
    for root, dirs, files, in os.walk('D:\\Documents\\Source\\MW-Pose\\test'):
        for file in files:
            print(root + '\\' + file)
    # loader = Loader('/Users/midora/Desktop/MW-Pose/test')
