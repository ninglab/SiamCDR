#### LOAD PACKAGES
import os

#### DEFINE CONSTANTS
file_dir = os.path.dirname(__file__)

#### Define classes
class createDirs( object ):
    """
    Create the subdirectories given the path to a parent directory
    """
    def __init__(self, parent, kind='models'):
        self.parent = parent
        self.paths = self.define_subdirs()
        self.mkdir()

    def define_subdirs(self):
        paths = {}
        paths['models'] = os.path.join(self.parent, 'models')
        
        subResDirs = ['fit', 'train', 'val', 'test', 'newcancer']
        for d in subResDirs:
            if d != 'fit':
                key = f"{d}_res"
            else:
                key = d
            paths[key] = os.path.join(self.parent, f"{d}_results")

        subPredDirs = ['train', 'val', 'test', 'newcancer']
        for d in subPredDirs:
            paths[f"{d}_res"] = os.path.join(self.parent, f"{d}_preds")

        return paths

    def mkdir(self):
        for p in [self.parent] + list(self.paths.values()):
            if not os.path.exists(p):
                os.mkdir(p)
