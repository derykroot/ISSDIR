import os

from torch.utils.data import Dataset
from PIL import Image  

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, classes=[]):
        """
        Args:
            data (list of tensors): Lista de tensors contendo as features.
            labels (list of int): Lista de inteiros contendo os rótulos.
            transform (callable, optional): Uma função/transformação opcional para aplicar aos dados.
        """
        self.data = data
        self.labels = labels
        self.targets = labels # Apenas para compatibilidade
        self.transform = transform

        self.feats = []
        self.classes = classes
        if type(classes)!=dict:
            self.classes = {c:i for i,c in enumerate(self.classes)}

        self.training = False
        self.filereturn = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx] if not self.training else self.feats[idx] 
        label = self.labels[idx]

        if self.training:
            return item
        elif self.transform and not self.filereturn:
            item = self.transform(Image.open(item))

        return item, label
    
class DatasetFeatsTrain():
    def __init__(self, featslabels, idclasses):

        self.feats = featslabels
        self.idclasses = idclasses
        self.lbclassrel = {i:[idx for idx,l in enumerate([l for _,l in self.feats]) if l==i ] for i in self.idclasses}
        self.clusters_ws = []

def getdata(root_dir):
    data = []
    classes = []
    
    for cls in [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]:
        if len([os.path.join(os.path.join(root_dir, cls), img) for img in os.listdir(os.path.join(root_dir, cls))])>0:
            classes.append(cls)

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    extensoes_imagem = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    for cls in classes:            
        cls_dir = os.path.join(root_dir, cls)
        cls_images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) if os.path.splitext(img)[1].lower() in extensoes_imagem]
        #loaded_images = [(img ,self.class_to_idx[cls]) for img in cls_images]
        loaded_images = [(img, class_to_idx[cls]) for img in cls_images]
        data.extend(loaded_images)

    return data, class_to_idx


def dsbuild(root_dir, transform=None):
    alldata, classes = getdata(root_dir)
    return CustomDataset([im for im, _ in alldata], [l for _, l in alldata], transform, classes) 