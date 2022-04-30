"""
CS 7643 Project. 
Similarity Aux files. 

Inspired by: https://docs.fast.ai/tutorial.siamese.html
"""
from fastai.vision.all import *

class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs): 
        if len(self) > 2:
            img1,img2,similarity = self
        else:
            img1,img2 = self
            similarity = 'Undetermined'
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), title=similarity, ctx=ctx, **kwargs)


class SiameseTransform(Transform):
    def __init__(self, files, is_valid=False):
        self.splbl2files = [{l:[f for f in files[splits[i]] if parent_label(f) == l] for l in labels} for i in range(2)]
        self.valid = {f: self._draw(f,1) for f in files[splits[1]]}
        
    def encodes(self, f):
        f2, same = self.valid.get(f, self._draw(f,0))
        img1, img2 = PILImage.create(f), PILImage.create(f2)
        return SiameseImage(img1, img2, same)
    
    def _draw(self, f, split=0):
        same = random.random() < 0.5
        cls = parent_label(f)
        if not same: cls = random.choice(L(l for l in labels if l != cls)) 
        return random.choice(self.splbl2files[split][cls]), 1*same
    
class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder,self.head = encoder,head
    
    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)
    
def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]


def loss_func(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())