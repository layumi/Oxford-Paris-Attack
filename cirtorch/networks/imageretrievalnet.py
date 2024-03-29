import os
import pdb

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable

import torchvision

from cirtorch.layers.pooling import MAC, SPoC, GeM, RMAC, Rpool
from cirtorch.layers.normalization import L2N, PowerLaw
from cirtorch.datasets.genericdataset import ImagesFromList
from cirtorch.utils.general import get_data_root

from PIL import Image
import numpy as np
# for some models, we have imported features (convolutions) from caffe because the image retrieval performance is higher for them
FEATURES = {
    'vgg16'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-vgg16-features-d369c8e.pth',
    'resnet50'      : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet50-features-ac468af.pth',
    'resnet101'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet101-features-10a101d.pth',
    'resnet152'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/imagenet/imagenet-caffe-resnet152-features-1011020.pth',
}

# TODO: pre-compute for more architectures and properly test variations (pre l2norm, post l2norm)
# pre-computed local pca whitening that can be applied before the pooling layer
L_WHITENING = {
    'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-9f830ef.pth', # no pre l2 norm
    # 'resnet101' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-lwhiten-da5c935.pth', # with pre l2 norm
}

# possible global pooling layers, each on of these can be made regional
POOLING = {
    'mac'  : MAC,
    'spoc' : SPoC,
    'gem'  : GeM,
    'rmac' : RMAC,
}

# TODO: pre-compute for: resnet50-gem-r, resnet50-mac-r, vgg16-mac-r, alexnet-mac-r
# pre-computed regional whitening, for most commonly used architectures and pooling methods
R_WHITENING = {
    'alexnet-gem-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-rwhiten-c8cf7e2.pth',
    'vgg16-gem-r'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-rwhiten-19b204e.pth',
    'resnet101-mac-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-rwhiten-7f1ed8c.pth',
    'resnet101-gem-r' : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-rwhiten-adace84.pth',
}

# TODO: pre-compute for more architectures
# pre-computed final (global) whitening, for most commonly used architectures and pooling methods
WHITENING = {
    'alexnet-gem'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-whiten-454ad53.pth',
    'alexnet-gem-r'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-alexnet-gem-r-whiten-4c9126b.pth',
    'vgg16-gem'         : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-whiten-eaa6695.pth',
    'vgg16-gem-r'       : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-vgg16-gem-r-whiten-83582df.pth',
    'resnet101-mac-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-mac-r-whiten-9df41d3.pth',
    'resnet101-gem'     : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-whiten-22ab0c1.pth',
    'resnet101-gem-r'   : 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/whiten/retrieval-SfM-120k/retrieval-SfM-120k-resnet101-gem-r-whiten-b379c0a.pth',
}

# output dimensionality for supported architectures
OUTPUT_DIM = {
    'alexnet'       :  256,
    'vgg11'         :  512,
    'vgg13'         :  512,
    'vgg16'         :  512,
    'vgg19'         :  512,
    'resnet18'      :  512,
    'resnet34'      :  512,
    'resnet50'      : 2048,
    'resnet101'     : 2048,
    'resnet152'     : 2048,
    'densenet121'   : 1024,
    'densenet161'   : 2208,
    'densenet169'   : 1664,
    'densenet201'   : 1920,
    'squeezenet1_0' :  512,
    'squeezenet1_1' :  512,
}


class ImageRetrievalNet(nn.Module):
    
    def __init__(self, features, lwhiten, pool, whiten, meta):
        super(ImageRetrievalNet, self).__init__()
        self.features = nn.Sequential(*features)
        self.lwhiten = lwhiten
        self.pool = pool
        self.whiten = whiten
        self.norm = L2N()
        self.meta = meta
    
    def forward(self, x):
        # x -> features
        o = self.features(x)

        # TODO: properly test (with pre-l2norm and/or post-l2norm)
        # if lwhiten exist: features -> local whiten
        if self.lwhiten is not None:
            # o = self.norm(o)
            s = o.size()
            o = o.permute(0,2,3,1).contiguous().view(-1, s[1])
            o = self.lwhiten(o)
            o = o.view(s[0],s[2],s[3],self.lwhiten.out_features).permute(0,3,1,2)
            # o = self.norm(o)

        # features -> pool -> norm
        o = self.norm(self.pool(o)).squeeze(-1).squeeze(-1)

        # if whiten exist: pooled features -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o))

        # permute so that it is Dx1 column vector per image (DxN if many images)
        return o.permute(1,0)

    def __repr__(self):
        tmpstr = super(ImageRetrievalNet, self).__repr__()[:-1]
        tmpstr += self.meta_repr()
        tmpstr = tmpstr + ')'
        return tmpstr

    def meta_repr(self):
        tmpstr = '  (' + 'meta' + '): dict( \n' # + self.meta.__repr__() + '\n'
        tmpstr += '     architecture: {}\n'.format(self.meta['architecture'])
        tmpstr += '     local_whitening: {}\n'.format(self.meta['local_whitening'])
        tmpstr += '     pooling: {}\n'.format(self.meta['pooling'])
        tmpstr += '     regional: {}\n'.format(self.meta['regional'])
        tmpstr += '     whitening: {}\n'.format(self.meta['whitening'])
        tmpstr += '     outputdim: {}\n'.format(self.meta['outputdim'])
        tmpstr += '     mean: {}\n'.format(self.meta['mean'])
        tmpstr += '     std: {}\n'.format(self.meta['std'])
        tmpstr = tmpstr + '  )\n'
        return tmpstr


def init_network(params):

    # parse params with default values
    architecture = params.get('architecture', 'resnet101')
    local_whitening = params.get('local_whitening', False)
    pooling = params.get('pooling', 'gem')
    regional = params.get('regional', False)
    whitening = params.get('whitening', False)
    mean = params.get('mean', [0.485, 0.456, 0.406])
    std = params.get('std', [0.229, 0.224, 0.225])
    pretrained = params.get('pretrained', True)

    # get output dimensionality size
    dim = OUTPUT_DIM[architecture]

    # loading network from torchvision
    if pretrained:
        if architecture not in FEATURES:
            # initialize with network pretrained on imagenet in pytorch
            net_in = getattr(torchvision.models, architecture)(pretrained=True)
        else:
            # initialize with random weights, later on we will fill features with custom pretrained network
            net_in = getattr(torchvision.models, architecture)(pretrained=False)
    else:
        # initialize with random weights
        net_in = getattr(torchvision.models, architecture)(pretrained=False)

    # initialize features
    # take only convolutions for features,
    # always ends with ReLU to make last activations non-negative
    if architecture.startswith('alexnet'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('vgg'):
        features = list(net_in.features.children())[:-1]
    elif architecture.startswith('resnet'):
        features = list(net_in.children())[:-2]
    elif architecture.startswith('densenet'):
        features = list(net_in.features.children())
        features.append(nn.ReLU(inplace=True))
    elif architecture.startswith('squeezenet'):
        features = list(net_in.features.children())
    else:
        raise ValueError('Unsupported or unknown architecture: {}!'.format(architecture))

    # initialize local whitening
    if local_whitening:
        lwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: lwhiten with possible dimensionality reduce

        if pretrained:
            lw = architecture
            if lw in L_WHITENING:
                print(">> {}: for '{}' custom computed local whitening '{}' is used"
                    .format(os.path.basename(__file__), lw, os.path.basename(L_WHITENING[lw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                lwhiten.load_state_dict(model_zoo.load_url(L_WHITENING[lw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no local whitening computed, random weights are used"
                    .format(os.path.basename(__file__), lw))

    else:
        lwhiten = None
    
    # initialize pooling
    pool = POOLING[pooling]()
    
    # initialize regional pooling
    if regional:
        rpool = pool
        rwhiten = nn.Linear(dim, dim, bias=True)
        # TODO: rwhiten with possible dimensionality reduce

        if pretrained:
            rw = '{}-{}-r'.format(architecture, pooling)
            if rw in R_WHITENING:
                print(">> {}: for '{}' custom computed regional whitening '{}' is used"
                    .format(os.path.basename(__file__), rw, os.path.basename(R_WHITENING[rw])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                rwhiten.load_state_dict(model_zoo.load_url(R_WHITENING[rw], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no regional whitening computed, random weights are used"
                    .format(os.path.basename(__file__), rw))

        pool = Rpool(rpool, rwhiten)

    # initialize whitening
    if whitening:
        whiten = nn.Linear(dim, dim, bias=True)
        # TODO: whiten with possible dimensionality reduce

        if pretrained:
            w = architecture
            if local_whitening:
                w += '-lw'
            w += '-' + pooling
            if regional:
                w += '-r'
            if w in WHITENING:
                print(">> {}: for '{}' custom computed whitening '{}' is used"
                    .format(os.path.basename(__file__), w, os.path.basename(WHITENING[w])))
                whiten_dir = os.path.join(get_data_root(), 'whiten')
                whiten.load_state_dict(model_zoo.load_url(WHITENING[w], model_dir=whiten_dir))
            else:
                print(">> {}: for '{}' there is no whitening computed, random weights are used"
                    .format(os.path.basename(__file__), w))
    else:
        whiten = None

    # create meta information to be stored in the network
    meta = {
        'architecture' : architecture, 
        'local_whitening' : local_whitening, 
        'pooling' : pooling, 
        'regional' : regional, 
        'whitening' : whitening, 
        'mean' : mean, 
        'std' : std,
        'outputdim' : dim,
    }

    # create a generic image retrieval network
    net = ImageRetrievalNet(features, lwhiten, pool, whiten, meta)

    # initialize features with custom pretrained network if needed
    if pretrained and architecture in FEATURES:
        print(">> {}: for '{}' custom pretrained features '{}' are used"
            .format(os.path.basename(__file__), architecture, os.path.basename(FEATURES[architecture])))
        model_dir = os.path.join(get_data_root(), 'networks')
        net.features.load_state_dict(model_zoo.load_url(FEATURES[architecture], model_dir=model_dir))

    return net


def extract_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = torch.zeros(net.meta['outputdim'], len(images))
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1:
                vecs[:, i] = extract_ss(net, input)
            else:
                vecs[:, i] = extract_ms(net, input, ms, msp)

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs

def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = inp * 255.0
    inp = np.clip(inp, 0, 255)
    return inp


def extract_vectors_aq(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    torch.autograd.set_detect_anomaly(True)
    # extracting vectors
    vecs = torch.zeros(net.meta['outputdim'], len(images))
    for i, input in enumerate(loader):
            input = Variable(input.cuda(), requires_grad=True)
            input_copy = Variable(input.data, requires_grad = False)
            diff = torch.FloatTensor(input.shape).zero_()
            diff = Variable(diff.cuda(), requires_grad = False)

            if len(ms) == 1:
                vecs_tmp = extract_ss(net, input, grad=True)
            else:
                vecs_tmp = extract_ms(net, input, ms, msp, grad=True)

            target = Variable(-vecs_tmp.data, requires_grad=False)
            criterion = nn.MSELoss()
            # The input has been whiten.
            #So when we recover, we need to use a alpha
            alpha = 1.0 / (0.226 * 255.0)
            # generate adversarial query
            rate = 16 
            #rate = 10 ablation 
            for iter in range( round(min(1.25 * rate, rate+4))):
                loss = criterion(vecs_tmp, target)
                loss.backward()
                diff += torch.sign(input.grad)
                mask_diff = diff.abs() > rate
                diff[mask_diff] = rate * torch.sign(diff[mask_diff])
                input = input_copy - diff * 1.0 * alpha
                input = Variable(input.data, requires_grad=True)
                if len(ms) == 1:
                    vecs_tmp = extract_ss(net, input, grad=True)
                else:
                    vecs_tmp = extract_ms(net, input, ms, msp, grad=True)

            if len(ms) == 1:
                vecs[:, i] = extract_ss(net, input)
            else:
                vecs[:, i] = extract_ms(net, input, ms, msp)

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
                print('')

            attack = input.cpu().detach()
            for j in range(input.shape[0]):
                im = recover(attack[j,:,:,:])
                im = Image.fromarray(im.astype('uint8'))
                save_path = images[j].replace('test', 'attack')
                if not os.path.isdir( os.path.dirname(os.path.dirname(os.path.dirname(save_path)))):
                    os.mkdir(os.path.dirname(os.path.dirname(os.path.dirname(save_path))))
                if not os.path.isdir( os.path.dirname(os.path.dirname(save_path))):
                    os.mkdir(os.path.dirname(os.path.dirname(save_path)))
                if not os.path.isdir( os.path.dirname(save_path)):
                    os.mkdir(os.path.dirname(save_path))
                im.save( save_path )
            
    return vecs


def extract_ss(net, input, grad=False):
    if grad:
        return net(input).squeeze()
    return net(input).cpu().data.squeeze()

def extract_ms(net, input, ms, msp, grad=False):

    for s in ms: 
        if s == 1:
            input_t = input
        else:    
            input_t = nn.functional.interpolate(input, scale_factor=s, mode='bilinear', align_corners=False)
        if s == ms[0]:
            v = net(input_t).pow(msp).squeeze()
        else:
            v += net(input_t).pow(msp).squeeze()
    v /= len(ms)
    v = v.pow(1./msp)
    v_norm = torch.norm(v, p=2, dim=0, keepdim=True) + 1e-6
    v = v/v_norm

    if grad == False:
        return v.cpu().data.squeeze()

    return v


def extract_vectors_pire(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10, iteration=20):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    torch.autograd.set_detect_anomaly(True)
    # extracting vectors
    vecs = torch.zeros(net.meta['outputdim'], len(images))
    for i, input in enumerate(loader):
            input = Variable(input.cuda(), requires_grad=True)
            input_copy = Variable(input.data, requires_grad = False)

            criterion = nn.MSELoss()
            # The input has been whiten.
            #So when we recover, we need to use a alpha
            alpha = 1.0 / (0.226 * 255.0)
            # generate adversarial query
            rate = 16
            init_v = (torch.rand(input.shape)-0.5)*2*rate  # -16 to 16
            gem_out = extract_ss(net, input, grad=False)
            gem_out = Variable(gem_out.data, requires_grad=False).cuda()
            diff = torch.autograd.Variable(init_v.cuda(),requires_grad=True)
            loss_fn = torch.nn.MSELoss(reduction='sum')
            learning_rate = 0.1 #Here we use one large lr Rate for quick converge
            optimizer = torch.optim.Adam([diff], lr=learning_rate)
            for iter in range(iteration):
                y_pred = extract_ss(net, input + diff * 1.0  * alpha, grad=True)
                           
                loss =  -1 * loss_fn(y_pred, gem_out)
                optimizer.zero_grad()
                diff.data = torch.clamp(diff.data,-rate, rate)
                loss.backward(retain_graph=True)
                optimizer.step()

            input = input_copy + diff * 1.0 * alpha
            input = Variable(input.data, requires_grad=False)
            if len(ms) == 1:
                vecs[:, i] = extract_ss(net, input)
            else:
                vecs[:, i] = extract_ms(net, input, ms, msp)

            attack = input.cpu().detach()
            for j in range(input.shape[0]):
                im = recover(attack[j,:,:,:])
                im = Image.fromarray(im.astype('uint8'))
                save_path = images[j].replace('test', 'attack+pire')
                if not os.path.isdir( os.path.dirname(os.path.dirname(os.path.dirname(save_path)))):
                    os.mkdir(os.path.dirname(os.path.dirname(os.path.dirname(save_path))))
                if not os.path.isdir( os.path.dirname(os.path.dirname(save_path))):
                    os.mkdir(os.path.dirname(os.path.dirname(save_path)))
                if not os.path.isdir( os.path.dirname(save_path)):
                    os.mkdir(os.path.dirname(save_path))
                im.save( save_path )

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
                print('')

    return vecs

def extract_regional_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssr(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msr(net, input, ms, msp))
                raise NotImplementedError

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs

def extract_ssr(net, input):
    return net.pool(net.features(input), aggregate=False).squeeze(0).squeeze(-1).squeeze(-1).permute(1,0).cpu().data


def extract_local_vectors(net, images, image_size, transform, bbxs=None, ms=[1], msp=1, print_freq=10):
    # moving network to gpu and eval mode
    net.cuda()
    net.eval()

    # creating dataset loader
    loader = torch.utils.data.DataLoader(
        ImagesFromList(root='', images=images, imsize=image_size, bbxs=bbxs, transform=transform),
        batch_size=1, shuffle=False, num_workers=8, pin_memory=True
    )

    # extracting vectors
    with torch.no_grad():
        vecs = []
        for i, input in enumerate(loader):
            input = input.cuda()

            if len(ms) == 1:
                vecs.append(extract_ssl(net, input))
            else:
                # TODO: not implemented yet
                # vecs.append(extract_msl(net, input, ms, msp))
                raise NotImplementedError

            if (i+1) % print_freq == 0 or (i+1) == len(images):
                print('\r>>>> {}/{} done...'.format((i+1), len(images)), end='')
        print('')

    return vecs

def extract_ssl(net, input):
    return net.norm(net.features(input)).squeeze(0).view(net.meta['outputdim'], -1).cpu().data
