import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime
import numpy as np
import logging

from utils.utils import AverageMeter, xyxy2xywh, xywh2xyxy
from models.detr import build_detr, build_VLFusion
from pytorch_pretrained_bert.modeling import BertModel
from models.backbone import build_backbone
from models.transformer import build_vis_transformer, build_transformer
from models.position_encoding import build_position_encoding
from utils.misc import NestedTensor
from torch.autograd import Variable


class VLFusion(nn.Module):
    def __init__(self, transformer, pos):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: no use
        """
        super().__init__()
        self.transformer = transformer
        self.pos = pos
        hidden_dim = transformer.d_model

        # TODO: Define a learnable token with shape (1, hidden_dim) and randomly initialize it
        # Then the token should be converted to a nn.Parameter for learning
        # See https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
        self.learnable_token = nn.Parameter(torch.randn(1, hidden_dim))

        self.v_proj = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),)
        self.l_proj = torch.nn.Sequential(
          nn.Linear(768, 256),
          nn.ReLU(),)

    def forward(self, fv, fl):
        """
        args:
            fv: vision features
            fl: language features
        """
        bs, c, h, w = fv.shape
        _, _, l = fl.shape

        pv = self.v_proj(fv.view(bs, c, -1).permute(0,2,1))  # [bs,400,256]
        pl = self.l_proj(fl)  # [bs, 40, 256]
        pv = pv.permute(0,2,1)  # [bs,256,400]
        pl = pl.permute(0,2,1)  # [bs,256,40]


        # TODO: Concat the learnable token with the vision tokens (pv) and language tokens (pl). The order is [learnable token, vision tokens, language tokens]
        # The shape of the learnable token is [1, hidden dim]
        # You should combine the expand and unsqueeze functions to make the shape [batch size, hidden dim, 1] for matching the dim with input tokens
        # Then concat them to get the final input tokens with shape [batch size, hidden dim, input token length+1]
        # See https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html and https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        hidden_dim = self.learnable_token.shape[-1]
        # TODO [batch size, hidden dim, 1] 
        learnable_token = self.learnable_token.unsqueeze(0).expand(bs, 1, hidden_dim).transpose(1, 2)
        # TODO [batch size, hidden dim, 1 + vision token length + language token length]
        input_tokens = torch.cat([learnable_token, pv, pl], dim=-1)
        
        pos = self.pos(input_tokens).to(input_tokens.dtype)  # [bs, 441, 256]
        mask = torch.zeros([bs, input_tokens.shape[2]]).cuda()
        mask = mask.bool()  # [bs, 441]
        
        out = self.transformer(input_tokens, mask, pos)  # [441, bs, 256]
        
        return out[-1]


class VGModel(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', tunebert=True, args=None):
        super(VGModel, self).__init__()
        self.tunebert = tunebert
        if bert_model=='bert-base-uncased':
            self.textdim=768
        else:
            self.textdim=1024
        ## Visual model
        self.visumodel = build_detr(args)
        self.visumodel = load_weights(self.visumodel, './saved_models/detr-r50-e632da11.pth')
        
        ## Text model
        self.textmodel = BertModel.from_pretrained(bert_model)

        ## Visual-linguistic Fusion model
        self.vlmodel = build_VLFusion(args)
        self.vlmodel = load_weights(self.vlmodel, './saved_models/detr-r50-e632da11.pth')
        
        ## Prediction Head
        self.Prediction_Head = torch.nn.Sequential(
          nn.Linear(256, 256),
          nn.ReLU(),
          # TODO: Define a bounding box regression layer
          nn.Linear(256, 4)
          # It should be a one-layer connected layer
          # The layer generates bounding boxes (coordinates), so the output dim is the dim of one bounding box
          # See https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        )

        for p in self.Prediction_Head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, image, mask, word_id, word_mask):
        ## Visual Module
        fv = self.visumodel(image, mask)  # [bs, 256, 20, 20]

        ## Language Module
        all_encoder_layers, _ = self.textmodel(word_id, \
            token_type_ids=None, attention_mask=word_mask)

        ## Sentence feature 
        fl = (all_encoder_layers[-1] + all_encoder_layers[-2]\
             + all_encoder_layers[-3] + all_encoder_layers[-4])/4  # [bs, 40, 768]
        if not self.tunebert:
            ## fix bert during training
            fl = fl.detach()

        ## Visual-linguistic Fusion Module
        x = self.vlmodel(fv, fl)

        ## Prediction Head
        outbox = self.Prediction_Head(x)  # (x; y;w; h)
        outbox = outbox.sigmoid()*2.-0.5

        return outbox

# Training and validation functions
def train_epoch(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    GIoU_losses = AverageMeter()
    # div_losses = AverageMeter()
    acc = AverageMeter()
    # acc_center = AverageMeter()
    miou = AverageMeter()

    model.train()
    end = time.time()

    for batch_idx, (imgs, masks, word_id, word_mask, gt_bbox) in enumerate(train_loader):
        # print('get data from train_loader...')
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:,:,:,0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        gt_bbox = gt_bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        gt_bbox = Variable(gt_bbox)

        # TODO: Calculate loss
        # Bounding boxes have several formats:
        # 1. x1y1x2y2. x1 and y1 are left and top; x2 and y2 are right and bottom.
        # 2. xywh. x and y are center coordinates; w and h are width and height.
        # The size of bounding boxes can be the original size and normalized size.
        # To compute the giou loss, the bounding boxes should be the original size and xywh format

        # pred_bbox has normalized size and xywh format
        # gt_bbox has original size and x1y1x2y2 format
        # TODO : Get the pred_box from the model.forward call. --> forward(self, image, mask, word_id, word_mask):
        pred_bbox = model.forward(image, masks, word_id, word_mask)
        scale = args.size-1
        gt_bbox = torch.clamp(gt_bbox,min=0, max=scale)

        loss = 0.
        # The inputs of GIoU loss are bounding boxes with original size and x1y1x2y2 format
        # You can multiply the scale variable to convert the normalized size to the original size
        # The xywh2xyxy function is utilized to convert the bounding box format from xywh to x1y1x2y2
        # TODO: convert the normalized size to the original size
        original_size_pred_bbox = pred_bbox * scale
        # TODO: convert the xywh format to the x1y1x2y2 format
        original_size_x1y1x2y2_pred_bbox = xywh2xyxy(original_size_pred_bbox)
        GIoU_loss = GIoU_Loss(original_size_x1y1x2y2_pred_bbox, gt_bbox)
        loss += GIoU_loss

        # The inputs of Reg_Loss are bounding boxes with normalized size and xywh format
        # You can divide the scale variable to convert the original size to the normalized size
        # The xyxy2xywh function is utilized to convert bounding box format form x1y1x2y2 to xywh
        # TODO: convert the x1y1x2y2 format to the xywh format
        xywh_gt_bbox = xyxy2xywh(gt_bbox)
        # TODO: convert the normalized size to theoriginal size
        normalized_size_xywh_gt_bbox = xywh_gt_bbox / scale
        l1_loss = Reg_Loss(pred_bbox, normalized_size_xywh_gt_bbox)
        loss += l1_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))
        l1_losses.update(l1_loss.item(), imgs.size(0))
        GIoU_losses.update(GIoU_loss.item(), imgs.size(0))
        
        # TODO: Calculate IoU
        # The inputs of bbox_iou are bounding boxes with original size and x1y1x2y2 format
        # xyxy_pred_bbox = # TODO: convert the xywh format to the x1y1x2y2 format
        # original_size_xyxy_pred_bbox = # TODO: convert the normalized size to the original size
        # iou = bbox_iou(original_size_xyxy_pred_bbox, gt_bbox, x1y1x2y2=True)
        iou = bbox_iou(original_size_x1y1x2y2_pred_bbox, gt_bbox, x1y1x2y2=True)
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size
        
        ## metrics
        miou.update(torch.mean(iou).item(), imgs.size(0))
        acc.update(accu, imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                'L1_Loss {l1_loss.val:.4f} ({l1_loss.avg:.4f})\t' \
                'GIoU_Loss {GIoU_loss.val:.4f} ({GIoU_loss.avg:.4f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                'vis_lr {vis_lr:.8f}\t' \
                'lang_lr {lang_lr:.8f}\t' \
                .format( \
                    epoch, batch_idx, len(train_loader), \
                    loss=losses, l1_loss = l1_losses, \
                    GIoU_loss = GIoU_losses, miou=miou, acc=acc, \
                    vis_lr = optimizer.param_groups[0]['lr'], lang_lr = optimizer.param_groups[2]['lr'])
            print(print_str)

            logging.info(print_str)


def validate_epoch(val_loader, model, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    # acc_center = AverageMeter()
    miou = AverageMeter()
    # pect_long = AverageMeter()
    # acc_long = AverageMeter()
    # acc_short = AverageMeter()

    model.eval()
    end = time.time()
    print(datetime.datetime.now())
    
    for batch_idx, (imgs, masks, word_id, word_mask, bbox) in enumerate(val_loader):
        imgs = imgs.cuda()
        masks = masks.cuda()
        masks = masks[:,:,:,0] == 255
        word_id = word_id.cuda()
        word_mask = word_mask.cuda()
        bbox = bbox.cuda()
        image = Variable(imgs)
        masks = Variable(masks)
        word_id = Variable(word_id)
        word_mask = Variable(word_mask)
        bbox = Variable(bbox)
        bbox = torch.clamp(bbox,min=0,max=args.size-1)

        with torch.no_grad():
            pred_bbox = model(image, masks, word_id, word_mask)
        
        scale = args.size - 1
        gt_bbox = bbox
        # TODO: Calculate IoU
        # The inputs of bbox_iou are bounding boxes with original size and x1y1x2y2 format
        # TODO: convert the xywh format to the x1y1x2y2 format
        # xyxy_pred_bbox = 
        original_size_pred_bbox = pred_bbox * scale
        # TODO: convert the normalized size to the original size
        original_size_xyxy_pred_bbox = xywh2xyxy(original_size_pred_bbox)
        iou = bbox_iou(original_size_xyxy_pred_bbox, gt_bbox, x1y1x2y2=True)
        # accu_center = np.sum(np.array((target_gi == np.array(pred_gi)) * (target_gj == np.array(pred_gj)), dtype=float))/args.batch_size
        accu = np.sum(np.array((iou.data.cpu().numpy()>0.5),dtype=float))/args.batch_size

        acc.update(accu, imgs.size(0))
        # acc_center.update(accu_center, imgs.size(0))
        miou.update(torch.mean(iou).item(), imgs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.print_freq == 0:
            print_str = '[{0}/{1}]\t' \
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                'Accu {acc.val:.4f} ({acc.avg:.4f})\t' \
                'Mean_iu {miou.val:.4f} ({miou.avg:.4f})\t' \
                .format( \
                    batch_idx, len(val_loader), batch_time=batch_time, \
                    data_time=data_time, \
                    acc=acc, miou=miou)
            print(print_str)
            logging.info(print_str)
            
    print(acc.avg, miou.avg)
    logging.info("%f,%f"%(acc.avg, float(miou.avg)))
    return acc.avg


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    box1 = box1.cpu()
    box2 = box2.cpu()
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # TODO: Calculate IoU. IoU = intersection area / union area
    # b1_x1 and b1_y1 are the left and top coordinates of box1; b1_x2 and b1_y2 are the right and bottom coordinates of box1
    # b2_x1 and b2_y1 are the left and top coordinates of box2; b2_x2 and b2_y2 are the right and bottom coordinates of box2
    # First, use torch.max or torch.min to compare coordinates between box1 and box2 to get the coordinates of the intersection rectangle, and calculate the intersection area
    # TODO: Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height
    # Next, calculate the union area. Hint: the union area is equal to box1 area + box2 area - intersection area.
    # TODO: Union area
    area_box1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_box2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area_box1 + area_box2 - inter_area
    union_area += 1e-16 # For numerical stability
    # Finally, calculate the IoU
    # TODO: IoU
    iou = inter_area / union_area

    return iou


def Reg_Loss(output, target):
    # target is (x1,y1,x2,y2)
    sm_l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
    
    loss_x1 = sm_l1_loss(output[:,0], target[:,0])
    loss_x2 = sm_l1_loss(output[:,1], target[:,1])
    loss_y1 = sm_l1_loss(output[:,2], target[:,2])
    loss_y2 = sm_l1_loss(output[:,3], target[:,3])

    return (loss_x1+loss_x2+loss_y1+loss_y2)


def GIoU_Loss(boxes1, boxes2):
    '''
    cal GIOU of two boxes or batch boxes
    '''

    # ===========cal IOU=============#
    # cal Intersection
    bs = boxes1.size(0)
    #boxes1 = torch.cat([boxes1[:,:2]-(boxes1[:,2:]/2), boxes1[:,:2]+(boxes1[:,2:]/2)], dim=1)
    boxes1 = torch.clamp(boxes1, min=0, max=639)
    max_xy = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    min_xy = torch.max(boxes1[:, :2], boxes2[:, :2])


    # max_xy = torch.min(boxes1[:, 2:].unsqueeze(1).expand(bs, bs, 2),
    #                    boxes2[:, 2:].unsqueeze(0).expand(bs, bs, 2))
    # min_xy = torch.max(boxes1[:, :2].unsqueeze(1).expand(bs, bs, 2),
    #                    boxes2[:, :2].unsqueeze(0).expand(bs, bs, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)  # make sure the low bound of tensor is 0
    inter = inter[:, 0] * inter[:, 1]
    # Calcualte areas of boxes1 and 2
    boxes1Area = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))
    boxes2Area = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]))

    union_area = boxes1Area + boxes2Area - inter + 1e-7
    ious = inter / union_area

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_right_down = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose = torch.clamp((enclose_right_down - enclose_left_up), min=0)
    enclose_area = enclose[:, 0] * enclose[:, 1] + 1e-7

    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area  # the range of giou is [-1, 1]

    # GIOU Loss
    giou_loss = ((1-gious).sum())/bs

    return giou_loss


def generate_coord(batch, height, width):
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

def load_weights(model, load_path):
    # Load DETR transformer encoder and resnet50 backbone
    dict_trained = torch.load(load_path)['model']
    dict_new = model.state_dict().copy()
    for key in dict_new.keys():
        if key in dict_trained.keys():
            dict_new[key] = dict_trained[key]
    model.load_state_dict(dict_new)
    del dict_new
    del dict_trained
    torch.cuda.empty_cache()
    return model


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            """
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

    def forward(self, img, mask):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        samples = NestedTensor(img, mask)
        # pos: position encoding
        features, pos = self.backbone(samples)  # pos:list, pos[-1]: [64, 256, 20, 20]

        src, mask = features[-1].decompose()  # src:[64, 2048, 20, 20]  mask:[64,20,20]
        assert mask is not None
        out = self.transformer(self.input_proj(src), mask, pos[-1])
        
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_detr(args):
    backbone = build_backbone(args) # ResNet 50
    transformer = build_vis_transformer(args)

    model = DETR(
        backbone,
        transformer,
    )
    return model

def build_VLFusion(args):
    transformer = build_transformer(args)
    pos = build_position_encoding(args, position_embedding = 'learned')

    model = VLFusion(
        transformer,
        pos,
    )
    return model

