# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device) #初始化各种loss，值均为0
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
        _tcls, _tbox, _indices, _anchors = self._build_targets(p, targets)  # todo 测试

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    # 原始版本
    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  #(3*441) same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
                        # target(441,6)->(3,441,6)     ai(3*441)->(3,441,1) => (3,441,7)
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        # for e in tcls:
        #     print(len(e))
        return tcls, tbox, indices, anch
        # tcls: gt_bbox对应的物体类别
        # tbox：
            # gt_bbox的 (中心点坐标x,y,            # 相对与网格的左上角，在（0，1）范围内
            #           归一化到特征图大小的w,h)    #
        # indices：
            # image索引
            # anchor索引
            # gt_bbox的中心点坐标x取整（对应网格cell的左上角）
            # gt_bbox的中心点坐标y取整（对应网格cell的左上角）

        # anch：匹配上的anchor的w，h （ 归一化到特征图大小）

    # ####################################################################################################
    # 【yolov3 vs yolov5（anchor的匹配机制）】 https://zhuanlan.zhihu.com/p/424984172
    # 【build_targets函数解读】 https://zhuanlan.zhihu.com/p/415071583
    '''
    下面的可能是不准确的，只是保留做可能的参考    
    描述1：【进击的后浪yolov5深度可视化解析】https://zhuanlan.zhihu.com/p/183838757【但实际应该是预设的Anchor与GT Bbox进行宽高比的匹配】
        (1) 对于任何一个输出层，抛弃了基于max iou匹配的规则，而是直接采用shape规则匹配，也就是该bbox和当前层的anchor计算宽高比，如果宽高比例大于设定阈值，则说明该bbox和anchor匹配度不够，将该bbox过滤暂时丢掉，在该层预测中认为是背景
        (2) 对于剩下的bbox，计算其落在哪个网格内，同时利用四舍五入规则，找出最近的两个网格，将这三个网格都认为是负责预测该bbox的，可以发现粗略估计正样本数相比前yolo系列，至少增加了三倍
    '''
    # 注释和测试版本，下面保留了原始版本
    def _build_targets(self, p, targets):
        # Build targets for compute_loss(),
        # input p: [tensor1(bs,na,80,80,85),tensor1(bs,na,40,40,85),tensor1(bs,na,40,40,85)]
        # input targets:(image_idx,class_id,x,y,w,h) ##Such as [ 0.00000, 45.00000,  0.17721,  0.60102,  0.35441,  0.34919]
        na, nt = self.na, targets.shape[0]  # number of anchors, number of targets
        tcls, tbox, indices, anch = [], [], [], []
        # ai(anchor indices的缩写) shape[na,nt] ai[0]全部为0，ai[1]全部为1，ai[2]全部为2
        # 以na=3 , nt = 441 举例
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  #shape(3*441)  #same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
                        # target(441,6)->(3,441,6)     ai(3*441)->(3,441,1) => (3,441,7)

        # (image_idx,class_id,x,y,w,h,ai).target复制3个(3,nt,6)，然后在最后一维 append ai(anchor indices).##Shape[na,nt,7],such as[ 0.00000, 45.00000,  0.17721,  0.60102,  0.35441,  0.34919, 0.00000(最后一位为ai)]
        # targets维度变为[na,nt,7],such as[ 0.00000:image_idx, 45.00000:class_id,  0.17721:x,  0.60102:y,  0.35441:w,  0.34919:h, 0.00000:anchor indices]
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain (与现在的target最后维度7一致，相当于权重系数)
        g = 0.5  # bias，cell左上角==> cell中心点的偏移
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        # off结果值
            # 0.00000, 0.00000
            # 0.50000, 0.00000
            # 0.00000, 0.50000
            # -0.50000, 0.00000
            # 0.00000, -0.50000
        print('')
        print("####开始特征图匹配........")
        for i in range(self.nl): # 输出3个尺度的特征图
            anchors = self.anchors[i] # 对应尺度特征图上预设的3个anchors
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain ##Such as [80,80,80,80]
            # gain的值变为[ 1.,  1., 80., 80., 80., 80.,  1.]
            # Match targets to anchors
            t = targets * gain
            print(f"----第{i}个输出特征图，targets共{targets.shape[1]=}个，每个target跟{na=}个anchor匹配，共有{targets.shape[1]*na=}个")
                # targetS[0][0]的值为 [ 0.00000, 45.00000,  0.17721,  0.60102,  0.35441,  0.34919,  0.00000]
                # t[0][0]      的值为 [ 0.00000, 45.00000, 14.17645, 48.08151, 28.35291, 27.93507,  0.00000]
            if nt:  # num targets
                # Matches
                r = t[:, :, 4:6] / anchors[:,None]
                # wh ratio  # anchors[:, None]与anchors[:, None,:]等价  r.shape=[na,nt,2],r[0][0]=[22.68232, 17.19081]
                # 只保留gt_box的w和h 与 anchor的w和h的比值均小于设置的超参数anchor_t的gt box。
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t'] #anchor_t=4.0
                # j.shape[na,nt],compare,宽高比小于self.hyp['anchor_t']=4.0
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter  ##t.shape[na,nt,7],j.shape[na,nt], t= t[j]后的shape[nt_keeped,7]
                _num_targets=len(t)
                print(f"............根据长宽比匹配后，还剩{_num_targets=}个输出特征图")
                # 得到了保留下来的target。shape为[nt_keeped,7]
                # nt_keeped可以大于nt(number of targets=targets.shape[0])
                # 每个scale特征图上，每个cell预设3个anchors，存在一个target匹配多个Anchor的情况，如
                # 可通过t.cpu().numpy()显示，然后排序进行查看
                    # 7.00000, 0.00000, 1.81211, 2.57524, 1.62025, 5.15048, 2.00000 # 最后一个数值为anchor indices
                    # 7.00000, 0.00000, 1.81211, 2.57524, 1.62025, 5.15048, 0.00000
                    # 7.00000, 0.00000, 1.81211, 2.57524, 1.62025, 5.15048, 1.00000

                #############  bool_ternsor_filter example #############
                # import torch
                # original=torch.randn((3,3,10))
                # print(f'{original.shape=}')  # [3, 3, 10]
                # bool_select = torch.randn((3,3))>0.5 # [3, 3]
                # print(f'{bool_select.shape=},{bool_select.sum()=}') #bool_select.shape=torch.Size([3, 3]),bool_select.sum()=tensor(2)
                # des=original[bool_select]
                # print(f'{des.shape=}')  #des.shape=torch.Size([2, 10])
                #############

                # Offsets
                gxy = t[:, 2:4]  # grid xy 取出过滤后的gt box的中心点浮点型的坐标
                gxi = gain[[2, 3]] - gxy  # inverse # 将以图像左上角为原点的坐标变换为以图像右下角为原点的坐标.
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                # 以图像左上角为原点的坐标，取中心点的小数部分，小数部分小于0.5的为ture，大于0.5的为false。
                # j和 k的shape都是(nt_keeped)，true的位置分别表示靠近方格左边的gt box和靠近方格上方的gt box。

                l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 以图像右下角为原点的坐标，取中心点的小数部分，
                # 小数部分小于0.5的为ture，大于0.5的为false。
                # l和m的shape都是(nt_keeped)，true的位置分别表示靠近方格右边的gt box和靠近方格下方的gt box。

                # 当所在位置元素 大于1的时候，对应位置的： j和 l的值是刚好相反的，k 和 m的值也是刚好相反的。
                # 以下是验证：
                #############  以下是验证 #############
                # ll = ~j
                # mm = ~k
                # if not (ll == l).all() or not (mm == m).all():
                #     print()
                #     print("##"*10)
                #     print(f"----请检查，一定是有元素<1.0 或 > {gain[[2, 3]]-1}  ----------------------")
                #     idx_l_not_equal = (ll == l) == False
                #     idx_m_not_equal = (mm == m) == False
                #     print(f"----{gxy[idx_l_not_equal].tolist()=}")
                #     print(f"----{gxi[idx_m_not_equal].tolist()=}")
                #     print("##" * 10)
                #############  以上是验证 #############
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            print(f"............增加最近邻的cell后，共计{len(t)}个输出特征图,理论上不应超出{_num_targets*3=},相差{_num_targets*3-len(t)}个")
            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
            # tcls: gt_bbox对应的物体类别
            # tbox：
                # gt_bbox的 (中心点坐标x,y,            # 相对与网格的左上角，在（0，1）范围内
                #           归一化到特征图大小的w,h)    #
            # indices：image, anchor, gridy, gridx
                # image索引
                # anchor索引
                # gt_bbox的中心点坐标x取整（对应网格cell的左上角）
                # gt_bbox的中心点坐标y取整（对应网格cell的左上角）
            # anch：匹配上的anchor的w，h （ 归一化到特征图大小）
        # ####################################################################################################
        # for e in tcls:
        #     print(len(e))
        print("####结束特征图匹配........")
        return tcls, tbox, indices, anch