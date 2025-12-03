from collections import deque
from math import sqrt
from mmseg.registry import MODELS
from .aspp_head import ASPPHead
from ..utils import resize
import torch
import torch.nn.functional as F

def resize_nochannel(t:torch.Tensor,size=None,scale_factor=None,mode='nearest'):
    dt = t.dtype
    t = t.unsqueeze(1).float()
    t = resize(t,size=size,scale_factor=scale_factor,mode=mode)
    t = t.squeeze(1)
    t = t.to(dtype=dt)
    return t

@MODELS.register_module()
class MCANetHead(ASPPHead):

    def __init__(self,
                 num_classes=4,
                 receive_attn_map=False,
                 label_downsample_rate=1/16,
                 ignore_index=None,
                 local_inter_attn_flag=True,
                 global_intra_attn_flag=True, 
                 lambda_l=0.7,
                 lambda_g=1,
                 **kwargs):
        super(MCANetHead, self).__init__(num_classes=num_classes,**kwargs)
        self.num_classes = num_classes
        self.receive_attn_map = receive_attn_map
        self.label_downsample_rate = label_downsample_rate
        self.ignore_index = ignore_index
        self.local_inter_attn_flag = local_inter_attn_flag
        self.global_intra_attn_flag = global_intra_attn_flag
        self.lambda_l = lambda_l
        self.lambda_g = lambda_g

    def calc_local_inter_mat(self,label):
        h,w = label.shape
        n = h * w
        mat = [[0 for j in range(n)] for i in range(n)]
        group_id = [[-1 for j in range(w)] for i in range(h)]
        group_nodes = []
        gid = 0
        dire = [[1,0],[-1,0],[0,1],[0,-1]]
        for i in range(h):
            for j in range(w):
                if group_id[i][j] != -1:
                    continue
                nodes = [(i,j)]
                q = deque([(i,j)])
                group_id[i][j] = gid
                while q:
                    y,x = q.popleft()
                    for ty,tx in dire:
                        ny,nx = y + ty,x + tx
                        if ny >= h or ny < 0 or nx >= w or nx < 0 or group_id[ny][nx] != -1 or label[ny][nx] != label[i][j]:
                            continue
                        q.append((ny,nx))
                        nodes.append((ny,nx))
                        group_id[ny][nx] = gid
                group_nodes.append(nodes)
                gid += 1

        relate_groups = [set() for _ in range(gid)]
        for i in range(h):
            for j in range(w):
                for ty,tx in dire:
                    ny,nx = ty + i,tx + j
                    while 0 <= ny < h and 0 <= nx < w:
                        if group_id[ny][nx] != group_id[i][j]:
                            relate_groups[group_id[i][j]].add(group_id[ny][nx])
                            break
                        ny += ty
                        nx += tx
        for i in range(h):
            for j in range(w):
                gid = group_id[i][j]
                for g in relate_groups[gid]:
                    for y,x in group_nodes[g]:
                        mat[i * w + j][y * w + x] = 1
        return torch.tensor(mat,dtype=torch.float)


    def local_inter_loss(self,attn_map,label):
        label = resize_nochannel(label,size=(64,32),mode='nearest')
        label = label.squeeze(0)
        mat = self.calc_local_inter_mat(label=label.cpu().numpy())
        assert torch.max(attn_map) <= 1 and torch.min(attn_map) >= 0,"局部类间注意力图数据范围错误"
        loss = F.binary_cross_entropy(attn_map,mat.to(attn_map.device))
        return loss * self.lambda_l

    def global_intra_loss(self,attn_maps,label):
        label = resize_nochannel(label,scale_factor=self.label_downsample_rate,mode='nearest')
        loss = torch.tensor(0.,device='cuda')
        # (B,H,W,C)
        if self.ignore_index:
            label[label == self.ignore_index] = self.num_classes
            label = F.one_hot(label,self.num_classes + 1).permute(0,3,1,2)
        else:
            label = F.one_hot(label,self.num_classes).permute(0,3,1,2)
        label = label.float()
        # (B,C,H,W)
        label = label
        # (B,C,N)
        label = label.flatten(2)
        # (B,N,C)
        label_t = label.permute(0,2,1)
        # (B,N,N)
        mat = label_t @ label
        for attn_map in attn_maps:
            rsmat = resize_nochannel(mat,attn_map.shape[-2:],mode='nearest')
            assert torch.max(attn_map) <= 1 and torch.min(attn_map) >= 0,"全局类内注意力图数据范围错误"
            assert torch.max(rsmat) <= 1 and torch.min(rsmat) >= 0,"全局类内矩阵数据范围错误"
            loss += F.binary_cross_entropy(attn_map,rsmat)
        loss = loss / len(attn_maps)
        return loss * self.lambda_g

    def predict(self, inputs, batch_img_metas, test_cfg):
        if not self.receive_attn_map:
            return super().predict(inputs, batch_img_metas, test_cfg)
        result,_,_ = self.forward(inputs)
        return self.predict_by_feat(result,batch_img_metas=batch_img_metas)
    
    # 因为自定义损失函数只能拿到预测结果拿不到解码器的输出,所以只能通过重写loss方法手动添加
    def loss(self, inputs, batch_data_samples, train_cfg):
        if not self.receive_attn_map:
            return super().loss(inputs, batch_data_samples, train_cfg)
        output,local_inter_attn_map,global_intra_attn_maps = self.forward(inputs)
        losses = self.loss_by_feat(output, batch_data_samples)
        if self.local_inter_attn_flag:
            loss_local_inter = torch.tensor(0.,device='cuda',requires_grad=True)
            for i,data_sample in enumerate(batch_data_samples):
                label = data_sample.gt_sem_seg.data
                loss_local_inter = loss_local_inter + self.local_inter_loss(local_inter_attn_map[i],label)        
            losses['loss_local_inter'] = loss_local_inter / len(batch_data_samples)
        if self.global_intra_attn_flag:
            label_cat = torch.cat([data_sample.gt_sem_seg.data for data_sample in batch_data_samples],dim=0)
            losses['loss_global_intra'] = self.global_intra_loss(global_intra_attn_maps,label_cat)
        return losses

    def forward(self, inputs):
        if not self.receive_attn_map:
            return super().forward(inputs)
        features,local_inter_attn_map,global_intra_attn_maps = inputs
        seg_result = super().forward(features)
        return [seg_result,local_inter_attn_map,global_intra_attn_maps]