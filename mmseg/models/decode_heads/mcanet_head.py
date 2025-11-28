from collections import deque
from mmseg.registry import MODELS
from .aspp_head import ASPPHead
from ..utils import resize
import torch
import torch.nn.functional as F



@MODELS.register_module()
class MCANetHead(ASPPHead):

    def __init__(self,num_classes=4,receive_attn_map=False, **kwargs):
        super(MCANetHead, self).__init__(num_classes=num_classes,**kwargs)
        self.num_classes = num_classes
        self.receive_attn_map = receive_attn_map

    def calc_local_inter_mat(self,label):
        h,w = label.shape
        n = h * w
        mat = [[0 for j in range(n)] for i in range(n)]
        group_id = [[-1 for j in range(w)] for i in range(h)]
        group_nodes = []
        gid = 0
        tab = [[1,0],[-1,0],[0,1],[0,-1]]
        for i in range(h):
            for j in range(w):
                if group_id[i][j] != -1:
                    continue
                nodes = [(i,j)]
                q = deque([(i,j)])
                group_id[i][j] = gid
                while q:
                    y,x = q.popleft()
                    for ty,tx in tab:
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
                for ty,tx in tab:
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
        return torch.from_numpy(mat).cuda()


    def local_inter_loss(self,attn_map,label):
        label = label.unsqueeze(0).unsqueeze(1).float()
        label = F.interpolate(label,size=(64,32),mode='nearest')
        label = label.squeeze(1).squeeze(0).long()
        mat = self.calc_local_inter_mat(label=label.cpu().numpy())
        loss = F.binary_cross_entropy(attn_map,mat)
        return loss

    def global_intra_loss(self,attn_maps,label):
        loss = torch.tensor(0,device='cuda')
        # (H,W,C)
        label = F.one_hot(label,self.num_classes)
        # (C,H,W)
        label = label.permute(2,0,1)
        # (C,N)
        label = label.flatten(1)
        # (N,C)
        label_t = label.permute(1,0)
        # (N,N)
        mat = label_t @ label
        for attn_map in attn_maps:
            rsmat = resize(mat,attn_map.shape)
            loss += F.binary_cross_entropy(attn_map,rsmat)
        loss = loss / len(attn_maps)
        return loss

    def predict(self, inputs, batch_img_metas, test_cfg):
        if not self.receive_attn_map:
            return super().predict(inputs, batch_img_metas, test_cfg)
        output = inputs
        # predict时只取特征图
        output = inputs[0]
        return super().predict(output, batch_img_metas, test_cfg)
    
    # 因为自定义损失函数只能拿到预测结果拿不到解码器的输出,所以只能通过重写loss方法手动添加
    def loss(self, inputs, batch_data_samples, train_cfg):
        if not self.receive_attn_map:
            return super().loss(inputs, batch_data_samples, train_cfg)
        output,local_inter_attn_map,global_intra_attn_maps = self.forward(inputs)
        losses = self.loss_by_feat(output, batch_data_samples)
        losses['LocalInterLoss'] = torch.tensor(0.)
        losses['GlobalIntraLoss'] = torch.tensor(0.)
        for i,data_sample in enumerate(batch_data_samples):
            label = data_sample.gt_sem_seg.data
            losses['LocalInterLoss'] = losses['LocalInterLoss'] + self.local_inter_loss(local_inter_attn_map[i],label)
            losses['GlobalIntraLoss'] = losses['GlobalIntraLoss'] + self.global_intra_loss([maps[i] for maps in global_intra_attn_maps],label)
        losses['LocalInterLoss'] = losses['LocalInterLoss'] / len(batch_data_samples)
        losses['GlobalIntraLoss'] = losses['GlobalIntraLoss'] / len(batch_data_samples)
        return losses

    def forward(self, inputs):
        if not self.receive_attn_map:
            return super().forward(inputs)
        features,local_inter_attn_map,global_intra_attn_maps = inputs
        seg_result = super().forward(features)
        return [seg_result,local_inter_attn_map,global_intra_attn_maps]