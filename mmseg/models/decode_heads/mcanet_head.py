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


    def parallel_grouping_batch(self,label: torch.Tensor):
        """
        计算 batch 中每个标签图的连通区域 ID。
        输入: label (B, H, W)
        输出: group_id (B, H, W)
        """
        B, H, W = label.shape
        N = H * W
        device = label.device

        # 1. 初始化每个像素的唯一 ID (0到N-1)
        # group_id 形状: (H, W) -> 扩展到 (B, H, W)
        initial_ids = torch.arange(N, device=device).view(H, W)
        group_id = initial_ids.unsqueeze(0).repeat(B, 1, 1).long() # (B, H, W)

        # 为了使用 F.unfold, 需要 (B, C, H, W) 形状
        label_4d = label.unsqueeze(1).float() # (B, 1, H, W)
        
        # 迭代传播 ID
        for _ in range(max(H, W)): 
            # 4D ID 
            group_id_4d = group_id.unsqueeze(1).float() # (B, 1, H, W)

            # 1. 展开 ID 和 Label (B, C*k*k, N)
            id_patches_folded = F.unfold(group_id_4d, kernel_size=3, padding=1)
            label_patches_folded = F.unfold(label_4d, kernel_size=3, padding=1)
            
            # 转换形状以方便操作: (B, N, k*k=9)
            id_patches = id_patches_folded.transpose(1, 2)
            label_patches = label_patches_folded.transpose(1, 2)
            
            current_id = id_patches[..., 4].unsqueeze(-1) # (B, N, 1)
            current_label = label_patches[..., 4].unsqueeze(-1) # (B, N, 1)
            
            # 邻居像素的 ID 和标签
            neighbor_ids = torch.cat([id_patches[..., :4], id_patches[..., 5:]], dim=-1) # (B, N, 8)
            neighbor_labels = torch.cat([label_patches[..., :4], label_patches[..., 5:]], dim=-1) # (B, N, 8)
            
            # 连通性条件: 邻居标签与中心标签相同
            is_same_label = (neighbor_labels == current_label) # (B, N, 8)
            
            # 传播规则: 如果标签相同，将邻居 ID 与中心 ID 取最小值
            propagated_ids = torch.where(is_same_label, neighbor_ids, current_id.repeat(1, 1, 8))
            
            # 新的 ID 是中心 ID 和所有满足条件邻居 ID 中的最小值
            # 沿 dim=-1 找到最小值, 然后再与中心 ID 取最小值
            min_neighbor_ids = torch.min(propagated_ids, dim=-1).values # (B, N)
            new_group_id_flat = torch.min(min_neighbor_ids, current_id.squeeze(-1)) # (B, N)

            new_group_id = new_group_id_flat.view(B, H, W).long()
            
            if torch.equal(group_id, new_group_id):
                break
            group_id = new_group_id

        return group_id
    
    def find_inter_group_relations_batch(self,group_id: torch.Tensor):
        """
        识别 Batch 中每个元素的 GxG 关系矩阵。
        输入: group_id (B, H, W)
        输出: list of R_i (list of torch.Tensor, size (G_i, G_i))
        """
        B, H, W = group_id.shape
        device = group_id.device
        
        # 1. 展开 ID 
        group_id_4d = group_id.unsqueeze(1).float() # (B, 1, H, W)
        id_patches_folded = F.unfold(group_id_4d, kernel_size=3, padding=1)
        id_patches = id_patches_folded.transpose(1, 2).long() # (B, N, 9)

        current_id = id_patches[..., 4].unsqueeze(-1) # (B, N, 1)
        neighbor_ids = torch.cat([id_patches[..., :4], id_patches[..., 5:]], dim=-1) # (B, N, 8)
        
        # 2. 识别类间边界关系: 邻居 group_id 与中心 group_id 不同
        is_inter_class_relation = (neighbor_ids != current_id) # (B, N, 8)
        
        # 3. 逐个 Batch 元素处理，构建 GxG 矩阵
        R_matrices = []
        for b in range(B):
            # 展平该 Batch 元素的布尔掩码和 IDs
            is_rel_flat = is_inter_class_relation[b].flatten() # (N*8,)
            
            rel_center_ids = current_id[b].repeat(1, 8).flatten()[is_rel_flat]
            rel_neighbor_ids = neighbor_ids[b].flatten()[is_rel_flat]
            
            # 关系对 (中心 GID, 邻居 GID)
            group_pairs = torch.stack([rel_center_ids, rel_neighbor_ids], dim=1)
            
            # 去重
            unique_group_pairs = torch.unique(group_pairs, dim=0)
            
            # 构造 G x G 关系矩阵 R_i
            if unique_group_pairs.numel() == 0:
                # 如果没有相邻区域，Max_GID 就是当前图像的最大 ID + 1
                max_gid = group_id[b].max().item() + 1
                R_i = torch.zeros((max_gid, max_gid), dtype=torch.bool, device=device)
            else:
                max_gid = unique_group_pairs.max().item() + 1
                R_i = torch.zeros((max_gid, max_gid), dtype=torch.bool, device=device)
                
                # 写入 G x G 矩阵
                R_i[unique_group_pairs[:, 0], unique_group_pairs[:, 1]] = True
                R_i[unique_group_pairs[:, 1], unique_group_pairs[:, 0]] = True # 对称
            
            R_matrices.append(R_i)
            
        return R_matrices # 返回一个包含 B 个 GxG 矩阵的列表
    
    def construct_attention_matrix_batch(self,group_id: torch.Tensor, R_matrices: list):
        """
        构造 Batch 中每个元素的 N x N 注意力矩阵。
        输入: group_id (B, H, W), R_matrices (list of R_i)
        输出: M (B, N, N)
        """
        B, H, W = group_id.shape
        N = H * W
        M_batch = []
        
        for b in range(B):
            # 获取当前 Batch 元素的 group_id 和 GxG 关系矩阵 R_i
            group_id_b = group_id[b].flatten() # (N,)
            R_i = R_matrices[b] # (G_i, G_i)
            
            # 1. 构造 N x N 查询索引 (N,) -> (N, N)
            # N*N 的查询 ID 矩阵 (行索引)
            G_query = group_id_b.unsqueeze(1).repeat(1, N) 
            # N*N 的键 ID 矩阵 (列索引)
            G_key = group_id_b.unsqueeze(0).repeat(N, 1)
            
            # 2. 并行查表 (核心操作)
            # M_b[i, j] = R_i[G_query[i, j], G_key[i, j]]
            M_b = R_i[G_query, G_key].float() # (N, N)
            
            M_batch.append(M_b)
        
        # 将列表拼接成一个大的 Batch 张量
        # 警告: 如果 N 很大，M_batch 会占用巨大的 GPU 显存
        M = torch.stack(M_batch, dim=0) # (B, N, N)
        
        return M
    
    def calc_local_inter_mat(self,label: torch.Tensor):
        
        # 步骤 1: 连通区域 ID 计算 (B, H, W)
        group_id = self.parallel_grouping_batch(label)
        
        # 步骤 2: GxG 关系矩阵 R 的计算 (list of R_i)
        R_matrices = self.find_inter_group_relations_batch(group_id)
        
        # 步骤 3: N x N 注意力矩阵 M 的构造 (B, N, N)
        M = self.construct_attention_matrix_batch(group_id, R_matrices)
        
        return M
    
    '''
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
    '''


    def local_inter_loss(self,attn_maps,label):
        label = resize_nochannel(label,size=(64,32),mode='nearest')
        mats = self.calc_local_inter_mat(label)
        assert torch.max(attn_maps) <= 1 and torch.min(attn_maps) >= 0,"局部类间注意力图数据范围错误"
        loss = F.binary_cross_entropy(attn_maps,mats)
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
        label_cat = torch.cat([data_sample.gt_sem_seg.data for data_sample in batch_data_samples],dim=0)
        if self.local_inter_attn_flag:
            losses['loss_local_inter'] = self.local_inter_loss(local_inter_attn_map,label_cat)
        if self.global_intra_attn_flag:
            losses['loss_global_intra'] = self.global_intra_loss(global_intra_attn_maps,label_cat)
        return losses

    def forward(self, inputs):
        if not self.receive_attn_map:
            return super().forward(inputs)
        features,local_inter_attn_map,global_intra_attn_maps = inputs
        seg_result = super().forward(features)
        return [seg_result,local_inter_attn_map,global_intra_attn_maps]