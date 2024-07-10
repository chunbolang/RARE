import torch
from torch import batch_norm, nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.resnet as models
import model.vgg as vgg_models
from model.ASPP import ASPP
from torch.cuda.amp import autocast as autocast

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005  

    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
    
        self.vgg = args.vgg
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        self.pretrained = True
        self.classes = 2
        
        assert self.layers in [50, 101, 152]
    
        if self.vgg:
            print('INFO: Using VGG_16 bn')
            vgg_models.BatchNorm = BatchNorm
            vgg16 = vgg_models.vgg16_bn(pretrained=self.pretrained)
            print(vgg16)
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = get_vgg16_layer(vgg16)
        else:
            print('INFO: Using ResNet {}'.format(self.layers))
            if self.layers == 50:
                resnet = models.resnet50(pretrained=self.pretrained)
            elif self.layers == 101:
                resnet = models.resnet101(pretrained=self.pretrained)
            else:
                resnet = models.resnet152(pretrained=self.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        mask_add_num = 3
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

        self.init_merge_1 = nn.Sequential(
            nn.Conv2d(reduce_dim*2+ mask_add_num , reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))

        self.ASPP = ASPP()

        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))


        self.res_h = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )   
        self.res_h_1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            # nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(inplace=True),                             
        )   
        self.res_w = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        ) 
        self.res_w_1 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            # nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            # nn.ReLU(inplace=True),                             
        )   
   
    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [
            {'params': model.down_query.parameters()},
            {'params': model.down_supp.parameters()},
            {'params': model.init_merge.parameters()},
            {'params': model.init_merge_1.parameters()},
            {'params': model.ASPP.parameters()},
            {'params': model.res1.parameters()},
            {'params': model.res2.parameters()},        
            {'params': model.cls.parameters()},
            {'params': model.res_h.parameters()}, 
            {'params': model.res_w.parameters()}, 
            {'params': model.res_h_1.parameters()}, 
            {'params': model.res_w_1.parameters()}, 
            ],
            
            lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)  # 2.5e-3, 0.9, 1e-4
        
        return optimizer


    def forward(self, x, s_x, s_y, y, cat_idx=None):
        
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)    # 473

        # Query Feature
        with torch.no_grad():
            query_feat_0 = self.layer0(x)                # [4, 128, 119, 119]
            query_feat_1 = self.layer1(query_feat_0)     # [4, 256, 119, 119]
            query_feat_2 = self.layer2(query_feat_1)     # [4, 512, 60, 60]
            query_feat_3 = self.layer3(query_feat_2)     # [4, 1024, 60, 60]
            query_feat_4 = self.layer4(query_feat_3)     # [4, 2048, 60, 60]
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)  # [4, 1536, 60, 60]
        query_feat = self.down_query(query_feat)                 # [4, 256, 60, 60]

        # Support Feature     
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        supp_res_list = []
        supp_feat_tem_list = []
        mask_res_list = []
        mask_down_list = []
        for i in range(self.shot):
            mask_ori = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask_ori)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)   # [4, 1024, 60, 60]
                mask = F.interpolate(mask_ori, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)  
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat_tem = self.down_supp(supp_feat)
            supp_feat_tem_list.append(supp_feat_tem)

            mask_down = F.interpolate(mask_ori, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
            mask_up = F.interpolate(mask_down, size=(mask_ori.size(2), mask_ori.size(3)), mode='nearest')
            mask_res = (mask_ori - mask_up)
            mask_res[mask_res == -1] = 0
            mask_res = F.interpolate(mask_res, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)

            supp_res = Weighted_GAP(supp_feat_tem, mask_res)
            supp_feat = Weighted_GAP(supp_feat_tem, mask)
            mask_res_list.append(mask_res)
            mask_down_list.append(mask)

            supp_feat_list.append(supp_feat)    # [4, 256, 1, 1]    
            supp_res_list.append(supp_res)

        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):  
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                   
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]  # [4, 2048, 60, 60]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)  # [4, 2048, 3600]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)  

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)    # [4, 3600, 2048]
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)  
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)

        if self.shot > 1:             
            supp_feat = supp_feat_list[0]
            supp_res = supp_res_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
                supp_res += supp_res_list[i]
            supp_feat /= len(supp_feat_list) 
            supp_res /= len(supp_res_list) 

        pro_map = torch.cat([supp_feat.unsqueeze(1) , supp_res.unsqueeze(1) ], 1)
        activation_map = self.query_region_activate(query_feat, pro_map , mode = 'Cosine').unsqueeze(2) # b,5,1,h,w

        if self.shot >1 :
            H_bin_list = []
            res_bin_list =[]
            tem_beta = 0

            for i in range (self.shot):
                tem_mask = mask_down_list[i]
                tem_mask_res = mask_res_list[i]
                supp_feat_bin = self.refine(supp_feat_tem_list [i], query_feat ,supp_feat_list[i], tem_mask)  
                H_bin_list.append(supp_feat_bin.unsqueeze(1))
                supp_feat_bin_res = self.refine(supp_feat_tem_list[i], query_feat , supp_res_list[i], tem_mask_res)  
                res_bin_list.append(supp_feat_bin_res.unsqueeze(1))
                tem_beta += self.GAP(tem_mask_res) / (self.GAP(tem_mask) + cosine_eps)
            beta = tem_beta / self.shot

            H_gather = torch.cat(H_bin_list , 1) # b*n*c*h*w
            res_gather = torch.cat(res_bin_list , 1) # b*n*c*h*w

            query_base = query_feat.unsqueeze(1) # b*1*c*h*w
            index_s_H = nn.CosineSimilarity(2)(H_gather , query_base).unsqueeze(2) # b*n*1*h*w
            index_s_res = nn.CosineSimilarity(2)(res_gather , query_base).unsqueeze(2) # b*n*1*h*w
            index_H = index_s_H.max(1)[1].unsqueeze(1).expand_as(query_base)  # b*1*c*h*w
            index_res = index_s_res.max(1)[1].unsqueeze(1).expand_as(query_base)  # b*1*c*h*w
            fuse_H = torch.gather(H_gather , 1 ,index_H).squeeze(1)
            fuse_res = torch.gather(res_gather , 1 ,index_res).squeeze(1)
        else:
            fuse_res = self.refine(supp_feat_tem_list[0], query_feat, supp_res, mask_res)
            fuse_H = self.refine(supp_feat_tem_list [0], query_feat, supp_feat, mask) 
            beta = self.GAP(mask_res) / (self.GAP(mask) + cosine_eps)

        query_feat_bin = query_feat
    
        corr_mask_bin = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)

        temp_fuse = torch.cat([query_feat_bin, fuse_res, corr_mask_bin, activation_map[:,0,...], activation_map[:,1,...] ] , 1)
        query_fuse = self.init_merge_1(temp_fuse)
        merge_feat_bin = torch.cat([query_feat_bin, fuse_H, corr_mask_bin, activation_map[:,0,...], activation_map[:,1,...]], 1)
        merge_feat_bin = self.init_merge(merge_feat_bin)  

        fuse_bin = (merge_feat_bin + query_fuse * beta)

        if self.training:
            mask_query = (y == 1).float().unsqueeze(1)
            mask_query = F.interpolate(mask_query, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)

            aux_loss_2 = self.loss_add(fuse_bin, mask_query) 

        query_feat = self.ASPP(fuse_bin )
        query_feat = self.res1(query_feat)               # 1024->256
        query_feat = self.res2(query_feat) + query_feat  # 2* 3*3Conv

        out = self.cls(query_feat)                       # 3*3 + 1*1 Conv  

        # Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            mask_ori_q = (y == 1).float().unsqueeze(1)
            mask_down_q = F.interpolate(mask_ori_q, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='nearest')
            mask_up_q = F.interpolate(mask_down_q, size=(mask_ori.size(2), mask_ori.size(3)), mode='nearest')
            mask_res_q = (mask_ori_q - mask_up_q)
            mask_res_q[mask_res_q == 0 ] = 255
            mask_res_q[mask_res_q == -1 ] = 0
            mask_res_tem = (mask_res_q == 1 ).float()
            alpha = (self.GAP(mask_res_tem) / (self.GAP(mask_ori_q) +cosine_eps)).mean()
            aux_loss_1 = self.criterion(out, mask_res_q.squeeze(1).long())* alpha
            
            main_loss = self.criterion(out, y.long())
      
            return out.max(1)[1], main_loss, aux_loss_1, aux_loss_2
        else:
            return out


    def refine(self, in_feat, feat_q, feat_pro , mask = None):
        if mask ==None:
            in_feat1 = in_feat
        else :
            in_feat1 = in_feat * mask
        b,c,h,w = in_feat1.size()
        in_h = self.pool_h(in_feat1).permute(0, 1, 3, 2)  # b*c*1*h
        in_w = self.pool_w(in_feat1).permute(0, 1, 3, 2)  # b*c*w*1

        in_h_q = self.pool_h(feat_q) # b*c*h*1
        in_w_q = self.pool_w(feat_q) # b*c*1*w

        a_h = in_h_q * in_h  # b*c*hq*hs
        a_h = a_h.view(b,c,-1)
        a_h = F.softmax(a_h, dim =-1)
        a_h = a_h.view(b,c,h,h)

        a_w = in_w * in_w_q  # b*c*ws*wq
        a_w = a_w.view(b,c,-1)
        a_w = F.softmax(a_w, dim =-1)
        a_w = a_w.view(b,c,w,w)

        a_h_new = self.res_h(a_h)
        a_h_new = self.res_h_1(a_h_new) + a_h_new
        a_w_new = self.res_w(a_w)
        a_w_new = self.res_w_1(a_w_new) +a_w_new

        tmp_pro =feat_pro.expand(-1, -1,h, w) 

        outfeat_h = torch.matmul(a_h_new , tmp_pro)  # (b,c,hq,hs)*(b,c,hs,ws)  
        outfeat = torch.matmul(outfeat_h , a_w_new)  # (b,c,hq,ws)*(b,c,ws,wq)

        return outfeat


    def query_region_activate(self, query_fea, prototypes, mode):
        """             
        Input:  query_fea:      [b, c, h, w]
                prototypes:     [b, n, c, 1, 1]
                mode:           Cosine/Conv/Learnable
        Output: activation_map: [b, n, h, w]
        """
        b, c, h, w = query_fea.shape
        n = prototypes.shape[1]
        que_temp = query_fea.reshape(b, c, h*w)

        if mode == 'Conv':
            map_temp = torch.bmm(prototypes.squeeze(-1).squeeze(-1), que_temp)  # [b, n, h*w]
            activation_map = map_temp.reshape(b, n, h, w)
            return activation_map

        elif mode == 'Cosine':
            que_temp = que_temp.unsqueeze(dim=1)           # [b, 1, c, h*w]
            prototypes_temp = prototypes.squeeze(dim=-1)   # [b, n, c, 1]
            map_temp = nn.CosineSimilarity(2)(que_temp, prototypes_temp)  # [n, c, h*w]
            activation_map = map_temp.reshape(b, n, h, w)       # Normalize to (0,1)
            return activation_map

        elif mode == 'Learnable':
            for p_id in range(n):
                prototypes_temp = prototypes[:,p_id,:,:,:]                         # [b, c, 1, 1]
                prototypes_temp = prototypes_temp.expand(b, c, h, w)
                concat_fea = torch.cat([query_fea, prototypes_temp], dim=1)        # [b, 2c, h, w]                
                if p_id == 0:
                    activation_map = self.relation_coding(concat_fea)              # [b, 1, h, w]
                else:
                    activation_map_temp = self.relation_coding(concat_fea)              # [b, 1, h, w]
                    activation_map = torch.cat([activation_map,activation_map_temp], dim=1)
            return activation_map
            

    def loss_add(self, query_feat, mask_q):
        '''
        expect 
        feat.size: b*c*h*w
        feat_fg_1 has the same class with feat_fg_2

        '''
        b, c, h, w = query_feat.size()
        mask_q = mask_q.squeeze(1)
        tmp_loss = 0
        feat_pro = Weighted_GAP(query_feat, mask_q.unsqueeze(1)).squeeze(2).squeeze(2)

        for i in range(b):
            tmp_location_fg = torch.where(mask_q[i]>0.5) # h*w
            tmp_location_bg = torch.where(mask_q[i]<=0.5)

            if len(tmp_location_fg[0])>0 and len(tmp_location_bg[0])>0:
                num_fg = torch.randint(0, len(tmp_location_fg[0]), [1])  # |P|=|N|=1
                num_bg = torch.randint(0, len(tmp_location_bg[0]), [1])

                feat_fg_1 = query_feat[i, :, tmp_location_fg[0][num_fg[0]], tmp_location_fg[1][num_fg[0]]] 
  
                feat_bg = query_feat[i, :, tmp_location_bg[0][num_bg[0]], tmp_location_bg[1][num_bg[0]]] 

                s = 1
                distance_in = nn.CosineSimilarity(0)(feat_fg_1, feat_pro[i])*s
                distance_out = nn.CosineSimilarity(0)(feat_fg_1, feat_bg)*s
                distance_out_1 = nn.CosineSimilarity(0)(feat_pro[i], feat_bg)*s
                tmp_loss += 1-distance_in + distance_out + distance_out_1
            else:
                tmp_loss += torch.tensor(0).float().cuda()
        fin_loss = tmp_loss/(b*3)

        return fin_loss
