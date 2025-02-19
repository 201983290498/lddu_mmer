from importlib.metadata import entry_points
from .models import *
import math
import torch
import torch.nn as nn

def getBinaryTensor(imgTensor, boundary = 0.21):
    one = torch.ones_like(imgTensor).fill_(1)
    zero = torch.zeros_like(imgTensor).fill_(0)
    return torch.where(imgTensor > boundary, one, zero)

def search_binary(pred_score, total_label, eval_threshold=None):
    if not (eval_threshold is None):
        total_pred = getBinaryTensor(pred_score, eval_threshold)
        total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_label)
        logger.info("Eval Threshold, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f",
                    total_micro_f1, total_micro_precision, total_micro_recall, total_acc)
    best_f1,best_threshold = 0,0
    best_result = []
    for threshold in range(150, 300, 5):
        total_pred = getBinaryTensor(pred_score, threshold / 1000.0)
        total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_label)
        if total_micro_f1 > best_f1:
            best_threshold = threshold
            best_f1 = total_micro_f1
            best_result = [total_micro_f1, total_micro_precision, total_micro_recall, total_acc]
    logger.info("Best Threshold, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f， threshold: %f",
                best_result[0], best_result[1], best_result[2], best_result[3], best_threshold / 100)
    return best_threshold / 1000.0

class LabelWiseModalModel(nn.Module):
    
    def __init__(self, label_dim, hidden_dim, pro_dim, dropout=0.1):
        super(LabelWiseModalModel, self).__init__()
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.pro_dim = pro_dim
        self.seq_projection = nn.ModuleList([nn.Linear(hidden_dim, pro_dim) for _ in range(3)]) # 3 modalities v,a,t
        self.label_projection = nn.ModuleList([nn.Linear(label_dim, pro_dim) for _ in range(3)]) # 3 modalities v,a,t

        self.out_projection = nn.ModuleList([nn.Linear(pro_dim, hidden_dim) for _ in range(3)]) # 3 modalities
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, label_features, visual_features, audio_features, text_features):
        # label_features: [batch_size, labels, label_dim]
        # modal_features: [batch_size, seq, hidden_dim]
        label_proj_features = [] # v, a, t
        for i in range(len(self.label_projection)):
            label_proj_features.append(self.label_projection[i](label_features)) # [batch_size, labels, pro_dim]
        
        modal_proj_features = [] # v, a, t
        modal_proj_features.append(self.seq_projection[0](visual_features))
        modal_proj_features.append(self.seq_projection[1](audio_features))
        modal_proj_features.append(self.seq_projection[2](text_features)) # [batch_size, seq, pro_dim]


        Vij = []
        for i in range(3):
            attention_scores = torch.matmul(label_proj_features[i], modal_proj_features[i].transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.pro_dim) # b,l,s
            attention_scores = self.softmax(attention_scores) # b,l,s
            feat = torch.matmul(attention_scores, modal_proj_features[i]) # b,l,d
            label_proj_features[i] = self.dropout(feat)
            # label_proj_features[i] = feat
            Vij.append(self.out_projection[i](label_proj_features[i]))

        return Vij # [[batch_size, labels, hidden_dim], [batch_size, labels, hidden_dim], [batch_size, labels, hidden_dim]]

class LatentEncoder(nn.Module):
    def __init__(self, state_list, output_size, latten_size, dropout=0.5):
        super(LatentEncoder, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s)
                                    for in_s, out_s in zip(state_list[:-1], state_list[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(state_list[-1], output_size)
        self.fc_mu = nn.Linear(output_size, latten_size)
        self.fc_var = nn.Linear(output_size, latten_size)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.xavier_uniform_(self.fc_mu.weight)
        nn.init.xavier_uniform_(self.fc_var.weight)
        self.dropout = nn.Dropout(dropout)
        
        
    def encode(self, inputs):
        linear_out = inputs
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        output = F.dropout(F.relu(self.output(linear_out)), p=0.1) # [B, labels, latent_size]
        output = F.relu(self.output(linear_out))
        mu = self.fc_mu(output) 
        logvar = self.fc_var(output) # [B, labels, latent_size]
        return [mu, logvar]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # return mu + eps*std
        return mu 

class LatentDecoder(nn.Module):
    def __init__(self, state_list, output_size, dropout=0.5):
        super(LatentDecoder, self).__init__()
        self.linear = nn.ModuleList(nn.Linear(in_s, out_s) for in_s, out_s in zip(state_list[:-1], state_list[1:]))
        for linear in self.linear:
            nn.init.xavier_uniform_(linear.weight)
        self.output = nn.Linear(state_list[-1], output_size)
        nn.init.xavier_uniform_(self.output.weight)
        self.dropout = nn.Dropout(dropout)
    
    def decode(self, latent_feature):
        linear_out = latent_feature
        for linear in self.linear:
            linear_out = F.relu(linear(linear_out))
        linear_out = F.dropout(linear_out, p=0.1)
        return self.output(linear_out)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask, neg_mask=None, batch_size=-1, device=None, other_features=None):
        if mask is not None:
            mask = mask.float().detach()
            if other_features is None:
                similarity = torch.matmul(features[:batch_size], features.T) / 2.0  # 计算相似度
                anchor_dot_contrast = torch.div(
                    similarity,
                    self.temperature)
            else:
                similarity = torch.matmul(features[:batch_size], other_features.T) / 2.0  # 计算相似度
                anchor_dot_contrast = torch.div(
                    similarity,
                    self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()
            if neg_mask is None:
                logits_mask = torch.ones_like(mask)
            else:
                logits_mask = torch.scatter(neg_mask, 1, torch.arange(batch_size).view(-1, 1).to(neg_mask.device), 0)
            mask = mask * logits_mask
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
            mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)
        return loss, ((similarity * mask).sum() / mask.sum(),(similarity * logits_mask).sum() / logits_mask.sum())

class MyModel(TAILORPreTrainedModel): 
    def __init__(self, bert_config, visual_config, audio_config, cross_config, decoder_config, task_config):
        super(MyModel, self).__init__(bert_config, visual_config, audio_config, cross_config, decoder_config)
        
        self.task_config = task_config 
        self.ignore_video_index = -1 
        self.num_classes = task_config.num_classes 
        self.aligned = task_config.aligned 
        
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_sequence <= audio_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "bert_num_hidden_layers")
        bert_config = update_attr("bert_config", bert_config, "vocab_size",
                                   self.task_config, "text_dim")

        self.bert = BertModel(bert_config) 
        self.embedding_layer = nn.Embedding(self.num_classes, task_config.hidden_size)
        nn.init.xavier_uniform_(self.embedding_layer.weight)
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        visual_config = update_attr("visual_config", visual_config, "vocab_size",
                                    self.task_config, "video_dim")

        self.visual = VisualModel(visual_config)
        audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                    self.task_config, "audio_num_hidden_layers")
        audio_config = update_attr("audio_config", audio_config, "vocab_size",
                                    self.task_config, "audio_dim")

        self.audio = AudioModel(audio_config)
        
        self.text_norm = NormalizeText(task_config) 
        self.visual_norm = NormalizeVideo(task_config)
        self.audio_norm = NormalizeAudio(task_config)
        
        self.label_wise_attention = LabelWiseModalModel(task_config.label_dim, task_config.hidden_size, task_config.pro_dim)
        
        self.common_classfier = nn.Sequential(
            nn.Linear(task_config.hidden_size * 3, task_config.hidden_size // 2) ,
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        self.final_classifier = nn.ModuleList([nn.Sequential( 
                nn.Linear(task_config.latent_size * 3, task_config.hidden_size // 2),
                nn.Dropout(0.1),
                nn.ReLU(),
                nn.Linear(task_config.hidden_size // 2, 1),
                nn.Sigmoid()
            ) for _ in range(6)])
        
        self.final_classifer2 = nn.Sequential(
            nn.Linear(task_config.latent_size * 6, task_config.hidden_size), 
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size, 6),
            nn.Sigmoid()
        )

        self.var_classifier1 = nn.ModuleList([nn.Sequential( 
            nn.Linear(task_config.latent_size * 3, task_config.hidden_size // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size // 2, 1),
            nn.Sigmoid()
        ) for _ in range(6)])

        self.var_classifier2 = nn.Sequential(
            nn.Linear(task_config.latent_size * 6, task_config.hidden_size ), 
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size, 6),
            nn.Sigmoid()
        )

        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.criterion_cl = SupConLoss(task_config.temperature, task_config.temperature)
        self.encoder = nn.ModuleList([LatentEncoder([task_config.hidden_size, task_config.hidden_size // 4 * 3, task_config.hidden_size // 2], task_config.hidden_size //2, task_config.latent_size) for _ in range(3)])

        self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.latent_size*2))
        # self.register_buffer('queue', torch.randn(task_config.moco_queue, task_config.latent_size))     
        self.register_buffer("queue_label", torch.randn(task_config.moco_queue, 1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=0)
        self.numCounts = {}
        self.correctCounts = {}

    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask, 
                label_input, label_mask, groundTruth_labels=None, training=True, samples_index=None): 
        """
        text: [B, L, Dt]
        visual: [B, L, Dv]
        audio: [B, L, Da]
        groundTruth_labels: [B, num_classes]

        """
        label_input = label_input.unsqueeze(0).repeat(text.shape[0], 1) # L=> b,L
        label_mask = label_mask.unsqueeze(0).repeat(text.shape[0], 1) # l => b,l
        label_features = self.embedding_layer(label_input) # B, L, D
        text = self.text_norm(text)
        visual = self.visual_norm(visual)  
        audio = self.audio_norm(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual, visual_mask, audio, audio_mask) #[B, L, D]
        
        # [batch_size, label_nums, hidden_size]
        label_features = self.label_wise_attention(label_features, visual_output, audio_output, text_output)

        mus, stds = [], []
        for idx in range(len(label_features)):
            label_feature = label_features[idx]
            mu, log_var = self.encoder[idx].encode(label_feature)
            mus.append(mu)
            stds.append(torch.exp(0.5 * log_var))

        common_features = torch.cat((label_features[0], label_features[1],label_features[2]), axis=-1)# [batch_size, label_nums, 3 * hidden_size] 
        common_result = self.common_classfier(common_features).squeeze(-1)  
        
        mus_features = torch.cat(mus, axis=-1)
        stds_features = torch.cat(stds, axis=-1)
        mus_result, stds_result = [], []
        for i in range(len(self.final_classifier)):
            mus_result.append(self.final_classifier[i](mus_features[:,i,:].unsqueeze(1)).squeeze(-1))
        mus_result = torch.cat(mus_result, dim=-1)
        for i in range(len(self.var_classifier1)):
            stds_result.append(self.var_classifier1[i](stds_features[:,i,:].unsqueeze(1)).squeeze(-1))
        stds_result = torch.cat(stds_result, dim=-1)
        final_result = mus_result * common_result + stds_result * (1 - common_result)

        final_labels = getBinaryTensor(final_result)
        l2_norms = torch.norm(stds_features, p=2, dim=2).data.clone().cpu()
        if training:
            common_loss = self.bce_loss(common_result, groundTruth_labels)
            final_loss = self.bce_loss(final_result, groundTruth_labels)
            cdl_loss, mean_similarity = self.SupContrasts(mus, stds, groundTruth_labels)
            crl_loss = self.CRLDis(common_result , groundTruth_labels, stds_features, samples_index)
            total_loss = self.task_config.cml*common_loss + self.task_config.final_loss * final_loss + self.task_config.cdl*cdl_loss + self.task_config.crl*crl_loss
            return total_loss, final_labels, groundTruth_labels, final_result, mean_similarity, (mus_features.data.clone(), stds_features.data.clone(), l2_norms)
        else:
            return final_labels, groundTruth_labels, final_result, (mus_features.data.clone(), stds_features.data.clone(), l2_norms)
    
    def get_similar(self, mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        term1 = 0.125 * (diff ** 2) / (0.5 * (sigma1 ** 2 + sigma2 ** 2))
        term2 = 0.5 * torch.log(0.5 * (sigma1 ** 2 + sigma2 ** 2) / (sigma1 * sigma2))
        distance = torch.sum(term1 + term2, dim=-1)
        coefficient = torch.exp(-distance)
        return coefficient.sum()

    def CRLDis(self, common_result , groundTruth_labels, stds_features, samples_index=None):
        var_norm2 = torch.norm(stds_features, p=2, dim=-1).view(-1)
        var_scores = F.softmax(var_norm2)
        common_scores = F.softmax(common_result)
        kl_loss1 = F.kl_div(var_scores.log(), common_scores.view(-1), reduction='batchmean') + F.kl_div(common_scores.view(-1).log(), var_scores, reduction='batchmean')
        c_scores = []
        for idx, index in enumerate(samples_index): # 记录历史得分
            if index not in self.numCounts:
                self.numCounts[index] = 1
                self.correctCounts[index] = common_result[idx]
            else:
                self.numCounts[index] += 1
                self.correctCounts[index] += common_result[idx]
            c_scores.append(self.correctCounts[index] / self.numCounts[index]) 
        c_scores = F.softmax(torch.cat(c_scores).view(-1))
        # kl_loss2 = F.kl_div(common_result.view(-1).log(), c_scores, reduction='batchmean') + F.kl_div(c_scores.log(), common_result.view(-1), reduction='batchmean')
        
        return kl_loss1 / 2

    def get_text_visual_audio_output(self, text, text_mask, visual, visual_mask, audio, audio_mask):
        """
        Uni-modal Extractor
        """
        text_layers, _ = self.bert(text, text_mask, output_all_encoded_layers=True)
        text_output = text_layers[-1] 
        visual_layers, _ = self.visual(visual, visual_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]  
        audio_layers, _ = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        audio_output = audio_layers[-1] 

        return text_output, visual_output, audio_output

    def SupContrasts(self, mus, stds, groundTruth_labels):
        mus = torch.stack(mus, dim=1) # [batch_size, 3, 6 labels,latent_dim]
        mus = F.normalize(mus, dim=-1)
        stds = torch.stack(stds, dim=1) # [batch_size, 3, 6 labels,latent_dim]
        stds = F.normalize(stds, dim=-1)
        total_proj = torch.cat((mus, stds), dim=-1) # [batch_size, 3, 6 labels, 2 * latent_dim]
        total_proj = total_proj.view(-1, total_proj.shape[-1])
        
        cl_labels = self.get_cl_labels(groundTruth_labels, times=1).view(-1).unsqueeze(-1)
        cl_feats = torch.cat((total_proj, self.queue.clone().detach()), dim=0)
        total_cl_labels = torch.cat((cl_labels, self.queue_label.clone().detach()), dim=0)
        batch_size = cl_feats.shape[0]
        cl_mask, cl_neg_mask = self.get_cl_mask(total_cl_labels, batch_size)
        cdl_loss, mean_similarity = self.criterion_cl(cl_feats, cl_mask, cl_neg_mask, batch_size)
        self.dequeue_and_enqueue(total_proj, cl_labels)
        return cdl_loss, mean_similarity

    def _get_cross_output(self, sequence_output, visual_output,  audio_output, common_feature, attention_mask, visual_mask, audio_mask, common_mask):
        # =============> visual audio fusion
        va_concat_features = torch.cat((audio_output, visual_output), dim=1)
        va_concat_mask = torch.cat((audio_mask, visual_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(visual_mask)
        audio_type_ =  torch.zeros_like(audio_mask)
        va_concat_type = torch.cat((audio_type_, video_type_), dim=1)
        va_cross_layers, va_pooled_output = self.va_cross(va_concat_features, va_concat_type, va_concat_mask)
        va_cross_output = va_cross_layers[-1]
        # <============= visual audio fusion

        # =============> VisualAudio and text fusion
        vat_concat_features = torch.cat((sequence_output, va_cross_output), dim=1)
        vat_concat_mask = torch.cat((attention_mask, va_concat_mask), dim=1)
        va_type_ = torch.ones_like(va_concat_mask)
        vat_concat_type = torch.cat((text_type_, va_type_), dim=1)
        vat_cross_layers, vat_pooled_output = self.vat_cross(vat_concat_features, vat_concat_type, vat_concat_mask)
        vat_cross_output = vat_cross_layers[-1]
        # <============= VisualAudio and text fusion

        # =============> private common fusion
        pc_concate_features = torch.cat((vat_cross_output, common_feature), dim=1)
        specific_type = torch.zeros_like(vat_concat_mask)
        common_type = torch.ones_like(common_mask)
        pc_concate_type = torch.cat((specific_type, common_type), dim=1)
        pc_concat_mask = torch.cat((vat_concat_mask, common_mask), dim=1)
        pc_cross_layers, pc_pooled_output = self.pc_cross(pc_concate_features, pc_concate_type, pc_concat_mask)
        pc_cross_output = pc_cross_layers[-1]
        # <============= private common fusion
 
        return  pc_pooled_output, pc_cross_output, pc_concat_mask
    
    def inference(self, text, text_mask, visual, visual_mask, audio, audio_mask, \
                label_input, label_mask, groundTruth_labels=None):
        label_input = label_input.unsqueeze(0)
        batch = text.size(0)
        label_input = label_input.repeat(batch, 1)
        label_mask = label_mask.unsqueeze(0).repeat(batch, 1)
        text = self.text_norm(text)
        visual = self.visual_norm(visual)   
        audio = self.audio_norm(audio)
        if self.aligned == False:
            visual, _ = self.v2t_ctc(visual)
            audio, _ = self.a2t_ctc(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual, visual_mask, audio, audio_mask)

        private_text = self.private_feature_extractor[0](text_output)
        private_visual = self.private_feature_extractor[1](visual_output)
        private_audio = self.private_feature_extractor[2](audio_output)

        common_text = self.common_feature_extractor(text_output)
        common_visual = self.common_feature_extractor(visual_output)
        common_audio = self.common_feature_extractor(audio_output)

        common_feature = (common_text + common_visual + common_audio) #[B, L, D]
        preivate_feature = private_text + private_visual + private_audio 
        pooled_common = common_feature[:, 0] #[B, D]
        pooled_preivate = preivate_feature[:, 0]
        common_pred = self.common_classfier(pooled_common)
        preivate_pred = self.common_classfier(pooled_preivate)
        common_mask = torch.ones_like(text_mask)


        pooled_output, cross_output, cross_mask = self._get_cross_output(private_text, private_visual, private_audio, common_feature, text_mask, visual_mask, audio_mask, common_mask)
        decoder_output = self.decoder(label_input, cross_output, label_mask, cross_mask)
        cross_predict_scores = self.cross_classifier(decoder_output)
        cross_predict_scores  = cross_predict_scores.view(-1, self.num_classes)     
        predict_scores = cross_predict_scores
        predict_labels = getBinaryTensor(predict_scores)
        groundTruth_labels = groundTruth_labels.view(-1, self.num_classes)
        

        return predict_labels, groundTruth_labels

    def calculate_orthogonality_loss(self, first_feature, second_feature):
        diff_loss = torch.norm(torch.bmm(first_feature, second_feature.transpose(1, 2)), dim=(1, 2)).pow(2).mean()
        return diff_loss
    
        # 更新对比学习队列

    def dequeue_and_enqueue(self, feats, labels):
        batch_size = feats.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size >= self.task_config.moco_queue:
            self.queue[ptr:,:] = feats[:self.task_config.moco_queue-ptr,:]
            self.queue[:batch_size - self.task_config.moco_queue + ptr,:] = feats[self.task_config.moco_queue-ptr:,:]
            self.queue_label[ptr:, :] = labels[:self.task_config.moco_queue - ptr, :]
            self.queue_label[:batch_size - self.task_config.moco_queue + ptr, :] = labels[self.task_config.moco_queue - ptr:,
                                                                             :]
        else:
            self.queue[ptr:ptr+batch_size, :] = feats
            self.queue_label[ptr:ptr + batch_size, :] = labels
        ptr = (ptr + batch_size) % self.task_config.moco_queue  
        self.queue_ptr[0] = ptr

    def get_cl_labels(self, labels, times = 1): # [batch_size, 6 labels], times
        text_labels = torch.zeros_like(labels) + labels
        visual_labels = torch.zeros_like(labels) + labels
        audio_labels = torch.zeros_like(labels) + labels

        text_cl_labels = torch.zeros_like(text_labels, dtype=torch.long)
        visual_cl_labels = torch.zeros_like(visual_labels, dtype=torch.long)
        audio_cl_labels = torch.zeros_like(audio_labels, dtype=torch.long)

        example_idx, label_idx = torch.where(text_labels >= 0.5) 
        text_cl_labels[example_idx, label_idx] = label_idx 
        example_idx, label_idx = torch.where(text_labels < 0.5) 
        text_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 1 

        example_idx, label_idx = torch.where(visual_labels >= 0.5) 
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 2
        example_idx, label_idx = torch.where(visual_labels < 0.5)
        visual_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 3

        example_idx, label_idx = torch.where(audio_labels >= 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 4
        example_idx, label_idx = torch.where(audio_labels < 0.5)
        audio_cl_labels[example_idx, label_idx] = label_idx + self.num_classes * 5

        cl_labels = torch.stack([text_cl_labels, visual_cl_labels, audio_cl_labels], dim=1) # 64, 3, 6
        # The above step divides labels for each modality into 0~11, with each label split into two values: 0,6; 1,7; 2,8; 3,9; 4,10; 5,11
        cl_labels = cl_labels.to(torch.int)
        if times > 1:
            final_cl_labels = torch.cat([cl_labels, cl_labels], dim=1)
            for i in range(2, times):
                final_cl_labels = torch.cat([final_cl_labels, cl_labels], dim=1)
        else:
            final_cl_labels = cl_labels
        return final_cl_labels
    def get_cl_mask(self, cl_labels, batch_size):
        mask = torch.eq(cl_labels[:batch_size], cl_labels.T).float() # Find elements with similar labels across rows and columns
        neg_mask = torch.ones_like(mask)
        return mask, neg_mask