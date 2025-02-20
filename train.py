from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import argparse     
from src.models.myModels import MyModel, getBinaryTensor, search_binary
from src.models.optimization import BertAdam
from src.utils.eval import get_metrics
from src.utils.eval_gap import *
from torch.utils.data import DataLoader, WeightedRandomSampler
from util import get_logger
from src.dataloaders.cmu_dataloader_ai import AlignedMoseiDataset, UnAlignedMoseiDataset
BASE_PATH = './'
global logger
from src.utils.modelBackUp import ModelBackUp, EarlyStopping

def get_args(description='Multi-modal Multi-label Emotion Recognition'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", default=True, action='store_true', help="Whether to run training.") 
    parser.add_argument("--do_test", action='store_true', help="whether to run test")
    parser.add_argument("--aligned", default=True, action='store_true', help="whether train align of unalign dataset")
    parser.add_argument("--data_path", type=str, default='./datasets/MMSA/aligned_501.pkl', help='cmu_mosei data_path')
    parser.add_argument("--output_dir", default=BASE_PATH+'/cpkt_align', type=str, required=False,
                            help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--num_thread_reader', type=int, default=1, help='') 
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit') 
    parser.add_argument('--unaligned_data_path', type=str, default='./datasets/MMSA/aligned_501.pkl', help='load unaligned dataset')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay') 
    parser.add_argument('--n_display', type=int, default=10, help='Information display frequence')
    parser.add_argument('--text_dim', type=int, default=768, help='text_feature_dimension')  # 文本的特征
    parser.add_argument('--video_dim', type=int, default=35, help='video feature dimension')  # 视频的特征
    parser.add_argument('--audio_dim', type=int, default=74, help='audio_feature_dimension')  # 音频的特征
    parser.add_argument('--seed', type=int, default=42, help='random seed') 
    parser.add_argument('--max_words', type=int, default=60, help='')
    parser.add_argument('--max_frames', type=int, default=60, help='')
    parser.add_argument('--max_sequence', type=int, default=60, help='')
    parser.add_argument('--max_label', type=int, default=6, help='')
    parser.add_argument("--bert_model", default="bert-base", type=str, required=False, help="Bert module")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--audio_model", default="audio-base", type=str, required=False, help="Audio module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.") 
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training") 
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--bert_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=3, help="Layer NO. of visual.")
    parser.add_argument('--audio_num_hidden_layers', type=int, default=3, help="Layer No. of audio")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=1, help="Layer NO. of decoder.")
    parser.add_argument("--num_classes", default=6, type=int, required=False, help='Number of emotion label categories')
    parser.add_argument("--hidden_size",type=int, default=256, help='Size of hidden layer')
    parser.add_argument("--latent_size",type=int, default=128, help='Size of latent variables after encoding')
    parser.add_argument("--label_dim",type=int, default=256, help='Dimension size of labels')
    parser.add_argument("--pro_dim",type=int, default=256, help='Size of projection layer')
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument("--cml",type=float, default=0.0, help='Classification loss for label-related vectors')
    parser.add_argument("--kll",type=float, default=0.00, help='KL divergence loss')
    parser.add_argument("--cdl",type=float, default=0.1, help='Contrastive loss')
    parser.add_argument("--rel",type=float, default=0.1, help='Reconstruction loss for encoder and decoder')
    parser.add_argument("--device", type=str, default="cuda", help='Device placement location')
    parser.add_argument("--final_loss", type=float, default=1.0, help='Classification loss', required=False) # Final classification loss
    parser.add_argument("--moco_queue", type=int, default=8192, help='Size of queue')
    parser.add_argument("--average", type=bool, default=False, help='Whether to calculate comprehensive modal scores')
    parser.add_argument("--temperature", type=float, default=0.07, help='Temperature coefficient')
    parser.add_argument("--crl",type=float, default=1, help='CRL loss')
    args = parser.parse_args()
    # Check paramenters
    if args.gradient_accumulation_steps < 1: 
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args): 
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  
    torch.cuda.set_device(args.local_rank) 
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, "log.txt"))
    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
    return args

def init_device(args, local_rank):
    global logger
    device = args.device
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0: 
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))
    return device, n_gpu

def init_model(args, device, n_gpu, local_rank): 

    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu') 
    else:
        model_state_dict = None
    # Prepare model 加载预训练模型
    model = MyModel.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model, args.decoder_model, task_config=args)
    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "audio." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "audio." not in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * 1.0},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * 1.0},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]
    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)
    return optimizer, scheduler, model

def prep_dataloader(args):
    Dataset = AlignedMoseiDataset if args.aligned else UnAlignedMoseiDataset
    train_dataset = Dataset(
        args.data_path,
        'train'
    )
    val_dataset = Dataset(
        args.data_path,
        'valid'
    )
    test_dataset = Dataset(
        args.data_path,
        'test'
    )
    label_input, label_mask = train_dataset._get_label_input()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=True,
        drop_last=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False   
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=False,
        drop_last=False,   
    )
    train_length = len(train_dataset)
    val_length = len(val_dataset)
    test_length = len(test_dataset)
    return train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask

def save_model(args, model, epoch):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model_{}.bin.".format(epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def load_model(model, model_file=None):
    print("model_file", model_file)
    if os.path.exists(model_file):
        model.load_state_dict(torch.load(model_file))
    return model

def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0, label_input=None, label_mask=None): 
    global logger
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    total_pred,total_true_label, total_pred_scores, total_mus, total_stds, total_labels = [], [], [], [], [], []
    
    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        pairs_text, pairs_mask, video, video_mask,audio, audio_mask, ground_label, samples_index = batch
        
        model_loss, batch_pred, true_label, pred_scores, mean_similarity, (mus, stds, _) = model(pairs_text, pairs_mask, video, video_mask, audio, audio_mask, label_input, label_mask, groundTruth_labels=ground_label, training=True, samples_index=samples_index)
        if n_gpu > 1:
            model_loss = model_loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            model_loss = model_loss / args.gradient_accumulation_steps
        optimizer.zero_grad()
        model_loss.backward() 
        total_loss += float(model_loss)
        total_pred.append(batch_pred)
        total_true_label.append(true_label)
        total_pred_scores.append(pred_scores)
        total_mus.append(mus)
        total_stds.append(stds)
        total_labels.append(ground_label)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                print("Epoch: %d/%d, Step: %d/%d, Lr: %s, loss: %f,  Time/step: %f, similarity: %f %f" % (epoch + 1,
                      args.epochs, step + 1,
                      len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),float(model_loss), 
                      (time.time() - start_time) / (log_step * args.gradient_accumulation_steps), mean_similarity[0].cpu(), mean_similarity[1].cpu()))
                start_time = time.time()
    total_loss = total_loss / len(train_dataloader)
    total_pred=torch.cat(total_pred,0)
    total_true_label = torch.cat(total_true_label, 0)
    total_pred_scores = torch.cat(total_pred_scores, 0)
    total_mus = torch.cat(total_mus, 0)
    total_stds = torch.cat(total_stds, 0)
    total_labels = torch.cat(total_labels, 0)
    # torch.save(total_mus, os.path.join(args.output_dir, "nocdl_train_mus.pt"))
    # torch.save(total_stds, os.path.join(args.output_dir, "nocdl_train_stds.pt"))
    # torch.save(total_labels, os.path.join(args.output_dir, "nocdl_train_labels.pt"))
    return total_loss, total_pred, total_true_label, total_pred_scores, model

def eval_epoch(model, val_dataloader, device, label_input, label_mask):
    model.eval()
    with torch.no_grad():
        total_pred = []
        total_true_label = []
        total_pred_scores = []
        total_mus = []
        total_stds = []
        total_labels = []
        total_var = []
        for _, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, groundTruth_labels, samples_index = batch
            batch_pred, true_label, pred_scores, (mus, stds, common_var) = model(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, groundTruth_labels=groundTruth_labels, training=False, samples_index=samples_index)
            total_pred.append(batch_pred)
            total_true_label.append(true_label)
            total_pred_scores.append(pred_scores)
            total_mus.append(mus)
            total_stds.append(stds)
            total_labels.append(groundTruth_labels)
            total_var.append(common_var)
        total_pred=torch.cat(total_pred,0)
        total_true_label = torch.cat(total_true_label, 0)
        total_pred_scores = torch.cat(total_pred_scores, 0)
        total_mus = torch.cat(total_mus, 0)
        total_stds = torch.cat(total_stds, 0)
        total_labels = torch.cat(total_labels, 0)
        total_var = torch.cat(total_var, 0)
        # torch.save(total_mus, os.path.join(args.output_dir, "nocdl_val_mus.pt"))
        # torch.save(total_stds, os.path.join(args.output_dir, "nocdl_val_stds.pt"))
        # torch.save(total_labels, os.path.join(args.output_dir, "nocdl_val_labels.pt"))
        return  total_pred, total_true_label, total_pred_scores, total_var
           
def main(args):
    global logger
    device, n_gpu = init_device(args, args.local_rank) # 确认GPU是否存在, 和设备的关系。
    model = init_model(args, device, n_gpu, args.local_rank)    
    model = model.to(device)
    if args.aligned == False:
        logger.warning("!!!!!!!!!!!!!! you start train unaligned dataset")
    else:
        logger.warning("!!!!!!!!!!!!!! you start train aligned dataset")
    print('***** dataloder preping ... *****')
    if args.do_train:
        train_dataloader, val_dataloader, test_dataloader, train_length, val_length, test_length, label_input, label_mask = prep_dataloader(args)
        label_input = label_input.to(device)
        label_mask = label_mask.to(device)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs
        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)
        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        
        best_score,best_test_score, global_step = 0.000, 0.000, 0
        best_output_model_file, best_model = None, None
        early_stopping = EarlyStopping(patience=5, delta=0)

        for epoch in range(args.epochs):
            total_loss, total_pred, total_label, total_pred_scores, model= train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                                scheduler, global_step, local_rank=args.local_rank, label_input=label_input, label_mask=label_mask)
            total_micro_f1, total_micro_precision, total_micro_recall, total_acc = get_metrics(total_pred, total_label)
            total_pred_scores = total_pred_scores.data.cpu().numpy()
            total_label = total_label.data.cpu().numpy()
            train_gap = calculate_gap(total_pred_scores, total_label) 
            if args.local_rank == 0:
                logger.info("Epoch %d/%d Finished, Train Loss: %f, Train_micro_f1: %f, Train_micro_precision: %f, Train_micro_recall: %f,  Train_acc: %f, train_gap: %f",  \
                    epoch + 1, args.epochs, total_loss, total_micro_f1, total_micro_precision, total_micro_recall,  total_acc, train_gap)

            if args.local_rank == 0:
                logger.info('***** Running testing *****')
                logger.info('  Num examples = %d', test_length)
                logger.info("  Batch_size = %d", args.batch_size)
                test_pred, test_label, test_pred_scores, common_var = eval_epoch(model, test_dataloader, device, label_input, label_mask)
                # best_test_threshold = search_binary(test_pred_scores, test_label)
                best_test_threshold = args.threshold
                test_pred = getBinaryTensor(test_pred_scores, best_test_threshold)
                test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(test_pred, test_label)
                test_pred1, test_label1, _, common_var = eval_epoch(model, test_dataloader, device, label_input, label_mask)
                test_pred1 = getBinaryTensor(_, best_test_threshold)
                test_micro_f11, test_micro_precision1, test_micro_recall1, test_acc1 = get_metrics(test_pred1, test_label1)
                test_micro_f1 = max(test_micro_f1, test_micro_f11)
                test_micro_precision = max(test_micro_precision, test_micro_precision1)
                test_micro_recall = max(test_micro_recall, test_micro_recall1)
                test_acc = max(test_acc, test_acc1)
                test_pred_scores = test_pred_scores.data.cpu().numpy()
                test_label = test_label.data.cpu().numpy()
                test_gap = calculate_gap(test_pred_scores, test_label)
                logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f, test_gap: %f, threshold: %f", \
                        test_micro_f1, test_micro_precision, test_micro_recall, test_acc, test_gap, best_test_threshold)
                early_stopping(test_micro_f1)
                if best_test_score <=  test_micro_f1:
                    best_test_score = test_micro_f1
                    best_model = model
                    output_model_file = save_model(args, best_model, epoch)
                    if best_output_model_file is not None and os.path.exists(best_output_model_file):
                        os.remove(best_output_model_file)
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the f1 is: {:.4f}".format(best_output_model_file, best_test_score))
            if total_micro_f1 - test_micro_f1 > 0.06 and early_stopping.early_stop:
                break
    print("best_output_model_file", best_output_model_file)
    best_model = load_model(model, best_output_model_file)
    test_pred, test_label, test_pred_scores, common_var  = eval_epoch(best_model, test_dataloader, device, label_input, label_mask)
    # best_test_threshold = search_binary(test_pred_scores, test_label)
    test_pred = getBinaryTensor(test_pred_scores, args.threshold)
    # torch.save({
    #     'test_pred': test_pred,
    #     'test_label': test_label,
    #     'test_pred_scores': common_var
    # }, os.path.join(args.output_dir, 'test_results.pt'))
    test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(test_pred, test_label)
    logger.info("Final test results: f1 %f,\tp %f,\tr %f,\tacc %f",
                test_micro_f1, test_micro_precision, test_micro_recall, test_acc)

      
if __name__ == "__main__":
    args = get_args() # 获取日志
    args = set_seed_logger(args)    
    main(args)