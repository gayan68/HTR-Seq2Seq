import torch
import subprocess as sub
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
#import loadData3 as loadData
#import loadData2_latest as loadData
#import loadData
import numpy as np
import time
import os
#from LogMetric import Logger
import argparse
#from models.encoder_plus import Encoder
#from models.encoder import Encoder
#from models.encoder_bn_relu import Encoder
from models.encoder_vgg import Encoder
from models.decoder import Decoder
from models.attention import locationAttention as Attention
#from models.attention import TroAttention as Attention
from models.seq2seq import Seq2Seq
from utils import visualizeAttn, writePredict, writeLoss, HEIGHT, WIDTH, output_max_len, vocab_size, FLIP, WORD_LEVEL, load_data_func, tokens

from utils2.metrics import CER, WER
from utils2.stike_remove_models import DenseGenerator
import datasetConfig

import tqdm
import wandb


parser = argparse.ArgumentParser(description='seq2seq net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--start_epoch', default=0, type=int, help='load saved weights from which epoch')
parser.add_argument("--dataset", default="", type=str)
parser.add_argument("--run_id", default="", type=str)
args = parser.parse_args()

dataset = args.dataset
run_id = args.run_id
log_root = "../../logs/HTR-Seq2Seq"
log_dir = f"{log_root}/pred_logs_{run_id}"

print(dataset)
print(args.start_epoch)

#torch.cuda.set_device(1)

LABEL_SMOOTH = True

Bi_GRU = True
VISUALIZE_TRAIN = False

BATCH_SIZE = 32
learning_rate = 2 * 1e-4
#lr_milestone = [30, 50, 70, 90, 120]
#lr_milestone = [20, 40, 60, 80, 100]
#lr_milestone = [15, 25, 35, 45, 55, 65]
#lr_milestone = [30, 40, 50, 60, 70]
#lr_milestone = [30, 40, 60, 80, 100]
lr_milestone = [20, 40, 60, 80, 100]
#lr_milestone = [20, 40, 46, 60, 80, 100]

lr_gamma = 0.5

START_TEST = 1e4 # 1e4: never run test 0: run test from beginning
FREEZE = False
freeze_milestone = [65, 90]
EARLY_STOP_EPOCH = 20 # None: no early stopping
HIDDEN_SIZE_ENC = 512
HIDDEN_SIZE_DEC = 512 # model/encoder.py SUM_UP=False: enc:dec = 1:2  SUM_UP=True: enc:dec = 1:1
CON_STEP = None # CON_STEP = 4 # encoder output squeeze step
CurriculumModelID = args.start_epoch
#CurriculumModelID = -1 # < 0: do not use curriculumLearning, train from scratch
#CurriculumModelID = 170 # 'save_weights/seq2seq-170.model.backup'
EMBEDDING_SIZE = 60 # IAM
TRADEOFF_CONTEXT_EMBED = None # = 5 tradeoff between embedding:context vector = 1:5
TEACHER_FORCING = True
MODEL_SAVE_EPOCH = 1

baseDir = datasetConfig.baseDir_word
# dataset = datasetConfig.dataset
ignore_chars = datasetConfig.ignore_chars

class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

log_softmax = torch.nn.LogSoftmax(dim=-1)
crit = LabelSmoothing(vocab_size, tokens['PAD_TOKEN'], 0.4)
# predict and gt follow the same shape of cross_entropy
# predict: 704, 83   gt: 704
def loss_label_smoothing(predict, gt):
    def smoothlabel_torch(x, amount=0.25, variance=5):
        mu = amount/x.shape[0]
        sigma = mu/variance
        noise = np.random.normal(mu, sigma, x.shape).astype('float32')
        smoothed = x*torch.from_numpy(1-noise.sum(1)).view(-1, 1).cuda() + torch.from_numpy(noise).cuda()
        return smoothed

    def one_hot(src): # src: torch.cuda.LongTensor
        ones = torch.eye(vocab_size).cuda()
        return ones.index_select(0, src)

    gt_local = one_hot(gt.data)
    gt_local = smoothlabel_torch(gt_local)
    loss_f = torch.nn.BCEWithLogitsLoss()
    gt_local = Variable(gt_local)
    res_loss = loss_f(predict, gt_local)
    return res_loss

def teacher_force_func(epoch):
    if epoch < 50:
        teacher_rate = 0.5
    elif epoch < 150:
        teacher_rate = (50 - (epoch-50)//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate

def teacher_force_func_2(epoch):
    if epoch < 200:
        teacher_rate = (100 - epoch//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate


def all_data_loader():
    data_train = load_data_func(baseDir, dataset, "train", ignore_chars)
    data_valid = load_data_func(baseDir, dataset, "val", ignore_chars)
    data_test = load_data_func(baseDir, "CLEAN", "test", ignore_chars)
    train_loader = torch.utils.data.DataLoader(data_train, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, valid_loader, test_loader

def test_data_loader_batch(batch_size_nuevo):
    _, _, data_test = load_data_func(baseDir, dataset)
    test_loader = torch.utils.data.DataLoader(data_test, collate_fn=sort_batch, batch_size=batch_size_nuevo, shuffle=False, num_workers=2, pin_memory=True)
    return test_loader

def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_in_len = []
    train_out = []
    for i in range(n_batch):
        idx, img, img_width, label = batch[i]
        train_index.append(idx)
        train_in.append(img)
        train_in_len.append(img_width)
        train_out.append(label)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)

    train_in_len, idx = train_in_len.sort(0, descending=True)
    train_in = train_in[idx]
    train_out = train_out[idx]
    train_index = train_index[idx]
    return train_index, train_in, train_in_len, train_out

def train(train_loader, seq2seq, opt, teacher_rate, epoch, log_dir):
    seq2seq.train()
    total_loss = 0
    t = tqdm.tqdm(train_loader)
    t.set_description('Epoch {}'.format(epoch+1))
    for num, (train_index, train_in, train_in_len, train_out) in enumerate(t):
        #train_in = train_in.unsqueeze(1)
        train_in, train_out = Variable(train_in).cuda(), Variable(train_out).cuda()
        output, attn_weights = seq2seq(train_in, train_out, train_in_len, teacher_rate=teacher_rate, train=True) # (100-1, 32, 62+1)
        batch_count_n = writePredict(log_dir, epoch, train_index, output, 'train')
        train_label = train_out.permute(1, 0)[1:].contiguous().view(-1)#remove<GO>
        output_l = output.view(-1, vocab_size) # remove last <EOS>

        if VISUALIZE_TRAIN:
            if 'e02-074-03-00,191' in train_index:
                b = train_index.tolist().index('e02-074-03-00,191')
                visualizeAttn(train_in.data[b,0], train_in_len[0], [j[b] for j in attn_weights], epoch, batch_count_n[b], 'train_e02-074-03-00')

        #loss = F.cross_entropy(output_l.view(-1, vocab_size),
        #                       train_label, ignore_index=tokens['PAD_TOKEN'])
        #loss = loss_label_smoothing(output_l.view(-1, vocab_size), train_label)
        if LABEL_SMOOTH:
            loss = crit(log_softmax(output_l.view(-1, vocab_size)), train_label)
        else:
            loss = F.cross_entropy(output_l.view(-1, vocab_size),
                               train_label, ignore_index=tokens['PAD_TOKEN'])
        t.set_postfix(values='train_loss: {:.2f}'.format(loss.item()))
                        
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()

    total_loss /= (num+1)
    return total_loss

def valid(valid_loader, seq2seq, epoch, log_dir):
    seq2seq.eval()
    total_loss_t = 0
    with torch.no_grad():
        t = tqdm.tqdm(valid_loader)
        t.set_description('Val Epoch {}'.format(epoch+1))
        for num, (test_index, test_in, test_in_len, test_out) in enumerate(t):
            #test_in = test_in.unsqueeze(1)
            #test_in, test_out = Variable(test_in, volatile=True).cuda(), Variable(test_out, volatile=True).cuda()
            test_in, test_out = Variable(test_in).cuda(), Variable(test_out).cuda()
            output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)
            batch_count_n = writePredict(log_dir, epoch, test_index, output_t, 'valid')
            test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)

            # if "p03-185-08-06" in test_index:
            #     print(f"IMAGE ID {test_index}")
            #     print(output_t)
            #     print(f"batch_count_n: {batch_count_n}")
            #loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
            #                         test_label, ignore_index=tokens['PAD_TOKEN'])
            #loss_t = loss_label_smoothing(output_t.view(-1, vocab_size), test_label)
            if LABEL_SMOOTH:
                loss_t = crit(log_softmax(output_t.view(-1, vocab_size)), test_label)
            else:
                loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                    test_label, ignore_index=tokens['PAD_TOKEN'])

            total_loss_t += loss_t.item()

            if 'n04-015-00-01,171' in test_index:
                b = test_index.tolist().index('n04-015-00-01,171')
                #visualizeAttn(test_in.data[b,0], test_in_len[0], [j[b] for j in attn_weights_t], epoch, batch_count_n[b], 'valid_n04-015-00-01')
    total_loss_t /= (num+1)
    return total_loss_t

# def loadStrikeRemovalModel(blockCount=1, model_name="epoch_30.pth"):
#     strremoveNet = DenseGenerator(1, 1, n_blocks=blockCount)
#     state_dict = torch.load(f"strike_removal_model/{model_name}")
#     if "model_state_dict" in state_dict.keys():
#         state_dict = state_dict['model_state_dict']
#     strremoveNet.load_state_dict(state_dict)
#     return strremoveNet.cuda()

def test2(gt, epoch_pred):
    cer = CER()

    decoded_list_img_id, decoded_list_transcr = [], []
    with open(epoch_pred, 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            img_id, transcr = l[0].split(',')[0], ' '.join(l[1:])
            decoded_list_img_id.append(img_id)
            decoded_list_transcr.append(transcr)

    with open(gt, 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            img_id, _, transcr = l[0].split(',')[0],l[0].split(',')[1], ' '.join(l[1:])
            if (len(ignore_chars) == 0) or (transcr not in ignore_chars):
                if img_id in decoded_list_img_id:
                    idx = decoded_list_img_id.index(img_id)
                    cer.update(transcr, decoded_list_transcr[idx])
                else:
                    print(f"No predictions for : {img_id}")
            
    return cer.score()

def test(test_loader, modelID, log_dir, showAttn=True, StrikeRemove=False):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    model_file = f"{log_root}/save_weights_{run_id}/seq2seq-{modelID}.model"
    print('Loading ' + model_file)
    seq2seq.load_state_dict(torch.load(model_file)) #load

    seq2seq.eval()
    total_loss_t = 0
    start_t = time.time()
    print("Test P")
    with torch.no_grad():
        t = tqdm.tqdm(test_loader)
        for num, (test_index, test_in, test_in_len, test_out) in enumerate(t):
            #test_in = test_in.unsqueeze(1)
            test_in, test_out = Variable(test_in).cuda(), Variable(test_out).cuda()
            # if StrikeRemove :
            #     print("##############3StrikeRemove")
            #     strremoveNet = loadStrikeRemovalModel()
            #     test_in[:,0:1,:,:] = strremoveNet(test_in[:,0:1,:,:-1])[:,:,:,:-1]
            #     test_in[:,1:2,:,:] = strremoveNet(test_in[:,1:2,:,:-1])[:,:,:,:-1]
            #     test_in[:,2:3,:,:] = strremoveNet(test_in[:,2:3,:,:-1])[:,:,:,:-1]

            output_t, attn_weights_t = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)
            batch_count_n = writePredict(log_dir, modelID, test_index, output_t, 'test')
            test_label = test_out.permute(1, 0)[1:].contiguous().view(-1)
            #loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
            #                        test_label, ignore_index=tokens['PAD_TOKEN'])
            #loss_t = loss_label_smoothing(output_t.view(-1, vocab_size), test_label)
            if LABEL_SMOOTH:
                loss_t = crit(log_softmax(output_t.view(-1, vocab_size)), test_label)
            else:
                loss_t = F.cross_entropy(output_t.view(-1, vocab_size),
                                    test_label, ignore_index=tokens['PAD_TOKEN'])

            total_loss_t += loss_t.item()

            if showAttn:
                global_index_t = 0
                for t_idx, t_in in zip(test_index, test_in):
                    #visualizeAttn(t_in.data[0], test_in_len[0], [j[global_index_t] for j in attn_weights_t], modelID, batch_count_n[global_index_t], 'test_'+t_idx.split(',')[0])
                    global_index_t += 1

    total_loss_t /= (num+1)
    writeLoss(log_dir, total_loss_t, 'test')
    print('    TEST loss=%.3f, time=%.3f' % (total_loss_t, time.time()-start_t))

def main(train_loader, valid_loader, test_loader, log_dir):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()    
    if CurriculumModelID > 0:
        model_file = f"{log_root}/save_weights_{run_id}/seq2seq-{CurriculumModelID}.model"
        #model_file = 'save_weights/words/seq2seq-' + str(CurriculumModelID) +'.model'
        print('Loading ' + model_file)
        seq2seq.load_state_dict(torch.load(model_file)) #load
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    #opt = optim.SGD(seq2seq.parameters(), lr=learning_rate, momentum=0.9)
    #opt = optim.RMSprop(seq2seq.parameters(), lr=learning_rate, momentum=0.9)

    #scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=1)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)
    epochs = 5000000
    if EARLY_STOP_EPOCH is not None:
        min_loss = 1e3
        min_loss_index = 0
        min_loss_count = 0

    if CurriculumModelID > 0 and WORD_LEVEL:
        start_epoch = CurriculumModelID + 1
        for i in range(start_epoch):
            scheduler.step()
    else:
        start_epoch = 0

    for epoch in range(start_epoch, epochs):
        scheduler.step()
        lr = scheduler.get_last_lr()[0] #get_lr()[0]
        teacher_rate = teacher_force_func(epoch) if TEACHER_FORCING else False
        start = time.time()
        loss = train(train_loader, seq2seq, opt, teacher_rate, epoch, log_dir)
        writeLoss(log_dir, loss, 'train')
        print('epoch %d/%d, loss=%.3f, lr=%.8f, teacher_rate=%.3f, time=%.3f' % (epoch, epochs, loss, lr, teacher_rate, time.time()-start))

        if epoch%MODEL_SAVE_EPOCH == 0:
            folder_weights = f'save_weights_{run_id}'
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            # torch.save(seq2seq.state_dict(), folder_weights+'/seq2seq-%d.model'%epoch)

        start_v = time.time()
        loss_v = valid(valid_loader, seq2seq, epoch, log_dir)
        writeLoss(log_dir, loss_v, 'valid')
        print('  Valid loss=%.3f, time=%.3f' % (loss_v, time.time()-start_v))

        gt = f"{baseDir}val/images/{dataset}/gt_RWTH.txt"
        decoded = f'{log_dir}/valid_predict_seq.{epoch}.log'
        cer_v = test2(gt, decoded)      
        print('  CER on ValidationSet=%.3f' % (cer_v))  
        #Upload logs to Wandb
        if wandb_log:
            wandb.log({"Train_loss": loss, "Valid_loss": loss_v, "current_lr": lr, "teacher_rate": teacher_rate, "CER": cer_v})
        
        if EARLY_STOP_EPOCH is not None:
            # gt = 'RWTH_partition/gt_val_no_threshold.txt'
            # decoded = 'pred_logs/valid_predict_seq.'+str(epoch)+'.log'
            # res_cer = sub.Popen(['./tasas_cer.sh', gt, decoded], stdout=sub.PIPE)
            # res_cer = res_cer.stdout.read().decode('utf8')
            # loss_v = float(res_cer)/100
            if cer_v < min_loss:
                min_loss = cer_v
                min_loss_index = epoch
                min_loss_count = 0
                torch.save(seq2seq.state_dict(), f"{log_root}/{folder_weights}/seq2seq-best.model")
                wandb.log({"Current best at": epoch})
            else:
                min_loss_count += 1
            if min_loss_count >= EARLY_STOP_EPOCH:
                print('Early Stopping at: %d. Best epoch is: %d' % (epoch, min_loss_index))
                wandb.log({"Early Stopping at": min_loss_index})
                return min_loss_index



if __name__ == '__main__':

    wandb_log = True  #Set wandb web log

    # ----------------------- initialize wandb ------------------------------- #
    if wandb_log:
        wandb.init(
            # set the wandb project where this run will be logged
            project="S2S_GRU",

            # track hyperparameters and run metadata
            config={
            "learning_rate": learning_rate,
            "architecture": "S2S+GRU",
            "dataset": "IAM",
            }
        )
        print("run name WANDB : " + str(wandb.run.name))
    

    print(time.ctime())
    train_loader, valid_loader, test_loader = all_data_loader()
    mejorModelID = main(train_loader, valid_loader, test_loader, log_dir)
    test(test_loader, mejorModelID, log_dir, True)
    os.system('./test.sh '+str(mejorModelID))
    print(time.ctime())
