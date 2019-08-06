from __future__ import print_function
import torch
import sys
sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric')
print(sys.path)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import *
from datasets.dataset_read import dataset_read
import numpy as np


class Solver(object):
    def __init__(self, args, batch_size=64, source='svhn',
                 target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.src_domain_code = np.repeat(np.array([[*([1]), *([0])]]), batch_size, axis=0)
        self.tgt_domain_code = np.repeat(np.array([[*([0]), *([1])]]), batch_size, axis=0)
        self.src_domain_code = Variable(torch.FloatTensor(self.src_domain_code).cuda(), requires_grad=False)
        self.tgt_domain_code = Variable(torch.FloatTensor(self.tgt_domain_code).cuda(), requires_grad=False)
        self.batch_size = batch_size

        self.source = source
        self.target = target
        self.num_k = num_k
        self.mi_k  = 1
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.belta = 0.01
        self.mi_para = 0.0001
        if self.source == 'svhn':
            self.scale = False
        else:
            self.scale = False
        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size, scale=self.scale,
                                                        all_use=self.all_use)
        print('load finished!')
        self.G = Generator(source=source, target=target)
        self.D0 = Disentangler()
        self.D1 = Disentangler()
        self.D2 = Disentangler()

        self.C0 = Classifier(source=source, target=target)
        self.C1 = Classifier(source=source, target=target)
        self.C2 = Classifier(source=source, target=target)
        self.FD = Feature_Discriminator()
        self.R = Reconstructor()
        # Mutual information network estimation
        self.M = Mine()

        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C0.cuda()
        self.C1.cuda()
        self.C2.cuda()

        self.D0.cuda()
        self.D1.cuda()
        self.D2.cuda()
        self.FD.cuda()
        self.R.cuda()
        self.M.cuda()

        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='adam', lr=0.001, momentum=0.9):
        self.opt_g = optim.Adam(self.G.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_c0 = optim.Adam(self.C0.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_c1 = optim.Adam(self.C1.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_c2 = optim.Adam(self.C2.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_d0 = optim.Adam(self.D0.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_d1 = optim.Adam(self.D1.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_d2 = optim.Adam(self.D2.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_fd = optim.Adam(self.FD.parameters(),lr=lr, weight_decay=0.0005)
        self.opt_r  = optim.Adam(self.R.parameters(), lr=lr, weight_decay=0.0005)
        self.opt_mi = optim.Adam(self.M.parameters(), lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c0.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()
        self.opt_d0.zero_grad()
        self.opt_d1.zero_grad()
        self.opt_d2.zero_grad()
        self.opt_fd.zero_grad()
        self.opt_r.zero_grad()
        self.opt_mi.zero_grad()

    def ent(self, output):
        return - torch.mean(torch.log(F.softmax(output + 1e-6)))

    def discrepancy(self, out1, out2):

        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def mutual_information_estimator(self, x, y, y_):
        joint, marginal = self.M(x, y), self.M(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def reconstruct_loss(self,src,tgt):
        return torch.sum((src-tgt)**2) / (src.shape[0]*src.shape[1])

    def group_step(self, step_list):
        for i in range(len(step_list)):
            step_list[i].step()
        self.reset_grad()

    def ring_loss(self, feat, type='geman'):
        x = feat.pow(2).sum(dim=1).pow(0.5)
        radius = x.mean()
        radius = radius.expand_as(x)
        # print(radius)
        if type=='geman':
            l2_loss = (x-radius).pow(2).sum(dim=0) / (x.shape[0]*0.5)
            return l2_loss

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        adv_loss = nn.BCEWithLogitsLoss().cuda()
        self.G.train()
        self.D0.train()
        self.D1.train()
        self.D2.train()
        self.C0.train()
        self.C1.train()
        self.C2.train()
        self.FD.train()
        self.R.train()
        self.M.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T']
            img_s = data['S']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            imgs = Variable(torch.cat((img_s,\
                                       img_t), 0))
            label_s = Variable(label_s.long().cuda())

            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()

            # first step: train three disentangler
            feat_s = self.G(img_s)
            loss_s0 = criterion(self.C0(self.D0(feat_s)), label_s)
            loss_s1 = criterion(self.C1(self.D1(feat_s)), label_s)
            loss_s2 = criterion(self.C2(self.D2(feat_s)), label_s)
            loss_s = loss_s0 + loss_s1 + loss_s2

            loss_s.backward()
            self.group_step([self.opt_g, self.opt_c0, self.opt_c1, self.opt_c2, self.opt_d0,self.opt_d1,self.opt_d2])


            loss_s0 = criterion(self.C0(self.D0(self.G(img_s))),label_s)
            loss_s1 = criterion(self.C1(self.D1(self.G(img_s))),label_s)
            loss_dis = self.discrepancy(self.C0(self.D0(self.G(img_t))),self.C1(self.D1(self.G(img_t))))
            loss = loss_s0 + loss_s1 - loss_dis
            loss.backward()
            self.group_step([self.opt_d0,self.opt_d1,self.opt_c0,self.opt_c1])

            #ring loss
            data = torch.cat((img_s, img_t), 0)
            feat = self.G(data)
            ring_loss = self.ring_loss(feat)
            ring_loss.backward()
            self.group_step([self.opt_g])




            # minimize mutual information between (d0, d2) and (d1, d2)
            for i in range(0, self.mi_k):
                output_d0_s, output_d0_t = self.D0(self.G(img_s)), self.D0(self.G(img_t))
                output_d1_s, output_d1_t = self.D1(self.G(img_s)), self.D1(self.G(img_t))
                output_d2_s, output_d2_t = self.D2(self.G(img_s)), self.D2(self.G(img_t))
                output_d2_s_shuffle = torch.index_select(output_d2_s,0,Variable(torch.randperm(output_d2_s.shape[0]).cuda()))
                output_d2_t_shuffle = torch.index_select(output_d2_t, 0, Variable(torch.randperm(output_d2_t.shape[0]).cuda()))

                MI_d0d2_s = self.mutual_information_estimator(output_d0_s, output_d2_s, output_d2_s_shuffle)
                MI_d0d2_t = self.mutual_information_estimator(output_d0_t, output_d2_t, output_d2_t_shuffle)
                MI_d1d2_s = self.mutual_information_estimator(output_d1_s, output_d2_s, output_d2_s_shuffle)
                MI_d1d2_t = self.mutual_information_estimator(output_d1_t, output_d2_t, output_d2_t_shuffle)
                MI = 0.25*(MI_d0d2_s + MI_d0d2_t + MI_d1d2_s + MI_d1d2_t) * self.mi_para
                MI.backward()
                self.group_step([self.opt_d0, self.opt_d1, self.opt_d2, self.opt_mi])
            # pred_d1d2_s = self.M(output_d1_s, output_d2_s)

            # adversarial training
            entropy_s2 = self.ent(self.C2(self.D2(self.G(img_s))))
            entropy_t2 = self.ent(self.C2(self.D2(self.G(img_t))))
            loss = entropy_s2+entropy_t2
            loss.backward()
            self.group_step([self.opt_d2, self.opt_g])



            #adversarial alignment
            src_domain_pred = self.FD(self.D1(self.G(img_s)))
            tgt_domain_pred = self.FD(self.D1(self.G(img_t)))

            df_loss_src = adv_loss(src_domain_pred, self.src_domain_code)
            df_loss_tgt = adv_loss(tgt_domain_pred, self.tgt_domain_code)
            loss = 0.01 * (df_loss_src + df_loss_tgt)
            loss.backward()
            self.group_step([self.opt_fd, self.opt_d1, self.opt_g])

            src_domain_pred = self.FD(self.D1(self.G(img_s)))
            tgt_domain_pred = self.FD(self.D1(self.G(img_t)))

            df_loss_src = adv_loss(src_domain_pred, 1-self.src_domain_code)
            df_loss_tgt = adv_loss(tgt_domain_pred, 1-self.tgt_domain_code)
            loss = 0.01 * (df_loss_src + df_loss_tgt)

            loss.backward()
            self.group_step([self.opt_fd, self.opt_d1, self.opt_g])

            for i in range(self.num_k):
                loss_dis = self.discrepancy(self.C0(self.D0(self.G(img_t))), self.C1(self.D1(self.G(img_t))))
                loss_dis.backward()
                self.group_step([self.opt_g])

            # reconstruction component
            feat_s = self.G(img_s)
            feat_t = self.G(img_t)

            d0_feat_s, d1_feat_s, d2_feat_s = self.D0(feat_s), self.D1(feat_s), self.D2(feat_s)
            d0_feat_t, d1_feat_t, d2_feat_t = self.D0(feat_t), self.D1(feat_t), self.D2(feat_t)
            d0_d2_s =torch.cat((d0_feat_s, d2_feat_s),1)
            d1_d2_s =torch.cat((d1_feat_s, d2_feat_s),1)
            d0_d2_t =torch.cat((d0_feat_t, d2_feat_t),1)
            d1_d2_t =torch.cat((d1_feat_t, d2_feat_t),1)

            recon_d0_d2_s, recon_d1_d2_s = self.R(d0_d2_s), self.R(d1_d2_s)
            recon_d0_d2_t, recon_d1_d2_t = self.R(d0_d2_t), self.R(d1_d2_t)

            recon_loss = self.reconstruct_loss(feat_s, recon_d0_d2_s)
            recon_loss += self.reconstruct_loss(feat_s, recon_d1_d2_s)
            recon_loss += self.reconstruct_loss(feat_t, recon_d0_d2_t )
            recon_loss += self.reconstruct_loss(feat_t, recon_d1_d2_t )
            recon_loss = (recon_loss/4)*self.belta
            recon_loss.backward()
            self.group_step([self.opt_d1, self.opt_d2, self.opt_d0, self.opt_r])

            if batch_idx > 500000:
                return batch_idx

            if batch_idx % self.interval == 0:
                loss_dis = loss_s1
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.data[0], loss_s2.data[0], loss_dis.data[0]))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.data[0], loss_s1.data[0], loss_s2.data[0]))
                    record.close()
        return batch_idx


    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.D1.eval()
        self.D0.eval()
        self.C1.eval()
        self.C0.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        for batch_idx, data in enumerate(self.dataset_test):
            img = data['T']
            label = data['T_label']
            img, label = img.cuda(), label.long().cuda()
            img, label = Variable(img, volatile=True), Variable(label)
            feat = self.G(img)
            output1 = self.C1(self.D1(feat))
            output2 = self.C0(self.D0(feat))
            test_loss += F.nll_loss(output1, label).data[0]
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
            record = open('conf_{}.txt'.format(epoch),'a')
            for tmp_index in range(0,len(pred1)):
                record.write('%d %d\n'%(label.data[tmp_index], pred1[tmp_index]))

            record.close()
        test_loss = test_loss / size
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                test_loss, correct1, size,
                100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
            record.close()
