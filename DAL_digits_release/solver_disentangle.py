from __future__ import print_function
import torch
import sys
import os
sys.path.append('./model')
sys.path.append('./datasets')
sys.path.append('./metric')
print(sys.path)

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.build_gen import Disentangler, Generator, Classifier, Feature_Discriminator, Reconstructor, Mine
from datasets.dataset_read import dataset_read
from utils.utils import _l2_rec, _ent, _discrepancy, _ring

from torch.utils.tensorboard import SummaryWriter
from time import gmtime, strftime
from tqdm import tqdm

class Solver():
    def __init__(self, args, batch_size=64, source='svhn',
                 target='mnist', learning_rate=0.0002, interval=1,
                 optimizer='adam', num_k=4, all_use=False,
                 checkpoint_dir=None, save_epoch=10):

        timestring = strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_%s" % args.exp_name
        self.logdir = os.path.join('./logs', timestring)
        self.logger = SummaryWriter(log_dir=self.logdir)
        self.device = torch.device("cuda" if args.use_cuda else "cpu")

        self.src_domain_code = np.repeat(
            np.array([[*([1]), *([0])]]), batch_size, axis=0)
        self.trg_domain_code = np.repeat(
            np.array([[*([0]), *([1])]]), batch_size, axis=0)
        self.src_domain_code = torch.FloatTensor(
            self.src_domain_code).to(self.device)
        self.trg_domain_code = torch.FloatTensor(
            self.trg_domain_code).to(self.device)

        self.source = source
        self.target = target
        self.num_k = num_k
        self.mi_k = 1
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        self.delta = 0.01
        self.mi_coeff = 0.0001
        self.interval = interval
        self.batch_size = batch_size
        self.lr = learning_rate
        self.scale = False

        print('dataset loading')
        self.datasets, self.dataset_test = dataset_read(
            source, target, self.batch_size, scale=self.scale, all_use=self.all_use)
        print('load finished!')

        self.G = Generator(source=source, target=target)
        self.FD = Feature_Discriminator()
        self.R = Reconstructor()
        self.MI = Mine()

        self.C = nn.ModuleDict({
            'ds': Classifier(source=source, target=target),
            'di': Classifier(source=source, target=target),
            'ci': Classifier(source=source, target=target)
        })

        self.D = nn.ModuleDict({
            'ds': Disentangler(), 'di': Disentangler(), 'ci': Disentangler()})

        # All modules in the same dict
        self.modules = nn.ModuleDict({
            'G': self.G, 'FD': self.FD, 'R': self.R, 'MI': self.MI
        })

        if args.eval_only:
            self.G.torch.load('%s/%s_to_%s_model_epoch%s_G.pt' % (
                self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load('%s/%s_to_%s_model_epoch%s_G.pt' % (
                self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load('%s/%s_to_%s_model_epoch%s_G.pt' % (
                self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.xent_loss = nn.CrossEntropyLoss().cuda()
        self.adv_loss = nn.BCEWithLogitsLoss().cuda()
        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.to_device()

    def to_device(self):
        for k, v in self.modules.items():
            self.modules[k] = v.cuda()

        for k, v in self.C.items():
            self.C[k] = v.cuda()

        for k, v in self.D.items():
            self.D[k] = v.cuda()

    def set_optimizer(self, which_opt='adam', lr=0.001, momentum=0.9):
        self.opt = {
            'C_ds': optim.Adam(self.C['ds'].parameters(), lr=lr, weight_decay=5e-4),
            'C_di': optim.Adam(self.C['di'].parameters(), lr=lr, weight_decay=5e-4),
            'C_ci': optim.Adam(self.C['ci'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ds': optim.Adam(self.D['ds'].parameters(), lr=lr, weight_decay=5e-4),
            'D_di': optim.Adam(self.D['di'].parameters(), lr=lr, weight_decay=5e-4),
            'D_ci': optim.Adam(self.D['ci'].parameters(), lr=lr, weight_decay=5e-4),
            'G': optim.Adam(self.G.parameters(), lr=lr, weight_decay=5e-4),
            'FD': optim.Adam(self.FD.parameters(), lr=lr, weight_decay=5e-4),
            'R': optim.Adam(self.R.parameters(), lr=lr, weight_decay=5e-4),
            'MI': optim.Adam(self.MI.parameters(), lr=lr, weight_decay=5e-4),
        }

    def reset_grad(self):
        for _, opt in self.opt.items():
            opt.zero_grad()

    def mi_estimator(self, x, y, y_):
        joint, marginal = self.MI(x, y), self.MI(x, y_)
        return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)))

    def group_opt_step(self, opt_keys):
        for k in opt_keys:
            self.opt[k].step()
        self.reset_grad()

    def optimize_classifier(self, img_src, label_src):
        feat_src = self.G(img_src)
        _loss = dict()
        for key in ['ds', 'di', 'ci']:
            _loss['class_src_' + key] = self.xent_loss(
                self.C[key](self.D[key](feat_src)), label_src)

        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()
        self.group_opt_step(['G', 'C_ds', 'C_di', 'C_ci', 'D_ds', 'D_di', 'D_ci'])
        return _loss

    def discrepancy_minimizer(self, img_src, img_trg, label_src):
        # ================================== #
        # NOTE: I'm still not sure why we need this

        _loss = dict()
        # on source domain
        _loss['ds_src'] = self.xent_loss(
            self.C['ds'](self.D['ds'](self.G(img_src))), label_src)
        _loss['di_src'] = self.xent_loss(
            self.C['di'](self.D['di'](self.G(img_src))), label_src)

        # on target domain
        _loss['discrepancy_ds_di_trg'] = _discrepancy(
            self.C['ds'](self.D['ds'](self.G(img_trg))),
            self.C['di'](self.D['di'](self.G(img_trg))))

        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()
        self.group_opt_step(['D_ds', 'D_di', 'C_ds', 'C_di'])
        return _loss

    def ring_loss_minimizer(self, img_src, img_trg):
        data = torch.cat((img_src, img_trg), 0)
        feat = self.G(data)
        ring_loss = _ring(feat)
        ring_loss.backward()
        self.group_opt_step(['G'])
        return ring_loss

    def mutual_information_minimizer(self, img_src, img_trg):
        # minimize mutual information between (ds, ci) and (di, ci)
        for i in range(0, self.mi_k):
            ds_src, ds_trg = self.D['ds'](self.G(img_src)), self.D['ds'](self.G(img_trg))
            di_src, di_trg = self.D['di'](self.G(img_src)), self.D['di'](self.G(img_trg))
            ci_src, ci_trg = self.D['ci'](self.G(img_src)), self.D['ci'](self.G(img_trg))

            ci_src_shuffle = torch.index_select(
                ci_src, 0, torch.randperm(ci_src.shape[0]).to(self.device))
            ci_trg_shuffle = torch.index_select(
                ci_trg, 0, torch.randperm(ci_trg.shape[0]).to(self.device))

            MI_ds_ci_src = self.mi_estimator(ds_src, ci_src, ci_src_shuffle)
            MI_ds_ci_trg = self.mi_estimator(ds_trg, ci_trg, ci_trg_shuffle)
            MI_di_ci_src = self.mi_estimator(di_src, ci_src, ci_src_shuffle)
            MI_di_ci_trg = self.mi_estimator(di_trg, ci_trg, ci_trg_shuffle)

            MI = 0.25 * (MI_ds_ci_src + MI_ds_ci_trg + MI_di_ci_src + MI_di_ci_trg) * self.mi_coeff
            MI.backward()
            self.group_opt_step(['D_ds', 'D_di', 'D_ci', 'MI'])
        # pred_di_ci_src = self.M(out_di_src, out_ci_src)

    def class_confusion(self, img_src, img_trg):
        # - adversarial training

        # f_ci = CI(G(im)) extracts features that are class irrelevant
        # by maximizing the entropy, given that the classifier is fixed
        _loss = dict()
        _loss['src_ci'] = _ent(self.C['ci'](self.D['ci'](self.G(img_src))))
        _loss['trg_ci'] = _ent(self.C['ci'](self.D['ci'](self.G(img_trg))))
        _sum_loss = sum([l for _, l in _loss.items()])
        _sum_loss.backward()
        self.group_opt_step(['D_ci', 'G'])
        return _loss

    def adversarial_alignment(self, img_src, img_trg):

        # FD should guess if the features extracted f_di = DI(G(im))
        # are from target or source domain. To win this game and fool FD,
        # DI should extract domain invariant features.

        # Loss measures features' ability to fool the discriminator
        src_domain_pred = self.FD(self.D['di'](self.G(img_src)))
        tgt_domain_pred = self.FD(self.D['di'](self.G(img_trg)))
        df_loss_src = self.adv_loss(src_domain_pred, self.src_domain_code)
        df_loss_trg = self.adv_loss(tgt_domain_pred, self.trg_domain_code)
        alignment_loss1 = 0.01 * (df_loss_src + df_loss_trg)
        alignment_loss1.backward()
        self.group_opt_step(['FD', 'D_di', 'G'])

        # Measure discriminator's ability to classify source from target samples
        src_domain_pred = self.FD(self.D['di'](self.G(img_src)))
        tgt_domain_pred = self.FD(self.D['di'](self.G(img_trg)))
        df_loss_src = self.adv_loss(src_domain_pred, 1 - self.src_domain_code)
        df_loss_trg = self.adv_loss(tgt_domain_pred, 1 - self.trg_domain_code)
        alignment_loss2 = 0.01 * (df_loss_src + df_loss_trg)
        alignment_loss2.backward()
        self.group_opt_step(['FD', 'D_di', 'G'])

        for _ in range(self.num_k):
            loss_dis = _discrepancy(
                self.C['ds'](self.D['ds'](self.G(img_trg))),
                self.C['di'](self.D['di'](self.G(img_trg))))
            loss_dis.backward()
            self.group_opt_step(['G'])
        return alignment_loss1, alignment_loss2, loss_dis

    def optimize_rec(self, img_src, img_trg):
        _feat_src = self.G(img_src)
        _feat_trg = self.G(img_trg)

        feat_src, feat_trg = dict(), dict()
        rec_src, rec_trg = dict(), dict()
        for k in ['ds', 'di', 'ci']:
            feat_src[k] = self.D[k](_feat_src)
            feat_trg[k] = self.D[k](_feat_trg)

        recon_loss = None
        rec_loss_src, rec_loss_trg = dict(), dict()
        for k1, k2 in [('ds', 'ci'), ('di', 'ci')]:
            k = '%s_%s' % (k1, k2)
            rec_src[k] = self.R(torch.cat([feat_src[k1], feat_src[k2]], 1))
            rec_trg[k] = self.R(torch.cat([feat_trg[k1], feat_trg[k2]], 1))
            rec_loss_src[k] = _l2_rec(rec_src[k], _feat_src)
            rec_loss_trg[k] = _l2_rec(rec_trg[k], _feat_trg)

            if recon_loss is None:
                recon_loss = rec_loss_src[k] + rec_loss_trg[k]
            else:
                recon_loss += rec_loss_src[k] + rec_loss_trg[k]

        recon_loss = (recon_loss / 4) * self.delta
        recon_loss.backward()
        self.group_opt_step(['D_di', 'D_ci', 'D_ds', 'R'])
        return rec_loss_src, rec_loss_trg

    def train_epoch(self, epoch, record_file=None):
        # set training
        for k in self.modules.keys():
            self.modules[k].train()
        for k in self.C.keys():
            self.C[k].train()
        for k in self.D.keys():
            self.D[k].train()

        # torch.cuda.manual_seed(1)
        total_batches = 500000
        pbar_descr_prefix = "Epoch %d" % (epoch)
        with tqdm(total=total_batches, ncols=80, dynamic_ncols=False,
                  desc=pbar_descr_prefix) as pbar:
            for batch_idx, data in enumerate(self.datasets):
                if batch_idx > total_batches:
                    return batch_idx

                img_trg = data['T'].to(self.device)
                img_src = data['S'].to(self.device)
                label_src = data['S_label'].long().to(self.device)

                if img_src.size()[0] < self.batch_size or img_trg.size()[0] < self.batch_size:
                    break

                self.reset_grad()
                # ================================== #
                class_loss = self.optimize_classifier(img_src, label_src)
                ring_loss = self.ring_loss_minimizer(img_src, img_trg)
                self.mutual_information_minimizer(img_src, img_trg)
                confusion_loss = self.class_confusion(img_src, img_trg)
                (alignment_loss1, alignment_loss2, discrepancy_loss) = self.adversarial_alignment(
                    img_src, img_trg)
                rec_loss_src, rec_loss_trg = self.optimize_rec(img_src, img_trg)
                # ================================== #

                if batch_idx % self.interval == 0:
                    # ================================== #
                    for key, val in class_loss.items():
                        self.logger.add_scalar(
                            "class_loss/%s" % key, val,
                            global_step=batch_idx)

                    for key, val in confusion_loss.items():
                        self.logger.add_scalar(
                            "confusion_loss/%s" % key, val,
                            global_step=batch_idx)

                    for key, val in rec_loss_src.items():
                        self.logger.add_scalar(
                            "rec_loss_src/%s" % key, val,
                            global_step=batch_idx)

                    for key, val in rec_loss_trg.items():
                        self.logger.add_scalar(
                            "rec_loss_trg/%s" % key, val,
                            global_step=batch_idx)

                    self.logger.add_scalar(
                        "extra_loss/alignment_loss1", alignment_loss1,
                        global_step=batch_idx)

                    self.logger.add_scalar(
                        "extra_loss/alignment_loss2", alignment_loss2,
                        global_step=batch_idx)

                    self.logger.add_scalar(
                        "extra_loss/discrepancy_ds_di_trg", discrepancy_loss,
                        global_step=batch_idx)

                    self.logger.add_scalar(
                        "extra_loss/ring", ring_loss,
                        global_step=batch_idx)
                    # ================================== #
                pbar.update()
        return batch_idx


    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.D['di'].eval()
        self.D['ds'].eval()
        self.C['di'].eval()
        self.C['ds'].eval()
        test_loss = 0
        size = 0
        correct1, correct2, correct3 = 0, 0, 0
        with torch.no_grad():
            for batch_idx, data in enumerate(self.dataset_test):
                img, label = data['T'], data['T_label'].long()
                img, label = img.to(self.device), label.to(self.device)

                feat = self.G(img)
                out1 = self.C['di'](self.D['di'](feat))
                out2 = self.C['ds'](self.D['ds'](feat))
                test_loss += F.nll_loss(out1, label).item()

                out_ensemble = out1 + out2
                predi = out1.data.max(1)[1]
                preci = out2.data.max(1)[1]
                pred_ensemble = out_ensemble.data.max(1)[1]

                k = label.data.size()[0]
                correct1 += predi.eq(label.data).cpu().sum()
                correct2 += preci.eq(label.data).cpu().sum()
                correct3 += pred_ensemble.eq(label.data).cpu().sum()
                size += k
                # record = open('conf_{}.txt'.format(epoch), 'a')
                # for tmp_index in range(0, len(predi)):
                #     record.write('%d %d\n' % (label.data[tmp_index], predi[tmp_index]))
                # record.close()

        test_loss = test_loss / size
        acc1 = 100. * correct1 / size
        acc2 = 100. * correct2 / size
        acc3 = 100. * correct3 / size

        print('\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
            test_loss,
            correct1, size, acc1,
            correct2, size, acc2,
            correct3, size, acc3))

        self.logger.add_scalar(
            "test_target_acc/di", acc1,
            global_step=epoch)

        self.logger.add_scalar(
            "test_target_acc/ds", acc2,
            global_step=epoch)

        self.logger.add_scalar(
            "test_target_acc/max_ensemble", acc3,
            global_step=epoch)

        # if record_file:
        #     record = open(record_file, 'a')
        #     print('recording %s', record_file)
        #     record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
        #     record.close()
