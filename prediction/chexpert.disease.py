import os

import sys
sys.path.append('../../chexploration')
from prediction.dataloader import CheXpertDataResampleModule,CheXpertDataset,CheXpertDataModule,CheXpertGIDataModule

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torchvision
import torchvision.transforms as T
from torchvision import models
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from skimage.io import imread
from skimage.io import imsave
from tqdm import tqdm
from argparse import ArgumentParser

from torchmetrics import Accuracy,AUROC
from torchmetrics.classification import MultilabelAUROC
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil

image_size = (224, 224)
batch_size = 64
epochs = 20
num_workers = 2
img_data_dir = '/work3/ninwe/dataset/'
model_name = 'resnet' # 'densenet' or 'resnet'
model_scale = '50' # resnet: 18,34,50,101,152
                   # densenet: 121,161,169,201
                   #

view_position='all'
gender = 'None' #'F','M',None
single_label = None
num_classes = 14 if single_label == None else 1
lr=1e-6


pretrained = True
augmentation = True
csv_file_img = '../datafiles/chexpert/'+'chexpert.sample.allrace.csv'

gi_split=True
fold_nums=np.arange(0,20)


resam=False
female_perc_in_training_set = [0,50,100]#
random_state_set = np.arange(0,10)
num_per_patient = 1
disease_list=['Pneumothorax','Pneumonia','Cardiomegaly']
#['Enlarged Cardiomediastinum','Cardiomegaly','Lung Opacity','Lung Lesion']
# chose_disease_str =  'Pneumothorax' #'Pneumonia','Pneumothorax'
random_state = 2022
if resam: num_classes = 1
# print('multi label training')
# num_classes = len(DISEASE_LABELS)
isMultilabel = True if num_classes!=1 else False

save_model_para = False
loss_func_type='BCE'







def get_cur_version(dir_path):
    i = 0
    while os.path.exists(dir_path+'/version_{}'.format(i)):
        i+=1
    return i



class ResNet(pl.LightningModule):
    def __init__(self, num_classes,lr,pretrained,model_scale):
        super().__init__()
        self.lr=lr
        self.num_classes = num_classes
        self.pretrained= pretrained
        self.model_scale=model_scale

        if self.model_scale == '18':
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.model_scale == '34':
            self.model = models.resnet34(pretrained=self.pretrained)
        elif self.model_scale == '50':
            self.model = models.resnet50(pretrained=self.pretrained)
        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        if self.num_classes == 1:
            self.accu_func = Accuracy(task="binary", num_labels=num_classes)
            self.auroc_func = AUROC(task='binary', num_labels=num_classes, average='macro', thresholds=None)
        elif self.num_classes > 1:
            self.accu_func = Accuracy(task="multilabel", num_labels=num_classes)
            self.auroc_func = MultilabelAUROC(num_labels=num_classes, average='macro', thresholds=None)

    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob,lab.long())
        return loss,multi_accu,multi_auroc

    def training_step(self, batch, batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('train_loss', loss)
        self.log('train_accu', multi_accu)
        self.log('train_auroc', multi_auroc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('val_loss', loss)
        self.log('val_accu', multi_accu)
        self.log('val_auroc', multi_auroc)

    def test_step(self, batch, batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('test_loss', loss)
        self.log('test_accu', multi_accu)
        self.log('test_auroc', multi_auroc)


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes,lr):
        super().__init__()
        self.lr=lr
        self.num_classes = num_classes
        self.model = models.densenet121(pretrained=True)
        # freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        #grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        #self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            out = model(img)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()


def embeddings(model, data_loader, device):
    model.eval()

    embeds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            img, lab = batch['image'].to(device), batch['label'].to(device)
            emb = model(img)
            embeds.append(emb)
            targets.append(lab)

        embeds = torch.cat(embeds, dim=0)
        targets = torch.cat(targets, dim=0)

    return embeds.cpu().numpy(), targets.cpu().numpy()


def main(hparams,female_perc_in_training=None,random_state=None,chose_disease_str=None,fold_num=None):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")
    print('DEVICE:{}'.format(device))



    if resam:
        run_config = '{}{}-lr{}-ep{}-pt{}-aug{}-{}%female-D{}-npp{}-ml{}-rs{}-imgs{}_mpara{}'.format(model_name,
                                                                                                    model_scale, lr,
                                                                                                    epochs,
                                                                                                    int(pretrained),
                                                                                                    int(augmentation),
                                                                                                    female_perc_in_training,
                                                                                                    chose_disease_str,
                                                                                                    num_per_patient,
                                                                                                    int(isMultilabel),
                                                                                                    random_state,
                                                                                                    image_size[0],
                                                                                                    int(save_model_para))

    elif gi_split:
        gender_setting = '{}%_female'.format(female_perc_in_training)
        run_config = '{}{}-lr{}-ep{}-pt{}-aug{}-VP{}-GIsplit-{}-Fold{}-imgs{}-mpara{}'.format(model_name,
                                                                                              model_scale,
                                                                                              lr,
                                                                                              epochs,
                                                                                              int(pretrained),
                                                                                              int(augmentation),
                                                                                              view_position,
                                                                                              gender_setting,
                                                                                              fold_num,
                                                                                              image_size[0],
                                                                                              int(save_model_para))
    else:
        run_config = '{}{}-sl{}-ep{}-lr{}-VP{}-SEX{}-mpara{}'.format(model_name, model_scale, str(single_label), epochs, lr,
                                                             view_position, gender,
                                                                     int(save_model_para))

    print('------------------------------------------\n'*3)
    print(run_config)

     # Create output directory
    out_dir = '/work3/ninwe/run/chexpert/disease/' + run_config
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cur_version = get_cur_version(out_dir)

    # data
    if resam:
        data=CheXpertDataResampleModule(img_data_dir=img_data_dir,
                                csv_file_img=csv_file_img,
                                image_size=image_size,
                                pseudo_rgb=False,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                augmentation=augmentation,
                                outdir=out_dir,
                                version_no=cur_version,
                                female_perc_in_training=female_perc_in_training,
                                chose_disease=chose_disease_str,
                                random_state=random_state,
                                num_classes=num_classes,
                                num_per_patient=num_per_patient

        )
    elif gi_split:
        data = CheXpertGIDataModule(img_data_dir=img_data_dir,
                                csv_file_img=csv_file_img,
                                image_size=image_size,
                                pseudo_rgb=False,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                augmentation=augmentation,
                                view_position = view_position,
                                vp_sample = view_position,
                                only_gender=gender,
                             save_split=True,
                             outdir=out_dir,
                             version_no=cur_version,
                             gi_split=gi_split,
                             gender_setting=gender_setting,
                             fold_num=fold_num)
    else:
        data = CheXpertDataModule(csv_train_img='../datafiles/chexpert/chexpert.sample.train.csv',
                                  csv_val_img='../datafiles/chexpert/chexpert.sample.val.csv',
                                  csv_test_img='../datafiles/chexpert/chexpert.sample.test.csv',
                                  image_size=image_size,
                                  pseudo_rgb=True,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  single_label=single_label,
                                  view_position=view_position,
                                  gender=gender,
                                  outdir=out_dir,
                                  version_no=cur_version)



    # model
    if model_name == 'densenet':
        model_type = DenseNet
    elif model_name == 'resnet':
        model_type = ResNet
    model = model_type(num_classes=num_classes,lr=lr,pretrained=pretrained,model_scale=model_scale)
    model = model.to(device)


    temp_dir = os.path.join(out_dir, 'temp_version_{}'.format(cur_version))
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    for idx in range(0, 5):
        if augmentation:
            sample = data.train_set.exam_augmentation(idx)
            sample = np.asarray(sample)
            sample = np.transpose(sample, (2, 1, 0))
            imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.png'), sample)
        else:
            sample = data.train_set.get_sample(idx)  # PIL
            sample = np.asarray(sample['image'])
            imsave(os.path.join(temp_dir, 'sample_' + str(idx) + '.png'), sample.astype(np.uint8))

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')

    # train
    trainer = pl.Trainer(
        #callbacks=[checkpoint_callback],S
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        log_every_n_steps = 1,
        max_epochs=epochs,
        gpus=hparams.gpus,
        logger=TensorBoardLogger('/work3/ninwe/run/chexpert/disease/', name=run_config,version=cur_version),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes,lr=lr,
                                            pretrained=pretrained,model_scale=model_scale)

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")
    # print('DEVICE:{}'.format(device))

    model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.version_{}.csv'.format(cur_version)), index=False)


    print('TESTING')
    preds_test, targets_test, logits_test = test(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.version_{}.csv'.format(cur_version)), index=False)


    if (True and resam):
        print('TESTING on tain set')
        # trainloader need to be non shuffled!
        preds_test, targets_test, logits_test = test(model, data.train_dataloader_nonshuffle(), device)
        df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
        df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
        df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
        df = pd.concat([df, df_logits, df_targets], axis=1)
        df.to_csv(os.path.join(out_dir, 'predictions.train.version_{}.csv'.format(cur_version)), index=False)

    # print('EMBEDDINGS')
    #
    # model.remove_head()
    #
    # embeds_val, targets_val = embeddings(model, data.val_dataloader(), device)
    # df = pd.DataFrame(data=embeds_val)
    # df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    # df = pd.concat([df, df_targets], axis=1)
    # df.to_csv(os.path.join(out_dir, 'embeddings.val.version_{}.csv'.format(cur_version)), index=False)
    #
    # embeds_test, targets_test = embeddings(model, data.test_dataloader(), device)
    # df = pd.DataFrame(data=embeds_test)
    # df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    # df = pd.concat([df, df_targets], axis=1)
    # df.to_csv(os.path.join(out_dir, 'embeddings.test.version_{}.csv'.format(cur_version)), index=False)

    if save_model_para == False:
        model_para_dir = os.path.join(out_dir,'version_{}'.format(cur_version))
        shutil.rmtree(model_para_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    print('START!')

    if resam:
        print('***********RESAMPLING EXPERIMENT**********\n' * 5)
        for d in disease_list:
            for female_perc_in_training in female_perc_in_training_set:
                for i in random_state_set:
                    main(args, female_perc_in_training=female_perc_in_training, random_state=i, chose_disease_str=d)
    elif gi_split:
        print('***********GI SPLIT EXPERIMENT**********\n' * 5)
        for f_perc in female_perc_in_training_set:
            for fold_i in fold_nums:
                main(args, female_perc_in_training=f_perc, fold_num=fold_i)

