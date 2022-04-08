from datasets.phoenix import Phoenix
from torch.utils.data import DataLoader
import os
from model import Model
import torch
from torch import optim
import tqdm
from util.metrics import get_wer_delsubins


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda')

def freeze(layer):
    for child in layer.children():
        #print(child)
        for param in child.parameters():
            param.requires_grad = False

def load_model(pretrained=True):
    model = Model(
        vocab_size=1296,
        dim=512,
        max_num_states=5,
        use_sfl=True,
        rdim=32,
        ent_coef=0.01
    )
    pretrained_dict = torch.load("phoenixpth/2014.pth",map_location=('cpu'))
    model_dict = model.state_dict()
    
    pretrained_dict["model"] = {k:v for k, v in pretrained_dict["model"].items() if (k in model_dict and "classifier" not in k)}
    model_dict.update(pretrained_dict["model"])
    
    model.load_state_dict(model_dict)
    #freeze(model.visual)
    #freeze(model.semantic)
    return model

def create_dataset(split='train'):
        
        #print(args.data_root)
    dataset = Phoenix(
            root='D:\Program\stochastic-cslr-main\stochastic-cslr-main\data\phoenix-2014-multisigner',
            split=split,
            p_drop=0.65,
            random_drop=True,
            random_crop=True,
            crop_size=[224,224],
            base_size=[225,225],
        )
    return dataset


def prepare_batch(batch):
       
        x, y = batch["video"], batch["label"]
        for i in range(len(x)):
            x[i] = x[i].to(device)
            y[i] = y[i].to(device)
        batch["video"] = x
        batch["label"] = y
        
        return batch

def test(model, test_loader, epoch):
    model.eval()
    prob = []
    wers = 0
    count = 0
    prob = []
    gt = []
    #result_dir = Path("results")
    #prob_path = result_dir / "prob.npz"
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader):
            video = list(map(lambda v: v.to(device), batch["video"]))
            #print(video)
            prob += [lpi.exp().cpu().numpy() for lpi in model(video)]
            gloss = batch['label']
            for g in gloss:
                gt += [g]
            break
    #prob_path.parent.mkdir(exist_ok=True, parents=True)
    #np.savez_compressed(prob_path, prob=np.array(prob, dtype="object"))

    hyp = model.decode(
            prob,
            beam_width=10, 
            prune=1e-2, 
            lm=None,
            nj=8
    )
    #with open('dict/wordtoix_final.txt', 'r', encoding='utf-8') as f2: 
    #    vocab = json.load(f2)
    #hyp = [" ".join([vocab[i] for i in hi]) for hi in hyp]
    
    #print(hyp)
    #print(gt)
    #print(len(hyp))
    #print(len(gt))
    for h,g in zip(hyp,gt):
        wers = wers + get_wer_delsubins(g,h)[0]  # wer()
        count = count+1
    a = wers/count  * 100
    with open('WER.txt','a+',encoding='utf-8') as f:
        print('epoch:{}|WER:{}'.format(epoch, a),file=f)
    return a

if __name__ == '__main__':
    dataset = create_dataset()#读数据，预处理
    dataset2 = create_dataset('test')
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False, num_workers=0, collate_fn=dataset.collate_fn)
    test_loader = DataLoader(dataset2, batch_size=1,shuffle=True, drop_last=False, num_workers=0, collate_fn=dataset2.collate_fn)

    model = load_model()
    print("model is on the {}!".format(device))
    model.to(device)
    model.train()
    optimizer = optim.Adam(params=filter(lambda p:p.requires_grad,model.parameters()), lr=0.0001, weight_decay = 1e-4)#
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    
    for epoch in range(100):
        all_loss = 0
        leng = 0
        l = 0
        for datas in tqdm.tqdm(train_loader,position=0):
        #print(datas['video'][0].shape)
            optimizer.zero_grad()
            leng += 1
            datas = prepare_batch(datas)
            #print(datas['video'][0],datas['gloss'][0])
            #exit()

            loss = model.compute_loss(datas['video'],datas['label'])['ctc_loss']
            #print(loss.item())
            #exit()
            loss.backward()
            optimizer.step()
            all_loss = all_loss+loss.item()
#   print('loss',loss.item())
        if not os.path.exists("train_loss"):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs("train_loss")
        with open('train_loss/unfre.txt', 'a+') as f:
            print(epoch,' ',all_loss/leng,file=f)
        print('epoch',epoch,'loss',all_loss/leng)
        if not os.path.exists("pth"):  #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs("pth")
        torch.save(model.state_dict(), 'pth/{:04d}.pth'.format(epoch))


        wer1 = test(model, test_loader, epoch)
        print(epoch,':',wer1)



