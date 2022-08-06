import pyarrow as pa
import pandas as pd
import os

import torch
import copy
import time
import io
import numpy as np
import re
from PIL import Image
import ipdb

from stanfordcorenlp import StanfordCoreNLP
import numpy as np




from vilt.modules import ViLTransformerSS

from vilt.modules.objectives import cost_matrix_cosine, ipot
from vilt.transforms import pixelbert_transform
from vilt.datamodules.datamodule_base import get_pretrained_tokenizer

zh_model = StanfordCoreNLP(r'/home/zzzqi/ViLT/nlp/stanford-corenlp-full-2018-10-05')
_config = {'exp_name': 'vilt', 'seed': 0, 'datasets': ['coco', 'vg', 'sbu', 'gcc'], 'loss_names': {'itm': 1, 'mlm': 1, 'mpp': 0, 'vqa': 0, 'nlvr2': 0, 'irtr': 0}, 'batch_size': 4096, 'train_transform_keys': ['pixelbert'], 'val_transform_keys': ['pixelbert'], 'image_size': 384, 'max_image_len': -1, 'patch_size': 32, 'draw_false_image': 1, 'image_only': False, 'vqav2_label_size': 3129, 'max_text_len': 40, 'tokenizer': 'bert-base-uncased', 'vocab_size': 30522, 'whole_word_masking': False, 'mlm_prob': 0.15, 'draw_false_text': 0, 'vit': 'vit_base_patch32_384', 'hidden_size': 768, 'num_heads': 12, 'num_layers': 12, 'mlp_ratio': 4, 'drop_rate': 0.1, 'optim_type': 'adamw', 'learning_rate': 0.0001, 'weight_decay': 0.01, 'decay_power': 1, 'max_epoch': 100, 'max_steps': 25000, 'warmup_steps': 2500, 'end_lr': 0, 'lr_mult': 1, 'get_recall_metric': False, 'resume_from': None, 'fast_dev_run': False, 'val_check_interval': 1.0, 'test_only': False, 'data_root': '', 'log_dir': 'result', 'per_gpu_batchsize': 0, 'num_gpus': 0, 'num_nodes': 1, 'load_path': 'weights/vilt_200k_mlm_itm.ckpt', 'num_workers': 8, 'precision': 16}
_config = copy.deepcopy(_config)


loss_names = {
    "itm": 0,
    "mlm": 0.5,
    "mpp": 0,
    "vqa": 0,
    "imgcls": 0,
    "nlvr2": 0,
    "irtr": 0,
    "arc": 0,
}
tokenizer = get_pretrained_tokenizer(_config["tokenizer"])


_config.update(
    {
        "loss_names": loss_names,
    }
)

model = ViLTransformerSS(_config)
model.setup("test")
#开启评估模式
model.eval()


device = "cuda:0" if _config["num_gpus"] > 0 else "cpu"
model.to(device)

print("加载模型")
data_dir = "/data/nfs/datasets/mscoco2014"
name = "coco_caption_karpathy_train"
names = ["coco_caption_karpathy_train", "coco_caption_karpathy_test", "coco_caption_karpathy_val", "coco_caption_karpathy_restval"]
# tables = [
#                 pa.ipc.RecordBatchFileReader(
#                     pa.memory_map(f"{data_dir}/{name}.arrow", "r")
#                 ).read_all()
#                 for name in names
#                 if os.path.isfile(f"{data_dir}/{name}.arrow")
#             ]

table = pa.ipc.open_file("/data/nfs/datasets/mscoco2014/coco_caption_karpathy_test.arrow").read_pandas()
print("加载dataset")
def infer(image, mp_text):
    try:
        # res = requests.get(url)
        # image = Image.open(io.BytesIO(res.content)).convert("RGB")
        image = Image.open(io.BytesIO(image)).convert("RGB")
        img = pixelbert_transform(size=384)(image)
        img = img.unsqueeze(0).to(device)
        # 完成img的预处理
    except:
        return False

    batch = {"text": [""], "image": [None]}
    tl = len(re.findall("\[MASK\]", mp_text))
    inferred_token = [mp_text]
    batch["image"][0] = img


    selected_token = ""
    encoded = tokenizer(inferred_token)
    # 完成text的预处理

    
    batch["text"] = inferred_token
    batch["text_ids"] = torch.tensor(encoded["input_ids"]).to(device)
    batch["text_labels"] = torch.tensor(encoded["input_ids"]).to(device)
    batch["text_masks"] = torch.tensor(encoded["attention_mask"]).to(device)
    infer = model(batch)
    
    txt_emb, img_emb = infer["text_feats"], infer["image_feats"]

    print(infer["cls_feats"].shape)

            
        
    return infer["raw_cls_feats"]

dataset_root="/home/zzzqi/ViLT/dataset"

def textToHeadAdj(textEn):

    s_zh = textEn
    dep_zh = zh_model.dependency_parse(s_zh)
    '''查找根结点对应的索引'''
    root_index=[]
    for i in range(len(dep_zh)):
        if dep_zh[i][0]=='ROOT':
            root_index.append(i)

    '''修改依存关系三元组'''
    new_dep_outputs=[]
    for i in range(len(dep_zh)):
        for index in root_index:
            if i+1>index:
                tag=index

        if dep_zh[i][0]=='ROOT':	
            dep_output=(dep_zh[i][0],dep_zh[i][1],dep_zh[i][2]+tag)
        else:
            dep_output = (dep_zh[i][0], dep_zh[i][1] + tag, dep_zh[i][2] + tag)
        new_dep_outputs.append(dep_output)

    head_list = []
    tokens = zh_model.word_tokenize(s_zh)
    # 求解headlist
    for i in range(len(tokens)):
        for dep_output in new_dep_outputs:
            if dep_output[-1] == i + 1:
                head_list.append(int(dep_output[1]))

    
    ret = np.zeros((20,20))
    for i in range(len(head_list)):
        j=head_list[i]
        if j!=0:
            ret[i,j-1]=1
            ret[j-1,i]=1

    return ret.reshape(400)

list = []

for row in table.itertuples():
  
    img = row[1]
    text = (row[2])[0]

    raw_cls_feature = infer(img, text)
    tempList = raw_cls_feature.detach().numpy().tolist()
    dep_tree = textToHeadAdj(text)
    tempList.append(dep_tree.tolist())
    # print(raw_cls_feature)
    list.append(tempList)


dataframe = pd.DataFrame(
            list, columns=["raw_cls_feature", "dep_tree_adj"],
        )

table = pa.Table.from_pandas(dataframe)
os.makedirs(dataset_root, exist_ok=True)
with pa.OSFile(
    f"{dataset_root}/Allfeature.arrow", "wb"
) as sink:
    with pa.RecordBatchFileWriter(sink, table.schema) as writer:
        writer.write_table(table)

