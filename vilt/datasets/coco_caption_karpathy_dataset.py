from .base_dataset import BaseDataset
from stanfordcorenlp import StanfordCoreNLP
import numpy as np

zh_model = StanfordCoreNLP(r'/home/zzq/stanford-corenlp-4.5.0')

class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            # names = ["coco_caption_karpathy_val"]
            names = ["coco_caption_karpathy_test"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)
        # print(suite.keys())
        if "text" in suite.keys():
            s_zh = suite["text"][0]
            dep_zh = zh_model.dependency_parse(s_zh)
            dep = ""
            for item in dep_zh:
                dep = dep + item[0]+ str(item[1])+ str(item[2])
            # suite["text"][0] = dep
            # suite.update({"text": dep})
            # print(dep)
            s_zh += dep
            listTup = list(suite['text'])
            listTup[0] = s_zh
            suite.update({"text": listTup})
            # print(suite['text'][0]+dep)

        if "test" in self.split:
            _index, _question_index = self.index_mapper[index]
            iid = self.table["image_id"][_index].as_py()
            iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite
