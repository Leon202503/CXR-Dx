"""离线评估适配器，与提交端共享 InferenceEngine。"""
from pathlib import Path
from data.dataset import CXRDataset, CXRTestDataset
from data.splits import assert_unseen
from predict.runtime import InferenceEngine


class Predictor(InferenceEngine):
    def __init__(self, ckpt_paths, device="", tta=False, batch_size=32, num_workers=0):
        if isinstance(ckpt_paths, str):
            ckpt_paths = [ckpt_paths]
        super().__init__(ckpt_paths, device, tta, batch_size)

    def predict_labeled(self, df, image_root=""):
        assert_unseen(df, self.metas)
        ds = CXRDataset.from_df(df, self.class_names, image_root)
        paths = [ds._full_path(str(p)) for p in ds.df[ds.img_col]]
        return self.predict_paths(paths), ds.labels

    def predict_test_csv(self, test_csv, image_root=""):
        ds = CXRTestDataset(test_csv, image_root)
        paths = [ds._full_path(str(p)) for p in ds.df[ds.img_col]]
        ids = ds.df[ds.id_col].astype(str).tolist() if ds.id_col else [Path(p).stem for p in paths]
        return ids, self.predict_paths(paths)
