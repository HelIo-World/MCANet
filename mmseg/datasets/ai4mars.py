from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class Ai4MarsDataSet(BaseSegDataset):

    METAINFO = dict(
        classes=('soil','bedrock','sand','big_rock','null'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153]])

    def __init__(self,
                 img_suffix='.JPG',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
