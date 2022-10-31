from dataloader import VeRiWildDataset, get_simple_transform
from datamodule import VeRiWildDataModule
from model_lit import LitClassifier
from trainer import trainer

from pathlib import Path
from torchvision.models import resnet18


if __name__ == '__main__':
    data_path = Path('/content/drive/MyDrive/VIT_REID/mini_data')
    data_module = VeRiWildDataModule(
        data_path=data_path,
        train_txt='train_list_start0.txt',
        query_txt='test_3000_id_query.txt',
        gallery_txt='test_3000_id.txt',
        train_transform=get_simple_transform(),
        test_transform=get_simple_transform(),
    )
    data_module.setup()

    model = LitClassifier(
        net=resnet18(num_classes=data_module.num_classes)
    )

    # Fit
    trainer.fit(model, data_module)
