from GAN import StainingGAN
from predict_images import PredictGAN
import os
import torch
from pathlib import Path

class DigitalStaining():
    def __init__(
            self,
            dir,  # train or test : 元データのpath, predict : 入力画像のpath
            main_dir=None,  # train or test : 前処理データ保存フォルダのpath, predict : なし
            original_dir=None,  # train or test : なし, predict : 入力画像のpath
            new_dir=None,  # train or test : なし, predict : 予測画像保存フォルダのpath
            name="Run",  # 名称
            produce_image=True,  # 一度前処理をしている場合はFalse
            train_folders=["1"],  # Trainデータのフォルダ名
            val_folders=["2"],  # Valデータのフォルダ名
            test_folders=["3"],  # Testデータのフォルダ名
            # produce_images
            img_n=100,
            img_size=256,
            # gan
            n_epoch=50,
            discriminator="Patch4",  # Patch4, Patch3, Patch5, ResnetPatch, Resnet, or U_Net
            num_workers=8,  # GPUのメモリが足りない場合は小さくしてください
            in_chans=2,
            w_l1=50,
            w_ssim=1.0,
            w_dice=1.0,
            crop_size=256,
            stride=128,
            learning_rate_g=0.0002,
            learning_rate_d=0.0002,
            betas=(0.5, 0.999),
            images_to_use="both",
            device=torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'),
            patches_per_epoch=200,
            val_epoch=1,
            batch_size=16,
            # predict
            test_id="lpips"
    ):
        self.dir = Path(dir)
        self.main_dir = main_dir
        self.original_dir = original_dir
        self.new_dir = new_dir
        self.name = name
        self.produce_image = produce_image
        self.train_folders = train_folders
        self.val_folders = val_folders
        self.test_folders = test_folders
        self.img_n = img_n
        self.img_size = img_size
        self.n_epoch = n_epoch
        self.discriminator = discriminator
        self.num_workers = num_workers
        self.in_chans = in_chans
        self.w_l1 = w_l1
        self.w_ssim = w_ssim
        self.w_dice = w_dice
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        self.betas = betas
        self.images_to_use = images_to_use
        self.device = device
        self.patches_per_epoch = patches_per_epoch
        self.val_epoch = val_epoch
        self.batch_size = batch_size
        self.test_id = test_id

    def _init_gan(self):
        self.gan = StainingGAN(
            self.main_dir,
            train_folders=self.train_folders,
            val_folders=self.val_folders,
            test_folders=self.test_folders,
            name=self.name,
            n_epoch=self.n_epoch,
            discriminator=self.discriminator,  # Patch4, Patch3, Patch5, ResnetPatch, Resnet, or U_Net
            num_workers=self.num_workers,  # GPUのメモリが足りない場合は小さくしてください
            in_chans=self.in_chans,
            w_l1=self.w_l1,
            w_ssim=self.w_ssim,
            w_dice=self.w_dice,
            crop_size=self.crop_size,
            stride=self.stride,
            learning_rate_g=self.learning_rate_g,
            learning_rate_d=self.learning_rate_d,
            betas=self.betas,
            images_to_use=self.images_to_use,
            device=self.device,
            patches_per_epoch=self.patches_per_epoch,
            val_epoch=self.val_epoch,
            batch_size=self.batch_size
        )

    def train(self):
        if self.produce_image:
            if self.main_dir is None:
                parent_dir = os.path.dirname(self.dir)
                self.main_dir = os.path.join(parent_dir, f"{self.name}_produced_images")
                os.makedirs(self.main_dir, exist_ok=True)
            from produce_images import produce_images
            produce_images(
                self.dir,
                self.main_dir,
                train_folders=self.train_folders,
                val_folders=self.val_folders,
                test_folders=self.test_folders,
                img_n=self.img_n,
                img_size=self.img_size
            )
        else:
            self.main_dir = self.dir

        self._init_gan()
        self.gan.train()

    def test(self):
        if self.produce_image:
            if self.main_dir is None:
                parent_dir = os.path.dirname(self.dir)
                self.main_dir = os.path.join(parent_dir, f"{self.name}_produced_images")
                os.makedirs(self.main_dir, exist_ok=True)
        else:
            self.main_dir = self.dir
        self._init_gan()
        self.gan.test()

    def predict(self):
        if self.original_dir is None:
            self.original_dir = self.dir
        PredictGAN(
            self.original_dir,
            new_dir=self.new_dir,
            name=self.name,
            test_id=self.test_id,
            in_chans=self.in_chans,
            crop_size=self.crop_size,
            stride=self.stride,
            images_to_use=self.images_to_use,
            device=self.device
        )