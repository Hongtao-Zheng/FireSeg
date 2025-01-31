import os
import yaml
import random
import argparse

import cv2
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from detectron2.structures import Boxes, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel, ConfigMixin
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from diffusers.schedulers import EulerDiscreteScheduler 

from ptp_utils import text2image, register_attention_control, encode_prompt
from model.unet import get_feature_dic, clear_feature_dic, ProxyUNet2D
from model.segment.transformer_decoder_semantic import SegDecoderOpenWord
from model.segment.criterion import SetCriterion
from model.segment.matcher import HungarianMatcher
from dataset import *

LOW_RESOURCE = False 



def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class dict2obj(object):
    def __init__(self, d):
        self.__dict__['d'] = d
 
    def __getattr__(self, key):
        value = self.__dict__['d'][key]
        if type(value) == type({}):
            return dict2obj(value)
 
        return value

class AttentionStore:
    def __init__(self):
        self.num_att_layers = -1
        self.cur_step = 0 
        self.cur_att_layer = 0
        self.step_store = self._get_empty_store()
        self.attention_store = {}
        self.activate = True
    
    def reset(self):
        self.cur_step = 0 
        self.cur_att_layer = 0
        self.step_store = self._get_empty_store()
        self.attention_store = {}
    
    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if self.activate:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn
    
    def between_steps(self):
        if self.activate:
            if len(self.attention_store) == 0:
                self.attention_store = self.step_store
            else:
                for key in self.attention_store:
                    for i in range(len(self.attention_store[key])):
                        self.attention_store[key][i] += self.step_store[key][i]
            self.step_store = self._get_empty_store()

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
            
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0 # reset
            self.cur_step += 1 if self.activate else 0
            self.between_steps()
        return attn
    
    def _get_empty_store(self):
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [],  "mid_self": [],  "up_self": []}
    
def freeze_params(params):
    for param in params:
        param.requires_grad = False
        



def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

    for i in range(1,semseg.shape[0]):
        if (semseg[i]*(semseg[i]>0.5)).sum()<5000:
            semseg[i] = 0

    return semseg

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, nargs="?", default="./config/", help="config for training")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--batch_size", type=int, default=1, help="the seed (for reproducible sampling)")
    parser.add_argument("--image_limitation", type=int, default=5, help="image_limitation")
    parser.add_argument("--dataset", type=str, default="Cityscapes", help="dataset: VOC/Cityscapes/MaskCut")
    parser.add_argument("--save_name", type=str, help="the save dir name", default="Test")
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    f = open(opt.config)
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = dict2obj(cfg)
    
    opt.dataset = cfg.DATASETS.dataset
    opt.batch_size = cfg.DATASETS.batch_size
    
    # dataset
    if opt.dataset == "VOC":
        dataset = Semantic_VOC(set="train",image_limitation = opt.image_limitation, is_zero = cfg.DATASETS.is_zero, is_long_tail = cfg.DATASETS.long_tail)
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    
    learning_rate = cfg.SOLVER.learning_rate
    total_epoch = cfg.SOLVER.total_epoch
    num_diffusion_steps = cfg.Diffusion.NUM_DIFFUSION_STEPS
    guidance_scale = 5.0

    save_dir = 'checkpoint'
    os.makedirs(save_dir, exist_ok=True)
    
    ckpt_dir = os.path.join(save_dir, opt.save_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    os.makedirs(os.path.join(ckpt_dir, 'training'), exist_ok=True)
    
    # ===================load model=====================
    sd_xl_path = "../../autodl-tmp/stable-diffusion-xl-base-1.0"
    
    tokenizer = CLIPTokenizer.from_pretrained(sd_xl_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(sd_xl_path, subfolder="tokenizer_2")
    

    vae = AutoencoderKL.from_pretrained(sd_xl_path, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(sd_xl_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(sd_xl_path, subfolder="text_encoder_2")
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(sd_xl_path, subfolder="scheduler")
    
    unet: UNet2DConditionModel = ProxyUNet2D.from_pretrained(sd_xl_path, subfolder="unet")
    
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    text_encoder_2 = text_encoder_2.to(device)
    unet = unet.to(device)

    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(text_encoder_2.parameters())
    freeze_params(unet.parameters())

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    print("VAE parameters:       ", count_parameters(vae))
    print("Text Encoder parameters:", count_parameters(text_encoder))
    print("Text Encoder_2 parameters:", count_parameters(text_encoder_2))
    print("UNet parameters:       ", count_parameters(unet))

    
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    unet.eval()
    
    # sd_xl_pipeline = StableDiffusionXLPipeline.from_pretrained(sd_xl_path)
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    
    # ===================segmentation Decoder=====================
    # building criterion
    matcher = HungarianMatcher(cost_class=cfg.SEG_Decoder.CLASS_WEIGHT, cost_mask=cfg.SEG_Decoder.MASK_WEIGHT, cost_dice=cfg.SEG_Decoder.DICE_WEIGHT, num_points=cfg.SEG_Decoder.TRAIN_NUM_POINTS)
    criterion = SetCriterion(cfg.SEG_Decoder.num_classes, matcher=matcher, weight_dict={"loss_ce": cfg.SEG_Decoder.CLASS_WEIGHT, "loss_mask": cfg.SEG_Decoder.MASK_WEIGHT, "loss_dice": cfg.SEG_Decoder.DICE_WEIGHT}, eos_coef=cfg.SEG_Decoder.no_object_weight, losses=["labels", "masks"], num_points=cfg.SEG_Decoder.TRAIN_NUM_POINTS, oversample_ratio=cfg.SEG_Decoder.OVERSAMPLE_RATIO, importance_sample_ratio=cfg.SEG_Decoder.IMPORTANCE_SAMPLE_RATIO)
    
    # ==============initializing open vocabulary model================
    seg_model = SegDecoderOpenWord(num_classes=cfg.SEG_Decoder.num_classes, num_queries=cfg.SEG_Decoder.num_queries).to(device)


    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("SegDecoderOpenWord Params (trainable):", count_params(seg_model))


    g_optim = optim.Adam([{"params": seg_model.parameters()}], lr=learning_rate)
    scheduler = StepLR(g_optim, step_size=350, gamma=0.1)
    
    controller = AttentionStore()
    register_attention_control(unet, controller)

    best_loss = 100000000
    for ep in range(total_epoch):
        print('Epoch ' +  str(ep) + '/' + str(total_epoch))
        avg_loss = []
        for step, batch in enumerate(dataloader):
            
            torch.cuda.empty_cache()
            unet = unet.to(device)
        
            g_cpu = torch.Generator().manual_seed(random.randint(1, 10000000))

            # clear all features and attention maps
            clear_feature_dic()
            controller.reset()

            image, prompts, class_name = batch["image"], batch["prompt"], batch["classes_str"]
            
            with torch.no_grad():
                latents = vae.encode(image.to(device)).latent_dist.sample().detach()
            latents = latents * vae.config.scaling_factor

            # sample noise that we'll add to the latents
            noise = torch.randn(latents.shape).to(latents.device)
            bsz = latents.shape[0]

            # set timesteps
            noise_scheduler.set_timesteps(num_diffusion_steps)
            timesteps = torch.ones(bsz).long().to(latents.device) * noise_scheduler.timesteps[-1]
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = noisy_latents.to(latents.device)
    
            text2image(unet, vae, tokenizer, text_encoder, tokenizer_2, text_encoder_2, noise_scheduler, prompts, controller, latent=noisy_latents, num_inference_steps=num_diffusion_steps, guidance_scale=guidance_scale, generator=g_cpu, vae_scale_factor=vae_scale_factor, is_train=True)
            
            torch.cuda.empty_cache()
            unet = unet.to("cpu")
            
            # train segmentation
            query_text = class_name[0]
            text_embeddings = encode_prompt(query_text, tokenizer, tokenizer_2, text_encoder, text_encoder_2)['prompt_embeds']
            text_embeddings = text_embeddings.to(latents.device)

            if text_embeddings.size()[1] > 1:
                text_embeddings = torch.unsqueeze(text_embeddings.mean(1), 1)

            diffusion_features = get_feature_dic()
            outputs = seg_model(diffusion_features, controller, prompts, tokenizer, text_embeddings)

            # bipartite matching-based loss
            losses = criterion(outputs, batch)
            loss = losses['loss_ce'] * cfg.SEG_Decoder.CLASS_WEIGHT + losses['loss_mask'] * cfg.SEG_Decoder.MASK_WEIGHT + losses['loss_dice']*cfg.SEG_Decoder.DICE_WEIGHT
            avg_loss.append(losses['loss_mask'].item() * cfg.SEG_Decoder.MASK_WEIGHT + losses['loss_dice'].item() * cfg.SEG_Decoder.DICE_WEIGHT)
            
            g_optim.zero_grad()
            print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}, loss_ce: {3:0.4f},loss_mask: {4:0.4f},loss_dice: {5:0.4f}, lr: {6:0.6f}, prompt: ".format(step, len(dataloader), loss, losses['loss_ce']*cfg.SEG_Decoder.CLASS_WEIGHT,losses['loss_mask']*cfg.SEG_Decoder.MASK_WEIGHT, losses['loss_dice']*cfg.SEG_Decoder.DICE_WEIGHT,float(g_optim.state_dict()['param_groups'][0]['lr'])),prompts)
            loss.backward()
            g_optim.step()
                        
        scheduler.step()
        avg_loss = sum(avg_loss) / len(avg_loss)
        best_loss = avg_loss if avg_loss < best_loss else best_loss
        print(f"Average loss: {avg_loss}, Best loss: {best_loss} at epoch {ep}")
        torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'latest_checkpoint.pth'))
        torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'best_checkpoint.pth')) if avg_loss == best_loss else None
        torch.save(seg_model.state_dict(), os.path.join(ckpt_dir, 'checkpoint_'+str(ep)+'.pth')) if ep % 10 == 0 else None
        
    print("Saving latest checkpoint to", os.path.join(ckpt_dir, 'latest_checkpoint.pth'))

if __name__ == "__main__":
    main()