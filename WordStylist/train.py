import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import json
import unicodedata
from diffusers import AutoencoderKL
from unet import UNetModel
import wandb

MAX_CHARS = 16
OUTPUT_MAX_LEN = MAX_CHARS
c_classes = ''.join(chr(n) for n in [40, 41, 44, 45, 46, 47, 58, 2433, 2434, 2435, 2437, 2438, 2439, 2441, 2443, 2447, 2451, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2467, 2468, 2469, 2470, 2471, 2472, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2482, 2486, 2487, 2488, 2489, 2492, 2494, 2495, 2496, 2497, 2498, 2499, 2503, 2504, 2507, 2508, 2509, 2510, 2537, 2540, 2542, 2543])
cdict = {c: i for i, c in enumerate(c_classes)}
icdict = {i: c for i, c in enumerate(c_classes)}


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def label_padding(labels, num_tokens):
    labels = unicodedata.normalize('NFC', labels)
    if len(labels) > OUTPUT_MAX_LEN:
        raise ValueError(f"Label length {len(labels)} exceeds OUTPUT_MAX_LEN={OUTPUT_MAX_LEN}: {labels!r}")
    unknown = sorted(set(labels) - set(letter2index))
    if unknown:
        raise ValueError(f"Unknown characters in label {labels!r}: {unknown!r}")
    ids = [letter2index[char] for char in labels]
    ids.extend([tokens["PAD_TOKEN"]] * (OUTPUT_MAX_LEN - len(ids)))
    return ids

def labelDictionary():
    letter2index = {label: n for n, label in enumerate(c_classes)}
    index2letter = {v: k for k, v in letter2index.items()}
    with open("letter2index.json", "w", encoding="utf-8") as f:
        json.dump(letter2index, f, ensure_ascii=False, indent=2)
    with open("index2letter.json", "w", encoding="utf-8") as f:
        json.dump(index2letter, f, ensure_ascii=False, indent=2)
    return len(c_classes), letter2index, index2letter

char_classes, letter2index, index2letter = labelDictionary()
tokens = {"PAD_TOKEN": char_classes}
num_tokens = 1
vocab_size = char_classes + 1

print("num_tokens", num_tokens)
print("num of character classes", char_classes)
print("padding token ID", tokens["PAD_TOKEN"])
print("vocabulary size", vocab_size)

def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im

class IAMDataset(Dataset):
    def __init__(self, full_dict, image_path, writer_dict, args, transforms=None):

        self.data_dict = full_dict
        self.image_path = image_path
        self.writer_dict = writer_dict
    
        self.transforms = transforms
        self.output_max_len = OUTPUT_MAX_LEN
        self.max_len = MAX_CHARS
        self.n_samples_per_class = 16
        self.indices = list(full_dict.keys())
        
            
    def __len__(self):
        return len(self.indices)
            

    
    def __getitem__(self, idx):
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        wr_id = torch.tensor(self.writer_dict[wr_id]).to(torch.int64)
        img_path = os.path.join(self.image_path, *image_name.replace("\\", "/").split("/"))
        
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        
        word_embedding = label_padding(label, num_tokens) 
        word_embedding = np.array(word_embedding, dtype="int64")
        word_embedding = torch.from_numpy(word_embedding).long()    
        
        return image, word_embedding, wr_id



class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 128), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


    def sampling(self, model, vae, n, x_text, labels, args, mix_rate=None, cfg_scale=3):
        model.eval()
        tensor_list = []
        #if mix_rate is not None:
         #   print('mix rate', mix_rate)
        with torch.no_grad():
            
            words = [x_text]*n
            for word in words:
                transcript = label_padding(word, num_tokens) #self.transform_text(transcript)
                word_embedding = np.array(transcript, dtype="int64")
                word_embedding = torch.from_numpy(word_embedding).long()#float()
                tensor_list.append(word_embedding)
            text_features = torch.stack(tensor_list)
            text_features = text_features.to(args.device)
            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                if cfg_scale > 0:
                    # uncond_predicted_noise = model(x, t, text_features, sid)
                    # predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                    uncond_predicted_noise = model(x, None, t, text_features, labels, mix_rate=mix_rate)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
    
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, num_classes, vocab_size, transforms, args, start_epoch=0):
    model.train()
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)

    print('Training started....')

    for epoch in range(start_epoch, args.epochs):
        print('Epoch:', epoch + 1)
        pbar = tqdm(loader)

        for i, (images, word, s_id) in enumerate(pbar):
            images = images.to(args.device)
            original_images = images
            text_features = word.to(args.device)
            s_id = s_id.to(args.device)

            if args.latent:
                images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215

            t = diffusion.sample_timesteps(images.shape[0]).to(args.device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(
                x_t,
                original_images=original_images,
                timesteps=t,
                context=text_features,
                y=s_id,
                or_images=None
            )

            loss = mse_loss(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)
            pbar.set_postfix(MSE=loss.item())

        completed_epochs = epoch + 1
        should_save = completed_epochs % 50 == 0 or completed_epochs == args.epochs

        if should_save:
            checkpoint_path = os.path.join(args.save_path, 'models', 'training_state.pt')
            torch.save({
                'completed_epochs': completed_epochs,
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)

            torch.save(model.state_dict(), os.path.join(args.save_path, 'models', 'ckpt.pt'))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, 'models', 'ema_ckpt.pt'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'models', 'optim.pt'))

            print(f'Checkpoint saved after {completed_epochs} epochs.')

            model.eval()
            ema_model.eval()

            n = min(16, num_classes)
            labels = torch.arange(n, dtype=torch.long, device=args.device)
            words = ['নিজস্ব', 'ঘন', 'উদ্দেশ্য']

            for x_text in words:
                with torch.no_grad():
                    ema_sampled_images = diffusion.sampling(
                        ema_model,
                        vae,
                        n=n,
                        x_text=x_text,
                        labels=labels,
                        args=args
                    )

                sampled_ema = save_images(
                    ema_sampled_images,
                    os.path.join(args.save_path, 'images', f'{x_text}_{completed_epochs}.jpg'),
                    args
                )

                if args.wandb_log:
                    wandb_sampled_ema = wandb.Image(
                        sampled_ema,
                        caption=f'{x_text}_{completed_epochs}'
                    )
                    wandb.log({'Sampled images': wandb_sampled_ema})

            model.train()
            ema_model.eval()
   


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam', help='iam or other dataset') 
    parser.add_argument('--iam_path', type=str, default='/path/to/iam/images/', help='path to iam dataset (images 64x256)')
    parser.add_argument('--gt_train', type=str, default='./gt/gan.iam.tr_va.gt.filter27')
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4, help='if latent is True channels should be 4, else 3')  
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./save_path/')
    parser.add_argument('--device', type=str, default='cuda:0') 
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--writer_dict', type=str, default='./writers_dict.json')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5', help='path to stable diffusion')
    
    
    args = parser.parse_args()
    if args.wandb_log==True:
        runs = wandb.init(project='DIFFUSION_IAM', name=f'{args.save_path}', config=args)

        wandb.config.update(args)
    
    #create save directories
    setup_logging(args)

    print('character vocabulary size', vocab_size)
    
    if args.dataset == 'iam':
        class_dict = {}
        for i, j in enumerate(os.listdir(f'{args.iam_path}')):
            class_dict[j] = i

        transforms = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])

        with open(args.gt_train, 'r', encoding='utf-8-sig') as f:
            train_data = [line.rstrip('\r\n') for line in f if line.strip()]

        wr_dict = {}
        full_dict = {}
        wr_index = 0

        for line in train_data:
            first_field, transcription = line.split(' ', 1)
            s_id, image = first_field.split(',', 1)
            if s_id not in wr_dict:
                wr_dict[s_id] = wr_index
                wr_index += 1
            full_dict[len(full_dict)] = {'image': image, 's_id': s_id, 'label': transcription}

        print('number of training samples', len(full_dict))
        print('number of train writer styles', len(wr_dict))
        style_classes = len(wr_dict)

        with open('writers_dict_train.json', 'w', encoding='utf-8') as f:
            json.dump(wr_dict, f, ensure_ascii=False)

        train_ds = IAMDataset(full_dict, args.iam_path, wr_dict, args, transforms=transforms)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, args=args, max_seq_len=OUTPUT_MAX_LEN).to(args.device)    
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    if args.latent==True:
        print('Latent is true - Working on latent space')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = vae.to(args.device)
        
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        print('Latent is false - Working on pixel space')
        vae = None
    
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        unet.load_state_dict(checkpoint['model'])
        ema_model.load_state_dict(checkpoint['ema_model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['completed_epochs']
        print(f"Resuming from completed epoch {start_epoch}")

    train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, style_classes, vocab_size, transforms, args, start_epoch=start_epoch)



if __name__ == "__main__":
    main()
  
  
