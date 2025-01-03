import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Criterion(nn.Module):
    """
    Batch-based classifcation loss
    """
    def __init__(self):
        super(Criterion, self).__init__()
    
    def forward(self, scores):
        return F.cross_entropy(
            scores, 
            torch.arange(scores.shape[0]).long().to(scores.device)
        )


class Combiner(nn.Module):
    """ TODO: Combiner module, which fuses textual and visual information.
    Given an image feature and a text feature, you should fuse them to get a fused feature. The dimension of the fused feature should be embed_dim.
    Hint: You can concatenate image and text features and feed them to a FC layer, or you can devise your own fusion module, e.g., add, multiply, or attention, to achieve a higher retrieval score.
    """
    def __init__(self, vision_feature_dim, text_feature_dim, embed_dim):
        super(Combiner, self).__init__()

        self.fc = nn.Linear(vision_feature_dim + text_feature_dim, embed_dim)

    def forward(self, image_features, text_features):
        #Write code for fused_features using image and text features, you can also use the init function.
        combined = torch.concat([image_features, text_features], dim=1)
        fused_features = self.fc(combined)
        return fused_features


class Model(nn.Module):
    """
    CLIP-based Composed Image Retrieval Model.
    """
    def __init__(self, vision_feature_dim, text_feature_dim, embed_dim):
        super(Model, self).__init__()
        self.vision_feature_dim = vision_feature_dim
        self.text_feature_dim = text_feature_dim
        self.embed_dim = embed_dim

        # Load clip model and freeze its parameters
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.combiner = Combiner(vision_feature_dim, text_feature_dim, embed_dim)
    
    def train(self):
        self.combiner.train()

    def eval(self):
        self.combiner.eval()
    
    def encode_image(self, image_paths):
        """ TODO: Encode images to get image features by the vision encoder of clip model. See https://github.com/openai/CLIP
        Note: The clip model has loaded in the __init__() function. You do not need to create and load it on your own.

        Args:
            Image_paths (list[str]): a list of image paths.
        
        Returns:
            vision_features (torch.Tensor): image features.
        """
        imgs = [
            self.preprocess(Image.open(img_path)) for img_path in image_paths
        ]

        img_input = torch.stack(imgs).to(device)
        vision_features = self.clip_model.encode_image(img_input)

        return vision_features.float() # Convert to float32 data type

    def encode_text(self, texts):
        """ TODO: Encode texts to get text features by the text encoder of clip model. See https://github.com/openai/CLIP
        Note: The clip model has loaded in the __init__() function. You do not need to create and load it on your own.

        Args:
            texts (list[str]): a list of captions.
        
        Returns:
            text_features (torch.Tensor): text features.
        """
        tokenized_texts = clip.tokenize(texts).to(device)
        text_features = self.clip_model.encode_text(tokenized_texts)

        return text_features.float() # Convert to float32 data type

    def inference(self, ref_image_paths, texts):
        with torch.no_grad():
            ref_vision_features = self.encode_image(ref_image_paths)
            text_features = self.encode_text(texts)
            fused_features = self.combiner(ref_vision_features, text_features)
        return fused_features
    
    def forward(self, ref_image_paths, texts, tgt_image_paths):
        """
        Args:
            ref_image_paths (list[str]): image paths of reference images.
            texts (list[str]): captions.
            tgt_image_paths (list[str]): image paths of reference images.
        
        Returns:
            scores (torch.Tensor): score matrix with shape batch_size * batch_size.
        """
        batch_size = len(ref_image_paths)

        # Extract vision and text features
        with torch.no_grad():
            ref_vision_features = self.encode_image(ref_image_paths)
            tgt_vision_features = self.encode_image(tgt_image_paths)
            text_features = self.encode_text(texts)
        assert ref_vision_features.shape == torch.Size([batch_size, self.vision_feature_dim])
        assert tgt_vision_features.shape == torch.Size([batch_size, self.vision_feature_dim])
        assert text_features.shape == torch.Size([batch_size, self.text_feature_dim])

        # Fuse vision and text features 
        fused_features = self.combiner(ref_vision_features, text_features)
        assert fused_features.shape == torch.Size([batch_size, self.embed_dim])

        # L2 norm
        fused_features = F.normalize(fused_features)
        tgt_vision_features = F.normalize(tgt_vision_features)

        # Calculate scores
        scores = self.temperature.exp() * fused_features @ tgt_vision_features.t()
        assert scores.shape == torch.Size([batch_size, batch_size])

        return scores

# Training function
def train(data_loader, model, criterion, optimizer, log_step=15):
    model.train()
    for i, (_, ref_img_paths, tgt_img_paths, raw_captions) in enumerate(data_loader):
        
        # TODO: Implement a training loop. You should clean gradients, calculate loss values, call the backpropagation algorithm, 
        # and call the optimizer to update the model, see https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

        optimizer.zero_grad()
        scores = model(ref_img_paths, raw_captions, tgt_img_paths)
        loss = criterion(scores)
        loss.backward()
        optimizer.step()

        if i % log_step == 0:
            print("training loss: {:.3f}".format(loss.item()))

# Validation function
def eval_batch(data_loader, model, ranker):
    model.eval()
    ranker.update_emb(model)
    rankings = []
    for meta_info, ref_img_paths, _, raw_captions in data_loader:
        with torch.no_grad():
            fused_features = model.inference(ref_img_paths, raw_captions)
            target_asins = [ meta_info[m]['target'] for m in range(len(meta_info)) ]
            rankings.append(ranker.compute_rank(fused_features, target_asins))
    metrics = {}
    rankings = torch.cat(rankings, dim=0)
    metrics['score'] = 1 - rankings.mean().item() / ranker.data_emb.size(0)
    model.train()
    return metrics

def val(data_loader, model, ranker, best_score):
    model.eval()
    metrics = eval_batch(data_loader, model, ranker)
    dev_score = metrics['score']
    best_score = max(best_score, dev_score)
    print('-' * 77)
    print('| score {:8.5f} / {:8.5f} '.format(dev_score, best_score))
    print('-' * 77)
    print('best_dev_score: {}'.format(best_score))
    return best_score