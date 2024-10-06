'''
    Module contains final Model and all pieces of it.
'''
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer

class ImageEncoder(nn.Module):
    '''
        Encodes image and returns it's embedding.
    '''

    def __init__(self, model, device='cpu'):
        super(ImageEncoder, self).__init__()
        
        self.device = device

        self.preprocessor = CLIPProcessor.from_pretrained(model)
        self.model = CLIPModel.from_pretrained(model).vision_model.to(self.device)

    def forward(self, image):
        # only one image at a time
        image = self.preprocessor(images=image, return_tensors='pt').to(self.device)
        image_features = self.model(**image)
        
        return image_features.pooler_output

class Mapping(nn.Module):
    '''
        Maps image embedding to GPT-2 embedding.
    '''
    
    def __init__(
        self, 
        ep_len,
        num_layers,
        embed_size, 
        n_heads, 
        forward_expansion, 
        dropout, 
        device='cpu'
    ):
        super(Mapping, self).__init__()
        
        self.ep_len = ep_len
        self.embed_size = embed_size

        self.device = device

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_size, 
                nhead=n_heads, 
                dim_feedforward=embed_size*forward_expansion, 
                dropout=dropout, 
                batch_first=True, 
                device=device
            ),
            num_layers=num_layers
        ).to(self.device)

        self.mapping = nn.Linear(embed_size, ep_len * embed_size).to(self.device)

        self.init_weights()

    def forward(self, img_embedded, train_mode=False):
        x = self.transformer_encoder(img_embedded)

        print("after transformer: ", x.shape)
        x = self.mapping(x)

        print("after linear: ", x.shape)

        x = x.view(
            *(
                [-1, self.ep_len, self.embed_size] 
                if train_mode else 
                [self.ep_len, self.embed_size]
            )
        ) # for batched input

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class TextDecoder(nn.Module):
    '''
        Processes embedding into caption.
    '''

    def __init__(self, model, device='cpu'):
        super(TextDecoder, self).__init__()
        
        self.device = device
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2LMHeadModel.from_pretrained(model).to(self.device)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, embedding, attention_mask=None):
        text_features = self.model(inputs_embeds=embedding, attention_mask=attention_mask)
        
        return text_features.logits

class CaptioningModel(nn.Module):
    '''
        Final Model class. Puts all pieces together and generates caption based on image.
    '''
    
    def __init__(self, clip_model, text_model, ep_len, num_layers, n_heads, forward_expansion, dropout, max_len, device='cpu'):
        '''
            Model constructor.
            Args:
                num_layers: number of layers in the TransformerEncoder
                n_heads: number of heads in the MultiHeadAttention
                forward_expansion: expansion factor for the feedforward layer
                dropout: dropout probability
                max_len: maximum length of the generated text
        '''
        super(CaptioningModel, self).__init__()

        self.device = device
        self.ep_len = ep_len

        self.image_encoder = ImageEncoder(model=clip_model, device=device)
        self.mapping = Mapping(ep_len=self.ep_len, num_layers=num_layers, embed_size=self.image_encoder.model.config.hidden_size, n_heads=n_heads, forward_expansion=forward_expansion, dropout=dropout, device=device)
        self.text_decoder = TextDecoder(model=text_model, device=device)
        
        assert self.image_encoder.model.config.hidden_size == self.text_decoder.model.config.n_embd, "Embedding size of models mismatch"

        self.max_len = max_len

        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.text_decoder.tokenizer.pad_token_id) # chanded on epoch 91
        self.criterion = nn.CrossEntropyLoss()

        self.freeze_layers()

    def freeze_layers(self):
        for p in [*list(self.image_encoder.parameters()), *list(self.text_decoder.parameters())[14:-14]]: # freeze everything, except 1st and last transformer layer in Decoder
            p.requires_grad = False

    def forward(self, img_embedded, temperature=1.0):
        '''
            Caption generation for a single image.
            Args:
                img_embedded: Image features.
            Returns:
                caption: generated caption [str]
                tokens: generated tokens [torch.Tensor]
        '''

        if temperature <= 0.0:
            temperature = 1.0
            print('Temperature must be positive. Setting it to 1.0')

        with torch.no_grad():
            # TODO: Map image to text space. See class Mapping in this file.
            img_mapped = self.mapping(img_embedded)

            B, E, D = img_mapped.shape

            print("shape of img_mapped: ", img_mapped.shape)


            # TODO: Obtain <BOS> token embedding. You should get the <BOS> token ID from the tokenizer of GPT-2. Then, the <BOS> token ID is converted to an embedding by an Embedding layer named "wte" in GPT-2.
            # See https://huggingface.co/docs/transformers/model_doc/gpt2.
            # Adjust the dimensions/shape of the bos token embeddding and then, concatenate the <BOS> embedding and mapped image embedding to get a start embedding.
            
            tokenizer_output = self.text_decoder.tokenizer('<|endoftext|>', return_tensors='pt')
            bos_input_id = tokenizer_output['input_ids'][0][0]

            gpt2_token_emb_layer = self.text_decoder.model.transformer.wte
            bos_emb = gpt2_token_emb_layer(bos_input_id) 

            print("shape of bos_emb: ", bos_emb.shape)

            # expanding bos embedding to match other dimensions of img embeddings
            bos_emb = bos_emb.expand(img_mapped.shape[0], 1, bos_emb.shape[0])

            start_emb = torch.concat([img_mapped, bos_emb], dim=1)

            return

            tokens = []
            for _ in range(self.max_len):
                if len(tokens):
                    tok_emb = self.text_decoder.model.transformer.wte(torch.tensor(tokens).to(self.device))

                    emb = torch.cat([start_emb, tok_emb], dim=0)
                else:
                    emb = start_emb

                # TODO: Obtain position embedding and add it into token embedding. The position embedding is stored as an Embedding layer named "wpe" in GPT-2.
                # See https://huggingface.co/docs/transformers/model_doc/gpt2.
                # pos_emb = # obtain position embedding
                # emb = # add position embedding to token embedding

                # TODO: Generate the next token. See https://huggingface.co/docs/transformers/model_doc/gpt2.
                # pred = # generate the next token

                pred = torch.softmax(pred / temperature, dim=-1)
                _, pred = torch.max(pred, dim=1)
                last_token = pred[-1].item()
                tokens.append(last_token)
                if last_token == self.text_decoder.tokenizer.eos_token_id:
                    break
            
            decoded = self.text_decoder.tokenizer.decode(tokens[:-1])
            
            decoded = decoded.strip()
            if not decoded:
                return None, None
            decoded = decoded[0].upper() + decoded[1:]

            return decoded, tokens

    def train_forward(self, img_emb, trg_cap, att_mask):
        # method should get embedded by CLIP images and trg_text without last token.
        # dataset should contain image, embedded image, text
        
        x, x_mask = trg_cap[:, :-1], att_mask[:, :-1]
        y = trg_cap[:, 1:]

        # TODO: Map image to text space. See class Mapping in this file.
        # Now we are training the model, so the train_model parameter should be set to True
        img_mapped = self.mapping(img_emb, train_mode=True) # N, E, embed_size

        # N, len, embed_size
        text_emb = self.text_decoder.model.transformer.wte(x)

        x = torch.concat([img_mapped, text_emb], dim=1)
        x_mask = torch.concat([torch.ones(x_mask.shape[0], self.ep_len).to(self.device), x_mask], dim=1)
        
        # TODO: Obtain position embedding and add it into token embedding. The position embedding is stored as an Embedding layer named "wpe" in GPT-2.
        # See https://huggingface.co/docs/transformers/model_doc/gpt2.
        
        position_ids = torch.arange(0, x.shape[1], dtype=torch.long)
        pos_emb = self.text_decoder.model.transformer.wpe(position_ids) 
        pos_emb = pos_emb.expand_as(x)
        x = x + pos_emb


        # TODO: Generate tokens. Note that it is slightly different from codes in forward() function because of attention masks.
        # See https://huggingface.co/docs/transformers/model_doc/gpt2.
        # res = # generate tokens

        res = torch.softmax(res, dim=2)
        res = res[:, self.ep_len:, :]

        # TODO: Calculate loss value, fill arguments inside the below function. 
        loss = self.criterion() # cross-entropy loss
        
        return loss
