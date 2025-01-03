import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm 


class Fusion(nn.Module):
    def __init__(self, input_dim, out_dim, num_layers):
        super(Fusion, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)

            hv.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.image_transformation_layers = nn.ModuleList(hv)
        #
        hq = []
        for i in range(self.num_layers):
            do = nn.Dropout(p=0.5)
            lin = nn.Linear(input_dim, out_dim)
            hq.append(nn.Sequential(do, lin, nn.Tanh()))
        #
        self.ques_transformation_layers = nn.ModuleList(hq)

    def forward(self, ques_emb, img_emb):
        # TODO: At each fusion layer, we transform the question and image embeddings
        # to a common embedding space, and the fusion is implemented by multipling
        # question and image embeddings.
        res = []
        x_q = ques_emb
        x_i = img_emb

        for layer in range(self.num_layers):
            # TODO: Transform question embeddings at each layer by ques_transformation_layers
            x_q = self.ques_transformation_layers[layer](x_q)
            assert x_q.shape[-1] == 1024

            # TODO: Transform image embeddings at each layer by image_transformation_layers
            x_i = self.image_transformation_layers[layer](x_i)
            assert x_i.shape[-1] == 1024

            # TODO: Fuse quesion and image embeddings by multipling
            # see: https://pytorch.org/docs/stable/generated/torch.mul.html#torch.mul
            fused_embed = x_q * x_i
            assert fused_embed.shape[-1] == 1024

            res.append(fused_embed)
        
        # TODO: Convert the list res to tensor type
        # see: https://pytorch.org/docs/stable/generated/torch.stack.html
        res = torch.stack(res)
        res = res.transpose(0, 1)
        assert list(res.shape[1:]) == [self.num_layers, 1024]

        res = res.sum(dim=1).view(img_emb.size(0), self.out_dim)
        res = F.tanh(res)
        return res


class Normalize(nn.Module):
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        # Pdb().set_trace()
        x = x / x.norm(p=self.p, dim=1, keepdim=True)
        return x


class ImageEmbedding(nn.Module):
    def __init__(self, image_channel_type='I', output_size=1024, mode='train',
                 extract_features=False, features_dir=None):
        super(ImageEmbedding, self).__init__()
        self.extractor = models.vgg16(pretrained=True)
        # freeze feature extractor (VGGNet) parameters
        for param in self.extractor.parameters():
            param.requires_grad = False

        extactor_fc_layers = list(self.extractor.classifier.children())[:-1]
        if image_channel_type.lower() == 'normi':
            extactor_fc_layers.append(Normalize(p=2))
        self.extractor.classifier = nn.Sequential(*extactor_fc_layers)

        self.fflayer = nn.Sequential(
            nn.Linear(4096, output_size),
            nn.Tanh())

        self.mode = mode
        self.extract_features = extract_features
        self.features_dir = features_dir

    def forward(self, image):
        # Pdb().set_trace()
        if not self.extract_features:
            image = self.extractor(image)
            # if self.features_dir is not None:
            #     utils.save_image_features(image, image_ids, self.features_dir)

        image_embedding = self.fflayer(image)
        return image_embedding


class QuesEmbedding(nn.Module):
    def __init__(self, input_size=300, hidden_size=512, output_size=1024, num_layers=2, batch_first=True):
        super(QuesEmbedding, self).__init__()
        self.bidirectional = True
        if num_layers == 1:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                batch_first=batch_first, bidirectional=self.bidirectional)

            if self.bidirectional:
                self.fflayer = nn.Sequential(
                    nn.Linear(2 * num_layers * hidden_size, output_size),
                    nn.Tanh())
        else:
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                num_layers=num_layers, batch_first=batch_first)
            self.fflayer = nn.Sequential(
                nn.Linear(2 * num_layers * hidden_size, output_size),
                nn.Tanh())

    def forward(self, ques):
        _, hx = self.lstm(ques)
        lstm_embedding = torch.cat([hx[0], hx[1]], dim=2)
        ques_embedding = lstm_embedding[0]
        if self.lstm.num_layers > 1 or self.bidirectional:
            for i in range(1, self.lstm.num_layers):
                ques_embedding = torch.cat(
                    [ques_embedding, lstm_embedding[i]], dim=1)
            ques_embedding = self.fflayer(ques_embedding)
        return ques_embedding


class VQAModel(nn.Module):

    def __init__(self, vocab_size=10000, word_emb_size=300, emb_size=1024, output_size=1000, image_channel_type='I', ques_channel_type='lstm', use_mutan=True, mode='train', extract_img_features=True, features_dir=None):
        super(VQAModel, self).__init__()
        self.mode = mode
        self.word_emb_size = word_emb_size
        self.image_encoder = ImageEmbedding(image_channel_type, output_size=emb_size, mode=mode,
                                            extract_features=extract_img_features, features_dir=features_dir)

        # NOTE the padding_idx below.
        self.word_embeddings = nn.Embedding(vocab_size, word_emb_size)

        if ques_channel_type.lower() == 'lstm':
            self.question_encoder = QuesEmbedding(
                input_size=word_emb_size, output_size=emb_size, num_layers=1, batch_first=False)
        elif ques_channel_type.lower() == 'deeplstm':
            self.question_encoder = QuesEmbedding(
                input_size=word_emb_size, output_size=emb_size, num_layers=2, batch_first=False)
        else:
            msg = 'ques channel type not specified. please choose one of -  lstm or deeplstm'
            print(msg)
            raise Exception(msg)

        self.fusion_module = Fusion(emb_size, emb_size, 5)
        self.mlp = nn.Sequential(nn.Linear(emb_size, output_size))

    def forward(self, images, questions, image_ids):
        # TODO: Extract images embeddings
        img_embeds = self.image_encoder(images)

        # TODO: Extract questions embeddings
        question_word_embeds = self.word_embeddings(questions)
        question_embeds = self.question_encoder(question_word_embeds)

        # TODO: Fuse image and question embeddings and pass them through MLP
        fused_embeds = self.fusion_module(question_embeds, img_embeds)

        outputs = self.mlp(fused_embeds)
        return outputs


def train(model, dataloader, optimizer, use_gpu=False):
    # TODO: Define a classification criterion to optimize the model
    # see: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    step = 0

    total_examples = len(dataloader.dataset)
    
    # Iterate over data.
    for questions, images, image_ids, answers, ques_ids in dataloader:
        # print('questions size: ', questions.size())
        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(0, 1), Variable(images), Variable(answers)

        # zero grad
        optimizer.zero_grad()
        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)

        # TODO: Calculate the loss by criterion
        loss = criterion(ans_scores, answers)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)
        step += 1
        if step % 50 == 0:
            print('({}/{}) - running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
                example_count, total_examples,
                running_loss / example_count, running_corrects, example_count, (float(running_corrects) / example_count) * 100))
        # if step * batch_size == 40000:
        #     break
    loss = running_loss / example_count
    acc = (running_corrects / len(dataloader.dataset)) * 100
    print('Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                           acc, running_corrects, example_count))
    return loss, acc



def validate(model, dataloader, use_gpu=False):
    # TODO: Define a classification criterion to optimize the model
    # see: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()

    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    # Iterate over data.
    for questions, images, image_ids, answers, ques_ids in dataloader:
        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(
            ), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(
            0, 1), Variable(images), Variable(answers)

        # zero grad
        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)

        # TODO: Calculate the loss by criterion
        loss = criterion(ans_scores, answers)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)
    loss = running_loss / example_count
    # acc = (running_corrects / example_count) * 100
    acc = (running_corrects / len(dataloader.dataset)) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                                acc, running_corrects, example_count))
    return loss, acc