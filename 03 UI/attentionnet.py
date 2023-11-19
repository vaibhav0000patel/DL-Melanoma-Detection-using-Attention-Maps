import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes

class AttentionNetwork(torch.nn.Module):

    def __init__(self, AttentionNet=None, AttentionFtrExtractor=None, GlobalNet=None, num_classes=2, size=224, ftr_size=512):
        super(AttentionNetwork, self).__init__()

        self.Attention_net = AttentionNet

        self.Attention_ftr_extrcator = AttentionFtrExtractor

        self.global_net = GlobalNet

        self.aggregate_net = nn.Sequential(nn.Linear(ftr_size*2, ftr_size),
                             nn.Sigmoid(),
                             nn.Linear(ftr_size, num_classes))

        self.size = size

        self.Attention_FC = nn.Sequential(nn.Linear(ftr_size, 4), nn.Sigmoid())
        self.classifier_FC = nn.Linear(ftr_size, num_classes)

    def draw_attention_map(self, img, x_min, y_min, x_max, y_max):
        images = []
        for i in range(img.shape[0]):
            img_uint8 = torch.round(255*img[i]).to(torch.uint8)
            bbox1 = [x_min[i].item(), y_min[i].item(), x_max[i].item(), y_max[i].item()]
            bbox = [bbox1]
            bbox = torch.tensor(bbox, dtype=torch.int)
            img_with_rect=draw_bounding_boxes(img_uint8, bbox,width=1,colors=[(255,0,0)],fill =False,font_size=5)
            img_with_rect = (img_with_rect / 255).to(torch.float32)
            images.append(img_with_rect)

        return images

    def crop_zoom(self, image, tx, ty, tl_x, tl_y):
        tx_r = (self.size * tx).int() # real tx (since 0 <= tx <= 1)
        ty_r = (self.size * ty).int()

        tl_x_r = ((self.size / 2) * tl_x + 1).int() # should be at least 1 pixel
        tl_y_r = ((self.size / 2) * tl_y + 1).int()

        x_min = (tx_r - tl_x_r).clamp(min=0)
        x_max = (tx_r + tl_x_r).clamp(max=self.size)
        y_min = (ty_r - tl_y_r).clamp(min=0)
        y_max = (ty_r + tl_y_r).clamp(max=self.size)
        imgs_with_rect = self.draw_attention_map(image, x_min, y_min, x_max, y_max)
        for i in range(image.shape[0]):
          img = image[i][:, y_min[i]:y_max[i], x_min[i]:x_max[i]]
          img = F.interpolate(img[None, :, :, :], (224, 224), mode='bilinear')[0]
          image[i] = img

        return image, imgs_with_rect

    def aggregate(self, vec1, vec2):
        return torch.cat((vec1, vec2), dim=1)

    def forward(self, image):
        # Apply attention, crop, and zoom
        vec = self.Attention_net(image)
        tx, ty, tl_x, tl_y = self.Attention_FC(vec).transpose(0,1)
        x_cropped, img_with_rect = self.crop_zoom(image, tx, ty, tl_x, tl_y)
        # Local and Global features
        ftr_vec_local = self.Attention_ftr_extrcator(x_cropped)
        ftr_vec_global = self.global_net(image)
        # aggregate local and global features and process them
        ftr_vec_final = self.aggregate(ftr_vec_local, ftr_vec_global)
        probs = self.aggregate_net(ftr_vec_final) #

        return probs, img_with_rect
