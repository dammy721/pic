### MiDaSを用いた深度推定
import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from midas.midas_net import MidasNet
from midas.transforms import Resize, NormalizeImage, PrepareForNet

class DepthEstimator:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MidasNet(model_path, non_negative=True).to(self.device)
        self.model.eval()
        self.transform = Compose([
            Resize(
                384, 
                384, 
                resize_target=None, 
                keep_aspect_ratio=True, 
                ensure_multiple_of=32, 
                resize_method="upper_bound", 
                image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            ToTensor()
        ])

    def estimate_depth(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = self.transform({"image": image})["image"].to(self.device)
        
        with torch.no_grad():
            depth = self.model(input_tensor.unsqueeze(0))

        depth = depth.squeeze().cpu().numpy()
        depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

        return depth

# 使用例
depth_estimator = DepthEstimator("path_to_midas_v2_model.pt")
depth_image = depth_estimator.estimate_depth("path_to_your_image.jpg")
cv2.imshow("Depth Image", depth_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
