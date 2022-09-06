import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class ToTensor(transforms.ToTensor):
    def __call__(self, pic, bbo, seg):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)