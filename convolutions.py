import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pytorch sanity check
class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        kernel = torch.ones([3,3,3,3]) #  in_c out_c kernel_size_h kerner_size_w
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  # pytorch weights
 
    def forward(self, x, stride):
        x = F.conv2d(x, self.weight, padding=1, stride=stride)
        return x

# navie conv
def conv_naive(img, kernel, s):
    h, w, c = img.shape
    in_c, out_c, kh, kw =  kernel.shape
    p_h, p_w = kh // 2, kw // 2

    img_pad = np.pad(img, ((p_h, p_h), (p_w, p_w), (0, 0))) # padding
    out_h, out_w = (h + 2*p_h - kh) // s + 1, (w + 2*p_w - kh) // s + 1 # output size
    out_img  = np.zeros([out_h, out_w, c]) # output placeholder

    # slide window sum
    for i in range(out_h):
        for j in range(out_w):
            for k in range(out_c):
                re_kernel = np.transpose(kernel[:, k, :, :], (1,2,0)) # channel position should be same as input image
                out_img[i, j, k] = np.sum(img_pad[i*s:i*s + kh, j*s:j*s+kw, :] * re_kernel)
    return out_img

# matrix conv
def conv_developed(img, kernel, s):
    h, w, c = img.shape
    in_c, out_c, kh, kw =  kernel.shape
    p_h, p_w = kh // 2, kw // 2

    img_pad = np.pad(img, ((p_h, p_h), (p_w, p_w), (0, 0))) # padding
    out_h, out_w = (h + 2*p_h - kh) // s + 1, (w + 2*p_w - kh) // s + 1 # output size 
    input_matrix = np.zeros([out_h * out_w, kh*kw*c]) # image placeholder
    kernel_matrix = np.zeros([kh*kw*c, out_c]) # kernel  placeholder

    # slide window reshape image (h,w,c) to (N, kh*kw*c ), N is the number of slide windows
    for i in range(out_h):
        for j in range(out_w):
            input_matrix[i*out_w + j, :] = np.reshape(img_pad[i*s:i*s + kh, j*s:j*s+kw, :], [-1, kh*kw*c])# fill image placeholder.
    
    for k in range(out_c):
        kernel_matrix[:, k] = kernel[:, k, :, :].reshape(kh*kw*c) # fill kernel placeholder

    res = np.matmul(input_matrix, kernel_matrix) # matrix multiply (N, kh*kw*c) * (kh*kw*c, out_c) =  (N, out_c)
    res = np.reshape(res, [out_h, out_w, out_c]) # reshape to image size out_h, out_w, out_c
    return res    


import time
kernel = np.ones([3, 3, 3, 3])
stride = 2
img = np.ones([36,36,3])
start = time.time()
res_my_d =  conv_developed(img, kernel, stride)
end = time.time()
print('Developed conv time: %.5f' % (end - start))

img = np.ones([36,36,3])
start = time.time()
res_my_n =  conv_naive(img, kernel, stride)
end = time.time()
print('Naive conv time: %.5f' %(end - start))

img_tensor = torch.tensor(np.transpose(img, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
conv_pytorch = Conv()
res_torch = conv_pytorch(img_tensor, stride)

print('Naive error: %.5f' % np.sum(res_my_n - np.transpose(res_torch[0].numpy(), (1,2,0))))
print('Developed error: %.5f' % np.sum(res_my_d - np.transpose(res_torch[0].numpy(), (1,2,0))))
