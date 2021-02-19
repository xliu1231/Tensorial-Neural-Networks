import torch

# \todo it's actually easier to read if I change this to
#       mikl_ijkl_to_mijl_bar_l
#       which is the same as
#       ijkl_jmkl_to_ijml_bar_l # is this right?
#       because then the batch indices appear in the right order
# \todo I should probably switch the order of the inputs to stick to convention
# Supports padding_mode - 'max_zeros', 'max_circular', 'zeros', 'reflect', 'replicate' or 'circular'
# If 'max_zeros' or 'max_circular' is chosen, then whatever is passed to padding will be ignored
def ijkl_mikl_to_mijl_bar_l(kernel, input_tens, max_mode_size=None, \
                            padding_mode='zeros', padding=0, stride=1, dilation=1, bias=False):
    # this is supposed to compute the most general, up to reshaping / permuting,
    # convolutional einsum with 1 convolution index which is computable by Conv1d alone.
    # This is the order the indices must appear in so that the operation can be done
    # without any calls to permute.

    # By the Conv1d documentation https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html
    # input = (batch_size = M, input_channel = groups*K = I*K, input_size = input_tens.size(3))
    # weight = (out_channels = groups * J = I * J, in_channels/groups = K, kernel_size = kernel.size(3))
    # output = (batch_size = M, out_channels = groups * J = I * J, conv_len = max(input_size, kernel_size))
    # and it appears the indices are laid out as (so, after reshaping)
    # input: M, I, K, L
    # weight: I, J, K, L
    # output: M, I, J, L

    batch_size = input_tens.size(0)
    kernel_size = kernel.size(3)

    input_size = input_tens.size(3) # image_size is perhaps a better name
    conv_len = max(kernel_size, input_size)
    if max_mode_size != None:
        conv_len = max(conv_len, max_mode_size)# \todo

    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)


    m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, \
                        kernel_size=kernel_size, \
                        stride=stride, padding=padding, padding_mode=padding_mode, \
                        dilation=dilation, \
                        groups=groups, bias=bias)
    kernel_flipped = torch.flip(kernel, [3])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size)

    output = m(input_tens.reshape(batch_size, in_channels, input_size))
    out_shape = (batch_size, groups, out_channels//groups, conv_len//stride)

    # \todo Cutting the mode down only makes sense for a max padding
    #       or if they're passing the maximum expected size
    return torch.reshape(output.data[:,:,:(conv_len//stride)], out_shape)



def ijklm_niklm_to_nijlm_bar_lm(kernel, input_tens, max_mode_sizes=None, \
                               padding_mode='zeros', \
                               padding=0, stride=1, dilation=1, bias=False):

    # This is supposed to compute the most general, up to reshaping / permuting,
    # convolutional einsum with 2 convolution indices which is computable by Conv2d alone.

    batch_size = input_tens.size(0)
    kernel_size = kernel.size()[3:5]
    input_size = input_tens.size()[3:5]

    max_h = max(kernel_size[0], input_size[0])
    max_w = max(kernel_size[1], input_size[1])
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)


    m = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
                        kernel_size=kernel_size, stride=stride, \
                        padding=padding, padding_mode=padding_mode, \
                        dilation=dilation, groups=groups, bias=bias)

    kernel_flipped = torch.flip(kernel, [3,4])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1])

    output = m(input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1]))

    try:
        stride[0]
    except TypeError:
        stride = [stride]*2

    out_shape = (batch_size, groups, out_channels//groups, max_h//stride[0], max_w//stride[1])
    return torch.reshape(output.data[:,:,:(max_h//stride[0]),:(max_w//stride[1])], out_shape)



def ijklmn_oiklmn_to_oijlmn_bar_lmn(kernel, input_tens, max_mode_sizes=0, \
                                    padding_mode='zeros', \
                                    padding=0, stride=1, dilation=1, bias=False):
    # i.e "ijklmn, oiklmn -> oijlmn | lmn"
    # This is supposed to compute the most general, up to reshaping / permuting, convolutional einsum
    # with 3 convolution indices, which is computable by Conv3d alone.
    # The index order is what's produced by Conv3d, without permuting.

    batch_size = input_tens.size(0)
    kernel_size = kernel.size()[3:6]
    input_size = input_tens.size()[3:6]
    max_d = max(kernel_size[0], input_size[0])
    max_h = max(kernel_size[1], input_size[1])
    max_w = max(kernel_size[2], input_size[2])
    groups = kernel.size(0)
    in_channels = groups*kernel.size(2)
    out_channels = groups*kernel.size(1)


    m = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, \
                        stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation, groups=groups, bias=bias)

    kernel_flipped = torch.flip(kernel, [3,4,5])
    m.weight.data = kernel_flipped.view(out_channels, in_channels//groups, kernel_size[0], kernel_size[1], kernel_size[2])

    output = m(input_tens.reshape(batch_size, in_channels, input_size[0], input_size[1], input_size[2]))

    try:
        stride[0]
    except TypeError:
        stride = [stride]*3

    out_shape = (batch_size, groups, out_channels//groups, max_d//stride[0], max_h//stride[1], max_w//stride[2])
    return torch.reshape(output.data[:,:,:(max_d//stride[0]),:(max_h//stride[1]),:(max_w//stride[2])], out_shape)

# \todo Should swap the order of kernel / input_tens to adhere to convention
# \todo I want these functions to not do any special padding. That should be done in
#       a higher level function, so
#
def convolution_atomic_operation(kernel, input_tens, num_convolutions, max_mode_sizes, \
                                 padding_mode='max_zeros', padding=0, stride=1, dilation=1, \
                                 bias=False):
    # This operation expects the inputs input_tens and kernel to be shaped/permuted according to
    # the "atomic forms" given by the following functions
    # note the convolution indices always appear at the end


    if padding_mode == 'max_zeros':
        kernel_size = kernel.size()[3:len(kernel.size())]
        input_size = input_tens.size()[3:len(input_tens.size())]
        padding = max_zeros_padding_nd(kernel_size, input_size, \
                                       max_mode_size=max_mode_sizes, \
                                       stride=stride, dilation=dilation)

        padding_mode = 'zeros'
    elif padding_mode == 'max_circular':
        #padding = ??
        print("padding_mode == max_circular not implemented")
        padding_mode = 'circular'

    if num_convolutions == 0:
        print("Error: convolution_atomic_operation expects at least one convolution index")
    elif num_convolutions == 1:
        stride = stride[0]
        dilation = dilation[0]
        padding = padding[0]
        max_mode_size = max_mode_sizes[0]

        return ijkl_mikl_to_mijl_bar_l(kernel, input_tens, max_mode_size=max_mode_size, \
                                       padding_mode=padding_mode, padding=padding, \
                                       stride=stride, dilation=dilation, bias=bias)
    elif num_convolutions == 2:
        return ijklm_niklm_to_nijlm_bar_lm(kernel, input_tens, max_mode_sizes=max_mode_sizes, \
                                           padding_mode=padding_mode, padding=padding, \
                                           stride=stride, dilation=dilation, bias=bias)
    elif num_convolutions == 3:
        return ijklmn_oiklmn_to_oijlmn_bar_lmn(kernel, input_tens, \
                                               max_mode_sizes=max_mode_sizes, \
                                               padding_mode=padding_mode, padding=padding, \
                                               stride=stride, dilation=dilation, bias=bias)
    else:
        print("convolution_atomic_operation num_convolutions >= 4 not implemented")
