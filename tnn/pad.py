import math

def padding_mode_zeros_L_out(kernel_size, input_size, padding, stride, dilation):
    # This is from the documentation for Conv1d
    # It returns the expected output length for a given convolution with
    # padding_mode='zeros'
    return math.floor(input_size + (2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

###
# This computes the zero padding to append to the input image vector so that
# the resulting convolution has the same length as the maximum length of the kernel
# and the input tensor. Note the kernel can be longer than the input tensor. Additionally,
# conv_einsum("i, i -> i | i", A, B, padding=) = conv_einsum("i, i -> i | i", B, A, padding=)
# when this padding is selecting. These two properties may not hold when dilation > 1.
# \todo
#
def max_zeros_padding_1d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):
    # see https://pytorch.org/docs/master/generated/torch.nn.Conv1d.html for the definition
    # under the Shape section for the definition of the padding. We add 1 so the integer division
    # by 2 errs large instead of small
    # for 1d convolutions ker_mode_size < input_mode_size but this is not true for higher
    # dimensions, and I use this function to compute those paddings

    # "input" tensor is synonymous with "image" tensor in much of this

    max_ker_input = max(ker_mode_size, input_mode_size, max_mode_size)

    # \todo it's not clear if this works if stride doesn't divide evenly into
    #           input_mode_size + 2*padding - dilation * (kernel_size - 1) - 1
    #       (see Conv1d documentation)
    #       It does however appear to work whenever dilation = 1, or when
    #       dilation*kernel_size < input_size + some factor involving stride
    # padding can only be negative if kernel_mode_size == 0
    twice_padding = (max_ker_input-1)*stride + 1 + dilation*(ker_mode_size-1) - input_mode_size

    return (twice_padding+1)//2 # add 1 so its never less than twice_padding/2


def max_zeros_padding_2d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):
    return (max_zeros_padding_1d(ker_mode_size[0], input_mode_size[0], \
                                 max_mode_size[0], stride[0], dilation[0]), \
            max_zeros_padding_1d(ker_mode_size[1], input_mode_size[1], \
                                 max_mode_size[1], stride[1], dilation[1]))

def max_zeros_padding_3d(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):

    return (max_zeros_padding_1d(ker_mode_size[0], input_mode_size[0], \
                                 max_mode_size[0], stride, dilation), \
            max_zeros_padding_1d(ker_mode_size[1], input_mode_size[1], \
                                 max_mode_size[1], stride[1], dilation[1]),
            max_zeros_padding_1d(ker_mode_size[2], input_mode_size[2], \
                                 max_mode_size[2], stride[2], dilation[2]))


def max_zeros_padding_nd(ker_mode_size, input_mode_size, \
                         max_mode_size, stride=1, dilation=1):
    return tuple(max_zeros_padding_1d(ker_mode_size[i], input_mode_size[i], \
                                   max_mode_size[i], stride[i], dilation[i]) \
                 for i in range(0,len(ker_mode_size)))


# \todo this function is not fully implemented
#       I'm not sure if we really need to support nontrivial hyperparameter cases though
# \todo How to tell the user to specify the maximum mode size of an N way convolution
#        to input_mode_size
def max_circular_padding_1d(ker_mode_size, input_mode_size, \
                            stride=1, dilation=1):


    # the formulas in this function were determined by experimentally computing the first pad
    # which did not cause a run time error
    # \todo These formulas should be justified rigorously
    if stride != 1:
        # the padding seems to be more complicated in this case
        print("Error: stride != 1, case not implemented")

    if dilation == 1:
        ker_img_ratio = (ker_mode_size + 1) / (input_mode_size)

        if ker_img_ratio >= 2:
            print("Error: stride == 1, dilation == 1, ker+1 >= 2*img \
                   implies no pad is possible, and \
                   proper handling of this case is not yet implemented")
        elif 1 < ker_img_ratio and ker_img_ratio < 2:
            return 2*ker_mode_size - input_mode_size - 1
        else:
            return ker_mode_size
    else:
        # \todo This formula is not actually correct (at least for dilation == 6)...
        #       I'm not sure if this is a bug with pytorch (meaning no possible formula
        #       will work), or if there's another factor I'm missing
         if ker_mode_size > input_mode_size:
             print("Error: dilation > 2 and ker_mode_size > input_mode_size \
                    implies no pad is possible, and \
                    proper handling of this case is not yet implemented")
         else:
             return dilation*(ker_mode_size-1)





def atomic_pad(tensor, padding):

    if len(tensor.size()) == 4:
        # ijkl_mikl_to_mijl_bar_l   ("ijkl, mikl -> mijl | l")
        # in this case we have to append 0s to the last mode
        #tensor = torch.cat(torch.cat(tensor
        print("temp")
    elif len(tensor.size()) == 5:
        # ijklm_niklm_to_nijlm_bar_lm    ("ijklm, niklm -> nijlm | lm")
        print("temp")
    elif len(tensor.size()) == 6:
        # ijklmn_oiklmn_to_oijlmn_bar_lmn   ("ijklmn, oiklmn -> oijlmn | lmn")
        print("temp")
    else:
        print("Error: atomic_circular_pad tensor order not implemented")
