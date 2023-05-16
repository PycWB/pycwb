from PIL import Image
import numpy as np


def resize_resolution(input, dt, df, dt_target, df_target):
    input_width, input_height = input.shape
    output_height = int(input_height * df / df_target)
    output_width = int(input_width * dt / dt_target)
    output = Image.fromarray(input.astype(np.double)).resize((output_height, output_width),
                                                             resample=Image.Resampling.NEAREST)
    return output


def resize_image(image, max_width, max_height, t_offset, f_offset=0):
    blank_image = Image.new("L", (max_width, max_height), 0)
    blank_image.paste(image, (f_offset, t_offset))
    return blank_image


def align_images(resized_maps, t_starts_shifted):
    # calculate shift
    dt_starts_shifted = np.array(t_starts_shifted) - min(t_starts_shifted)

    max_width = max([m.size[0] for m in resized_maps])
    max_height = max([m.size[1] for m in resized_maps])

    aligned_images = [resize_image(m, max_width, max_height, dt_starts_shifted[i]) for i, m in enumerate(resized_maps)]
    return aligned_images


def merge_images(images):
    return np.sum([np.array(img, dtype=np.double) for img in images], axis=0)


# def resize_resolution_cv(input, dt, df, dt_target, df_target):
#     import cv2
#
#     input_height, input_width = input.shape
#
#     output_height = int(input_height * df / df_target)
#     output_width = int(input_width * dt / dt_target)
#
#     output = cv2.resize(input, (output_width, output_height), interpolation=cv2.INTER_NEAREST)
#
#     return output
