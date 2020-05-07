"""Losses folder.
Contains functions to calculate training loss on batches of images.
"""

from .binary_crossentropy import binary_crossentropy
from .categorical_crossentropy import categorical_crossentropy
from .dice_coefficient import dice_coef, dice_coef_loss
from .f1_score import f1_score, f1_score_loss
from .l2_norm import l2_norm, f1_l2_combined_loss
