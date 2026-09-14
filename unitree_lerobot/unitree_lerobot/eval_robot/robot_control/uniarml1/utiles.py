import math
from .constants import (
        BAUDRATE,
        HEAD_TX,
        HEAD_RX,
        RX_LEN,
        GEAR_RATIO,
        K_POS,
        K_SPEED,
        K_TORQUE,
        K_KP,
        K_KD,
        K_VOLTAGE,
        OUT_POS_RES,
        ERROR_MAP,
    )


def OutPosRaw2OutPos_rad(outpos: int) -> float:
    """Convert output shaft raw value to radians"""
    return (outpos / OUT_POS_RES) * 2.0 * math.pi

def rad2deg(rad: float) -> float:
    """Convert radians to degrees"""
    return rad * (180.0 / math.pi)
def deg2rad(deg: float) -> float:
    """Convert degrees to radians"""
    return deg * (math.pi / 180.0)

def OutPos_rad2OutPosRaw(outpos_rad: float) -> int:
    """Convert output shaft radians value to raw value"""
    return int((outpos_rad / (2.0 * math.pi)) * OUT_POS_RES)

