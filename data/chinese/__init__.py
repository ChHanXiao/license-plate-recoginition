'''
Date: 2022-01-05 22:53:32
Author: ChHanXiao
Github: https://github.com/ChHanXiao
LastEditors: ChHanXiao
LastEditTime: 2022-01-09 14:52:35
FilePath: /license-plate-recoginition/data/chinese/__init__.py
'''
from . import black_plate
from . import blue_plate
from . import yellow_plate
from . import green_plate
from . import white_plate
from . import farm_plate
from . import airport_plate
from .black_plate import Black_Type
from .blue_plate import Blue_Type
from .yellow_plate import Yellow_Type
from .green_plate import Green_Type
from .white_plate import White_Type
from .farm_plate import Farm_Type
from .airport_plate import Airport_Type

__all__ = [
    "black_plate", "Black_Type",
    "blue_plate", "Blue_Type",
    "yellow_plate", "Yellow_Type",
    "green_plate", "Green_Type",
    "white_plate", "White_Type",
    "farm_plate", "Farm_Type",
    "airport_plate", "Airport_Type",
    "PlateCommon"
    ]
