# Copyright (c) 2024 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from .base_oracle import BaseOracle, StandardOracle, Standard1DOracle, Standard2DOracle
from .base_constrained_oracle import StandardConstrainedOracle
from .base_object_oracle import BaseObjectOracle
from .ackley import Ackley, Ackley3D, Ackley4D
from .branin_hoo import BraninHoo
from .dkt_test_function import DKTOracleType
from .eggholder import Eggholder
from .exponential_2d_extended import Exponential2DExtended
from .exponential_2d import Exponential2D
from .flexible_oracle import Flexible1DOracle, Flexible2DOracle, FlexibleOracle
from .gp_model_bic_oracle import GPModelBICOracle
from .gp_model_cv_oracle import GPModelCVOracle
from .gp_model_evidence_oracle import GPModelEvidenceOracle
from .gp_oracle_1d import GPOracle1D
from .gp_oracle_2d import GPOracle2D
from .gp_oracle_hd import GPOracleHD
from .gp_oracle_from_data import GPOracleFromData
from .griewangk import Griewangk
from .hartmann3 import Hartmann3
from .hartmann6 import Hartmann6
from .high_dim_additive import HighDimAdditive
from .mnist_svm import MnistSVM
from .mnist_svm_2d import MnistSVM2d
from .mogp_oracle_1d import MOGP1DOracle
from .mogp_oracle_2d import MOGP2DOracle
from .mogp_oracle import MOGPOracle
from .piece_wise_linear import PieceWiseLinear
from .piece_wise_linear import FunctionType as PWLFunctionType
#from .regularized_gp_oracle import RegularizedGPOracle # this Oracle import nonexist kernel configs, please fix
from .rosenbrock import Rosenbrock, Rosenbrock3D, Rosenbrock4D
from .safe_test_func import SafeTestFunc
from .safe_test_func import FunctionType as STFFunctionType
from .sinus import Sinus
from .lsq import LSQMain, LSQConstraint1, LSQConstraint2, LSQ
from .simionescu import SimionescuMain, SimionescuConstraint, Simionescu
from .townsend import TownsendMain, TownsendConstraint, Townsend
from .cart_pole import CartPole, CartPoleConstrained
from .swing_up import SwingUp, SwingUpConstrained
from .gas_transmission_compressor_design import GasTransmissionCompressorDesignMain, GasTransmissionCompressorDesignConstraint, GasTransmissionCompressorDesign
from .high_pressure_fluid_system import HighPressureFluidSystemMain, HighPressureFluidSystem, HighPressureFluidSystemReduce, HighPressureFluidSystemReduceGPInterpolated

from .helpers.normalize_decorator import OracleNormalizer
from .helpers.constrained_sampler import ConstrainedSampler

__all__=[
    "BaseOracle",
    "BaseObjectOracle",
    "StandardOracle",
    "Standard1DOracle",
    "Standard2DOracle",
    "StandardConstrainedOracle",
    "OracleNormalizer",
    "ConstrainedSampler",

    "Ackley", "Ackley3D", "Ackley4D",
    "BraninHoo",
    "DKTOracleType",
    "Eggholder",
    "Exponential2DExtended",
    "Exponential2D",
    "Flexible1DOracle",
    "Flexible2DOracle",
    "FlexibleOracle",
    "GPModelBICOracle",
    "GPModelCVOracle",
    "GPModelEvidenceOracle",
    "GPOracle1D",
    "GPOracle2D",
    "GPOracleHD",
    "GPOracleFromData",
    "Griewangk",
    "Hartmann3",
    "Hartmann6",
    "HighDimAdditive",
    "MnistSVM"
    "MnistSVM2d",
    "MOGP1DOracle",
    "MOGP2DOracle",
    "MOGPOracle",
    "PieceWiseLinear", "PWLFunctionType",
    #"RegularizedGPOracle", # this Oracle import nonexist kernel configs, please fix
    "Rosenbrock", "Rosenbrock3D", "Rosenbrock4D",
    "SafeTestFunc", "STFFunctionType",
    "Sinus",
    "LSQMain", "LSQConstraint1", "LSQConstraint2", "LSQ",
    "SimionescuMain", "SimionescuConstraint", "Simionescu",
    "TownsendMain", "TownsendConstraint", "Townsend",
    "CartPole", "CartPoleConstrained",
    "SwingUp", "SwingUpConstrained",
    "HighPressureFluidSystemMain", "HighPressureFluidSystem", "HighPressureFluidSystemReduce", "HighPressureFluidSystemReduceGPInterpolated",
    "GasTransmissionCompressorDesignMain", "GasTransmissionCompressorDesignConstraint", "GasTransmissionCompressorDesign"
]