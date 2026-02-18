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

from alef.configs.acquisition_functions import (
    BasicRandomConfig,
    BasicNosafeRandomConfig,
    BasicSafeRandomConfig,
    BasicPredVarianceConfig,
    BasicPredEntropyConfig,
    BasicPredSigmaConfig,
    BasicSafePredEntropyConfig,
    BasicSafePredEntropyAllConfig,
    BasicSafeDiscountPredEntropyConfig,
    BasicSafeDiscountPredEntropyAllConfig,
    BasicMinUnsafePredEntropyConfig,
    MinUnsafePredEntropyLambda01Config,
    MinUnsafePredEntropyLambda05Config,
    MinUnsafePredEntropyLambda09Config,
    MinUnsafePredEntropyLambda2Config,
    MinUnsafePredEntropyLambda3Config,
    MinUnsafePredEntropyLambda4Config,
    MinUnsafePredEntropyLambda5Config,
    MinUnsafePredEntropyLambda10Config,
    MinUnsafePredEntropyLambda100Config,
    BasicMinUnsafePredEntropyAllConfig,
    BasicSafeDiscoverConfig,
    BasicSafeDiscoverQuantileConfig,
    BasicSafeDiscoverEIConfig,
    BasicSafeDiscoverQuantileEIConfig,
    BasicSafeOptConfig,
    BasicSafeGPUCBConfig,
    BasicEIConfig,
    BasicSafeEIConfig,
    BasicSafeDiscoverOptConfig,
    BasicSafeDiscoverOptQuantileConfig,
)
from alef.configs.acquisition_functions.bo_acquisition_functions.gp_ucb_config import BasicGPUCBConfig
from alef.configs.acquisition_functions.bo_acquisition_functions.integrated_ei_config import BasicIntegratedEIConfig
from alef.configs.active_learners.oracle_active_learner_configs import (
    PredEntropyOracleActiveLearnerConfig,
    PredVarOracleActiveLearnerConfig,
    PredSigmaOracleActiveLearnerConfig,
    RandomOracleActiveLearnerConfig,
)
from alef.configs.active_learners.oracle_policy_active_learner_configs import BasicOraclePolicyActiveLearnerConfig
from alef.configs.active_learners.pool_policy_active_learner_configs import BasicPoolPolicyActiveLearnerConfig
from alef.configs.active_learners.oracle_policy_safe_active_learner_configs import BasicOraclePolicySafeActiveLearnerConfig
from alef.configs.active_learners.pool_policy_safe_active_learner_configs import BasicPoolPolicySafeActiveLearnerConfig
from alef.configs.bayesian_optimization.bayesian_optimizer_configs import (
    BOExpectedImprovementConfig,
    BOGPUCBConfig,
    BOIntegratedExpectedImprovementConfig,
)
from alef.configs.bayesian_optimization.bayesian_optimizer_objects_configs import (
    ObjectBOExpectedImprovementConfig,
    ObjectBOExpectedImprovementEAConfig,
    ObjectBOExpectedImprovementEAFewerStepsConfig,
    ObjectBOExpectedImprovementEAFlatWideConfig,
    ObjectBOExpectedImprovementPerSecondConfig,
    ObjectBOExpectedImprovementPerSecondEAConfig,
)
from alef.configs.bayesian_optimization.greedy_kernel_search_configs import (
    BaseGreedyKernelSearchConfig,
    GreedyKernelSearchBaseInitialConfig,
    GreedyKernelSearchNumNeighboursLimitedConfig,
)
from alef.configs.bayesian_optimization.treeGEP_optimizer_configs import (
    TreeGEPEvolutionaryOptimizerConfig,
    TreeGEPEvolutionaryOptimizerSmallPopulationConfig,
)
from alef.configs.kernels.grammar_tree_kernel_kernel_configs import (
    KernelGrammarSubtreeKernelConfig,
    OTWeightedDimsExtendedGrammarKernelConfig,
    OTWeightedDimsExtendedKernelWithHyperpriorConfig,
    OTWeightedDimsInvarianceGrammarKernelConfig,
    OptimalTransportGrammarKernelConfig,
    TreeBasedOTGrammarKernelConfig,
)
from alef.configs.kernels.hellinger_kernel_kernel_configs import BasicHellingerKernelKernelConfig, HellingerKernelKernelSobolVirtualPoints
from alef.configs.kernels.kernel_grammar_generators.cks_high_dim_generator_config import CKSHighDimGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import (
    CKSWithRQGeneratorConfig,
    CKSWithRQTimeSeriesGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.compositional_kernel_search_configs import (
    CKSTimeSeriesGeneratorConfig,
    CompositionalKernelSearchGeneratorConfig,
)
from alef.configs.kernels.kernel_grammar_generators.dynamic_hhk_generator_config import DynamicHHKGeneratorConfig
from alef.configs.kernels.kernel_grammar_generators.local_kernel_search_generator_config import (
    BigLocalNDimFullKernelsGrammarGeneratorConfig,
    FlatLocalKernelSearchSpaceConfig,
)
from alef.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import (
    BigNDimFullKernelsGrammarGeneratorConfig,
    NDimFullKernelsGrammarGeneratorConfig,
)
from alef.configs.kernels.matern32_configs import BasicMatern32Config, Matern32WithPriorConfig
from alef.configs.kernels.rational_quadratic_configs import BasicRQConfig, RQWithPriorConfig
from alef.configs.kernels.spectral_mixture_kernel_config import BasicSMKernelConfig
from alef.configs.kernels.weighted_additive_kernel_config import BasicWeightedAdditiveKernelConfig, WeightedAdditiveKernelWithPriorConfig
from alef.configs.models.ahgp_model_config import AHGPModelConfig
from alef.configs.models.gp_model_amortized_ensemble_config import ExperimentalAmortizedEnsembleConfig
from alef.configs.models.gp_model_amortized_structured_config import ExperimentalAmortizedStructuredConfig
from alef.configs.means import (
    BaseMeanConfig,
    BasicZeroMeanConfig,
    BasicLinearMeanConfig,
    BasicQuadraticMeanConfig,
    BasicPeriodicMeanConfig,
    BasicSechMeanConfig,
)
from alef.configs.means.pytorch_means import (
    BaseMeanPytorchConfig,
    BasicZeroMeanPytorchConfig,
    BasicLinearMeanPytorchConfig,
    BasicQuadraticMeanPytorchConfig,
    BasicPeriodicMeanPytorchConfig,
    BasicSechMeanPytorchConfig,
)
from alef.configs.models.gp_model_config import (
    BasicGPModelConfig,
    GPModelExtenseOptimization,
    GPModelFastConfig,
    GPModelFixedNoiseConfig,
    GPModelSmallPertubationConfig,
    GPModelWithNoisePriorConfig,
)
from alef.configs.models.gp_model_kernel_search_config import (
    GPFlatLocalKernelSearchConfig,
    GPKernelSearchCKSwithHighDim,
    GPKernelSearchCKSwithHighDimEvidence,
    GPKernelSearchCKSwithRQ,
    GPKernelSearchCKSwithRQEvidence,
)
from alef.configs.models.gp_model_marginalized_config import (
    BasicGPModelMarginalizedConfig,
    GPModelMarginalizedConfigMoreThinningConfig,
    GPModelMarginalizedConfigMoreSamplesConfig,
    GPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
)
from alef.configs.models.gp_model_mixture_config import BasicGPModelMixtureConfig
from alef.configs.models.gp_model_marginalized_config import GPModelMarginalizedConfigMAPInitialized, GPModelMarginalizedConfigFast
from alef.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from alef.configs.kernels.hhk_configs import HHKEightLocalDefaultConfig, HHKFourLocalDefaultConfig, HHKTwoLocalDefaultConfig
from alef.configs.kernels.wami_configs import BasicWamiConfig
from alef.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from alef.configs.kernels.rbf_for_cartpole_configs import (
    CartpoleRBFConfig,
    CartpoleRBFSafetyConfig,
    CartpoleReduceRBFConfig,
    CartpoleReduceRBFSafetyConfig,
)
from alef.configs.models.svgp_model_config import BasicSVGPConfig
from alef.configs.active_learners.pool_active_learner_configs import (
    PredEntropyPoolActiveLearnerConfig,
    PredVarPoolActiveLearnerConfig,
    PredSigmaPoolActiveLearnerConfig,
    RandomPoolActiveLearnerConfig,
)
from alef.configs.models.mogp_model_config import BasicMOGPModelConfig
from alef.configs.active_learners.pool_active_learner_batch_configs import (
    EntropyBatchPoolActiveLearnerConfig,
    RandomBatchPoolActiveLearnerConfig,
)
from alef.configs.kernels.additive_kernel_configs import BasicAdditiveKernelConfig, AdditiveKernelWithPriorConfig
from alef.configs.kernels.matern52_configs import BasicMatern52Config, Matern52WithPriorConfig
from alef.configs.kernels.matern52_for_cartpole_configs import (
    CartpoleMatern52Config,
    CartpoleMatern52SafetyConfig,
    CartpoleReduceMatern52Config,
    CartpoleReduceMatern52SafetyConfig,
)
from alef.configs.kernels.linear_configs import BasicLinearConfig, LinearWithPriorConfig
from alef.configs.kernels.neural_kernel_network_config import BasicNKNConfig
from alef.configs.kernels.multi_output_kernels.coregionalization_1latent_kernel_configs import BasicCoregionalization1LConfig, Coregionalization1LWithPriorConfig
from alef.configs.kernels.multi_output_kernels.coregionalization_Platent_kernel_configs import BasicCoregionalizationPLConfig, CoregionalizationPLWithPriorConfig
from alef.configs.kernels.multi_output_kernels.coregionalization_kernel_configs import (
    BasicCoregionalizationSOConfig,
    CoregionalizationSOWithPriorConfig,
    BasicCoregionalizationMOConfig,
    CoregionalizationMOWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.multi_source_additive_kernel_configs import (
    BasicMIAdditiveConfig,
    MIAdditiveWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.flexible_transfer_kernel_config import (
    BasicFlexibleTransferConfig,
    FlexibleTransferWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.coregionalization_transfer_kernel_config import (
    BasicCoregionalizationTransferConfig,
    CoregionalizationTransferWithPriorConfig,
)
from alef.configs.kernels.multi_output_kernels.fpacoh_kernel_config import (
    BasicFPACOHKernelConfig
)
from alef.configs.models.metagp_model_config import BasicMetaGPModelConfig
from alef.configs.models.sparse_gp_model_config import (
    BasicSparseGPModelConfig,
    SparseGPModelFastConfig,
    SparseGPModelFixedNoiseConfig,
    SparseGPModel300IPConfig,
    SparseGPModel500IPConfig,
    SparseGPModel700IPConfig,
    SparseGPModel700IPExtenseConfig,
)
from alef.configs.models.deep_gp_config import DeepGPConfig, FiveLayerDeepGPConfig, ThreeLayerDeepGPConfig
from alef.configs.models.gp_model_scalable_config import (
    BasicScalableGPModelConfig,
    GPRAdamConfig,
    GPRAdamWithValidationSet,
    GPRAdamWithValidationSetNLL,
)
from alef.configs.models.mogp_model_config import BasicMOGPModelConfig
from alef.configs.models.mogp_model_so_config import BasicSOMOGPModelConfig
from alef.configs.models.mogp_sparse_model_so_config import BasicSOMOSparseGPModelConfig
from alef.configs.models.mogp_model_transfer_config import BasicTransferGPModelConfig
from alef.configs.models.mogp_model_so_marginalized_config import (
    BasicSOMOGPModelMarginalizedConfig,
    SOMOGPModelMarginalizedConfigMoreThinningConfig,
    SOMOGPModelMarginalizedConfigMoreSamplesConfig,
    SOMOGPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
    SOMOGPModelMarginalizedConfigMAPInitialized,
    SOMOGPModelMarginalizedConfigFast,
)
from alef.configs.models.gp_model_for_engine1_config import (
    Engine1GPModelBEConfig,
    Engine1GPModelTExConfig,
    Engine1GPModelPI0vConfig,
    Engine1GPModelPI0sConfig,
    Engine1GPModelHCConfig,
    Engine1GPModelNOxConfig,
)
from alef.configs.models.gp_model_for_engine2_config import (
    Engine2GPModelBEConfig,
    Engine2GPModelTExConfig,
    Engine2GPModelPI0vConfig,
    Engine2GPModelPI0sConfig,
    Engine2GPModelHCConfig,
    Engine2GPModelNOxConfig,
)
from alef.configs.models.gp_model_for_cartpole_config import (
    CartpoleGPModelConfig,
    CartpoleGPSafetyModelConfig
)
from alef.configs.models.metagp_model_for_cartpole_config import (
    CartpoleMetaGPModelConfig,
    CartpoleMetaGPSafetyModelConfig
)
from alef.configs.models.mogp_model_transfer_for_cartpole_config import (
    CartpoleTransferGPModelConfig,
    CartpoleTransferGPSafetyModelConfig
)
from alef.configs.models.mogp_model_so_for_cartpole_config import (
    CartpoleMOGPModelConfig,
    CartpoleMOGPSafetyModelConfig
)
from alef.configs.kernels.deep_kernels.invertible_resnet_kernel_configs import (
    BasicInvertibleResnetKernelConfig,
    ExploreRegularizedIResnetKernelConfig,
    InvertibleResnetKernelWithPriorConfig,
    CurlRegularizedIResnetKernelConfig,
    AxisRegularizedIResnetKernelConfig,
    InvertibleResnetWithLayerNoiseKernelConfig,
)
from alef.configs.kernels.deep_kernels.mlp_deep_kernel_config import (
    BasicMLPDeepKernelConfig,
    MLPWithPriorDeepKernelConfig,
    SmallMLPWithPriorDeepKernelConfig,
)
from alef.configs.kernels.kernel_list_configs import (
    SEKernelViaKernelListConfig,
    PERKernelViaKernelListConfig,
    ExperimentalKernelListConfig,
)
from alef.configs.experiment.simulator_configs.base_simulator_config import BaseSimulatorConfig
from alef.configs.experiment.simulator_configs.single_task_1d_illustrate_config import SingleTaskIllustrateConfig
from alef.configs.experiment.simulator_configs.transfer_task_1d_illustrate_config import TransferTaskIllustrateConfig
from alef.configs.experiment.simulator_configs.single_task_branin_config import (
    SingleTaskBraninConfig, SingleTaskBranin0Config, SingleTaskBranin1Config, SingleTaskBranin2Config, SingleTaskBranin3Config, SingleTaskBranin4Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_branin_config import (
    TransferTaskBraninBaseConfig,
    TransferTaskBranin0Config,
    TransferTaskBranin1Config,
    TransferTaskBranin2Config,
    TransferTaskBranin3Config,
    TransferTaskBranin4Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_multi_sources_branin_config import (
    TransferTaskMultiSourcesBraninBaseConfig,
    TransferTaskMultiSourcesBranin0Config,
    TransferTaskMultiSourcesBranin1Config,
    TransferTaskMultiSourcesBranin2Config,
    TransferTaskMultiSourcesBranin3Config,
    TransferTaskMultiSourcesBranin4Config,
)
from alef.configs.experiment.simulator_configs.single_task_hartmann3_config import (
    SingleTaskHartmann3Config, SingleTaskHartmann3_0Config, SingleTaskHartmann3_1Config, SingleTaskHartmann3_2Config, SingleTaskHartmann3_3Config, SingleTaskHartmann3_4Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_hartmann3_config import (
    TransferTaskHartmann3BaseConfig,
    TransferTaskHartmann3_0Config,
    TransferTaskHartmann3_1Config,
    TransferTaskHartmann3_2Config,
    TransferTaskHartmann3_3Config,
    TransferTaskHartmann3_4Config,
)
from alef.configs.experiment.simulator_configs.single_task_hartmann6_config import (
    SingleTaskHartmann6Config, SingleTaskHartmann6_0Config, SingleTaskHartmann6_1Config, SingleTaskHartmann6_2Config, SingleTaskHartmann6_3Config, SingleTaskHartmann6_4Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_hartmann6_config import (
    TransferTaskHartmann6BaseConfig,
    TransferTaskHartmann6_0Config,
    TransferTaskHartmann6_1Config,
    TransferTaskHartmann6_2Config,
    TransferTaskHartmann6_3Config,
    TransferTaskHartmann6_4Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_multi_sources_hartmann6_config import (
    TransferTaskMultiSourcesHartmann6BaseConfig,
    TransferTaskMultiSourcesHartmann6_0Config,
    TransferTaskMultiSourcesHartmann6_1Config,
    TransferTaskMultiSourcesHartmann6_2Config,
    TransferTaskMultiSourcesHartmann6_3Config,
    TransferTaskMultiSourcesHartmann6_4Config,
)
from alef.configs.experiment.simulator_configs.single_task_cartpole_config import SingleTaskCartPoleConfig
from alef.configs.experiment.simulator_configs.transfer_task_cartpole_config import (
    TransferTaskCartPoleConfig,
    TransferTaskCartPole0Config,
    TransferTaskCartPole1Config,
    TransferTaskCartPole2Config,
    TransferTaskCartPole3Config,
    TransferTaskCartPole4Config
)
from alef.configs.experiment.simulator_configs.single_task_mogp1d_config import (
    SingleTaskMOGP1DBaseConfig,
    SingleTaskMOGP1D0Config, SingleTaskMOGP1D1Config, SingleTaskMOGP1D2Config, SingleTaskMOGP1D3Config, SingleTaskMOGP1D4Config,
    SingleTaskMOGP1D5Config, SingleTaskMOGP1D6Config, SingleTaskMOGP1D7Config, SingleTaskMOGP1D8Config, SingleTaskMOGP1D9Config,
    SingleTaskMOGP1D10Config, SingleTaskMOGP1D11Config, SingleTaskMOGP1D12Config, SingleTaskMOGP1D13Config, SingleTaskMOGP1D14Config,
    SingleTaskMOGP1D15Config, SingleTaskMOGP1D16Config, SingleTaskMOGP1D17Config, SingleTaskMOGP1D18Config, SingleTaskMOGP1D19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp1d_config import (
    TransferTaskMOGP1DBaseConfig,
    TransferTaskMOGP1D0Config, TransferTaskMOGP1D1Config, TransferTaskMOGP1D2Config, TransferTaskMOGP1D3Config, TransferTaskMOGP1D4Config,
    TransferTaskMOGP1D5Config, TransferTaskMOGP1D6Config, TransferTaskMOGP1D7Config, TransferTaskMOGP1D8Config, TransferTaskMOGP1D9Config,
    TransferTaskMOGP1D10Config, TransferTaskMOGP1D11Config, TransferTaskMOGP1D12Config, TransferTaskMOGP1D13Config, TransferTaskMOGP1D14Config,
    TransferTaskMOGP1D15Config, TransferTaskMOGP1D16Config, TransferTaskMOGP1D17Config, TransferTaskMOGP1D18Config, TransferTaskMOGP1D19Config,
)
from alef.configs.experiment.simulator_configs.single_task_mogp1dz_config import (
    SingleTaskMOGP1DzBaseConfig,
    SingleTaskMOGP1Dz0Config, SingleTaskMOGP1Dz1Config, SingleTaskMOGP1Dz2Config, SingleTaskMOGP1Dz3Config, SingleTaskMOGP1Dz4Config,
    SingleTaskMOGP1Dz5Config, SingleTaskMOGP1Dz6Config, SingleTaskMOGP1Dz7Config, SingleTaskMOGP1Dz8Config, SingleTaskMOGP1Dz9Config,
    SingleTaskMOGP1Dz10Config, SingleTaskMOGP1Dz11Config, SingleTaskMOGP1Dz12Config, SingleTaskMOGP1Dz13Config, SingleTaskMOGP1Dz14Config,
    SingleTaskMOGP1Dz15Config, SingleTaskMOGP1Dz16Config, SingleTaskMOGP1Dz17Config, SingleTaskMOGP1Dz18Config, SingleTaskMOGP1Dz19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp1dz_config import (
    TransferTaskMOGP1DzBaseConfig,
    TransferTaskMOGP1Dz0Config, TransferTaskMOGP1Dz1Config, TransferTaskMOGP1Dz2Config, TransferTaskMOGP1Dz3Config, TransferTaskMOGP1Dz4Config,
    TransferTaskMOGP1Dz5Config, TransferTaskMOGP1Dz6Config, TransferTaskMOGP1Dz7Config, TransferTaskMOGP1Dz8Config, TransferTaskMOGP1Dz9Config,
    TransferTaskMOGP1Dz10Config, TransferTaskMOGP1Dz11Config, TransferTaskMOGP1Dz12Config, TransferTaskMOGP1Dz13Config, TransferTaskMOGP1Dz14Config,
    TransferTaskMOGP1Dz15Config, TransferTaskMOGP1Dz16Config, TransferTaskMOGP1Dz17Config, TransferTaskMOGP1Dz18Config, TransferTaskMOGP1Dz19Config,
)

from alef.configs.experiment.simulator_configs.single_task_mogp2d_config import (
    SingleTaskMOGP2DBaseConfig,
    SingleTaskMOGP2D0Config, SingleTaskMOGP2D1Config, SingleTaskMOGP2D2Config, SingleTaskMOGP2D3Config, SingleTaskMOGP2D4Config,
    SingleTaskMOGP2D5Config, SingleTaskMOGP2D6Config, SingleTaskMOGP2D7Config, SingleTaskMOGP2D8Config, SingleTaskMOGP2D9Config,
    SingleTaskMOGP2D10Config, SingleTaskMOGP2D11Config, SingleTaskMOGP2D12Config, SingleTaskMOGP2D13Config, SingleTaskMOGP2D14Config,
    SingleTaskMOGP2D15Config, SingleTaskMOGP2D16Config, SingleTaskMOGP2D17Config, SingleTaskMOGP2D18Config, SingleTaskMOGP2D19Config,
)
from alef.configs.experiment.simulator_configs.single_task_mogp4dz_config import (
    SingleTaskMOGP4DzBaseConfig,
    SingleTaskMOGP4Dz0Config, SingleTaskMOGP4Dz1Config, SingleTaskMOGP4Dz2Config, SingleTaskMOGP4Dz3Config, SingleTaskMOGP4Dz4Config,
    SingleTaskMOGP4Dz5Config, SingleTaskMOGP4Dz6Config, SingleTaskMOGP4Dz7Config, SingleTaskMOGP4Dz8Config, SingleTaskMOGP4Dz9Config,
    SingleTaskMOGP4Dz10Config, SingleTaskMOGP4Dz11Config, SingleTaskMOGP4Dz12Config, SingleTaskMOGP4Dz13Config, SingleTaskMOGP4Dz14Config,
    SingleTaskMOGP4Dz15Config, SingleTaskMOGP4Dz16Config, SingleTaskMOGP4Dz17Config, SingleTaskMOGP4Dz18Config, SingleTaskMOGP4Dz19Config,
)
from alef.configs.experiment.simulator_configs.single_task_mogp5dz_config import (
    SingleTaskMOGP5DzBaseConfig,
    SingleTaskMOGP5Dz0Config, SingleTaskMOGP5Dz1Config, SingleTaskMOGP5Dz2Config, SingleTaskMOGP5Dz3Config, SingleTaskMOGP5Dz4Config,
    SingleTaskMOGP5Dz5Config, SingleTaskMOGP5Dz6Config, SingleTaskMOGP5Dz7Config, SingleTaskMOGP5Dz8Config, SingleTaskMOGP5Dz9Config,
    SingleTaskMOGP5Dz10Config, SingleTaskMOGP5Dz11Config, SingleTaskMOGP5Dz12Config, SingleTaskMOGP5Dz13Config, SingleTaskMOGP5Dz14Config,
    SingleTaskMOGP5Dz15Config, SingleTaskMOGP5Dz16Config, SingleTaskMOGP5Dz17Config, SingleTaskMOGP5Dz18Config, SingleTaskMOGP5Dz19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp2d_config import (
    TransferTaskMOGP2DBaseConfig,
    TransferTaskMOGP2D0Config, TransferTaskMOGP2D1Config, TransferTaskMOGP2D2Config, TransferTaskMOGP2D3Config, TransferTaskMOGP2D4Config,
    TransferTaskMOGP2D5Config, TransferTaskMOGP2D6Config, TransferTaskMOGP2D7Config, TransferTaskMOGP2D8Config, TransferTaskMOGP2D9Config,
    TransferTaskMOGP2D10Config, TransferTaskMOGP2D11Config, TransferTaskMOGP2D12Config, TransferTaskMOGP2D13Config, TransferTaskMOGP2D14Config,
    TransferTaskMOGP2D15Config, TransferTaskMOGP2D16Config, TransferTaskMOGP2D17Config, TransferTaskMOGP2D18Config, TransferTaskMOGP2D19Config,
)
from alef.configs.experiment.simulator_configs.single_task_mogp2dz_config import (
    SingleTaskMOGP2DzBaseConfig,
    SingleTaskMOGP2Dz0Config, SingleTaskMOGP2Dz1Config, SingleTaskMOGP2Dz2Config, SingleTaskMOGP2Dz3Config, SingleTaskMOGP2Dz4Config,
    SingleTaskMOGP2Dz5Config, SingleTaskMOGP2Dz6Config, SingleTaskMOGP2Dz7Config, SingleTaskMOGP2Dz8Config, SingleTaskMOGP2Dz9Config,
    SingleTaskMOGP2Dz10Config, SingleTaskMOGP2Dz11Config, SingleTaskMOGP2Dz12Config, SingleTaskMOGP2Dz13Config, SingleTaskMOGP2Dz14Config,
    SingleTaskMOGP2Dz15Config, SingleTaskMOGP2Dz16Config, SingleTaskMOGP2Dz17Config, SingleTaskMOGP2Dz18Config, SingleTaskMOGP2Dz19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp2dz_config import (
    TransferTaskMOGP2DzBaseConfig,
    TransferTaskMOGP2Dz0Config, TransferTaskMOGP2Dz1Config, TransferTaskMOGP2Dz2Config, TransferTaskMOGP2Dz3Config, TransferTaskMOGP2Dz4Config,
    TransferTaskMOGP2Dz5Config, TransferTaskMOGP2Dz6Config, TransferTaskMOGP2Dz7Config, TransferTaskMOGP2Dz8Config, TransferTaskMOGP2Dz9Config,
    TransferTaskMOGP2Dz10Config, TransferTaskMOGP2Dz11Config, TransferTaskMOGP2Dz12Config, TransferTaskMOGP2Dz13Config, TransferTaskMOGP2Dz14Config,
    TransferTaskMOGP2Dz15Config, TransferTaskMOGP2Dz16Config, TransferTaskMOGP2Dz17Config, TransferTaskMOGP2Dz18Config, TransferTaskMOGP2Dz19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp4dz_config import (
    TransferTaskMOGP4DzBaseConfig,
    TransferTaskMOGP4Dz0Config, TransferTaskMOGP4Dz1Config, TransferTaskMOGP4Dz2Config, TransferTaskMOGP4Dz3Config, TransferTaskMOGP4Dz4Config,
    TransferTaskMOGP4Dz5Config, TransferTaskMOGP4Dz6Config, TransferTaskMOGP4Dz7Config, TransferTaskMOGP4Dz8Config, TransferTaskMOGP4Dz9Config,
    TransferTaskMOGP4Dz10Config, TransferTaskMOGP4Dz11Config, TransferTaskMOGP4Dz12Config, TransferTaskMOGP4Dz13Config, TransferTaskMOGP4Dz14Config,
    TransferTaskMOGP4Dz15Config, TransferTaskMOGP4Dz16Config, TransferTaskMOGP4Dz17Config, TransferTaskMOGP4Dz18Config, TransferTaskMOGP4Dz19Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_mogp5dz_config import (
    TransferTaskMOGP5DzBaseConfig,
    TransferTaskMOGP5Dz0Config, TransferTaskMOGP5Dz1Config, TransferTaskMOGP5Dz2Config, TransferTaskMOGP5Dz3Config, TransferTaskMOGP5Dz4Config,
    TransferTaskMOGP5Dz5Config, TransferTaskMOGP5Dz6Config, TransferTaskMOGP5Dz7Config, TransferTaskMOGP5Dz8Config, TransferTaskMOGP5Dz9Config,
    TransferTaskMOGP5Dz10Config, TransferTaskMOGP5Dz11Config, TransferTaskMOGP5Dz12Config, TransferTaskMOGP5Dz13Config, TransferTaskMOGP5Dz14Config,
    TransferTaskMOGP5Dz15Config, TransferTaskMOGP5Dz16Config, TransferTaskMOGP5Dz17Config, TransferTaskMOGP5Dz18Config, TransferTaskMOGP5Dz19Config,
)
from alef.configs.experiment.simulator_configs.single_task_engine_interpolated_config import (
    SingleTaskEngineInterpolatedBaseConfig,
    SingleTaskEngineInterpolated_be_Config,
    SingleTaskEngineInterpolated_TEx_Config,
    SingleTaskEngineInterpolated_PI0v_Config,
    SingleTaskEngineInterpolated_PI0s_Config,
    SingleTaskEngineInterpolated_HC_Config,
    SingleTaskEngineInterpolated_NOx_Config,
)
from alef.configs.experiment.simulator_configs.transfer_task_engine_interpolated_config import (
    TransferTaskEngineInterpolatedBaseConfig,
    TransferTaskEngineInterpolated_be_Config,
    TransferTaskEngineInterpolated_TEx_Config,
    TransferTaskEngineInterpolated_PI0v_Config,
    TransferTaskEngineInterpolated_PI0s_Config,
    TransferTaskEngineInterpolated_HC_Config,
    TransferTaskEngineInterpolated_NOx_Config,
)
from alef.configs.experiment.simulator_configs.single_task_gengine_config import (
    SingleTaskGEngineConfig,
    SingleTaskGEngineTestConfig,
)
from alef.configs.experiment.simulator_configs.transfer_task_gengine_config import (
    TransferTaskGEngineConfig,
    TransferTaskGEngineTestConfig,
)
from alef.configs.models.pfn_config import PFNTorchConfig
from alef.configs.models.pfn_model_config import BasicPFNModelConfig, PFNModelGPUConfig


class ConfigPicker:
    means_configs_dict = {
        c.__name__: c
        for c in [
            BaseMeanConfig,
            BasicZeroMeanConfig,
            BasicLinearMeanConfig,
            BasicQuadraticMeanConfig,
            BasicPeriodicMeanConfig,
            BasicSechMeanConfig,
            #
            BaseMeanPytorchConfig,
            BasicZeroMeanPytorchConfig,
            BasicLinearMeanPytorchConfig,
            BasicQuadraticMeanPytorchConfig,
            BasicPeriodicMeanPytorchConfig,
            BasicSechMeanPytorchConfig,
        ]
    }
    models_configs_dict = {
        c.__name__: c
        for c in [
            BasicGPModelConfig,
            GPModelFastConfig,
            GPModelWithNoisePriorConfig,
            GPModelSmallPertubationConfig,
            GPModelExtenseOptimization,
            GPModelFixedNoiseConfig,
            BasicGPModelMarginalizedConfig,
            GPModelMarginalizedConfigMoreThinningConfig,
            GPModelMarginalizedConfigMoreSamplesConfig,
            GPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
            BasicGPModelMixtureConfig,
            BasicGPModelLaplaceConfig,
            GPModelMarginalizedConfigMAPInitialized,
            GPModelMarginalizedConfigFast,
            BasicSVGPConfig,
            BasicMOGPModelConfig,
            BasicSparseGPModelConfig,
            SparseGPModelFastConfig,
            SparseGPModelFixedNoiseConfig,
            SparseGPModel300IPConfig,
            SparseGPModel500IPConfig,
            SparseGPModel700IPConfig,
            SparseGPModel700IPExtenseConfig,
            DeepGPConfig,
            ThreeLayerDeepGPConfig,
            FiveLayerDeepGPConfig,
            BasicScalableGPModelConfig,
            GPRAdamConfig,
            GPRAdamWithValidationSet,
            GPRAdamWithValidationSetNLL,
            GPKernelSearchCKSwithHighDim,
            GPKernelSearchCKSwithRQ,
            GPKernelSearchCKSwithRQEvidence,
            GPKernelSearchCKSwithHighDimEvidence,
            GPFlatLocalKernelSearchConfig,
            AHGPModelConfig,
            BasicMOGPModelConfig,
            BasicSOMOGPModelConfig,
            BasicSOMOSparseGPModelConfig,
            BasicTransferGPModelConfig,
            BasicSOMOGPModelMarginalizedConfig,
            SOMOGPModelMarginalizedConfigMoreThinningConfig,
            SOMOGPModelMarginalizedConfigMoreSamplesConfig,
            SOMOGPModelMarginalizedConfigMoreSamplesMoreThinningConfig,
            SOMOGPModelMarginalizedConfigMAPInitialized,
            SOMOGPModelMarginalizedConfigFast,
            BasicMetaGPModelConfig,
            Engine1GPModelBEConfig,
            Engine1GPModelTExConfig,
            Engine1GPModelPI0vConfig,
            Engine1GPModelPI0sConfig,
            Engine1GPModelHCConfig,
            Engine1GPModelNOxConfig,
            Engine2GPModelBEConfig,
            Engine2GPModelTExConfig,
            Engine2GPModelPI0vConfig,
            Engine2GPModelPI0sConfig,
            Engine2GPModelHCConfig,
            Engine2GPModelNOxConfig,
            ExperimentalAmortizedStructuredConfig,
            ExperimentalAmortizedEnsembleConfig,
            CartpoleGPModelConfig,
            CartpoleGPSafetyModelConfig,
            CartpoleMetaGPModelConfig,
            CartpoleMetaGPSafetyModelConfig,
            CartpoleTransferGPModelConfig,
            CartpoleTransferGPSafetyModelConfig,
            CartpoleMOGPModelConfig,
            CartpoleMOGPSafetyModelConfig,
            PFNTorchConfig, # this one can be found also in models
            BasicPFNModelConfig,
            PFNModelGPUConfig,
        ]
    }

    kernels_configs_dict = {
        c.__name__: c
        for c in [
            HHKEightLocalDefaultConfig,
            HHKFourLocalDefaultConfig,
            HHKTwoLocalDefaultConfig,
            BasicWamiConfig,
            RBFWithPriorConfig,
            BasicRBFConfig,
            CartpoleRBFConfig,
            CartpoleRBFSafetyConfig,
            CartpoleReduceRBFConfig,
            CartpoleReduceRBFSafetyConfig,
            BasicAdditiveKernelConfig,
            AdditiveKernelWithPriorConfig,
            BasicMatern52Config,
            Matern52WithPriorConfig,
            CartpoleMatern52Config,
            CartpoleMatern52SafetyConfig,
            CartpoleReduceMatern52Config,
            CartpoleReduceMatern52SafetyConfig,
            BasicMatern32Config,
            Matern32WithPriorConfig,
            BasicLinearConfig,
            LinearWithPriorConfig,
            BasicRQConfig,
            RQWithPriorConfig,
            BasicNKNConfig,
            BasicMLPDeepKernelConfig,
            MLPWithPriorDeepKernelConfig,
            SmallMLPWithPriorDeepKernelConfig,
            BasicInvertibleResnetKernelConfig,
            InvertibleResnetKernelWithPriorConfig,
            InvertibleResnetWithLayerNoiseKernelConfig,
            CurlRegularizedIResnetKernelConfig,
            AxisRegularizedIResnetKernelConfig,
            KernelGrammarSubtreeKernelConfig,
            ExploreRegularizedIResnetKernelConfig,
            BasicHellingerKernelKernelConfig,
            HellingerKernelKernelSobolVirtualPoints,
            OptimalTransportGrammarKernelConfig,
            TreeBasedOTGrammarKernelConfig,
            OTWeightedDimsExtendedGrammarKernelConfig,
            OTWeightedDimsInvarianceGrammarKernelConfig,
            OTWeightedDimsExtendedKernelWithHyperpriorConfig,
            BasicWeightedAdditiveKernelConfig,
            WeightedAdditiveKernelWithPriorConfig,  #
            BasicSMKernelConfig,
            BasicCoregionalization1LConfig,
            Coregionalization1LWithPriorConfig,
            BasicCoregionalizationPLConfig,
            CoregionalizationPLWithPriorConfig,
            BasicCoregionalizationSOConfig,
            CoregionalizationSOWithPriorConfig,
            BasicCoregionalizationMOConfig,
            CoregionalizationMOWithPriorConfig,
            BasicMIAdditiveConfig,
            MIAdditiveWithPriorConfig,
            BasicFlexibleTransferConfig,
            FlexibleTransferWithPriorConfig,
            BasicCoregionalizationTransferConfig,
            CoregionalizationTransferWithPriorConfig,
            BasicFPACOHKernelConfig,
            SEKernelViaKernelListConfig,
            PERKernelViaKernelListConfig,
            ExperimentalKernelListConfig,
            PFNTorchConfig, # this one can be found also in models
        ]
    }

    acquisition_function_configs_dict = {
        c.__name__: c
        for c in [
            BasicRandomConfig,
            BasicNosafeRandomConfig,
            BasicSafeRandomConfig,
            BasicPredVarianceConfig,
            BasicPredSigmaConfig,
            BasicPredEntropyConfig,
            BasicSafePredEntropyConfig,
            BasicSafePredEntropyAllConfig,
            BasicSafeDiscountPredEntropyConfig,
            BasicSafeDiscountPredEntropyAllConfig,
            BasicMinUnsafePredEntropyConfig,
            MinUnsafePredEntropyLambda01Config,
            MinUnsafePredEntropyLambda05Config,
            MinUnsafePredEntropyLambda09Config,
            MinUnsafePredEntropyLambda2Config,
            MinUnsafePredEntropyLambda3Config,
            MinUnsafePredEntropyLambda4Config,
            MinUnsafePredEntropyLambda5Config,
            MinUnsafePredEntropyLambda10Config,
            MinUnsafePredEntropyLambda100Config,
            BasicMinUnsafePredEntropyAllConfig,
            BasicSafeDiscoverConfig,
            BasicSafeDiscoverQuantileConfig,
            BasicSafeDiscoverEIConfig,
            BasicSafeDiscoverQuantileEIConfig,
            BasicSafeOptConfig,
            BasicSafeGPUCBConfig,
            BasicEIConfig,
            BasicGPUCBConfig,
            BasicIntegratedEIConfig,
            BasicSafeEIConfig,
            BasicSafeDiscoverOptConfig,
            BasicSafeDiscoverOptQuantileConfig,
        ]
    }

    active_learner_configs_dict = {
        c.__name__: c
        for c in [
            PredEntropyPoolActiveLearnerConfig,
            PredVarPoolActiveLearnerConfig,
            PredSigmaPoolActiveLearnerConfig,
            RandomPoolActiveLearnerConfig,
            PredEntropyOracleActiveLearnerConfig,
            PredVarOracleActiveLearnerConfig,
            PredSigmaOracleActiveLearnerConfig,
            RandomOracleActiveLearnerConfig,
            BasicOraclePolicyActiveLearnerConfig,
            BasicPoolPolicyActiveLearnerConfig,
            BasicOraclePolicySafeActiveLearnerConfig,
            BasicPoolPolicySafeActiveLearnerConfig,
        ]
    }

    batch_active_learner_configs_dict = {
        c.__name__: c for c in [
            EntropyBatchPoolActiveLearnerConfig,
            RandomBatchPoolActiveLearnerConfig
        ]
    }

    bayesian_optimization_configs_dict = {
        c.__name__: c
        for c in [
            BOExpectedImprovementConfig,
            BOGPUCBConfig,
            BOIntegratedExpectedImprovementConfig,
            ObjectBOExpectedImprovementConfig,
            ObjectBOExpectedImprovementEAConfig,
            ObjectBOExpectedImprovementEAFewerStepsConfig,
            ObjectBOExpectedImprovementEAFlatWideConfig,
            ObjectBOExpectedImprovementPerSecondEAConfig,
            ObjectBOExpectedImprovementPerSecondConfig,
        ]
    }

    greedy_kernel_seach_configs_dict = {
        c.__name__: c
        for c in [
            GreedyKernelSearchNumNeighboursLimitedConfig,
            BaseGreedyKernelSearchConfig,
            TreeGEPEvolutionaryOptimizerConfig,
            TreeGEPEvolutionaryOptimizerSmallPopulationConfig,
            GreedyKernelSearchBaseInitialConfig,
        ]
    }

    kernel_grammar_generator_configs_dict = {
        c.__name__: c
        for c in [
            NDimFullKernelsGrammarGeneratorConfig,
            CompositionalKernelSearchGeneratorConfig,
            CKSTimeSeriesGeneratorConfig,
            CKSWithRQGeneratorConfig,
            CKSWithRQTimeSeriesGeneratorConfig,
            CKSHighDimGeneratorConfig,
            DynamicHHKGeneratorConfig,
            FlatLocalKernelSearchSpaceConfig,
            BigLocalNDimFullKernelsGrammarGeneratorConfig,
            BigNDimFullKernelsGrammarGeneratorConfig,
        ]
    }

    experiment_simulator_configs_dict = {
        c.__name__: c
        for c in [
            BaseSimulatorConfig,
            SingleTaskIllustrateConfig,
            TransferTaskIllustrateConfig,
            SingleTaskBraninConfig,
            SingleTaskBranin0Config, SingleTaskBranin1Config, SingleTaskBranin2Config, SingleTaskBranin3Config, SingleTaskBranin4Config,
            TransferTaskBraninBaseConfig,
            TransferTaskBranin0Config, TransferTaskBranin1Config, TransferTaskBranin2Config, TransferTaskBranin3Config, TransferTaskBranin4Config,
            TransferTaskMultiSourcesBranin0Config, TransferTaskMultiSourcesBranin1Config, TransferTaskMultiSourcesBranin2Config, TransferTaskMultiSourcesBranin3Config, TransferTaskMultiSourcesBranin4Config,
            SingleTaskHartmann3Config,
            SingleTaskHartmann3_0Config, SingleTaskHartmann3_1Config, SingleTaskHartmann3_2Config, SingleTaskHartmann3_3Config, SingleTaskHartmann3_4Config,
            TransferTaskHartmann3BaseConfig,
            TransferTaskHartmann3_0Config, TransferTaskHartmann3_1Config, TransferTaskHartmann3_2Config, TransferTaskHartmann3_3Config, TransferTaskHartmann3_4Config,
            SingleTaskHartmann6Config,
            SingleTaskHartmann6_0Config, SingleTaskHartmann6_1Config, SingleTaskHartmann6_2Config, SingleTaskHartmann6_3Config, SingleTaskHartmann6_4Config,
            TransferTaskHartmann6BaseConfig,
            TransferTaskHartmann6_0Config, TransferTaskHartmann6_1Config, TransferTaskHartmann6_2Config, TransferTaskHartmann6_3Config, TransferTaskHartmann6_4Config,
            TransferTaskMultiSourcesHartmann6BaseConfig,
            TransferTaskMultiSourcesHartmann6_0Config, TransferTaskMultiSourcesHartmann6_1Config, TransferTaskMultiSourcesHartmann6_2Config, TransferTaskMultiSourcesHartmann6_3Config, TransferTaskMultiSourcesHartmann6_4Config,
            SingleTaskCartPoleConfig,
            TransferTaskCartPoleConfig,
            TransferTaskCartPole0Config, TransferTaskCartPole1Config, TransferTaskCartPole2Config, TransferTaskCartPole3Config, TransferTaskCartPole4Config,
            SingleTaskMOGP1DBaseConfig,
            SingleTaskMOGP1D0Config, SingleTaskMOGP1D1Config, SingleTaskMOGP1D2Config, SingleTaskMOGP1D3Config, SingleTaskMOGP1D4Config,
            SingleTaskMOGP1D5Config, SingleTaskMOGP1D6Config, SingleTaskMOGP1D7Config, SingleTaskMOGP1D8Config, SingleTaskMOGP1D9Config,
            SingleTaskMOGP1D10Config, SingleTaskMOGP1D11Config, SingleTaskMOGP1D12Config, SingleTaskMOGP1D13Config, SingleTaskMOGP1D14Config,
            SingleTaskMOGP1D15Config, SingleTaskMOGP1D16Config, SingleTaskMOGP1D17Config, SingleTaskMOGP1D18Config, SingleTaskMOGP1D19Config,
            TransferTaskMOGP1DBaseConfig,
            TransferTaskMOGP1D0Config, TransferTaskMOGP1D1Config, TransferTaskMOGP1D2Config, TransferTaskMOGP1D3Config, TransferTaskMOGP1D4Config,
            TransferTaskMOGP1D5Config, TransferTaskMOGP1D6Config, TransferTaskMOGP1D7Config, TransferTaskMOGP1D8Config, TransferTaskMOGP1D9Config,
            TransferTaskMOGP1D10Config, TransferTaskMOGP1D11Config, TransferTaskMOGP1D12Config, TransferTaskMOGP1D13Config, TransferTaskMOGP1D14Config,
            TransferTaskMOGP1D15Config, TransferTaskMOGP1D16Config, TransferTaskMOGP1D17Config, TransferTaskMOGP1D18Config, TransferTaskMOGP1D19Config,
            SingleTaskMOGP1DzBaseConfig,
            SingleTaskMOGP1Dz0Config, SingleTaskMOGP1Dz1Config, SingleTaskMOGP1Dz2Config, SingleTaskMOGP1Dz3Config, SingleTaskMOGP1Dz4Config,
            SingleTaskMOGP1Dz5Config, SingleTaskMOGP1Dz6Config, SingleTaskMOGP1Dz7Config, SingleTaskMOGP1Dz8Config, SingleTaskMOGP1Dz9Config,
            SingleTaskMOGP1Dz10Config, SingleTaskMOGP1Dz11Config, SingleTaskMOGP1Dz12Config, SingleTaskMOGP1Dz13Config, SingleTaskMOGP1Dz14Config,
            SingleTaskMOGP1Dz15Config, SingleTaskMOGP1Dz16Config, SingleTaskMOGP1Dz17Config, SingleTaskMOGP1Dz18Config, SingleTaskMOGP1Dz19Config,
            TransferTaskMOGP1DzBaseConfig,
            TransferTaskMOGP1Dz0Config, TransferTaskMOGP1Dz1Config, TransferTaskMOGP1Dz2Config, TransferTaskMOGP1Dz3Config, TransferTaskMOGP1Dz4Config,
            TransferTaskMOGP1Dz5Config, TransferTaskMOGP1Dz6Config, TransferTaskMOGP1Dz7Config, TransferTaskMOGP1Dz8Config, TransferTaskMOGP1Dz9Config,
            TransferTaskMOGP1Dz10Config, TransferTaskMOGP1Dz11Config, TransferTaskMOGP1Dz12Config, TransferTaskMOGP1Dz13Config, TransferTaskMOGP1Dz14Config,
            TransferTaskMOGP1Dz15Config, TransferTaskMOGP1Dz16Config, TransferTaskMOGP1Dz17Config, TransferTaskMOGP1Dz18Config, TransferTaskMOGP1Dz19Config,
            SingleTaskMOGP2DBaseConfig,
            SingleTaskMOGP2D0Config, SingleTaskMOGP2D1Config, SingleTaskMOGP2D2Config, SingleTaskMOGP2D3Config, SingleTaskMOGP2D4Config,
            SingleTaskMOGP2D5Config, SingleTaskMOGP2D6Config, SingleTaskMOGP2D7Config, SingleTaskMOGP2D8Config, SingleTaskMOGP2D9Config,
            SingleTaskMOGP2D10Config, SingleTaskMOGP2D11Config, SingleTaskMOGP2D12Config, SingleTaskMOGP2D13Config, SingleTaskMOGP2D14Config,
            SingleTaskMOGP2D15Config, SingleTaskMOGP2D16Config, SingleTaskMOGP2D17Config, SingleTaskMOGP2D18Config, SingleTaskMOGP2D19Config,
            TransferTaskMOGP2DBaseConfig,
            TransferTaskMOGP2D0Config, TransferTaskMOGP2D1Config, TransferTaskMOGP2D2Config, TransferTaskMOGP2D3Config, TransferTaskMOGP2D4Config,
            TransferTaskMOGP2D5Config, TransferTaskMOGP2D6Config, TransferTaskMOGP2D7Config, TransferTaskMOGP2D8Config, TransferTaskMOGP2D9Config,
            TransferTaskMOGP2D10Config, TransferTaskMOGP2D11Config, TransferTaskMOGP2D12Config, TransferTaskMOGP2D13Config, TransferTaskMOGP2D14Config,
            TransferTaskMOGP2D15Config, TransferTaskMOGP2D16Config, TransferTaskMOGP2D17Config, TransferTaskMOGP2D18Config, TransferTaskMOGP2D19Config,
            SingleTaskMOGP2DzBaseConfig,
            SingleTaskMOGP2Dz0Config, SingleTaskMOGP2Dz1Config, SingleTaskMOGP2Dz2Config, SingleTaskMOGP2Dz3Config, SingleTaskMOGP2Dz4Config,
            SingleTaskMOGP2Dz5Config, SingleTaskMOGP2Dz6Config, SingleTaskMOGP2Dz7Config, SingleTaskMOGP2Dz8Config, SingleTaskMOGP2Dz9Config,
            SingleTaskMOGP2Dz10Config, SingleTaskMOGP2Dz11Config, SingleTaskMOGP2Dz12Config, SingleTaskMOGP2Dz13Config, SingleTaskMOGP2Dz14Config,
            SingleTaskMOGP2Dz15Config, SingleTaskMOGP2Dz16Config, SingleTaskMOGP2Dz17Config, SingleTaskMOGP2Dz18Config, SingleTaskMOGP2Dz19Config,
            TransferTaskMOGP2DzBaseConfig,
            TransferTaskMOGP2Dz0Config, TransferTaskMOGP2Dz1Config, TransferTaskMOGP2Dz2Config, TransferTaskMOGP2Dz3Config, TransferTaskMOGP2Dz4Config,
            TransferTaskMOGP2Dz5Config, TransferTaskMOGP2Dz6Config, TransferTaskMOGP2Dz7Config, TransferTaskMOGP2Dz8Config, TransferTaskMOGP2Dz9Config,
            TransferTaskMOGP2Dz10Config, TransferTaskMOGP2Dz11Config, TransferTaskMOGP2Dz12Config, TransferTaskMOGP2Dz13Config, TransferTaskMOGP2Dz14Config,
            TransferTaskMOGP2Dz15Config, TransferTaskMOGP2Dz16Config, TransferTaskMOGP2Dz17Config, TransferTaskMOGP2Dz18Config, TransferTaskMOGP2Dz19Config,
            SingleTaskMOGP4DzBaseConfig,
            SingleTaskMOGP4Dz0Config, SingleTaskMOGP4Dz1Config, SingleTaskMOGP4Dz2Config, SingleTaskMOGP4Dz3Config, SingleTaskMOGP4Dz4Config,
            SingleTaskMOGP4Dz5Config, SingleTaskMOGP4Dz6Config, SingleTaskMOGP4Dz7Config, SingleTaskMOGP4Dz8Config, SingleTaskMOGP4Dz9Config,
            SingleTaskMOGP4Dz10Config, SingleTaskMOGP4Dz11Config, SingleTaskMOGP4Dz12Config, SingleTaskMOGP4Dz13Config, SingleTaskMOGP4Dz14Config,
            SingleTaskMOGP4Dz15Config, SingleTaskMOGP4Dz16Config, SingleTaskMOGP4Dz17Config, SingleTaskMOGP4Dz18Config, SingleTaskMOGP4Dz19Config,
            TransferTaskMOGP4DzBaseConfig,
            TransferTaskMOGP4Dz0Config, TransferTaskMOGP4Dz1Config, TransferTaskMOGP4Dz2Config, TransferTaskMOGP4Dz3Config, TransferTaskMOGP4Dz4Config,
            TransferTaskMOGP4Dz5Config, TransferTaskMOGP4Dz6Config, TransferTaskMOGP4Dz7Config, TransferTaskMOGP4Dz8Config, TransferTaskMOGP4Dz9Config,
            TransferTaskMOGP4Dz10Config, TransferTaskMOGP4Dz11Config, TransferTaskMOGP4Dz12Config, TransferTaskMOGP4Dz13Config, TransferTaskMOGP4Dz14Config,
            TransferTaskMOGP4Dz15Config, TransferTaskMOGP4Dz16Config, TransferTaskMOGP4Dz17Config, TransferTaskMOGP4Dz18Config, TransferTaskMOGP4Dz19Config,
            SingleTaskMOGP5DzBaseConfig,
            SingleTaskMOGP5Dz0Config, SingleTaskMOGP5Dz1Config, SingleTaskMOGP5Dz2Config, SingleTaskMOGP5Dz3Config, SingleTaskMOGP5Dz4Config,
            SingleTaskMOGP5Dz5Config, SingleTaskMOGP5Dz6Config, SingleTaskMOGP5Dz7Config, SingleTaskMOGP5Dz8Config, SingleTaskMOGP5Dz9Config,
            SingleTaskMOGP5Dz10Config, SingleTaskMOGP5Dz11Config, SingleTaskMOGP5Dz12Config, SingleTaskMOGP5Dz13Config, SingleTaskMOGP5Dz14Config,
            SingleTaskMOGP5Dz15Config, SingleTaskMOGP5Dz16Config, SingleTaskMOGP5Dz17Config, SingleTaskMOGP5Dz18Config, SingleTaskMOGP5Dz19Config,
            TransferTaskMOGP5DzBaseConfig,
            TransferTaskMOGP5Dz0Config, TransferTaskMOGP5Dz1Config, TransferTaskMOGP5Dz2Config, TransferTaskMOGP5Dz3Config, TransferTaskMOGP5Dz4Config,
            TransferTaskMOGP5Dz5Config, TransferTaskMOGP5Dz6Config, TransferTaskMOGP5Dz7Config, TransferTaskMOGP5Dz8Config, TransferTaskMOGP5Dz9Config,
            TransferTaskMOGP5Dz10Config, TransferTaskMOGP5Dz11Config, TransferTaskMOGP5Dz12Config, TransferTaskMOGP5Dz13Config, TransferTaskMOGP5Dz14Config,
            TransferTaskMOGP5Dz15Config, TransferTaskMOGP5Dz16Config, TransferTaskMOGP5Dz17Config, TransferTaskMOGP5Dz18Config, TransferTaskMOGP5Dz19Config,
            SingleTaskEngineInterpolatedBaseConfig,
            SingleTaskEngineInterpolated_be_Config,
            SingleTaskEngineInterpolated_TEx_Config,
            SingleTaskEngineInterpolated_PI0v_Config,
            SingleTaskEngineInterpolated_PI0s_Config,
            SingleTaskEngineInterpolated_HC_Config,
            SingleTaskEngineInterpolated_NOx_Config,
            TransferTaskEngineInterpolatedBaseConfig,
            TransferTaskEngineInterpolated_be_Config,
            TransferTaskEngineInterpolated_TEx_Config,
            TransferTaskEngineInterpolated_PI0v_Config,
            TransferTaskEngineInterpolated_PI0s_Config,
            TransferTaskEngineInterpolated_HC_Config,
            TransferTaskEngineInterpolated_NOx_Config,
            SingleTaskGEngineConfig,
            SingleTaskGEngineTestConfig,
            TransferTaskGEngineConfig,
            TransferTaskGEngineTestConfig,
        ]
    }

    @staticmethod
    def pick_kernel_config(config_class_name):
        return ConfigPicker.kernels_configs_dict[config_class_name]

    @staticmethod
    def pick_mean_config(config_class_name):
        return ConfigPicker.means_configs_dict[config_class_name]

    @staticmethod
    def pick_model_config(config_class_name):
        return ConfigPicker.models_configs_dict[config_class_name]

    @staticmethod
    def pick_acquisition_function_config(config_class_name):
        return ConfigPicker.acquisition_function_configs_dict[config_class_name]

    @staticmethod
    def pick_active_learner_config(config_class_name):
        return ConfigPicker.active_learner_configs_dict[config_class_name]

    @staticmethod
    def pick_batch_active_learner_config(config_class_name):
        return ConfigPicker.batch_active_learner_configs_dict[config_class_name]

    @staticmethod
    def pick_bayesian_optimization_config(config_class_name):
        return ConfigPicker.bayesian_optimization_configs_dict[config_class_name]

    @staticmethod
    def pick_kernel_grammar_generator_config(config_class_name):
        return ConfigPicker.kernel_grammar_generator_configs_dict[config_class_name]

    @staticmethod
    def pick_greedy_kernel_search_config(config_class_name):
        return ConfigPicker.greedy_kernel_seach_configs_dict[config_class_name]

    @staticmethod
    def pick_experiment_simulator_config(config_class_name):
        return ConfigPicker.experiment_simulator_configs_dict[config_class_name]

if __name__ == "__main__":
    print(ConfigPicker.pick_acquisition_function_config("BasicPredEntropyConfig"))
    print(ConfigPicker.pick_active_learner_config("PredEntropyActiveLearnerConfig"))
    print(ConfigPicker.pick_batch_active_learner_config("RandomBatchActiveLearnerConfig"))
    print(ConfigPicker.pick_bayesian_optimization_config("BOExpectedImprovementConfig"))
    print(ConfigPicker.pick_experiment_simulator_config("SingleTaskIllustrateConfig"))
    print(ConfigPicker.pick_greedy_kernel_search_config("BaseGreedyKernelSearchConfig"))
    print(ConfigPicker.pick_kernel_config("HHKEightLocalDefaultConfig")(input_dimension=2))
    print(ConfigPicker.pick_kernel_grammar_generator_config("NDimFullKernelsGrammarGeneratorConfig"))
    print(ConfigPicker.pick_model_config("BasicGPModelConfig"))

