import numpy as np
import torch
import time
from typing import Tuple, List, Union, Optional
from pathlib import Path
from alef.active_learners.amortized_policies.nn.aline import load_config_and_model, calculate_gmm_variance, create_target_mask, select_targets_by_mask, compute_ll
from alef.enums.active_learner_enums import ValidationType
from alef.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot
from attrdictionary import AttrDict

from .base_active_learners import BasePoolActiveLearner

class PoolALINE(BasePoolActiveLearner):

    def __init__(
        self,
        validation_type: Union[ValidationType, List[ValidationType]],
        policy_path: str='',
        validation_at: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self.validation_type = validation_type
        self.validation_at = validation_at
        if isinstance(self.validation_type, ValidationType):
            assert not self.validation_type == ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
            _metric_names = [self.validation_type.name]
        else:
            assert not any([v == ValidationType.RMSE_MULTIOUTPUT for v in self.validation_type]), NotImplementedError
            _metric_names = [v.name for v in self.validation_type]
        self.initialize_validation_metrics(metric_name_list=[*_metric_names, 'inference_time', 'query_time', 'validate_time'])
        self.load_policy(policy_path)
        self.inference_time = {}
        self.query_time = {}

    def load_policy(self, path):
        model_path = Path(path)
        folder = model_path.parent
        file_name = model_path.name
        cfg, model = load_config_and_model(
            path=folder,
            file_name=file_name,
            device='cpu',
        )
        self.policy = model
        self.policy.eval()
        self.policy_config = cfg

    def run_aline(self, x_data: np.ndarray, y_data: np.ndarray, x_pool: np.ndarray, current_step: int, total_steps: int):
        cfg = self.policy_config

        X = torch.from_numpy(x_data).to(torch.get_default_dtype()) # [N, D]
        Y = torch.from_numpy(y_data).to(torch.get_default_dtype()) # [N, 1]
        Xq = torch.from_numpy(x_pool).to(torch.get_default_dtype()) # [N_pool, D]
        Xtest = torch.from_numpy(self.x_test).to(torch.get_default_dtype()) # [N_test, D]
        Ytest = torch.from_numpy(self.y_test).to(torch.get_default_dtype()) # [N_test, 1]
        n_target_theta = X.shape[-1] + 1
        target_mask = create_target_mask(
            cfg.task.mask_type[0],
            cfg.task.embedding_type,
            Xtest.shape[0],
            n_target_theta,
            None,
            None,
            None,
            None,
            attend_to="data",
        )

        batch = AttrDict(
            context_x=X.unsqueeze(0), # [1, N, D]
            context_y=Y.unsqueeze(0), # [1, N, 1]
            query_x=Xq.unsqueeze(0), # [1, N_pool, D]
            target_x=Xtest.unsqueeze(0), # [1, N_test, D]
            target_y=Ytest.unsqueeze(0), # [1, N_test, 1]
            target_theta=torch.zeros(1, n_target_theta, 1), # [1, D + 1, 1]
            target_mask=target_mask,
        )
        batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1) # [1, N_test + D + 1, 1]
        if cfg.time_token:
            batch.t = torch.tensor([current_step/total_steps])

        outs = self.policy.forward(batch)

        design_out = outs.design_out
        posterior_attr = outs.posterior_out

        query = x_pool[design_out.idx.squeeze().cpu().numpy(), :] # [D]

        posterior_attr.observations = batch.target_all # to compute ll on test set
        posterior_attr.target_mask = batch.target_mask # to compute validation values on test set
        return query, posterior_attr

    def update_gp_model(self):
        raise NotImplementedError

    def learn(self, n_steps: Union[int, List[int]]):
        """
        Arguments:
            n_steps : number of active learning iteration/number of queries
                if a list is given, query n_steps[i] consecutively, 
                this would be different from providing one single sum(n_steps) if the policy takes budget input
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        if self.do_plotting and self.pool.get_dimension() <=2:
            if not self.plot_data_available:
                self.set_plot_data(*self.get_plot_data())
        self.n_steps = sum(n_steps) if hasattr(n_steps, '__len__') else n_steps
        n_steps_iter = iter(n_steps) if hasattr(n_steps, '__len__') else iter([n_steps])
        budget = next(n_steps_iter)
        for i in range(0, self.n_steps):
            validate_this_iter = self.validation_at is None or len(self.validation_at)==0 or i in self.validation_at

            t1 = time.perf_counter()
            query, posterior_attr = self.run_aline(self.x_data, self.y_data, self.pool.possible_queries(), i, self.n_steps)
            t2 = time.perf_counter()
            budget = budget - 1 if budget > 1 or i == self.n_steps - 1 else next(n_steps_iter)
            self.inference_time[i] = 0
            self.query_time[i] = t2 - t1

            print(f"Iter {i}: Query")
            print(query)
            new_y = self.pool.query(query)
            if self.do_plotting and self.pool.get_dimension() <=2 and validate_this_iter:
                self.plot(query, new_y, i)
            self.add_train_point(i, query, new_y)
            if validate_this_iter:
                self.validate(i, posterior_attr)
            else:
                self.empty_validate(i) # timer can still be recorded
        if self.save_result:
            self.save_experiment_summary()
        return self.validation_metrics, self.x_data

    def empty_validate(self, idx: int):
        metrics = super().compute_empty_validation_on_y()
        self.add_validation_value(idx, [*metrics, self.inference_time[idx], self.query_time[idx], 0.0])

    def compute_validation_on_y(self, posterior_attr: AttrDict):
        r"""
        :return: [metric values]
        """
        def compute_validation_one_type(v_type: ValidationType, posterior_attr: AttrDict):
            if v_type == ValidationType.MAE:
                weighted_means = torch.sum(
                    posterior_attr.mixture_means * posterior_attr.mixture_weights,
                    dim=-1
                )  # [batch_size, n_targets]
                abs_errors = (posterior_attr.observations.squeeze(-1) - weighted_means).abs()  # [batch_size, n_targets]
                masked_abs_errors = select_targets_by_mask(abs_errors, posterior_attr.target_mask)  # [batch_size, selected_targets]
                return [masked_abs_errors.mean().item()]
            elif v_type == ValidationType.RMSE:
                weighted_means = torch.sum(
                    posterior_attr.mixture_means * posterior_attr.mixture_weights,
                    dim=-1
                )  # [batch_size, n_targets]
                squared_errors = (posterior_attr.observations.squeeze(-1) - weighted_means) ** 2  # [batch_size, n_targets]
                masked_squared_errors = select_targets_by_mask(squared_errors, posterior_attr.target_mask)  # [batch_size, selected_targets]
                return [masked_squared_errors.mean().sqrt().item()]
            elif v_type == ValidationType.NEG_LOG_LIKELI:
                target_ll = compute_ll(
                    posterior_attr.observations,
                    posterior_attr.mixture_means,
                    posterior_attr.mixture_stds,
                    posterior_attr.mixture_weights
                )  # [batch_size, n_targets]
                masked_target_ll = select_targets_by_mask(target_ll, posterior_attr.target_mask)  # [batch_size, selected_targets]
                return [-1 * masked_target_ll.mean().item()]
            else:
                raise NotImplementedError(f"Validation type {v_type} not implemented.")

        if isinstance(self.validation_type, ValidationType):
            return compute_validation_one_type(self.validation_type, posterior_attr)
        else: # self.validation_type: List[ValidationType]
            metrics = []
            for v in self.validation_type:
                assert not v==ValidationType.RMSE_MULTIOUTPUT, NotImplementedError
                metrics.append(*compute_validation_one_type(v, posterior_attr))
            return metrics

    def validate(self, idx: int, posterior_attr: AttrDict):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used
        posterior_attr contains the predictive GMM parameter tensors on the test set
        """
        print('Validate')

        t_start = time.perf_counter()
        metrics = self.compute_validation_on_y(posterior_attr)
        t_end = time.perf_counter()
        validate_time = t_end - t_start

        self.add_validation_value(idx, [*metrics, self.inference_time[idx], self.query_time[idx], validate_time])

    def plot(self, query: np.array, new_y: float, step: int):
        raise NotImplementedError
        dimension = self.pool.get_dimension()
        x_plot, y_plot = self.get_plot_data()
        if dimension == 1:
            pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
            plot_name = "query_" + str(step) + ".png"
            active_learning_1d_plot(
                x_plot, pred_mu, pred_sigma, self.x_data, self.y_data, query, new_y, True, x_plot, y_plot,
                save_plot=self.save_plots,
                file_name=plot_name,
                file_path=self.plot_path,
            )
        elif dimension == 2:
            pred_mu, pred_sigma = self.model.predictive_dist(x_plot)
            acquisition_function = self.model.entropy_predictive_dist(x_plot)
            plot_name = "query_" + str(step) + ".png"
            active_learning_2d_plot(
                x_plot,
                acquisition_function,
                pred_mu,
                y_plot,
                self.x_data,
                query,
                save_plot=self.save_plots,
                file_name=plot_name,
                file_path=self.plot_path,
            )
        else:
            print("Dimension too high for plotting")
