# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import argparse
import dataclasses
import logging
import os
import pickle
import shutil
import sys
import types

import numpy as np
import torch
from ai_models.model import Model
from ai_models.sensitivity import add_sensitivity_parser_arguments
from ai_models.sensitivity import parse_target_area
from ai_models.sensitivity import SensitivityTarget
from ai_models.sensitivity import target_slug
from torch.utils.checkpoint import checkpoint

try:
    import timm.models.layers.helpers as _timm_helpers  # noqa: F401
except ModuleNotFoundError:
    try:
        from timm.layers.helpers import to_2tuple

        helpers_module = types.ModuleType("timm.models.layers.helpers")
        helpers_module.to_2tuple = to_2tuple
        sys.modules.setdefault("timm.models.layers.helpers", helpers_module)
    except Exception:
        pass


LOG = logging.getLogger(__name__)

try:
    from aurora import Batch
    from aurora import Metadata
    from aurora.model.aurora import Aurora
    from aurora.model.aurora import AuroraHighRes
    from huggingface_hub import hf_hub_download
except ModuleNotFoundError as e:
    msg = "You need microsoft-aurora and huggingface_hub installed to use this model."
    LOG.error(msg)
    raise ModuleNotFoundError(f"{msg}\n{e}")


class AuroraModel(Model):
    download_url = "https://huggingface.co/microsoft/aurora/resolve/main/{file}"

    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]

    surf_vars = ("2t", "10u", "10v", "msl")
    atmos_vars = ("z", "u", "v", "t", "q")
    levels = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)

    lagged = (-6, 0)

    param_sfc = surf_vars
    param_level_pl = (atmos_vars, levels)

    ordering = []

    expver = "auro"
    lora = None
    supported_attribution_methods = ("gradient", "integrated-gradients")

    surf_scales = {
        "2t": 2.122036e01,
        "10u": 5.547512e00,
        "10v": 4.765339e00,
        "msl": 1.332246e03,
    }

    atmos_scales = {
        "z": {
            50: 5.875553e03,
            100: 5.510640e03,
            150: 5.823912e03,
            200: 5.820169e03,
            250: 5.536585e03,
            300: 5.091916e03,
            400: 4.150851e03,
            500: 3.353187e03,
            600: 2.695808e03,
            700: 2.136436e03,
            850: 1.470321e03,
            925: 1.228997e03,
            1000: 1.072307e03,
        },
        "u": {
            50: 1.529281e01,
            100: 1.352611e01,
            150: 1.604335e01,
            200: 1.767630e01,
            250: 1.796710e01,
            300: 1.711917e01,
            400: 1.434276e01,
            500: 1.198419e01,
            600: 1.033421e01,
            700: 9.168821e00,
            850: 8.188043e00,
            925: 7.940808e00,
            1000: 6.141778e00,
        },
        "v": {
            50: 7.058931e00,
            100: 7.479310e00,
            150: 9.571990e00,
            200: 1.188069e01,
            250: 1.338039e01,
            300: 1.334044e01,
            400: 1.122955e01,
            500: 9.181708e00,
            600: 7.803569e00,
            700: 6.871040e00,
            850: 6.264443e00,
            925: 6.470644e00,
            1000: 5.308203e00,
        },
        "t": {
            50: 1.026284e01,
            100: 1.252901e01,
            150: 8.928709e00,
            200: 7.189547e00,
            250: 8.529282e00,
            300: 1.071679e01,
            400: 1.269102e01,
            500: 1.306447e01,
            600: 1.342046e01,
            700: 1.476523e01,
            850: 1.558880e01,
            925: 1.608798e01,
            1000: 1.713983e01,
        },
        "q": {
            50: 3.571687e-07,
            100: 5.703754e-07,
            150: 3.794077e-06,
            200: 2.267534e-05,
            250: 7.446644e-05,
            300: 1.684361e-04,
            400: 5.078644e-04,
            500: 1.079294e-03,
            600: 1.769722e-03,
            700: 2.549169e-03,
            850: 4.112368e-03,
            925: 5.071058e-03,
            1000: 5.913548e-03,
        },
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ordering = list(self.surf_vars) + [
            f"{param}{level}" for param in self.atmos_vars for level in self.levels
        ]
        self.level_to_index = {int(level): index for index, level in enumerate(self.levels)}
        self._sensitivity_latitudes = None
        self._sensitivity_longitudes = None
        self._warned_rollout_checkpointing = False

        default_sensitivity_path = f"{self.__class__.__name__.lower()}-sensitivity-{self.lead_time:03d}h.nc"
        self.init_sensitivity("aurora", default_sensitivity_path)

    def download_assets(self, **kwargs):
        del kwargs
        for filename in self.download_files:
            asset_path = os.path.realpath(os.path.join(self.assets, filename))
            if os.path.exists(asset_path):
                continue

            os.makedirs(os.path.dirname(asset_path), exist_ok=True)
            LOG.info("Downloading %s", asset_path)
            cache_path = hf_hub_download(repo_id="microsoft/aurora", filename=filename)
            shutil.copy2(cache_path, asset_path)

    def parse_model_args(self, args):
        parser = argparse.ArgumentParser(add_help=False)
        add_sensitivity_parser_arguments(parser)
        parser.add_argument(
            "--lora",
            type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
            nargs="?",
            const=True,
            default=None,
            help="Use LoRA model (true/false). Default depends on the model.",
        )
        parser.add_argument(
            "--model-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable Aurora activation checkpointing. Defaults to on in sensitivity mode.",
        )
        parser.add_argument(
            "--rollout-checkpointing",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Enable rollout checkpointing. Reserved for consistency with shared sensitivity options.",
        )
        return parser.parse_args(args)

    @property
    def latitudes(self):
        if self._sensitivity_latitudes is not None:
            return self._sensitivity_latitudes
        return np.asarray(self.fields_sfc[0].metadata("distinctLatitudes"), dtype=np.float32)

    @property
    def longitudes(self):
        if self._sensitivity_longitudes is not None:
            return self._sensitivity_longitudes
        values = np.asarray(self.fields_sfc[0].metadata("distinctLongitudes"), dtype=np.float32)
        return np.mod(values, 360.0)

    def parse_target_field(self, target_field=None, target_param=None, target_level=None):
        if target_field and target_param:
            raise ValueError("Use either --target-field or --target-param, not both")

        if target_field:
            if target_field not in self.ordering:
                raise ValueError(
                    f"Unknown target field '{target_field}'. Expected one of: {', '.join(self.ordering)}"
                )
            return target_field

        if target_param is None:
            if target_level is not None:
                raise ValueError("--target-level requires --target-param")
            return None

        if target_param in self.surf_vars:
            if target_level is not None:
                raise ValueError(f"Surface target '{target_param}' does not take --target-level")
            return target_param

        if target_param in self.atmos_vars:
            if target_level is None:
                raise ValueError(f"Pressure-level target '{target_param}' requires --target-level")
            if target_level not in self.levels:
                raise ValueError(
                    f"Unsupported level {target_level} for '{target_param}'. Expected one of: {self.levels}"
                )
            return f"{target_param}{target_level}"

        raise ValueError(f"Unsupported target parameter '{target_param}'")

    def default_target(self):
        field = self.parse_target_field(self.target_field, self.target_param, self.target_level)
        area = parse_target_area(self.target_area)
        if area is not None and field is None:
            raise ValueError("--target-area requires --target-field or --target-param")

        name = target_slug(field, "full-state")
        return SensitivityTarget(name=name, field=field, area=area, metric=self.sensitivity_metric)

    def config_targets(self, config):
        targets = config.get("targets")
        if not targets:
            return [self.default_target()]

        parsed_targets = []
        for index, target in enumerate(targets):
            if not isinstance(target, dict):
                raise ValueError("Each target entry must be a mapping")

            field = self.parse_target_field(
                target_field=target.get("field"),
                target_param=target.get("param"),
                target_level=target.get("level"),
            )
            area = parse_target_area(target.get("area"))
            if area is not None and field is None:
                raise ValueError("Target area requires a specific target field")

            metric = target.get("metric", self.sensitivity_metric)
            if metric not in ("mean", "mean-square"):
                raise ValueError(f"Unsupported metric '{metric}' for target {index + 1}")

            name = target_slug(target.get("name"), field or f"target-{index + 1}")
            parsed_targets.append(SensitivityTarget(name=name, field=field, area=area, metric=metric))

        return parsed_targets

    @staticmethod
    def _pressure_field_parts(field):
        for param in ("z", "u", "v", "t", "q"):
            if field.startswith(param):
                level_text = field[len(param) :]
                if level_text.isdigit():
                    return param, int(level_text)
                break
        raise ValueError(f"Could not parse pressure-level field '{field}'")

    def channel_to_variable_level(self, field):
        if field in self.surf_vars:
            return ("surf", field, None)

        param, level = self._pressure_field_parts(field)
        if param not in self.atmos_vars:
            raise ValueError(f"Unsupported atmospheric variable '{param}'")
        if level not in self.level_to_index:
            raise ValueError(f"Unsupported pressure level '{level}' for field '{field}'")
        return ("atmos", param, level)

    def load_model_instance(self, use_lora):
        LOG.info("Model is %s, use_lora=%s", self.__class__.__name__, use_lora)
        model = self.klass(use_lora=use_lora)
        model = model.to(self.device)

        checkpoint_path = os.path.join(self.assets, os.path.basename(self.checkpoint))
        if os.path.exists(checkpoint_path):
            LOG.info("Loading Aurora model from %s", checkpoint_path)
            model.load_checkpoint_local(checkpoint_path, strict=False)
        else:
            LOG.info("Downloading Aurora model %s", self.checkpoint)
            model.load_checkpoint("microsoft/aurora", self.checkpoint, strict=False)

        if self.model_checkpointing and hasattr(model, "configure_activation_checkpointing"):
            LOG.info("Model checkpointing enabled: configuring Aurora activation checkpointing")
            model.configure_activation_checkpointing()

        LOG.info("Loading Aurora model to device %s", self.device)
        model = model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        return model

    def load_static_vars(self):
        path = os.path.join(self.assets, self.download_files[0])
        if not os.path.exists(path):
            cache_path = hf_hub_download(repo_id="microsoft/aurora", filename=self.download_files[0])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            shutil.copy2(cache_path, path)

        with open(path, "rb") as file_handle:
            static_vars = pickle.load(file_handle)

        return {name: torch.from_numpy(values.astype(np.float32)) for name, values in static_vars.items()}

    def build_input_batch(self):
        fields_pl = self.fields_pl
        fields_sfc = self.fields_sfc

        nj, ni = fields_pl[0].shape
        templates = {}
        surf_inputs = {}

        for param in self.surf_vars:
            field = fields_sfc.sel(param=param).order_by(valid_datetime="ascending")
            templates[param] = field[-1]
            values = field.to_numpy(dtype=np.float32)
            surf_inputs[param] = torch.from_numpy(values).unsqueeze(0)

        static_vars = self.load_static_vars()

        atmos_inputs = {}
        for param in self.atmos_vars:
            field = fields_pl.sel(param=param).order_by(valid_datetime="ascending", level=self.levels)
            for level in self.levels:
                templates[(param, level)] = field.sel(level=level)[-1]

            values = field.to_numpy(dtype=np.float32).reshape(len(self.lagged), len(self.levels), nj, ni)
            atmos_inputs[param] = torch.from_numpy(values).unsqueeze(0)

        north, west, south, east = self.area
        metadata = Metadata(
            lat=torch.linspace(north, south, nj),
            lon=torch.linspace(west, east, ni),
            time=(self.start_datetime,),
            atmos_levels=self.levels,
        )

        batch = Batch(
            surf_vars=surf_inputs,
            static_vars=static_vars,
            atmos_vars=atmos_inputs,
            metadata=metadata,
        )

        return batch, templates, nj, ni

    def nan_extend(self, data):
        return np.concatenate(
            (data, np.full_like(data[[-1], :], np.nan, dtype=data.dtype)),
            axis=0,
        )

    def write_prediction_batch(self, prediction, templates, step, nj, ni):
        for param, values in prediction.surf_vars.items():
            data = np.squeeze(values.detach().cpu().numpy())
            data = self.nan_extend(data)
            assert data.shape == (nj, ni)
            self.write(data, template=templates[param], step=step, check_nans=True)

        for param, values in prediction.atmos_vars.items():
            tensor = values.detach().cpu().numpy()
            for level_index, level in enumerate(self.levels):
                data = np.squeeze(tensor[:, :, level_index])
                data = self.nan_extend(data)
                assert data.shape == (nj, ni)
                self.write(data, template=templates[(param, level)], step=step, check_nans=True)

    def forecast_step_count(self):
        if self.lead_time <= 0:
            raise ValueError(f"lead_time must be positive, got {self.lead_time}")
        if self.lead_time % 6 != 0:
            raise ValueError(f"For Aurora, lead_time must be a multiple of 6 hours; got {self.lead_time}")
        return self.lead_time // 6

    @staticmethod
    def advance_rollout(current_batch, prediction):
        device = next(iter(prediction.surf_vars.values())).device
        current_batch = current_batch.to(device)
        return dataclasses.replace(
            prediction,
            surf_vars={
                name: torch.cat([current_batch.surf_vars[name][:, 1:], values], dim=1)
                for name, values in prediction.surf_vars.items()
            },
            atmos_vars={
                name: torch.cat([current_batch.atmos_vars[name][:, 1:], values], dim=1)
                for name, values in prediction.atmos_vars.items()
            },
        )

    def model_step(self, model, batch):
        if self.rollout_checkpointing and torch.is_grad_enabled():
            if not self._warned_rollout_checkpointing:
                LOG.info("Rollout checkpointing enabled: rematerializing each Aurora rollout step")
                self._warned_rollout_checkpointing = True

            def _forward(batch_value):
                return model(batch_value)

            return checkpoint(_forward, batch, use_reentrant=False)

        return model(batch)

    @staticmethod
    def _target_weights(latitudes, longitudes, area, *, device, dtype):
        latitudes = torch.as_tensor(latitudes, dtype=dtype, device=device)
        longitudes = torch.as_tensor(np.mod(longitudes, 360.0), dtype=dtype, device=device)

        lat_weights = torch.cos(torch.deg2rad(latitudes)).clamp_min(0)
        lat_mask = torch.ones_like(lat_weights, dtype=torch.bool)
        lon_mask = torch.ones_like(longitudes, dtype=torch.bool)

        if area is not None:
            north, west, south, east = area
            lat_mask = (latitudes <= north) & (latitudes >= south)
            if west <= east:
                lon_mask = (longitudes >= west) & (longitudes <= east)
            else:
                lon_mask = (longitudes >= west) | (longitudes <= east)

        weights = lat_weights[:, None] * lat_mask.to(dtype)[:, None] * lon_mask.to(dtype)[None, :]
        total = weights.sum()
        if total <= 0:
            raise ValueError("Selected target area does not overlap the Aurora grid")
        return weights / total

    def _target_tensor(self, prediction, field):
        kind, param, level = self.channel_to_variable_level(field)
        if kind == "surf":
            return prediction.surf_vars[param][0, -1]

        level_index = self.level_to_index[int(level)]
        return prediction.atmos_vars[param][0, -1, level_index]

    def sensitivity_objective(self, prediction, target):
        metric = target.metric
        if metric not in ("mean", "mean-square"):
            raise ValueError(f"Unsupported sensitivity metric: {metric}")

        if target.field is None:
            total_sum = torch.zeros((), dtype=torch.float32, device=self.device)
            total_count = 0

            for values in prediction.surf_vars.values():
                field_values = values[:, -1]
                if metric == "mean":
                    total_sum = total_sum + field_values.sum()
                else:
                    total_sum = total_sum + field_values.square().sum()
                total_count += int(field_values.numel())

            for values in prediction.atmos_vars.values():
                field_values = values[:, -1]
                if metric == "mean":
                    total_sum = total_sum + field_values.sum()
                else:
                    total_sum = total_sum + field_values.square().sum()
                total_count += int(field_values.numel())

            if total_count <= 0:
                raise ValueError("Aurora output contained no values for sensitivity objective")
            return total_sum / float(total_count)

        field_values = self._target_tensor(prediction, target.field)

        # Aurora returns unnormalized outputs. For consistent sensitivity scaling
        # with other plugins, compute objectives in normalized units.
        kind, param, level = self.channel_to_variable_level(target.field)
        if kind == "surf":
            scale = self.surf_scales[param]
        else:
            scale = self.atmos_scales[param][int(level)]
        field_values = field_values / float(scale)
        if target.area is None:
            if metric == "mean":
                return field_values.mean()
            return field_values.square().mean()

        latitudes = prediction.metadata.lat.detach().cpu().numpy().astype(np.float32)
        longitudes = prediction.metadata.lon.detach().cpu().numpy().astype(np.float32)
        weights = self._target_weights(
            latitudes,
            longitudes,
            target.area,
            device=field_values.device,
            dtype=field_values.dtype,
        )
        if metric == "mean":
            return (field_values * weights).sum()
        return (field_values.square() * weights).sum()

    def gradients_to_channel_maps(self, gradients, lag_zero_index):
        channel_maps = []
        surf_count = len(self.surf_vars)

        for index in range(surf_count):
            gradient = gradients[index]
            channel_maps.append(gradient[0, lag_zero_index].detach().cpu().numpy().astype(np.float32))

        for index in range(len(self.atmos_vars)):
            gradient = gradients[surf_count + index]
            for level_index in range(len(self.levels)):
                channel_maps.append(
                    gradient[0, lag_zero_index, level_index].detach().cpu().numpy().astype(np.float32)
                )

        return np.stack(channel_maps, axis=0)

    @staticmethod
    def _clone_batch_to_device(batch, device):
        return Batch(
            surf_vars={name: tensor.to(device).detach().clone() for name, tensor in batch.surf_vars.items()},
            static_vars={name: tensor.to(device) for name, tensor in batch.static_vars.items()},
            atmos_vars={name: tensor.to(device).detach().clone() for name, tensor in batch.atmos_vars.items()},
            metadata=Metadata(
                lat=batch.metadata.lat.to(device),
                lon=batch.metadata.lon.to(device),
                time=batch.metadata.time,
                atmos_levels=batch.metadata.atmos_levels,
                rollout_step=batch.metadata.rollout_step,
            ),
        )

    @staticmethod
    def _batch_with_requires_grad(batch):
        return Batch(
            surf_vars={name: tensor.detach().clone().requires_grad_(True) for name, tensor in batch.surf_vars.items()},
            static_vars=batch.static_vars,
            atmos_vars={name: tensor.detach().clone().requires_grad_(True) for name, tensor in batch.atmos_vars.items()},
            metadata=batch.metadata,
        )

    def _batch_delta(self, batch, baseline):
        return Batch(
            surf_vars={name: batch.surf_vars[name] - baseline.surf_vars[name] for name in self.surf_vars},
            static_vars=baseline.static_vars,
            atmos_vars={name: batch.atmos_vars[name] - baseline.atmos_vars[name] for name in self.atmos_vars},
            metadata=batch.metadata,
        )

    def _batch_add_scaled(self, baseline, delta, alpha):
        return Batch(
            surf_vars={
                name: baseline.surf_vars[name] + alpha * delta.surf_vars[name]
                for name in self.surf_vars
            },
            static_vars=baseline.static_vars,
            atmos_vars={
                name: baseline.atmos_vars[name] + alpha * delta.atmos_vars[name]
                for name in self.atmos_vars
            },
            metadata=baseline.metadata,
        )

    def _zero_baseline_batch(self, batch):
        return Batch(
            surf_vars={name: torch.zeros_like(tensor) for name, tensor in batch.surf_vars.items()},
            static_vars=batch.static_vars,
            atmos_vars={name: torch.zeros_like(tensor) for name, tensor in batch.atmos_vars.items()},
            metadata=batch.metadata,
        )

    def _climatology_baseline_batch(self, batch):
        means = torch.zeros((len(self.ordering),), dtype=torch.float32, device=self.device)
        for index, field in enumerate(self.ordering):
            kind, param, level = self.channel_to_variable_level(field)
            if kind == "surf":
                tensor = batch.surf_vars[param]
            else:
                level_index = self.level_to_index[int(level)]
                tensor = batch.atmos_vars[param][:, :, level_index]
            means[index] = tensor.mean()

        surf_baseline = {}
        for param in self.surf_vars:
            channel_index = self.ordering.index(param)
            surf_baseline[param] = torch.ones_like(batch.surf_vars[param]) * means[channel_index]

        atmos_baseline = {}
        for param in self.atmos_vars:
            channels = [self.ordering.index(f"{param}{level}") for level in self.levels]
            channel_means = means[channels].view(1, 1, len(self.levels), 1, 1)
            atmos_baseline[param] = torch.ones_like(batch.atmos_vars[param]) * channel_means

        return Batch(
            surf_vars=surf_baseline,
            static_vars=batch.static_vars,
            atmos_vars=atmos_baseline,
            metadata=batch.metadata,
        )

    def integrated_gradients_baseline(self, batch):
        baseline_mode = getattr(self, "ig_baseline", "zero")
        if baseline_mode == "zero":
            return self._zero_baseline_batch(batch)
        if baseline_mode == "climatology":
            return self._climatology_baseline_batch(batch)
        raise ValueError(f"Unsupported integrated gradients baseline: {baseline_mode}")

    def _objective_from_batch(self, model, current_batch, forecast_steps, target):
        current = current_batch.crop(model.patch_size)
        objective_batch = None

        for _ in range(forecast_steps):
            prediction = self.model_step(model, current)
            objective_batch = prediction
            current = self.advance_rollout(current, prediction)

        if objective_batch is None:
            raise ValueError("Sensitivity rollout produced no prediction steps")

        self.sensitivity_manager.current_target = target
        return self.sensitivity_manager.objective(objective_batch, target)

    def integrated_gradients(self, model, input_batch, forecast_steps, target):
        steps = int(self.ig_steps)
        baseline = self.integrated_gradients_baseline(input_batch)
        delta = self._batch_delta(input_batch, baseline)

        surf_accum = {name: torch.zeros_like(delta.surf_vars[name]) for name in self.surf_vars}
        atmos_accum = {name: torch.zeros_like(delta.atmos_vars[name]) for name in self.atmos_vars}

        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            interpolated = self._batch_add_scaled(baseline, delta, alpha)
            interpolated = self._batch_with_requires_grad(interpolated)

            objective = self._objective_from_batch(model, interpolated, forecast_steps, target)

            grad_inputs = [interpolated.surf_vars[name] for name in self.surf_vars] + [
                interpolated.atmos_vars[name] for name in self.atmos_vars
            ]
            grads = torch.autograd.grad(objective, grad_inputs)

            for idx, name in enumerate(self.surf_vars):
                surf_accum[name] = surf_accum[name] + grads[idx]
            base_idx = len(self.surf_vars)
            for idx, name in enumerate(self.atmos_vars):
                atmos_accum[name] = atmos_accum[name] + grads[base_idx + idx]

        surf_attr = {
            name: delta.surf_vars[name] * (surf_accum[name] / float(steps))
            for name in self.surf_vars
        }
        atmos_attr = {
            name: delta.atmos_vars[name] * (atmos_accum[name] / float(steps))
            for name in self.atmos_vars
        }

        return Batch(
            surf_vars=surf_attr,
            static_vars=input_batch.static_vars,
            atmos_vars=atmos_attr,
            metadata=input_batch.metadata,
        )

    def run_forecast(self, model, batch, templates, nj, ni):
        forecast_steps = self.forecast_step_count()
        current_batch = batch.crop(model.patch_size)

        with torch.inference_mode():
            with self.stepper(6) as stepper:
                for index in range(forecast_steps):
                    prediction = self.model_step(model, current_batch)
                    step = (index + 1) * 6
                    self.write_prediction_batch(prediction, templates, step, nj, ni)
                    current_batch = self.advance_rollout(current_batch, prediction)
                    stepper(index, step)

    def run_sensitivity(self, model, batch, templates, nj, ni):
        forecast_steps = self.forecast_step_count()
        lag_zero_index = list(self.lagged).index(0) if 0 in self.lagged else len(self.lagged) - 1

        base_batch = self._clone_batch_to_device(batch, self.device)
        base_batch = base_batch.crop(model.patch_size)
        current_batch = self._batch_with_requires_grad(base_batch)
        gradient_inputs = current_batch

        self._sensitivity_latitudes = base_batch.metadata.lat.detach().cpu().numpy().astype(np.float32)
        self._sensitivity_longitudes = np.mod(
            base_batch.metadata.lon.detach().cpu().numpy().astype(np.float32),
            360.0,
        )

        objective_batch = None
        with self.stepper(6) as stepper:
            for index in range(forecast_steps):
                prediction = self.model_step(model, current_batch)
                step = (index + 1) * 6
                self.write_prediction_batch(prediction, templates, step, nj, ni)
                objective_batch = prediction
                current_batch = self.advance_rollout(current_batch, prediction)
                stepper(index, step)

        if objective_batch is None:
            raise ValueError("Sensitivity rollout produced no prediction steps")

        objectives = []
        for target in self.sensitivity_manager.targets:
            self.sensitivity_manager.current_target = target
            objectives.append(self.sensitivity_manager.objective(objective_batch, target))

        gradients = []
        objective_values = []
        if self.attribution_method == "gradient":
            input_tensors = [gradient_inputs.surf_vars[name] for name in self.surf_vars] + [
                gradient_inputs.atmos_vars[name] for name in self.atmos_vars
            ]
            for index, objective in enumerate(objectives):
                gradient_tensors = torch.autograd.grad(
                    objective,
                    input_tensors,
                    retain_graph=(index < len(objectives) - 1),
                )
                gradients.append(self.gradients_to_channel_maps(gradient_tensors, lag_zero_index))
                objective_values.append(float(objective.detach().cpu()))
        elif self.attribution_method == "integrated-gradients":
            for target in self.sensitivity_manager.targets:
                self.sensitivity_manager.current_target = target
                attribution_batch = self.integrated_gradients(model, base_batch, forecast_steps, target)
                objective = self._objective_from_batch(model, base_batch, forecast_steps, target)

                tensors = [attribution_batch.surf_vars[name] for name in self.surf_vars] + [
                    attribution_batch.atmos_vars[name] for name in self.atmos_vars
                ]
                gradients.append(self.gradients_to_channel_maps(tensors, lag_zero_index))
                objective_values.append(float(objective.detach().cpu()))
        else:
            raise ValueError(f"Unsupported attribution method: {self.attribution_method}")

        stacked_gradients = np.stack(gradients, axis=0)
        self.sensitivity_manager.save(stacked_gradients, objective_values)

    def run(self):
        if self.sensitivity_config and not self.sensitivity:
            self.sensitivity = True

        use_lora = self.lora if self.lora is not None else self.use_lora
        model = self.load_model_instance(use_lora=use_lora)

        self.write_input_fields(self.fields_pl + self.fields_sfc)
        batch, templates, nj, ni = self.build_input_batch()

        LOG.info("Starting inference")
        if self.sensitivity:
            self.run_sensitivity(model, batch, templates, nj, ni)
            return

        self.run_forecast(model, batch, templates, nj, ni)


class Aurora0p25(AuroraModel):
    expver = "au25"

    klass = Aurora
    download_files = ("aurora-0.25-static.pickle",)
    area = [90, 0, -90, 360 - 0.25]
    grid = [0.25, 0.25]


class Aurora0p25Pretrained(Aurora0p25):
    use_lora = False
    checkpoint = "aurora-0.25-pretrained.ckpt"


class Aurora0p25FineTuned(Aurora0p25):
    use_lora = True
    checkpoint = "aurora-0.25-finetuned.ckpt"

    def patch_retrieve_request(self, request):
        if request.get("class", "od") != "od":
            return

        if request.get("type", "an") not in ("an", "fc"):
            return

        if request.get("stream", "oper") not in ("oper", "scda"):
            return

        request["type"] = "fc"

        time = request.get("time", 12)
        request["stream"] = {
            0: "oper",
            6: "scda",
            12: "oper",
            18: "scda",
        }[time]


class Aurora0p1FineTuned(AuroraModel):
    download_files = ("aurora-0.1-static.pickle",)
    area = [90, 0, -90, 360 - 0.1]
    grid = [0.1, 0.1]

    klass = AuroraHighRes
    use_lora = True
    checkpoint = "aurora-0.1-finetuned.ckpt"


def model(model_version, **kwargs):
    models = {
        "0.25-pretrained": Aurora0p25Pretrained,
        "0.25-finetuned": Aurora0p25FineTuned,
        "0.1-finetuned": Aurora0p1FineTuned,
        "default": Aurora0p1FineTuned,
        "latest": Aurora0p1FineTuned,
    }

    if model_version not in models:
        LOG.error("Model version %s not found, using default", model_version)
        LOG.error("Available models: %s", list(models.keys()))
        raise ValueError(f"Model version {model_version} not found")

    return models[model_version](**kwargs)
